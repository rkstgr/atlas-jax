"""Minimal JAX port of Gated DeltaNet (Yang 2024, arXiv 2412.06464).

Purpose: reference linear-attention baseline for MQAR in JAX, mirroring the
FLA reference. If this passes our JAX MQAR bench, we confirm the bench +
training regime are correct in JAX and isolate any remaining issue to the
Atlas layer specifically.

This is NOT intended as a production GDN — no chunked parallelism, no Triton.
A plain recurrent `lax.scan` over tokens. Correctness > speed.

Key architectural choices (ported from fla.layers.gated_deltanet):
  - QKV projections + ShortConv(kernel=4) + SiLU activation
  - L2-normalized Q, K per head
  - beta = sigmoid(b_proj(x))                      — per-step delta-rule scale
  - g = -exp(A_log) · softplus(a_proj(x) + dt_bias) — per-step forget gate (Mamba2 init)
  - Delta rule: S_t = exp(g_t)·S_{t-1} + beta_t·(v_t - S_{t-1}·k_t)·k_t^T
  - Retrieval: o_t = S_t · q_t
  - Gated RMSNorm output: norm(o) * silu(g_out)
  - MLP (SwiGLU) as channel mixer

Usage:
    python bench/jax_gdn.py [--n-train 100000 --epochs 16 --batch-size 256 ...]
Pass: test accuracy ≥ 0.95.
"""

import argparse, math, sys, time
import numpy as np
import jax, jax.numpy as jnp, jax.lax as lax
import equinox as eqx
import optax

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.mqar import generate_mqar, masked_loss, batch_accuracy, evaluate
from atlas_jax.model import rms_norm, layer_norm, gelu_derivative


# =============================================================================
# Stateless helpers.
# =============================================================================

def l2_norm(x, eps=1e-6):
    """L2 normalize over the last axis."""
    return x * lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


class ShortConvSiLU(eqx.Module):
    """Causal depthwise 1D conv followed by SiLU (matches FLA's ShortConvolution)."""
    weight: jax.Array
    bias: jax.Array
    kernel_size: int = eqx.field(static=True)

    def __init__(self, dim, kernel_size, *, key):
        self.kernel_size = kernel_size
        self.weight = jax.random.normal(key, (dim, 1, kernel_size)) * 0.02
        self.bias = jnp.zeros(dim)

    def __call__(self, x):
        B, T, D = x.shape
        x = jnp.transpose(x, (0, 2, 1))
        x = jnp.pad(x, ((0, 0), (0, 0), (self.kernel_size - 1, 0)))
        x = lax.conv_general_dilated(
            x, self.weight.astype(x.dtype),
            window_strides=(1,), padding='VALID',
            dimension_numbers=('NCW', 'OIW', 'NCW'),
            feature_group_count=D,
        )
        x = x + self.bias.astype(x.dtype)[:, jnp.newaxis]
        x = jax.nn.silu(x)
        return jnp.transpose(x, (0, 2, 1))


# =============================================================================
# GDN mixer.
# =============================================================================

class GDNLayer(eqx.Module):
    c_q: eqx.nn.Linear
    c_k: eqx.nn.Linear
    c_v: eqx.nn.Linear
    c_g: eqx.nn.Linear
    c_o: eqx.nn.Linear
    c_a: eqx.nn.Linear
    c_b: eqx.nn.Linear
    conv_q: ShortConvSiLU
    conv_k: ShortConvSiLU
    conv_v: ShortConvSiLU
    A_log: jax.Array
    dt_bias: jax.Array
    o_norm_scale: jax.Array
    poly_coeffs: jax.Array
    W1_init: jax.Array
    W2_init: jax.Array
    c_gamma: eqx.nn.Linear
    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    expand_dim: int = eqx.field(static=True)
    qk_norm: str = eqx.field(static=True)
    poly_degree: int = eqx.field(static=True)
    memory_type: str = eqx.field(static=True)
    omega_window: int = eqx.field(static=True)

    def __init__(self, C, H, D, conv_size=4, qk_norm='l2', poly_degree=1,
                 memory_type='linear', memory_expand=1, omega_window=1, *, key):
        self.qk_norm = qk_norm
        self.poly_degree = poly_degree
        self.memory_type = memory_type
        self.omega_window = omega_window
        E = memory_expand * D
        self.expand_dim = E
        keys = jax.random.split(key, 15)
        inner = H * D
        self.n_head = H
        self.head_dim = D
        self.c_q = eqx.nn.Linear(C, inner, use_bias=False, key=keys[0])
        self.c_k = eqx.nn.Linear(C, inner, use_bias=False, key=keys[1])
        self.c_v = eqx.nn.Linear(C, inner, use_bias=False, key=keys[2])
        self.c_g = eqx.nn.Linear(C, inner, use_bias=False, key=keys[3])
        self.c_o = eqx.nn.Linear(inner, C, use_bias=False, key=keys[4])
        self.c_a = eqx.nn.Linear(C, H, use_bias=False, key=keys[5])
        self.c_b = eqx.nn.Linear(C, H, use_bias=False, key=keys[6])
        self.conv_q = ShortConvSiLU(inner, conv_size, key=keys[7])
        self.conv_k = ShortConvSiLU(inner, conv_size, key=keys[8])
        self.conv_v = ShortConvSiLU(inner, conv_size, key=keys[9])
        # A_log: learnable, init A ~ U(0, 16), store log.
        A = jax.random.uniform(keys[10], (H,), minval=0.0, maxval=16.0)
        self.A_log = jnp.log(A + 1e-6)
        # dt_bias: Mamba2-style init — softplus(dt_bias) ~ U[0.001, 0.1] in log space
        dt_min, dt_max = 0.001, 0.1
        dt = jnp.exp(
            jax.random.uniform(keys[11], (H,)) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.dt_bias = dt + jnp.log(-jnp.expm1(-dt))
        self.o_norm_scale = jnp.ones(D)
        # Atlas-style polynomial feature map coefficients; identity init (a_1=1, a_{>=2}=0).
        coeffs = jnp.zeros(max(self.poly_degree, 1))
        self.poly_coeffs = coeffs.at[0].set(1.0)
        # MLP-memory init weights (Xavier). Unused when memory_type='linear'.
        bound1 = math.sqrt(6.0 / (D + E))
        bound2 = math.sqrt(6.0 / (E + D))
        self.W1_init = jax.random.uniform(keys[12], (H, D, E), minval=-bound1, maxval=bound1)
        self.W2_init = jax.random.uniform(keys[13], (H, E, D), minval=-bound2, maxval=bound2)
        # Omega gate gamma (per-head); bias=-2 critical (sigmoid(-2)≈0.12 at init).
        gamma_lin = eqx.nn.Linear(C, H, use_bias=True, key=keys[14])
        gamma_lin = eqx.tree_at(lambda l: l.weight, gamma_lin, jnp.zeros_like(gamma_lin.weight))
        gamma_lin = eqx.tree_at(lambda l: l.bias, gamma_lin, jnp.full_like(gamma_lin.bias, -2.0))
        self.c_gamma = gamma_lin

    def _poly(self, x):
        """phi(x) = sum_{i=1}^{p} a_i * x^i, element-wise (Atlas feature map)."""
        if self.poly_degree <= 1:
            return x
        result = self.poly_coeffs[0] * x
        x_pow = x
        for i in range(1, self.poly_degree):
            x_pow = x_pow * x
            result = result + self.poly_coeffs[i] * x_pow
        return result

    def __call__(self, x):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        # Projections.
        q = x @ self.c_q.weight.T
        k = x @ self.c_k.weight.T
        v = x @ self.c_v.weight.T
        g_out = x @ self.c_g.weight.T

        # Conv + SiLU on QKV.
        q = self.conv_q(q); k = self.conv_k(k); v = self.conv_v(v)

        # Reshape to heads.
        q = q.reshape(B, T, H, D); k = k.reshape(B, T, H, D); v = v.reshape(B, T, H, D)
        g_out = g_out.reshape(B, T, H, D)

        # Q, K normalization per-head (L2 or RMS).
        if self.qk_norm == 'l2':
            q = l2_norm(q); k = l2_norm(k)
        elif self.qk_norm == 'rms':
            q = rms_norm(q); k = rms_norm(k)
        elif self.qk_norm == 'none':
            pass
        else:
            raise ValueError(f'unknown qk_norm: {self.qk_norm}')

        # Atlas polynomial feature map phi(x) = sum_i a_i x^i, applied post-norm.
        q = self._poly(q); k = self._poly(k)

        # Gating params.
        beta = jax.nn.sigmoid(x @ self.c_b.weight.T)                     # (B, T, H)
        a = x @ self.c_a.weight.T + self.dt_bias                         # (B, T, H)
        g = -jnp.exp(self.A_log) * jax.nn.softplus(a)                    # (B, T, H) log-decay
        decay = jnp.exp(g)                                               # (B, T, H), ∈ (0, 1]
        gamma = jax.nn.sigmoid(x @ self.c_gamma.weight.T + self.c_gamma.bias)  # (B, T, H)

        # Transpose to time-major for scan.
        q_T = jnp.transpose(q, (1, 0, 2, 3))
        k_T = jnp.transpose(k, (1, 0, 2, 3))
        v_T = jnp.transpose(v, (1, 0, 2, 3))
        beta_T = jnp.transpose(beta, (1, 0, 2))
        decay_T = jnp.transpose(decay, (1, 0, 2))
        gamma_T = jnp.transpose(gamma, (1, 0, 2))

        if self.memory_type == 'linear':
            # GDN linear delta-rule scan: S_t = decay·S_{t-1} + beta·(v - S_{t-1}·k)·k^T.
            def step(S_prev, xs):
                q_t, k_t, v_t, beta_t, decay_t = xs
                kS = jnp.einsum('bhd,bhde->bhe', k_t, S_prev)                # (B, H, D)
                delta = v_t - kS
                write = beta_t[..., None, None] * jnp.einsum('bhd,bhe->bhde', k_t, delta)
                S_new = decay_t[..., None, None] * S_prev + write
                o_t = jnp.einsum('bhd,bhde->bhe', q_t, S_new)                # (B, H, D)
                return S_new, o_t

            S_init = jnp.zeros((B, H, D, D), dtype=x.dtype)
            _, o_T = lax.scan(step, S_init, (q_T, k_T, v_T, beta_T, decay_T))
        elif self.memory_type == 'mlp':
            # Atlas-style MLP memory with residual on error + LN-residual retrieval.
            # Forward (frozen W):   y_pred = k + W1·GELU(W2·k),  err = y_pred - v
            # Analytic MSE grads (scale 2/D):
            #   u_W1 = err ⊗ GELU(W2·k)
            #   u_W2 = (W1^T·err ⊙ GELU'(W2·k)) ⊗ k
            # Omega aggregation: G_t = sum_{i=t-w+1..t} gamma_i · u_i (via ring buffer).
            # Delta-rule update: W_t = decay_t·W_{t-1} - beta_t·G_t
            # Retrieval:  o_t = q + LN(W1·GELU(W2·q))
            E = self.expand_dim
            W_omega = max(self.omega_window, 1)
            grad_scale = 2.0 / D

            def step(state_prev, xs):
                W1_prev, W2_prev, u1_buf, u2_buf = state_prev
                q_t, k_t, v_t, beta_t, decay_t, gamma_t = xs
                h = jnp.einsum('bhed,bhd->bhe', W2_prev, k_t)                # (B, H, E)
                act = jax.nn.gelu(h)
                y_pred = k_t + jnp.einsum('bhde,bhe->bhd', W1_prev, act)     # (B, H, D)
                err = y_pred - v_t
                u_W1 = grad_scale * jnp.einsum('bhd,bhe->bhde', err, act)
                chain = jnp.einsum('bhde,bhd->bhe', W1_prev, err) * gelu_derivative(h)
                u_W2 = grad_scale * jnp.einsum('bhe,bhd->bhed', chain, k_t)
                # Gamma-weight the gradients, then maintain a sliding window.
                gW1 = gamma_t[..., None, None] * u_W1
                gW2 = gamma_t[..., None, None] * u_W2
                if W_omega == 1:
                    G_W1, G_W2 = gW1, gW2
                    u1_buf_new, u2_buf_new = u1_buf, u2_buf
                else:
                    u1_buf_new = jnp.concatenate([u1_buf[1:], gW1[None]], axis=0)
                    u2_buf_new = jnp.concatenate([u2_buf[1:], gW2[None]], axis=0)
                    G_W1 = u1_buf_new.sum(axis=0)
                    G_W2 = u2_buf_new.sum(axis=0)
                W1_new = decay_t[..., None, None] * W1_prev - beta_t[..., None, None] * G_W1
                W2_new = decay_t[..., None, None] * W2_prev - beta_t[..., None, None] * G_W2
                h_q = jnp.einsum('bhed,bhd->bhe', W2_new, q_t)
                mem_out = jnp.einsum('bhde,bhe->bhd', W1_new, jax.nn.gelu(h_q))
                o_t = q_t + layer_norm(mem_out)
                return (W1_new, W2_new, u1_buf_new, u2_buf_new), o_t

            W1_0 = jnp.broadcast_to(self.W1_init[None].astype(x.dtype), (B, H, D, E))
            W2_0 = jnp.broadcast_to(self.W2_init[None].astype(x.dtype), (B, H, E, D))
            u1_buf_0 = jnp.zeros((W_omega, B, H, D, E), dtype=x.dtype)
            u2_buf_0 = jnp.zeros((W_omega, B, H, E, D), dtype=x.dtype)
            init_state = (W1_0, W2_0, u1_buf_0, u2_buf_0)
            _, o_T = lax.scan(step, init_state,
                              (q_T, k_T, v_T, beta_T, decay_T, gamma_T))
        else:
            raise ValueError(f'unknown memory_type: {self.memory_type}')

        o = jnp.transpose(o_T, (1, 0, 2, 3))                             # (B, T, H, D)

        # Gated RMSNorm output (FusedRMSNormGated equivalent):  norm(o) * silu(g_out).
        o = rms_norm(o) * self.o_norm_scale
        o = o * jax.nn.silu(g_out)

        # Output projection.
        o = o.reshape(B, T, H * D) @ self.c_o.weight.T
        return o


# =============================================================================
# Block + LM.
# =============================================================================

class MLP(eqx.Module):
    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear

    def __init__(self, C, *, key):
        k1, k2 = jax.random.split(key)
        h = int(C * 4 * 2 / 3)
        self.c_fc = eqx.nn.Linear(C, h * 2, use_bias=False, key=k1)
        self.c_proj = eqx.nn.Linear(h, C, use_bias=False, key=k2)

    def __call__(self, x):
        h = x @ self.c_fc.weight.T
        a, b = jnp.split(h, 2, axis=-1)
        return (jax.nn.silu(a) * b) @ self.c_proj.weight.T


class Block(eqx.Module):
    mixer: GDNLayer
    mlp: MLP

    def __init__(self, C, H, D, qk_norm='l2', poly_degree=1,
                 memory_type='linear', memory_expand=1, omega_window=1, *, key):
        k1, k2 = jax.random.split(key)
        self.mixer = GDNLayer(C, H, D, qk_norm=qk_norm, poly_degree=poly_degree,
                              memory_type=memory_type, memory_expand=memory_expand,
                              omega_window=omega_window, key=k1)
        self.mlp = MLP(C, key=k2)

    def __call__(self, x):
        x = x + self.mixer(rms_norm(x))
        x = x + self.mlp(rms_norm(x))
        return x


class GDN_LM(eqx.Module):
    wte: eqx.nn.Embedding
    blocks: list
    lm_head: eqx.nn.Linear

    def __init__(self, vocab, C, n_layer, H, D, qk_norm='l2', poly_degree=1,
                 memory_type='linear', memory_expand=1, omega_window=1, *, key):
        keys = jax.random.split(key, n_layer + 2)
        self.wte = eqx.nn.Embedding(vocab, C, key=keys[0])
        self.blocks = [Block(C, H, D, qk_norm=qk_norm, poly_degree=poly_degree,
                             memory_type=memory_type, memory_expand=memory_expand,
                             omega_window=omega_window, key=keys[i + 1])
                       for i in range(n_layer)]
        self.lm_head = eqx.nn.Linear(C, vocab, use_bias=False, key=keys[-1])

    def __call__(self, idx, *, dropout_key=None):
        x = self.wte.weight[idx]
        for b in self.blocks:
            x = b(x)
        x = rms_norm(x)
        return (x @ self.lm_head.weight.T).astype(jnp.float32)


# =============================================================================
# Training loop.
# =============================================================================

@eqx.filter_jit(donate='all')
def train_step(model, opt_state, optimizer, inputs, labels, dropout_key):
    loss, grads = eqx.filter_value_and_grad(masked_loss)(model, inputs, labels, dropout_key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab', type=int, default=8192)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--n-kv', type=int, default=4)
    p.add_argument('--n-train', type=int, default=100000)
    p.add_argument('--n-test', type=int, default=1000)
    p.add_argument('--n-layer', type=int, default=2)
    p.add_argument('--n-head', type=int, default=2)
    p.add_argument('--n-embd', type=int, default=128)
    p.add_argument('--head-dim', type=int, default=64)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threshold', type=float, default=0.95)
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--qk-norm', choices=['l2', 'rms', 'none'], default='l2',
                   help='Q/K normalization: l2 (GDN default), rms (Atlas-style), or none.')
    p.add_argument('--poly-degree', type=int, default=1,
                   help='Atlas polynomial feature map degree on Q/K (1=identity, 2=Atlas default).')
    p.add_argument('--memory-type', choices=['linear', 'mlp'], default='linear',
                   help='linear: GDN matrix S (default); mlp: Atlas 2-layer MLP memory.')
    p.add_argument('--memory-expand', type=int, default=1,
                   help='MLP hidden expansion factor (paper uses 4; only used if memory-type=mlp).')
    p.add_argument('--omega-window', type=int, default=1,
                   help='Sliding-window size for Omega gradient aggregation (1=off, Atlas uses 4).')
    args = p.parse_args()

    jax.config.update('jax_default_matmul_precision', 'float32')
    print(f'JAX {jax.__version__} | devices: {jax.devices()}')

    train_in, train_lb = generate_mqar(args.n_train, args.seq_len, args.vocab,
                                        args.n_kv, seed=args.seed)
    test_in, test_lb = generate_mqar(args.n_test, args.seq_len, args.vocab,
                                      args.n_kv, seed=args.seed + 1)
    print(f'train {train_in.shape} | test {test_in.shape}')

    key = jax.random.PRNGKey(args.seed)
    key, mkey = jax.random.split(key)
    model = GDN_LM(args.vocab, args.n_embd, args.n_layer, args.n_head, args.head_dim,
                   qk_norm=args.qk_norm, poly_degree=args.poly_degree,
                   memory_type=args.memory_type, memory_expand=args.memory_expand,
                   omega_window=args.omega_window, key=mkey)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f'JAX-GDN: {n_params / 1e6:.2f}M params')

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=100, decay_steps=args.epochs * (args.n_train // args.batch_size),
        end_value=args.lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    steps_per_epoch = args.n_train // args.batch_size
    rng = np.random.default_rng(args.seed + 10)
    t0 = time.time(); step = 0
    print(f'Training: {args.epochs} epochs × {steps_per_epoch} steps = '
          f'{args.epochs * steps_per_epoch}')
    print('-' * 72)

    for epoch in range(args.epochs):
        perm = rng.permutation(args.n_train)
        for s in range(steps_per_epoch):
            step += 1
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            bi = jnp.asarray(train_in[idx]); bl = jnp.asarray(train_lb[idx])
            key, dk = jax.random.split(key)
            model, opt_state, loss = train_step(model, opt_state, optimizer, bi, bl, dk)
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            acc = evaluate(model, test_in[:256], test_lb[:256], args.batch_size)
            print(f'epoch {epoch + 1:3d} | step {step:5d} | loss {float(loss):.4f} | '
                  f'test_acc(256) {acc:.4f} | {time.time() - t0:.1f}s')

    print('-' * 72)
    test_acc = evaluate(model, test_in, test_lb, args.batch_size)
    status = 'PASS' if test_acc >= args.threshold else 'FAIL'
    print(f'[jax-gdn] test_acc={test_acc:.4f} threshold={args.threshold} {status} '
          f'({time.time() - t0:.1f}s)')
    sys.exit(0 if test_acc >= args.threshold else 1)


if __name__ == '__main__':
    main()
