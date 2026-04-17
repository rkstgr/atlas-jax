"""Atlas — minimal JAX/Equinox reference implementation.

Single-file architecture. Paper defaults are hardcoded; only shape knobs and
a few structural toggles are exposed via AtlasConfig. Optimizations (fused
Triton kernels, D==E scan fusion, multi-GPU) are intentionally omitted.

References:
- arXiv 2505.23735 — Atlas: Learning to Optimally Memorize the Context at Test Time
- arXiv 2505.16932 — Polar Express Sign Method
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx


# =============================================================================
# Config
# =============================================================================

@dataclass
class AtlasConfig:
    """Minimal knobs. Paper-fixed values (poly_degree=2, ns_steps=5, pe_ste=True,
    deep_memory=True, dropout=0.1, gate_bias=-2.0) live as constants below."""
    vocab_size: int = 32768
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 448
    seq_len: int = 1024

    chunk_size: int = 64
    memory_expand: int = 1   # MLP hidden expansion. Paper uses 4; pure-JAX slow with >1.
    omega_window: int = 4
    conv_kernel: int = 4     # ShortConv kernel size (0 disables).

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def expand_dim(self) -> int:
        return self.memory_expand * self.head_dim


# Paper-fixed constants.
POLY_DEGREE = 2
NS_STEPS = 5
DROPOUT = 0.1
GATE_BIAS_INIT = -2.0   # sigmoid(-2) ≈ 0.12 at init — critical; see CLAUDE.md.
MAX_LR = 0.1            # upper bound on eta gate.


# =============================================================================
# Stateless helpers
# =============================================================================

def rms_norm(x, eps=1e-6):
    """RMS normalization over the last axis, computed in f32 for bf16 stability."""
    dtype = x.dtype
    x = x.astype(jnp.float32)
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return (x * lax.rsqrt(ms + eps)).astype(dtype)


def dropout(x, rate, key):
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def gelu_derivative(x):
    """Exact derivative of GELU(x) = x * Phi(x), computed in f32."""
    x = x.astype(jnp.float32)
    cdf = 0.5 * (1.0 + lax.erf(x * 0.7071067811865476))
    pdf = jnp.exp(-0.5 * x * x) * 0.3989422804014327
    return cdf + x * pdf


# =============================================================================
# Polar Express orthogonalization (Newton-Schulz, degree-2 polynomial)
# =============================================================================
# Coefficients from arXiv 2505.16932.

_PE_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def polar_express(X, steps=NS_STEPS):
    """Orthogonalize X via Newton-Schulz; converges to the polar factor of X."""
    dtype = X.dtype
    X = X.astype(jnp.float32)
    frob = jnp.sqrt(jnp.sum(X * X, axis=(-2, -1), keepdims=True) + 1e-12)
    X = X / (frob * 1.01 + 1e-6)

    d1, d2 = X.shape[-2], X.shape[-1]
    if d1 > d2:
        for a, b, c in _PE_COEFFS[:steps]:
            A = jnp.einsum('...ji,...jk->...ik', X, X)   # X^T X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in _PE_COEFFS[:steps]:
            A = X @ jnp.swapaxes(X, -2, -1)              # X X^T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X.astype(dtype)


def polar_express_ste(X, steps=NS_STEPS):
    """STE: forward uses orthogonalized value, backward passes gradients as identity.
    Paper validates this is equivalent in practice; 60x faster backward."""
    return X + lax.stop_gradient(polar_express(X, steps) - X)


# =============================================================================
# Linear scan: h_t = g_t * h_{t-1} + x_t — O(log T) via associative scan
# =============================================================================

def linear_scan(h_init, gates, inputs):
    """Parallel linear recurrence.

    Args:
        h_init: (B, H, ...) starting state
        gates:  (B, T, H) scalar gates per timestep
        inputs: (B, T, H, ...) per-timestep inputs

    Returns:
        h_all: (B, T, H, ...) state at every timestep
        h_final: (B, H, ...) state at T-1
    """
    # Broadcast gates to inputs' trailing shape.
    extra = inputs.ndim - gates.ndim
    g = gates
    for _ in range(extra):
        g = g[..., jnp.newaxis]

    # Fold h_init into timestep 0: h_1 = g_1 * h_init + x_1.
    # Then scan from zero state.
    first = g[:, 0:1] * h_init[:, jnp.newaxis] + inputs[:, 0:1]
    mod_inputs = jnp.concatenate([first, inputs[:, 1:]], axis=1)
    mod_gates = jnp.concatenate([jnp.zeros_like(g[:, 0:1]), g[:, 1:]], axis=1)

    def combine(a, b):
        ga, xa = a
        gb, xb = b
        return (ga * gb, gb * xa + xb)

    _, h_all = lax.associative_scan(combine, (mod_gates, mod_inputs), axis=1)
    return h_all, h_all[:, -1]


# =============================================================================
# Omega aggregation: sliding-window weighted sum of per-token gradients
# =============================================================================

def omega_aggregate(u, gamma, window):
    """G_t = sum_{i=max(0,t-w+1)}^{t} gamma_i * u_i, via cumsum subtraction."""
    T = u.shape[1]
    weighted = gamma * u
    cum = jnp.cumsum(weighted, axis=1)
    if window >= T:
        return cum
    shifted = jnp.concatenate([jnp.zeros_like(cum[:, :window]), cum[:, :-window]], axis=1)
    return cum - shifted


# =============================================================================
# ShortConv: causal depthwise 1D convolution
# =============================================================================

class ShortConv(eqx.Module):
    weight: jax.Array       # (D, 1, K)
    bias: jax.Array         # (D,)
    kernel_size: int = eqx.field(static=True)

    def __init__(self, dim, kernel_size, *, key):
        self.kernel_size = kernel_size
        self.weight = jax.random.normal(key, (dim, 1, kernel_size)) * 0.02
        self.bias = jnp.zeros(dim)

    def __call__(self, x):
        """x: (B, T, D) -> (B, T, D), causal 1D conv along T."""
        B, T, D = x.shape
        x = jnp.transpose(x, (0, 2, 1))                                     # NCW
        x = jnp.pad(x, ((0, 0), (0, 0), (self.kernel_size - 1, 0)))          # causal
        x = lax.conv_general_dilated(
            x, self.weight.astype(x.dtype),
            window_strides=(1,), padding='VALID',
            dimension_numbers=('NCW', 'OIW', 'NCW'),
            feature_group_count=D,
        )
        x = x + self.bias.astype(x.dtype)[:, jnp.newaxis]
        return jnp.transpose(x, (0, 2, 1))


# =============================================================================
# Memory state (deep 2-layer MLP)
# =============================================================================

class MemoryState(NamedTuple):
    W1: jax.Array    # (B, H, D, E)
    W2: jax.Array    # (B, H, E, D)
    S_W1: jax.Array  # (B, H, D, E) momentum
    S_W2: jax.Array  # (B, H, E, D) momentum


# =============================================================================
# Atlas memory layer
# =============================================================================

class AtlasMemoryLayer(eqx.Module):
    """Multi-head memory: Omega-rule gradient + momentum + PolarExpress + weight decay.

    Per token, per head:
      u_t  = grad_{M} || M(phi(k_t)) - v_t ||^2  (analytical, no autodiff inside scan)
      G_t  = sum_{i in window} gamma_i * u_i
      S_t  = theta_t * S_{t-1} - eta_t * G_t
      S'_t = PolarExpress(S_t)
      M_t  = alpha_t * M_{t-1} + S'_t
      y_t  = M_t(phi(q_t))

    Between chunks, the carry state is passed through stop_gradient so gradients
    don't compound across chunk boundaries (paper: frozen boundary).
    """
    c_q: eqx.nn.Linear
    c_k: eqx.nn.Linear
    c_v: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    conv_q: ShortConv | None
    conv_k: ShortConv | None
    conv_v: ShortConv | None
    gate_alpha: eqx.nn.Linear
    gate_eta: eqx.nn.Linear
    gate_theta: eqx.nn.Linear
    gate_gamma: eqx.nn.Linear | None
    poly_coeffs: jax.Array
    W1_init: jax.Array
    W2_init: jax.Array

    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    expand_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    omega_window: int = eqx.field(static=True)

    def __init__(self, config: AtlasConfig, *, key):
        keys = jax.random.split(key, 12)
        H, D, E = config.n_head, config.head_dim, config.expand_dim
        dim_inner = H * D
        self.n_head = H
        self.head_dim = D
        self.expand_dim = E
        self.chunk_size = config.chunk_size
        self.omega_window = config.omega_window

        C = config.n_embd
        self.c_q = eqx.nn.Linear(C, dim_inner, use_bias=False, key=keys[0])
        self.c_k = eqx.nn.Linear(C, dim_inner, use_bias=False, key=keys[1])
        self.c_v = eqx.nn.Linear(C, dim_inner, use_bias=False, key=keys[2])
        self.c_proj = eqx.nn.Linear(dim_inner, C, use_bias=False, key=keys[3])

        if config.conv_kernel > 0:
            self.conv_q = ShortConv(dim_inner, config.conv_kernel, key=keys[4])
            self.conv_k = ShortConv(dim_inner, config.conv_kernel, key=keys[5])
            self.conv_v = ShortConv(dim_inner, config.conv_kernel, key=keys[6])
        else:
            self.conv_q = self.conv_k = self.conv_v = None

        self.gate_alpha = eqx.nn.Linear(C, H, use_bias=True, key=keys[7])
        self.gate_eta = eqx.nn.Linear(C, H, use_bias=True, key=keys[8])
        self.gate_theta = eqx.nn.Linear(C, H, use_bias=True, key=keys[9])
        self.gate_gamma = (
            eqx.nn.Linear(C, H, use_bias=True, key=keys[10])
            if config.omega_window > 1 else None
        )

        self.poly_coeffs = jnp.ones(POLY_DEGREE)

        k_w1, k_w2 = jax.random.split(keys[11])
        bound1 = (6.0 / (D + E)) ** 0.5
        bound2 = (6.0 / (E + D)) ** 0.5
        self.W1_init = jax.random.uniform(k_w1, (H, D, E), minval=-bound1, maxval=bound1)
        self.W2_init = jax.random.uniform(k_w2, (H, E, D), minval=-bound2, maxval=bound2)

    def _poly(self, x):
        """phi(x) = sum_{i=1}^{p} a_i * x^i, element-wise."""
        result = self.poly_coeffs[0] * x
        x_pow = x
        for i in range(1, POLY_DEGREE):
            x_pow = x_pow * x
            result = result + self.poly_coeffs[i] * x_pow
        return result

    def _process_chunk(self, state: MemoryState, q_c, k_c, v_c, a_c, e_c, t_c, g_c):
        """One chunk. Shapes:
        state.W*, state.S_W*: (B, H, D|E, E|D)
        q_c, k_c, v_c:       (B, cs, H, D)
        a_c, e_c, t_c, g_c:  (B, cs, H, 1)
        """
        W1, W2, S_W1, S_W2 = state
        D = k_c.shape[-1]

        # 1. Forward with frozen W, compute error.
        h = jnp.einsum('bhed,bchd->bche', W2, k_c)          # (B,cs,H,E)
        act = jax.nn.gelu(h)
        y_pred = k_c + jnp.einsum('bhde,bche->bchd', W1, act)
        err = y_pred - v_c

        # 2. Analytical gradient of (1/D) * || y_pred - v ||^2.
        scale = 2.0 / D
        err_f = err.astype(jnp.float32)
        act_f = act.astype(jnp.float32)
        k_f   = k_c.astype(jnp.float32)
        u_W1 = (scale * jnp.einsum('bchd,bche->bchde', err_f, act_f)).astype(err.dtype)
        chain = jnp.einsum('bhde,bchd->bche', W1.astype(jnp.float32), err_f) * gelu_derivative(h)
        u_W2 = (scale * jnp.einsum('bche,bchd->bched', chain, k_f)).astype(err.dtype)

        # 3. Omega aggregation (sliding-window).
        if self.omega_window > 1 and g_c is not None:
            gw = g_c[..., jnp.newaxis]
            u_W1 = omega_aggregate(u_W1, gw, self.omega_window)
            u_W2 = omega_aggregate(u_W2, gw, self.omega_window)

        # 4. Build momentum inputs (scaled by -eta).
        mom_W1 = -(e_c[..., jnp.newaxis] * u_W1)
        mom_W2 = -(e_c[..., jnp.newaxis] * u_W2)

        # 5. Momentum scan  S_t = theta_t * S_{t-1} + mom_t.
        theta = jnp.squeeze(t_c, axis=-1)
        all_S_W1, S_W1 = linear_scan(S_W1, theta, mom_W1)
        all_S_W2, S_W2 = linear_scan(S_W2, theta, mom_W2)

        # 6. Polar Express orthogonalization (STE).
        all_S_W1_orth = polar_express_ste(all_S_W1, NS_STEPS)
        all_S_W2_orth = polar_express_ste(all_S_W2, NS_STEPS)

        # 7. Weight decay scan  W_t = alpha_t * W_{t-1} + S'_t.
        alpha = jnp.squeeze(a_c, axis=-1)
        all_W1, W1 = linear_scan(W1, alpha, all_S_W1_orth)
        all_W2, W2 = linear_scan(W2, alpha, all_S_W2_orth)

        # 8. Retrieve with per-timestep weights:  y_t = q_t + W1_t @ GELU(W2_t @ q_t).
        h_q = jnp.einsum('bched,bchd->bche', all_W2, q_c)
        y_c = q_c + jnp.einsum('bchde,bche->bchd', all_W1, jax.nn.gelu(h_q))

        return y_c, MemoryState(W1=W1, W2=W2, S_W1=S_W1, S_W2=S_W2)

    def __call__(self, x, state: MemoryState | None = None):
        """x: (B, T, C) -> (B, T, C). Returns (output, new_state)."""
        B, T, C = x.shape
        H, D, E = self.n_head, self.head_dim, self.expand_dim
        cs = self.chunk_size

        # QKV + optional ShortConv + RMS norm + polynomial features.
        q = (x @ self.c_q.weight.T)
        k = (x @ self.c_k.weight.T)
        v = (x @ self.c_v.weight.T)
        if self.conv_q is not None:
            q = self.conv_q(q); k = self.conv_k(k); v = self.conv_v(v)
        q = q.reshape(B, T, H, D); k = k.reshape(B, T, H, D); v = v.reshape(B, T, H, D)

        q = self._poly(rms_norm(q))
        k = rms_norm(self._poly(k))

        def _gate(layer, inp):
            out = inp @ layer.weight.T + layer.bias
            return jax.nn.sigmoid(out.reshape(B, T, H, 1))
        alpha = _gate(self.gate_alpha, x)
        eta = _gate(self.gate_eta, x) * MAX_LR
        theta = _gate(self.gate_theta, x)
        gamma = _gate(self.gate_gamma, x) if self.gate_gamma is not None else None

        # Initialize state from learnable W1_init / W2_init; momentum from zero.
        if state is None:
            W1 = jnp.broadcast_to(self.W1_init[jnp.newaxis].astype(x.dtype), (B, H, D, E))
            W2 = jnp.broadcast_to(self.W2_init[jnp.newaxis].astype(x.dtype), (B, H, E, D))
            state = MemoryState(
                W1=W1, W2=W2,
                S_W1=jnp.zeros((B, H, D, E), x.dtype),
                S_W2=jnp.zeros((B, H, E, D), x.dtype),
            )

        # Pad sequence to a multiple of chunk_size.
        T_orig = T
        if T % cs != 0:
            pad = cs - T % cs
            def p(y, v=0.0): return jnp.pad(y, ((0, 0), (0, pad)) + ((0, 0),) * (y.ndim - 2), constant_values=v)
            q, k, v = p(q), p(k), p(v)
            alpha = p(alpha, 1.0); eta = p(eta, 0.0); theta = p(theta, 0.0)
            if gamma is not None: gamma = p(gamma, 0.0)
            T = q.shape[1]

        n_chunks = T // cs

        # Pre-chunk into (n_chunks, B, cs, ...) for lax.scan.
        def _chunk(y):
            return y.reshape(B, n_chunks, cs, *y.shape[2:]).transpose(1, 0, 2, *range(3, y.ndim + 1))
        xs = tuple(_chunk(y) for y in (q, k, v, alpha, eta, theta, gamma if gamma is not None else alpha))

        def body(carry, chunk):
            # stop_gradient on carry: gradients don't compound across chunks (paper).
            carry = jax.tree.map(lax.stop_gradient, carry)
            q_c, k_c, v_c, a_c, e_c, t_c, g_c = chunk
            if gamma is None:
                g_c = None
            y_c, new_state = self._process_chunk(carry, q_c, k_c, v_c, a_c, e_c, t_c, g_c)
            return new_state, y_c

        final_state, all_y = lax.scan(body, state, xs)

        # (n_chunks, B, cs, H, D) -> (B, T, H, D)
        y = jnp.transpose(all_y, (1, 0, 2, 3, 4)).reshape(B, T, H, D)
        y = y[:, :T_orig]

        # Output projection.
        y = (y.reshape(B, T_orig, H * D) @ self.c_proj.weight.T)
        return y, final_state


# =============================================================================
# MLP (GEGLU)
# =============================================================================

class MLP(eqx.Module):
    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear

    def __init__(self, config: AtlasConfig, *, key):
        k1, k2 = jax.random.split(key)
        dim_inner = int(config.n_embd * 4 * 2 / 3)
        self.c_fc = eqx.nn.Linear(config.n_embd, dim_inner * 2, use_bias=False, key=k1)
        self.c_proj = eqx.nn.Linear(dim_inner, config.n_embd, use_bias=False, key=k2)

    def __call__(self, x):
        h = x @ self.c_fc.weight.T
        gate, v = jnp.split(h, 2, axis=-1)
        return (jax.nn.silu(gate) * v) @ self.c_proj.weight.T


# =============================================================================
# Block: PreNorm(memory) + PreNorm(MLP), each with residual + optional dropout.
# =============================================================================

class Block(eqx.Module):
    memory: AtlasMemoryLayer
    mlp: MLP

    def __init__(self, config: AtlasConfig, *, key):
        k1, k2 = jax.random.split(key)
        self.memory = AtlasMemoryLayer(config, key=k1)
        self.mlp = MLP(config, key=k2)

    def __call__(self, x, state=None, *, dropout_key=None):
        mem_out, new_state = self.memory(rms_norm(x), state)
        if dropout_key is not None:
            k1, k2 = jax.random.split(dropout_key)
            mem_out = dropout(mem_out, DROPOUT, k1)
        x = x + mem_out
        mlp_out = self.mlp(rms_norm(x))
        if dropout_key is not None:
            mlp_out = dropout(mlp_out, DROPOUT, k2)
        return x + mlp_out, new_state


# =============================================================================
# Weight init (paper Section 9)
# =============================================================================

def _init_weights(block, n_layer, key):
    """GPT-2-style scaled init for output projections; Xavier for QKV;
    zero weights + bias=-2.0 for gates; Xavier uniform is already correct for W1_init/W2_init."""
    keys = jax.random.split(key, 16)
    ki = 0
    proj_std = 0.02 / math.sqrt(2 * n_layer)

    # Output projections: scaled init.
    block = eqx.tree_at(
        lambda b: b.memory.c_proj.weight, block,
        jax.random.normal(keys[ki], block.memory.c_proj.weight.shape) * proj_std); ki += 1
    block = eqx.tree_at(
        lambda b: b.mlp.c_proj.weight, block,
        jax.random.normal(keys[ki], block.mlp.c_proj.weight.shape) * proj_std); ki += 1

    # Q/K/V: Xavier.
    fan_in = block.memory.c_q.weight.shape[1]
    xv_std = 1.0 / math.sqrt(fan_in)
    for name in ('c_q', 'c_k', 'c_v'):
        w = getattr(block.memory, name).weight
        block = eqx.tree_at(
            lambda b, n=name: getattr(b.memory, n).weight, block,
            jax.random.uniform(keys[ki], w.shape, minval=-xv_std, maxval=xv_std)); ki += 1

    # Gates: zero weight, bias = -2.0. Critical — without it gates start at 0.5 and explode.
    for name in ('gate_alpha', 'gate_eta', 'gate_theta'):
        layer = getattr(block.memory, name)
        block = eqx.tree_at(lambda b, n=name: getattr(b.memory, n).weight, block, jnp.zeros_like(layer.weight))
        block = eqx.tree_at(lambda b, n=name: getattr(b.memory, n).bias,   block, jnp.full_like(layer.bias, GATE_BIAS_INIT))
    if block.memory.gate_gamma is not None:
        layer = block.memory.gate_gamma
        block = eqx.tree_at(lambda b: b.memory.gate_gamma.weight, block, jnp.zeros_like(layer.weight))
        block = eqx.tree_at(lambda b: b.memory.gate_gamma.bias,   block, jnp.full_like(layer.bias, GATE_BIAS_INIT))

    return block


# =============================================================================
# Atlas (full model)
# =============================================================================

class Atlas(eqx.Module):
    wte: eqx.nn.Embedding
    blocks: list
    lm_head: eqx.nn.Linear
    config: AtlasConfig = eqx.field(static=True)

    def __init__(self, config: AtlasConfig, *, key):
        keys = jax.random.split(key, config.n_layer + 2)
        self.config = config

        self.wte = eqx.nn.Embedding(config.vocab_size, config.n_embd, key=keys[0])

        blocks = []
        for i in range(config.n_layer):
            bkey, ikey = jax.random.split(keys[i + 1])
            b = Block(config, key=bkey)
            b = _init_weights(b, config.n_layer, ikey)
            blocks.append(b)
        self.blocks = blocks

        self.lm_head = eqx.nn.Linear(config.n_embd, config.vocab_size, use_bias=False, key=keys[-1])
        self.lm_head = eqx.tree_at(
            lambda m: m.weight, self.lm_head,
            jax.random.normal(keys[-1], self.lm_head.weight.shape) * 0.02,
        )

    def __call__(self, idx, *, dropout_key=None):
        """idx: (B, T) int32 -> logits (B, T, vocab_size) in f32."""
        x = self.wte.weight[idx]
        drop_keys = (
            jax.random.split(dropout_key, self.config.n_layer)
            if dropout_key is not None else [None] * self.config.n_layer
        )
        for block, dk in zip(self.blocks, drop_keys):
            x, _ = block(x, state=None, dropout_key=dk)
        x = rms_norm(x)
        logits = (x @ self.lm_head.weight.T).astype(jnp.float32)
        return logits
