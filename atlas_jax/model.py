"""Atlas model: memory layer, MLP, block, and full model in Equinox.

Reference: arXiv 2505.23735 — Atlas: Learning to Optimally Memorize the Context at Test Time

Key architecture:
- Deep MLP memory per head (2-layer with residual, GELU activation)
- Omega rule: sliding window gradient aggregation with per-token context gates
- Polar Express (Newton-Schulz) orthogonalization on momentum
- Input-dependent gates: alpha (forget), eta (lr), theta (momentum), gamma (context)
- Short causal convolution on Q, K, V
- Chunk-parallel computation with gradient checkpointing
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx

from atlas_jax.config import AtlasConfig
from atlas_jax.polar_express import polar_express, polar_express_ste
from atlas_jax.state import LinearMemoryState, DeepMemoryState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rms_norm(x):
    """RMS normalization over the last axis."""
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + 1e-6)


def _gelu_derivative(x):
    """Exact derivative of GELU(x) = x * Phi(x)."""
    cdf = 0.5 * (1.0 + jax.lax.erf(x * 0.7071067811865476))
    pdf = jnp.exp(-0.5 * x * x) * 0.3989422804014327
    return cdf + x * pdf


def _omega_aggregate(u, gamma, omega_window):
    """Sliding window aggregation with per-position context gates.

    For position t: sum_{i=max(0,t-w+1)}^{t} gamma_i * u_i
    Uses cumsum for O(n) computation.

    Args:
        u: (B, cs, H, ...) per-position gradient values
        gamma: (B, cs, H, 1, ...) per-position context gates, broadcastable to u
        omega_window: sliding window size
    """
    cs = u.shape[1]
    weighted = gamma * u
    cum = jnp.cumsum(weighted, axis=1)
    if omega_window >= cs:
        return cum
    # Subtract the cumsum from omega_window positions ago
    padded = jnp.concatenate([jnp.zeros_like(cum[:, :1]), cum[:, :-1]], axis=1)
    # Shift: result[t] = cum[t] - cum[t - omega_window]
    shifted = jnp.concatenate([
        jnp.zeros_like(cum[:, :omega_window]),
        cum[:, :-omega_window]
    ], axis=1)
    return cum - shifted


# ---------------------------------------------------------------------------
# Linear scan via associative_scan (parallel, O(log n) depth)
# ---------------------------------------------------------------------------

def linear_scan(h_init, gates, inputs):
    """Linear recurrence: h_t = gate_t * h_{t-1} + input_t.

    Uses associative scan for O(log n) parallel depth instead of O(n).
    The monoid is: (g1, x1) ⊕ (g2, x2) = (g1*g2, g2*x1 + x2)

    Args:
        h_init: (B, H, ...) initial state
        gates: (B, T, H) scalar gates per timestep
        inputs: (B, T, H, ...) per-timestep inputs

    Returns:
        h_all: (B, T, H, ...) all intermediate states
        h_final: (B, H, ...) final state
    """
    # Broadcast scalar gates to match input shape: (B, T, H) -> (B, T, H, 1, ...)
    extra_dims = inputs.ndim - gates.ndim
    gates_expanded = gates
    for _ in range(extra_dims):
        gates_expanded = gates_expanded[..., jnp.newaxis]

    # Incorporate initial state into first position:
    # h_0 = g_0 * h_init + x_0
    # For subsequent positions, the scan handles it:
    # h_t = g_t * h_{t-1} + x_t
    first_x = gates_expanded[:, 0:1] * h_init[:, jnp.newaxis] + inputs[:, 0:1]
    modified_inputs = jnp.concatenate([first_x, inputs[:, 1:]], axis=1)

    # Set first gate to 0 since h_init is already folded into modified_inputs
    zeros = jnp.zeros_like(gates_expanded[:, 0:1])
    modified_gates = jnp.concatenate([zeros, gates_expanded[:, 1:]], axis=1)

    def associative_fn(a, b):
        ga, xa = a
        gb, xb = b
        return (ga * gb, gb * xa + xb)

    # Run parallel associative scan over time axis (axis=1)
    _, h_all = jax.lax.associative_scan(
        associative_fn,
        (modified_gates, modified_inputs),
        axis=1,
    )

    h_final = h_all[:, -1]
    return h_all, h_final


# ---------------------------------------------------------------------------
# ShortConv: causal depthwise 1D convolution
# ---------------------------------------------------------------------------

class ShortConv(eqx.Module):
    """Causal depthwise 1D convolution (per Titans / Based convention)."""
    weight: jax.Array   # (D, 1, K)
    bias: jax.Array     # (D,)
    kernel_size: int = eqx.field(static=True)

    def __init__(self, dim, kernel_size=4, *, key):
        self.kernel_size = kernel_size
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (dim, 1, kernel_size)) * 0.02
        self.bias = jnp.zeros(dim)

    def __call__(self, x):
        """x: (B, T, D) -> (B, T, D), causal conv over time dim."""
        B, T, D = x.shape
        # Transpose to (B, D, T) for conv
        x = jnp.transpose(x, (0, 2, 1))  # (B, D, T)
        # Causal left-padding
        x = jnp.pad(x, ((0, 0), (0, 0), (self.kernel_size - 1, 0)))
        # Depthwise conv1d: each channel independently
        # JAX conv_general_dilated with feature_group_count=D
        x = lax.conv_general_dilated(
            x,                                          # (B, D, T+K-1)
            self.weight.astype(x.dtype),                # (D, 1, K)
            window_strides=(1,),
            padding='VALID',
            dimension_numbers=('NCW', 'OIW', 'NCW'),   # batch=N, channel=C, spatial=W
            feature_group_count=D,
        )  # (B, D, T)
        x = x + self.bias.astype(x.dtype)[:, jnp.newaxis]
        return jnp.transpose(x, (0, 2, 1))  # (B, T, D)


# ---------------------------------------------------------------------------
# AtlasMemoryLayer
# ---------------------------------------------------------------------------

class AtlasMemoryLayer(eqx.Module):
    """Multi-head memory layer with Omega rule + Polar Express update.

    Paper formulation (per head, per token):
      1. Gradient:     u_t = 2 * d/dM ||M(phi(k_t)) - v_t||^2
      2. Omega agg:    G_t = sum_{i in window} gamma_i * u_i
      3. Momentum:     S_t = theta_t * S_{t-1} - eta_t * G_t
      4. Orthogonalize: S'_t = PolarExpress(S_t)
      5. Memory:       M_t = alpha_t * M_{t-1} + S'_t
      6. Output:       y_t = M_t(q_t)
    """
    # Projections
    c_q: eqx.nn.Linear
    c_k: eqx.nn.Linear
    c_v: eqx.nn.Linear
    c_proj: eqx.nn.Linear

    # Short causal convolutions
    conv_q: ShortConv
    conv_k: ShortConv
    conv_v: ShortConv

    # Gates
    gate_alpha: eqx.nn.Linear
    gate_eta: eqx.nn.Linear
    gate_theta: eqx.nn.Linear
    gate_gamma: eqx.nn.Linear | None

    # Polynomial feature mapping coefficients
    poly_coeffs: jax.Array | None

    # Static config
    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    expand_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    ns_steps: int = eqx.field(static=True)
    omega_window: int = eqx.field(static=True)
    poly_degree: int = eqx.field(static=True)
    deep_memory: bool = eqx.field(static=True)
    pe_ste: bool = eqx.field(static=True)
    use_checkpoint: bool = eqx.field(static=True)

    def __init__(self, config: AtlasConfig, *, key):
        keys = jax.random.split(key, 12)
        H = config.n_head
        D = config.n_embd // H
        E = config.memory_expand * D if config.deep_memory else D

        self.n_head = H
        self.head_dim = D
        self.expand_dim = E
        self.chunk_size = config.chunk_size
        self.ns_steps = config.ns_steps
        self.omega_window = config.omega_window
        self.poly_degree = config.poly_degree
        self.deep_memory = config.deep_memory
        self.pe_ste = config.pe_ste
        self.use_checkpoint = config.use_checkpoint

        C = config.n_embd
        self.c_q = eqx.nn.Linear(C, C, use_bias=False, key=keys[0])
        self.c_k = eqx.nn.Linear(C, C, use_bias=False, key=keys[1])
        self.c_v = eqx.nn.Linear(C, C, use_bias=False, key=keys[2])
        self.c_proj = eqx.nn.Linear(C, C, use_bias=False, key=keys[3])

        self.conv_q = ShortConv(C, config.conv_kernel, key=keys[4])
        self.conv_k = ShortConv(C, config.conv_kernel, key=keys[5])
        self.conv_v = ShortConv(C, config.conv_kernel, key=keys[6])

        self.gate_alpha = eqx.nn.Linear(C, H, use_bias=False, key=keys[7])
        self.gate_eta = eqx.nn.Linear(C, H, use_bias=False, key=keys[8])
        self.gate_theta = eqx.nn.Linear(C, H, use_bias=False, key=keys[9])

        if self.omega_window > 1:
            self.gate_gamma = eqx.nn.Linear(C, H, use_bias=False, key=keys[10])
        else:
            self.gate_gamma = None

        if self.poly_degree > 0:
            coeffs = jnp.array([1.0 / math.factorial(i) for i in range(1, self.poly_degree + 1)])
            self.poly_coeffs = coeffs
        else:
            self.poly_coeffs = None

    def _poly_features(self, x):
        """Element-wise polynomial: phi(x) = sum_{i=1}^{p} a_i * x^i."""
        result = self.poly_coeffs[0] * x
        x_pow = x
        for i in range(1, self.poly_degree):
            x_pow = x_pow * x
            result = result + self.poly_coeffs[i] * x_pow
        return result

    def _process_chunk_linear(self, M, S, q_c, k_c, v_c, a_c, e_c, t_c, g_c):
        """Process one chunk with linear memory.

        Linear memory: M is a matrix, M(x) = M @ x (no residual/activation).
        Gradient: u_t = 2 * (M @ k_t - v_t) outer k_t^T
        """
        _pe = polar_express_ste if self.pe_ste else polar_express

        # Parallel: per-position gradients w.r.t. frozen M
        pred = jnp.einsum('bhvk,bchk->bchv', M, k_c)
        err = pred - v_c
        u = 2.0 * jnp.einsum('bchv,bchk->bchvk', err, k_c)

        # Omega rule
        if self.omega_window > 1 and g_c is not None:
            u = _omega_aggregate(u, g_c[..., jnp.newaxis], self.omega_window)

        # Momentum scan: S_t = theta_t * S_{t-1} - eta_t * u_t
        theta = jnp.squeeze(t_c, axis=-1)         # (B, cs, H)
        mom_input = -(e_c[..., jnp.newaxis] * u)  # (B, cs, H, D, D)
        chunk_S, S = linear_scan(S, theta, mom_input)

        # Polar Express orthogonalization
        chunk_S_orth = _pe(chunk_S, self.ns_steps)

        # Memory scan: M_t = alpha_t * M_{t-1} + PE(S_t)
        alpha = jnp.squeeze(a_c, axis=-1)          # (B, cs, H)
        M_all, M = linear_scan(M, alpha, chunk_S_orth)

        # Output: y_t = M_t @ q_t
        y_c = jnp.einsum('bchvk,bchk->bchv', M_all, q_c)

        return y_c, LinearMemoryState(M=M, S=S)

    def _process_chunk_deep(self, state, q_c, k_c, v_c, a_c, e_c, t_c, g_c):
        """Process one chunk with deep MLP memory.

        Memory: M(x) = x + W1 @ GELU(W2 @ x) with residual connection.
        Gradients computed analytically w.r.t. W1 and W2.
        """
        W1, W2, S_W1, S_W2 = state.W1, state.W2, state.S_W1, state.S_W2
        _pe = polar_express_ste if self.pe_ste else polar_express

        # Forward through frozen MLP memory
        h = jnp.einsum('bhed,bchd->bche', W2, k_c)          # (B, cs, H, E)
        act = jax.nn.gelu(h)                                   # (B, cs, H, E)
        y_pred = k_c + jnp.einsum('bhde,bche->bchd', W1, act) # (B, cs, H, D)
        err = y_pred - v_c                                      # (B, cs, H, D)

        # Gradient w.r.t. W1: 2 * err outer GELU(W2 @ k)
        u_W1 = 2.0 * jnp.einsum('bchd,bche->bchde', err, act)  # (B, cs, H, D, E)

        # Gradient w.r.t. W2: chain through GELU' and W1^T
        gelu_prime = _gelu_derivative(h)                          # (B, cs, H, E)
        w1t_err = jnp.einsum('bhde,bchd->bche', W1, err)         # (B, cs, H, E)
        chain = w1t_err * gelu_prime                               # (B, cs, H, E)
        u_W2 = 2.0 * jnp.einsum('bche,bchd->bched', chain, k_c)  # (B, cs, H, E, D)

        # Omega rule
        if self.omega_window > 1 and g_c is not None:
            g_w1 = g_c[..., jnp.newaxis]  # broadcast for (B, cs, H, D, E)
            g_w2 = g_c[..., jnp.newaxis]  # broadcast for (B, cs, H, E, D)
            u_W1 = _omega_aggregate(u_W1, g_w1, self.omega_window)
            u_W2 = _omega_aggregate(u_W2, g_w2, self.omega_window)

        theta = jnp.squeeze(t_c, axis=-1)  # (B, cs, H)
        alpha = jnp.squeeze(a_c, axis=-1)  # (B, cs, H)

        # Fused momentum -> PE -> memory scan for both W1 and W2
        # Process both weight matrices with shared gates
        mom_W1 = -(e_c[..., jnp.newaxis] * u_W1)
        mom_W2 = -(e_c[..., jnp.newaxis] * u_W2)

        def _fused_scan(S_init, W_init, theta, alpha, mom_input):
            """Fused: momentum scan -> PE -> memory scan in one pass."""
            chunk_S, S_final = linear_scan(S_init, theta, mom_input)
            chunk_S_orth = _pe(chunk_S, self.ns_steps)
            W_all, W_final = linear_scan(W_init, alpha, chunk_S_orth)
            return W_all, W_final, S_final

        W1_all, W1, S_W1 = _fused_scan(S_W1, W1, theta, alpha, mom_W1)
        W2_all, W2, S_W2 = _fused_scan(S_W2, W2, theta, alpha, mom_W2)

        # Output: y_t = q_t + W1_t @ GELU(W2_t @ q_t)
        h_q = jnp.einsum('bched,bchd->bche', W2_all, q_c)
        g_q = jax.nn.gelu(h_q)
        y_c = q_c + jnp.einsum('bchde,bche->bchd', W1_all, g_q)

        return y_c, DeepMemoryState(W1=W1, W2=W2, S_W1=S_W1, S_W2=S_W2)

    def __call__(self, x, memory_state=None):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        E = self.expand_dim
        cs = self.chunk_size

        # Project (direct matmul, no vmap overhead) + short causal conv + multi-head reshape
        q = self.conv_q((x @ self.c_q.weight.T).reshape(B, T, C)).reshape(B, T, H, D)
        k = self.conv_k((x @ self.c_k.weight.T).reshape(B, T, C)).reshape(B, T, H, D)
        v = self.conv_v((x @ self.c_v.weight.T).reshape(B, T, C)).reshape(B, T, H, D)

        # Normalize Q, K for stable memory operations
        q, k = rms_norm(q), rms_norm(k)

        # Polynomial feature mapping
        if self.poly_degree > 0:
            q = self._poly_features(q)
            k = self._poly_features(k)

        # Input-dependent gates via sigmoid (direct matmul)
        alpha = jax.nn.sigmoid((x @ self.gate_alpha.weight.T).reshape(B, T, H, 1))
        eta = jax.nn.sigmoid((x @ self.gate_eta.weight.T).reshape(B, T, H, 1))
        theta = jax.nn.sigmoid((x @ self.gate_theta.weight.T).reshape(B, T, H, 1))

        gamma = None
        if self.omega_window > 1 and self.gate_gamma is not None:
            gamma = jax.nn.sigmoid((x @ self.gate_gamma.weight.T).reshape(B, T, H, 1))

        # Initialize memory state if needed
        if memory_state is None:
            if self.deep_memory:
                W1 = jnp.zeros((B, H, D, E), dtype=x.dtype)
                W2 = jnp.zeros((B, H, E, D), dtype=x.dtype)
                eye = jnp.eye(min(E, D), dtype=x.dtype)
                W2 = W2.at[:, :, :min(E, D), :min(E, D)].set(eye)
                S_W1 = jnp.zeros((B, H, D, E), dtype=x.dtype)
                S_W2 = jnp.zeros((B, H, E, D), dtype=x.dtype)
                memory_state = DeepMemoryState(W1=W1, W2=W2, S_W1=S_W1, S_W2=S_W2)
            else:
                M = jnp.zeros((B, H, D, D), dtype=x.dtype)
                S = jnp.zeros((B, H, D, D), dtype=x.dtype)
                memory_state = LinearMemoryState(M=M, S=S)

        # Pad sequence to multiple of chunk_size
        T_orig = T
        if T % cs != 0:
            pad = cs - T % cs
            q = jnp.pad(q, ((0, 0), (0, pad), (0, 0), (0, 0)))
            k = jnp.pad(k, ((0, 0), (0, pad), (0, 0), (0, 0)))
            v = jnp.pad(v, ((0, 0), (0, pad), (0, 0), (0, 0)))
            # alpha=1 carries memory unchanged, eta=0 no update, theta=0 kill momentum
            alpha = jnp.pad(alpha, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=1.0)
            eta = jnp.pad(eta, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=0.0)
            theta = jnp.pad(theta, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=0.0)
            if gamma is not None:
                gamma = jnp.pad(gamma, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=0.0)
            T = q.shape[1]

        n_chunks = T // cs

        # Pre-chunk arrays: (B, T, ...) -> (n_chunks, B, cs, ...)
        def _chunk(x):
            return x.reshape(B, n_chunks, cs, *x.shape[2:]).transpose(1, 0, 2, *range(3, x.ndim + 1))

        q_chunks = _chunk(q)          # (n_chunks, B, cs, H, D)
        k_chunks = _chunk(k)
        v_chunks = _chunk(v)
        a_chunks = _chunk(alpha)      # (n_chunks, B, cs, H, 1)
        e_chunks = _chunk(eta)
        t_chunks = _chunk(theta)
        if gamma is not None:
            g_chunks = _chunk(gamma)
        else:
            g_chunks = jnp.zeros_like(a_chunks)

        xs = (q_chunks, k_chunks, v_chunks, a_chunks, e_chunks, t_chunks, g_chunks)

        def chunk_body(carry, chunk_data):
            mem_state = jax.lax.stop_gradient(carry)
            q_c, k_c, v_c, a_c, e_c, t_c, g_c = chunk_data
            if self.deep_memory:
                y_c, new_state = self._process_chunk_deep(
                    mem_state, q_c, k_c, v_c, a_c, e_c, t_c, g_c)
            else:
                y_c, new_state = self._process_chunk_linear(
                    mem_state.M, mem_state.S, q_c, k_c, v_c, a_c, e_c, t_c, g_c)
            return new_state, y_c

        process_fn = jax.checkpoint(chunk_body) if self.use_checkpoint else chunk_body

        final_state, all_y = lax.scan(process_fn, memory_state, xs)
        # all_y: (n_chunks, B, cs, H, D) -> (B, T, H, D)
        y = jnp.transpose(all_y, (1, 0, 2, 3, 4)).reshape(B, T, H, D)

        # Trim padding and project back to residual stream
        y = y[:, :T_orig].reshape(B, T_orig, -1)
        y = (y @ self.c_proj.weight.T).reshape(B, T_orig, C)

        return y, final_state


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(eqx.Module):
    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear

    def __init__(self, config: AtlasConfig, *, key):
        k1, k2 = jax.random.split(key)
        self.c_fc = eqx.nn.Linear(config.n_embd, 4 * config.n_embd, use_bias=False, key=k1)
        self.c_proj = eqx.nn.Linear(4 * config.n_embd, config.n_embd, use_bias=False, key=k2)

    def __call__(self, x):
        """x: (B, T, C) -> (B, T, C)"""
        x = x @ self.c_fc.weight.T
        x = jax.nn.gelu(x)
        x = x @ self.c_proj.weight.T
        return x


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class Block(eqx.Module):
    memory: AtlasMemoryLayer
    mlp: MLP

    def __init__(self, config: AtlasConfig, *, key):
        k1, k2 = jax.random.split(key)
        self.memory = AtlasMemoryLayer(config, key=k1)
        self.mlp = MLP(config, key=k2)

    def __call__(self, x, memory_state=None):
        mem_out, new_state = self.memory(rms_norm(x), memory_state)
        x = x + mem_out
        x = x + self.mlp(rms_norm(x))
        return x, new_state


# ---------------------------------------------------------------------------
# Atlas (full model)
# ---------------------------------------------------------------------------

class Atlas(eqx.Module):
    wte: eqx.nn.Embedding
    blocks: list[Block]
    lm_head: eqx.nn.Linear
    config: AtlasConfig = eqx.field(static=True)
    padded_vocab_size: int = eqx.field(static=True)

    def __init__(self, config: AtlasConfig, *, key, pad_vocab_size_to=64):
        keys = jax.random.split(key, config.n_layer + 3)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.padded_vocab_size = padded_vocab_size
        self.config = config

        self.wte = eqx.nn.Embedding(padded_vocab_size, config.n_embd, key=keys[0])
        self.blocks = [Block(config, key=keys[i + 1]) for i in range(config.n_layer)]
        self.lm_head = eqx.nn.Linear(config.n_embd, padded_vocab_size, use_bias=False, key=keys[-1])

        # Re-initialize weights following the paper / nanochat convention
        self = self._init_weights(keys[-2])

    def _init_weights(self, key):
        """Reinitialize for stable training.

        Critical: zero output projections so residual blocks start as identity.
        Small gates so sigmoid ≈ 0.5. Small lm_head so initial logits are small.
        """
        model = self
        keys = jax.random.split(key, 10 + 2 * len(model.blocks))
        ki = 0

        # lm_head: small init
        model = eqx.tree_at(lambda m: m.lm_head.weight,
            model, jax.random.normal(keys[ki], model.lm_head.weight.shape) * 0.02)
        ki += 1

        for i in range(len(model.blocks)):
            # Memory output projection: ZERO (blocks start as identity)
            model = eqx.tree_at(lambda m, j=i: m.blocks[j].memory.c_proj.weight,
                model, jnp.zeros_like(model.blocks[i].memory.c_proj.weight))
            # MLP output projection: ZERO
            model = eqx.tree_at(lambda m, j=i: m.blocks[j].mlp.c_proj.weight,
                model, jnp.zeros_like(model.blocks[i].mlp.c_proj.weight))
            # Gates: small init
            for attr in ['gate_alpha', 'gate_eta', 'gate_theta']:
                w = getattr(model.blocks[i].memory, attr).weight
                model = eqx.tree_at(
                    lambda m, a=attr, j=i: getattr(m.blocks[j].memory, a).weight,
                    model, jax.random.normal(keys[ki], w.shape) * 0.01)
                ki += 1
            if model.blocks[i].memory.gate_gamma is not None:
                w = model.blocks[i].memory.gate_gamma.weight
                model = eqx.tree_at(lambda m, j=i: m.blocks[j].memory.gate_gamma.weight,
                    model, jax.random.normal(keys[ki], w.shape) * 0.01)
                ki += 1

        return model

    def __call__(self, idx, memory_states=None):
        """Forward pass.

        Args:
            idx: (B, T) integer token indices
            memory_states: optional list of per-layer memory states for inference

        Returns:
            logits: (B, T, vocab_size) with soft capping
            new_memory_states: list of updated memory states
        """
        B, T = idx.shape

        # Embed + normalize — direct index into weight matrix
        x = self.wte.weight[idx]  # (B, T, C)
        x = rms_norm(x)

        # Forward through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            layer_state = memory_states[i] if memory_states is not None else None
            x, new_state = block(x, layer_state)
            new_states.append(new_state)

        x = rms_norm(x)

        # Logits with soft capping
        logits = x @ self.lm_head.weight.T  # (B, T, padded_vocab)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.astype(jnp.float32)
        softcap = 15.0
        logits = softcap * jnp.tanh(logits / softcap)

        return logits, new_states
