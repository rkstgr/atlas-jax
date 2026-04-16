"""Atlas memory layer: everything related to the memory subsystem.

Consolidates polar_express, state, ops, kernel dispatch, fused_chunk,
ShortConv, and AtlasMemoryLayer into a single module.

Reference: arXiv 2505.23735 — Atlas: Learning to Optimally Memorize the
Context at Test Time.
"""

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx

from atlas_jax.config import AtlasConfig


# ============================================================================
# Polar Express orthogonalization (Newton-Schulz iteration)
# ============================================================================
# Reference: arXiv 2505.16932 (Polar Express Sign Method)
#
# Per-step iteration (for square/wide matrices):
#   A = X @ X^T
#   B = b*A + c*(A @ A)
#   X_{i+1} = a*X + B @ X
#
# Input is Frobenius-normalized before iteration.

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def polar_express(X, steps=5):
    """Batched Polar Express orthogonalization.

    Args:
        X: (..., D1, D2) batch of matrices.
        steps: number of Newton-Schulz iterations (1-5).

    Returns:
        Approximate orthogonal polar factor, same shape as X.
    """
    orig_dtype = X.dtype
    X = X.astype(jnp.float32)
    frob_norm = jnp.sqrt(jnp.sum(X * X, axis=(-2, -1), keepdims=True) + 1e-12)
    X = X / (frob_norm * 1.01 + 1e-6)

    d1, d2 = X.shape[-2], X.shape[-1]

    if d1 > d2:
        for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
            A = jnp.einsum('...ji,...jk->...ik', X, X)
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
            A = X @ jnp.swapaxes(X, -2, -1)
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    return X.astype(orig_dtype)


def polar_express_ste(X, steps=5):
    """Polar Express with straight-through estimator for backward pass."""
    return X + jax.lax.stop_gradient(polar_express(X, steps) - X)


def frobenius_clip(X, steps=None):
    """Frobenius norm clipping: X / max(||X||_F, 1.0).

    The `steps` argument is accepted but ignored (API compat with PE).
    """
    orig_dtype = X.dtype
    X = X.astype(jnp.float32)
    frob_norm = jnp.sqrt(jnp.sum(X * X, axis=(-2, -1), keepdims=True) + 1e-12)
    X = X / jnp.maximum(frob_norm, 1.0)
    return X.astype(orig_dtype)


def frobenius_clip_ste(X, steps=None):
    """Frobenius clipping with straight-through estimator."""
    return X + jax.lax.stop_gradient(frobenius_clip(X) - X)


# ============================================================================
# Memory state containers
# ============================================================================

class LinearMemoryState(NamedTuple):
    """State for linear (matrix-valued) memory per head."""
    M: jax.Array    # (B, H, D, D) memory matrix
    S: jax.Array    # (B, H, D, D) momentum matrix


class DeepMemoryState(NamedTuple):
    """State for deep MLP memory per head.

    Memory is a 2-layer MLP: M(x) = x + W1 @ GELU(W2 @ x)
    W2 is initialized to [I; 0] so GELU(W2 @ k) != 0 from step 1.
    """
    W1: jax.Array     # (B, H, D, E) first layer weights
    W2: jax.Array     # (B, H, E, D) second layer weights
    S_W1: jax.Array   # (B, H, D, E) momentum for W1
    S_W2: jax.Array   # (B, H, E, D) momentum for W2


def init_memory_state(config: AtlasConfig, batch_size: int, dtype=jnp.bfloat16):
    """Create a fresh zero-initialized memory state for all heads.

    Returns one state per layer (list of LinearMemoryState or DeepMemoryState).
    """
    H = config.n_head
    D = config.n_embd // config.n_head
    E = config.memory_expand * D if config.deep_memory else D
    B = batch_size

    states = []
    for _ in range(config.n_layer):
        if config.deep_memory:
            W1 = jnp.zeros((B, H, D, E), dtype=dtype)
            W2 = jnp.zeros((B, H, E, D), dtype=dtype)
            eye = jnp.eye(min(E, D), dtype=dtype)
            W2 = W2.at[:, :, :min(E, D), :min(E, D)].set(eye)
            S_W1 = jnp.zeros((B, H, D, E), dtype=dtype)
            S_W2 = jnp.zeros((B, H, E, D), dtype=dtype)
            states.append(DeepMemoryState(W1=W1, W2=W2, S_W1=S_W1, S_W2=S_W2))
        else:
            M = jnp.zeros((B, H, D, D), dtype=dtype)
            S = jnp.zeros((B, H, D, D), dtype=dtype)
            states.append(LinearMemoryState(M=M, S=S))

    return states


# ============================================================================
# Optional GPU kernel dispatch
# ============================================================================
# Centralizes try/except imports for Triton and Pallas kernel backends.
# Consumers branch on HAS_* flags when a faster path is available.

# --- FlashATLAS fused chunk scan (Triton preferred, Pallas fallback) ---
fused_chunk_scan_kernel = None
HAS_FUSED_CHUNK = False

try:
    from atlas_jax.kernels.fused_chunk import (
        fused_chunk_scan as _fcs_triton,
        fused_chunk_available as _fcs_triton_available,
    )
    if _fcs_triton_available():
        fused_chunk_scan_kernel = _fcs_triton
        HAS_FUSED_CHUNK = True
except ImportError:
    pass

if not HAS_FUSED_CHUNK:
    try:
        from atlas_jax.kernels.pallas_fused import (
            fused_chunk_scan as _fcs_pallas,
            pallas_available as _pallas_available,
        )
        if _pallas_available():
            fused_chunk_scan_kernel = _fcs_pallas
            HAS_FUSED_CHUNK = True
    except ImportError:
        pass

# --- Fused Triton Polar Express ---
triton_polar_express = None
triton_polar_express_ste = None
HAS_TRITON_PE = False

try:
    from atlas_jax.kernels.triton_pe import (
        triton_polar_express as _tpe,
        triton_polar_express_ste as _tpe_ste,
    )
    triton_polar_express = _tpe
    triton_polar_express_ste = _tpe_ste
    HAS_TRITON_PE = True
except ImportError:
    pass

# --- Fused Triton linear scan ---
triton_linear_scan = None
HAS_TRITON_SCAN = False

try:
    from atlas_jax.kernels.triton_scan import triton_linear_scan as _tls
    triton_linear_scan = _tls
    HAS_TRITON_SCAN = True
except ImportError:
    pass


# ============================================================================
# Stateless tensor ops
# ============================================================================

def rms_norm(x):
    """RMS normalization over the last axis. Computed in f32 for bf16 stability."""
    dtype = x.dtype
    x = x.astype(jnp.float32)
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(ms + 1e-6)).astype(dtype)


def _dropout(x, rate, key):
    """Apply dropout with inverted scaling."""
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


def _gelu_derivative(x):
    """Exact derivative of GELU(x) = x * Phi(x). Computed in f32."""
    x = x.astype(jnp.float32)
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
    shifted = jnp.concatenate([
        jnp.zeros_like(cum[:, :omega_window]),
        cum[:, :-omega_window]
    ], axis=1)
    return cum - shifted


def linear_scan(h_init, gates, inputs):
    """Linear recurrence: h_t = gate_t * h_{t-1} + input_t.

    Uses fused Triton kernel when available (2-4x faster than associative scan),
    falls back to jax.lax.associative_scan otherwise.

    Args:
        h_init: (B, H, ...) initial state
        gates: (B, T, H) scalar gates per timestep
        inputs: (B, T, H, ...) per-timestep inputs

    Returns:
        h_all: (B, T, H, ...) all intermediate states
        h_final: (B, H, ...) final state
    """
    if HAS_TRITON_SCAN:
        return triton_linear_scan(h_init, gates, inputs)

    extra_dims = inputs.ndim - gates.ndim
    gates_expanded = gates
    for _ in range(extra_dims):
        gates_expanded = gates_expanded[..., jnp.newaxis]

    first_x = gates_expanded[:, 0:1] * h_init[:, jnp.newaxis] + inputs[:, 0:1]
    modified_inputs = jnp.concatenate([first_x, inputs[:, 1:]], axis=1)
    zeros = jnp.zeros_like(gates_expanded[:, 0:1])
    modified_gates = jnp.concatenate([zeros, gates_expanded[:, 1:]], axis=1)

    def associative_fn(a, b):
        ga, xa = a
        gb, xb = b
        return (ga * gb, gb * xa + xb)

    _, h_all = jax.lax.associative_scan(
        associative_fn,
        (modified_gates, modified_inputs),
        axis=1,
    )

    h_final = h_all[:, -1]
    return h_all, h_final


# ============================================================================
# ShortConv: causal depthwise 1D convolution
# ============================================================================

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
        x = jnp.transpose(x, (0, 2, 1))
        x = jnp.pad(x, ((0, 0), (0, 0), (self.kernel_size - 1, 0)))
        x = lax.conv_general_dilated(
            x,
            self.weight.astype(x.dtype),
            window_strides=(1,),
            padding='VALID',
            dimension_numbers=('NCW', 'OIW', 'NCW'),
            feature_group_count=D,
        )
        x = x + self.bias.astype(x.dtype)[:, jnp.newaxis]
        return jnp.transpose(x, (0, 2, 1))


# ============================================================================
# AtlasMemoryLayer
# ============================================================================

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

    # ResidualNorm gamma: LayerNorm(MLP(x)) * (gamma+1) + x (per-head, init=0)
    ln_gamma: jax.Array  # (H, D)

    # Learnable initial memory weights (Xavier init, like PyTorch)
    W1_init: jax.Array  # (H, D, E)
    W2_init: jax.Array  # (H, E, D)

    # Retrieve gate: per-head sigmoid gate on memory output (PyTorch: when heads > 1)
    retrieve_gate: eqx.nn.Linear | None

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
    fused_chunk: bool = eqx.field(static=True)
    max_lr: float = eqx.field(static=True)
    stop_grad_chunks: bool = eqx.field(static=True)

    def __init__(self, config: AtlasConfig, *, key):
        keys = jax.random.split(key, 14)
        H = config.n_head
        D = config.dim_head if config.dim_head is not None else config.n_embd // H
        E = config.memory_expand * D if config.deep_memory else D
        dim_inner = H * D

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
        self.max_lr = config.max_lr
        self.stop_grad_chunks = config.stop_grad_chunks
        _d_is_pow2 = (D & (D - 1) == 0)
        _e_is_pow2 = (E & (E - 1) == 0)
        self.fused_chunk = config.fused_chunk and HAS_FUSED_CHUNK and _d_is_pow2 and _e_is_pow2
        if config.fused_chunk and HAS_FUSED_CHUNK and not (_d_is_pow2 and _e_is_pow2):
            import warnings
            warnings.warn(
                f"Fused chunk kernel disabled: D={D}, E={E} must both be powers of 2. "
                f"Falling back to regular scan ops."
            )

        C = config.n_embd
        self.c_q = eqx.nn.Linear(C, dim_inner, use_bias=False, key=keys[0])
        self.c_k = eqx.nn.Linear(C, dim_inner, use_bias=False, key=keys[1])
        self.c_v = eqx.nn.Linear(C, dim_inner, use_bias=False, key=keys[2])
        self.c_proj = eqx.nn.Linear(dim_inner, C, use_bias=False, key=keys[3])

        self.conv_q = ShortConv(dim_inner, config.conv_kernel, key=keys[4])
        self.conv_k = ShortConv(dim_inner, config.conv_kernel, key=keys[5])
        self.conv_v = ShortConv(dim_inner, config.conv_kernel, key=keys[6])

        self.gate_alpha = eqx.nn.Linear(C, H, use_bias=True, key=keys[7])
        self.gate_eta = eqx.nn.Linear(C, H, use_bias=True, key=keys[8])
        self.gate_theta = eqx.nn.Linear(C, H, use_bias=True, key=keys[9])

        if self.omega_window > 1:
            self.gate_gamma = eqx.nn.Linear(C, H, use_bias=True, key=keys[10])
        else:
            self.gate_gamma = None

        if self.poly_degree > 0:
            coeffs = jnp.ones(self.poly_degree)
            self.poly_coeffs = coeffs
        else:
            self.poly_coeffs = None

        self.ln_gamma = jnp.zeros((H, D))

        k_w1, k_w2 = jax.random.split(keys[11])
        bound1 = (6.0 / (D + E)) ** 0.5
        bound2 = (6.0 / (E + D)) ** 0.5
        self.W1_init = jax.random.uniform(k_w1, (H, D, E), minval=-bound1, maxval=bound1)
        self.W2_init = jax.random.uniform(k_w2, (H, E, D), minval=-bound2, maxval=bound2)

        if H > 1:
            self.retrieve_gate = eqx.nn.Linear(C, H, use_bias=False, key=keys[12])
        else:
            self.retrieve_gate = None

    def _poly_features(self, x):
        """Element-wise polynomial: phi(x) = sum_{i=1}^{p} a_i * x^i."""
        result = self.poly_coeffs[0] * x
        x_pow = x
        for i in range(1, self.poly_degree):
            x_pow = x_pow * x
            result = result + self.poly_coeffs[i] * x_pow
        return result

    def _process_chunk_linear(self, M, S, q_c, k_c, v_c, a_c, e_c, t_c, g_c):
        """Process one chunk with linear memory."""
        _pe = polar_express_ste if self.pe_ste else polar_express

        pred = jnp.einsum('bhvk,bchk->bchv', M, k_c)
        err = pred - v_c
        D = k_c.shape[-1]
        orig_dtype = err.dtype
        err_f = err.astype(jnp.float32)
        k_f = k_c.astype(jnp.float32)
        u = ((2.0 / D) * jnp.einsum('bchv,bchk->bchvk', err_f, k_f)).astype(orig_dtype)

        if self.omega_window > 1 and g_c is not None:
            u = _omega_aggregate(u, g_c[..., jnp.newaxis], self.omega_window)

        theta = jnp.squeeze(t_c, axis=-1)
        mom_input = -(e_c[..., jnp.newaxis] * u)
        chunk_S, S = linear_scan(S, theta, mom_input)

        chunk_S_orth = _pe(chunk_S, self.ns_steps)

        alpha = jnp.squeeze(a_c, axis=-1)
        M_all, M = linear_scan(M, alpha, chunk_S_orth)

        y_c = jnp.einsum('bchvk,bchk->bchv', M_all, q_c)

        return y_c, LinearMemoryState(M=M, S=S)

    def _process_chunk_deep(self, state, q_c, k_c, v_c, a_c, e_c, t_c, g_c):
        """Process one chunk with deep MLP memory."""
        W1, W2, S_W1, S_W2 = state.W1, state.W2, state.S_W1, state.S_W2
        if HAS_TRITON_PE:
            _pe = triton_polar_express_ste if self.pe_ste else triton_polar_express
        else:
            _pe = polar_express_ste if self.pe_ste else polar_express

        with jax.named_scope("mem_fwd"):
            h = jnp.einsum('bhed,bchd->bche', W2, k_c)
            act = jax.nn.gelu(h)
            y_pred = k_c + jnp.einsum('bhde,bche->bchd', W1, act)
            err = y_pred - v_c

        with jax.named_scope("mem_grad"):
            orig_dtype = err.dtype
            err_f = err.astype(jnp.float32)
            act_f = act.astype(jnp.float32)
            D = k_c.shape[-1]
            scale = 2.0 / D
            u_W1 = (scale * jnp.einsum('bchd,bche->bchde', err_f, act_f)).astype(orig_dtype)
            gelu_prime = _gelu_derivative(h)
            w1t_err = jnp.einsum('bhde,bchd->bche', W1.astype(jnp.float32), err_f)
            chain = w1t_err * gelu_prime
            u_W2 = (scale * jnp.einsum('bche,bchd->bched', chain, k_c.astype(jnp.float32))).astype(orig_dtype)

        with jax.named_scope("omega"):
            if self.omega_window > 1 and g_c is not None:
                g_w1 = g_c[..., jnp.newaxis]
                g_w2 = g_c[..., jnp.newaxis]
                u_W1 = _omega_aggregate(u_W1, g_w1, self.omega_window)
                u_W2 = _omega_aggregate(u_W2, g_w2, self.omega_window)

        theta = jnp.squeeze(t_c, axis=-1)
        alpha = jnp.squeeze(a_c, axis=-1)

        mom_W1 = -(e_c[..., jnp.newaxis] * u_W1)
        mom_W2 = -(e_c[..., jnp.newaxis] * u_W2)

        if self.fused_chunk:
            with jax.named_scope("flash_atlas"):
                y_c, new_state = fused_chunk_scan_kernel(
                    W1, W2, S_W1, S_W2,
                    mom_W1, mom_W2, theta, alpha, q_c,
                    self.ns_steps, self.pe_ste)
            return y_c, new_state

        D, E = self.head_dim, self.expand_dim
        if D == E:
            with jax.named_scope("batched_momentum_scan"):
                S_cat = jnp.concatenate(
                    [S_W1.reshape(*S_W1.shape[:2], -1),
                     S_W2.reshape(*S_W2.shape[:2], -1)], axis=-1)
                mom_cat = jnp.concatenate(
                    [mom_W1.reshape(*mom_W1.shape[:3], -1),
                     mom_W2.reshape(*mom_W2.shape[:3], -1)], axis=-1)
                chunk_S_cat, S_final_cat = linear_scan(S_cat, theta, mom_cat)
                DD = D * E
                chunk_S_W1 = chunk_S_cat[..., :DD].reshape(*chunk_S_cat.shape[:3], D, E)
                chunk_S_W2 = chunk_S_cat[..., DD:].reshape(*chunk_S_cat.shape[:3], E, D)
                S_W1 = S_final_cat[..., :DD].reshape(*S_final_cat.shape[:2], D, E)
                S_W2 = S_final_cat[..., DD:].reshape(*S_final_cat.shape[:2], E, D)

            with jax.named_scope("batched_polar_express"):
                stacked_S = jnp.stack([chunk_S_W1, chunk_S_W2], axis=3)
                stacked_orth = _pe(stacked_S, self.ns_steps)
                chunk_S_W1_orth = stacked_orth[:, :, :, 0]
                chunk_S_W2_orth = stacked_orth[:, :, :, 1]

            with jax.named_scope("batched_memory_scan"):
                W_cat = jnp.concatenate(
                    [W1.reshape(*W1.shape[:2], -1),
                     W2.reshape(*W2.shape[:2], -1)], axis=-1)
                orth_cat = jnp.concatenate(
                    [chunk_S_W1_orth.reshape(*chunk_S_W1_orth.shape[:3], -1),
                     chunk_S_W2_orth.reshape(*chunk_S_W2_orth.shape[:3], -1)], axis=-1)
                W_all_cat, W_final_cat = linear_scan(W_cat, alpha, orth_cat)
                W1_all = W_all_cat[..., :DD].reshape(*W_all_cat.shape[:3], D, E)
                W2_all = W_all_cat[..., DD:].reshape(*W_all_cat.shape[:3], E, D)
                W1 = W_final_cat[..., :DD].reshape(*W_final_cat.shape[:2], D, E)
                W2 = W_final_cat[..., DD:].reshape(*W_final_cat.shape[:2], E, D)
        else:
            def _fused_scan(S_init, W_init, theta, alpha, mom_input, name):
                with jax.named_scope(f"{name}/momentum_scan"):
                    chunk_S, S_final = linear_scan(S_init, theta, mom_input)
                with jax.named_scope(f"{name}/polar_express"):
                    chunk_S_orth = _pe(chunk_S, self.ns_steps)
                with jax.named_scope(f"{name}/memory_scan"):
                    W_all, W_final = linear_scan(W_init, alpha, chunk_S_orth)
                return W_all, W_final, S_final

            with jax.named_scope("fused_W1"):
                W1_all, W1, S_W1 = _fused_scan(S_W1, W1, theta, alpha, mom_W1, "W1")
            with jax.named_scope("fused_W2"):
                W2_all, W2, S_W2 = _fused_scan(S_W2, W2, theta, alpha, mom_W2, "W2")

        with jax.named_scope("mem_output"):
            h_q = jnp.einsum('bched,bchd->bche', W2_all, q_c)
            g_q = jax.nn.gelu(h_q)
            y_c = q_c + jnp.einsum('bchde,bche->bchd', W1_all, g_q)

        return y_c, DeepMemoryState(W1=W1, W2=W2, S_W1=S_W1, S_W2=S_W2)

    def __call__(self, x, memory_state=None):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        E = self.expand_dim
        cs = self.chunk_size
        dim_inner = H * D

        with jax.named_scope("qkv_proj"):
            q = self.conv_q((x @ self.c_q.weight.T).reshape(B, T, dim_inner)).reshape(B, T, H, D)
            k = self.conv_k((x @ self.c_k.weight.T).reshape(B, T, dim_inner)).reshape(B, T, H, D)
            v = self.conv_v((x @ self.c_v.weight.T).reshape(B, T, dim_inner)).reshape(B, T, H, D)

        with jax.named_scope("qk_norm_poly"):
            if self.poly_degree > 0:
                q = rms_norm(q)
                q = self._poly_features(q)
                k = self._poly_features(k)
                k = rms_norm(k)
            else:
                q, k = rms_norm(q), rms_norm(k)

        with jax.named_scope("gates"):
            def _gate(layer, x):
                out = x @ layer.weight.T
                if layer.bias is not None:
                    out = out + layer.bias
                return jax.nn.sigmoid(out.reshape(B, T, H, 1))

            alpha = _gate(self.gate_alpha, x)
            eta = _gate(self.gate_eta, x) * self.max_lr
            theta = _gate(self.gate_theta, x)

            gamma = None
            if self.omega_window > 1 and self.gate_gamma is not None:
                gamma = _gate(self.gate_gamma, x)

        if memory_state is None:
            if self.deep_memory:
                W1 = jnp.broadcast_to(
                    self.W1_init[jnp.newaxis].astype(x.dtype), (B, H, D, E))
                W2 = jnp.broadcast_to(
                    self.W2_init[jnp.newaxis].astype(x.dtype), (B, H, E, D))
                S_W1 = jnp.zeros((B, H, D, E), dtype=x.dtype)
                S_W2 = jnp.zeros((B, H, E, D), dtype=x.dtype)
                memory_state = DeepMemoryState(W1=W1, W2=W2, S_W1=S_W1, S_W2=S_W2)
            else:
                M = jnp.zeros((B, H, D, D), dtype=x.dtype)
                S = jnp.zeros((B, H, D, D), dtype=x.dtype)
                memory_state = LinearMemoryState(M=M, S=S)

        T_orig = T
        if T % cs != 0:
            pad = cs - T % cs
            q = jnp.pad(q, ((0, 0), (0, pad), (0, 0), (0, 0)))
            k = jnp.pad(k, ((0, 0), (0, pad), (0, 0), (0, 0)))
            v = jnp.pad(v, ((0, 0), (0, pad), (0, 0), (0, 0)))
            alpha = jnp.pad(alpha, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=1.0)
            eta = jnp.pad(eta, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=0.0)
            theta = jnp.pad(theta, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=0.0)
            if gamma is not None:
                gamma = jnp.pad(gamma, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=0.0)
            T = q.shape[1]

        n_chunks = T // cs

        def chunk_body(carry, chunk_data):
            mem_state = jax.lax.stop_gradient(carry) if self.stop_grad_chunks else carry
            q_c, k_c, v_c, a_c, e_c, t_c, g_c = chunk_data
            if self.deep_memory:
                y_c, new_state = self._process_chunk_deep(
                    mem_state, q_c, k_c, v_c, a_c, e_c, t_c, g_c)
            else:
                y_c, new_state = self._process_chunk_linear(
                    mem_state.M, mem_state.S, q_c, k_c, v_c, a_c, e_c, t_c, g_c)
            return new_state, y_c

        process_fn = jax.checkpoint(chunk_body) if self.use_checkpoint else chunk_body

        if self.fused_chunk:
            def _chunk_bt(x):
                return x.reshape(B, n_chunks, cs, *x.shape[2:])

            q_bt = _chunk_bt(q)
            k_bt = _chunk_bt(k)
            v_bt = _chunk_bt(v)
            a_bt = _chunk_bt(alpha)
            e_bt = _chunk_bt(eta)
            t_bt = _chunk_bt(theta)
            g_bt = _chunk_bt(gamma) if gamma is not None else jnp.zeros_like(a_bt)

            ys = []
            state = memory_state
            for i in range(n_chunks):
                chunk_data = (q_bt[:, i], k_bt[:, i], v_bt[:, i],
                              a_bt[:, i], e_bt[:, i], t_bt[:, i], g_bt[:, i])
                state, y_c = process_fn(state, chunk_data)
                ys.append(y_c)
            final_state = state
            all_y = jnp.stack(ys, axis=1)
        else:
            def _chunk(x):
                return x.reshape(B, n_chunks, cs, *x.shape[2:]).transpose(1, 0, 2, *range(3, x.ndim + 1))

            q_chunks = _chunk(q)
            k_chunks = _chunk(k)
            v_chunks = _chunk(v)
            a_chunks = _chunk(alpha)
            e_chunks = _chunk(eta)
            t_chunks = _chunk(theta)
            g_chunks = _chunk(gamma) if gamma is not None else jnp.zeros_like(a_chunks)

            xs = (q_chunks, k_chunks, v_chunks, a_chunks, e_chunks, t_chunks, g_chunks)
            final_state, all_y = lax.scan(process_fn, memory_state, xs)

        if self.fused_chunk:
            y = all_y.reshape(B, T, H, D)
        else:
            y = jnp.transpose(all_y, (1, 0, 2, 3, 4)).reshape(B, T, H, D)

        y = y[:, :T_orig]

        if self.retrieve_gate is not None:
            gate = jax.nn.sigmoid(x @ self.retrieve_gate.weight.T)
            y = y * gate[..., jnp.newaxis]

        y = y.reshape(B, T_orig, -1)
        y = (y @ self.c_proj.weight.T).reshape(B, T_orig, C)

        return y, final_state
