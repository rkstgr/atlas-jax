"""Stateless tensor ops used across the Atlas model.

Pure functions (no modules, no parameters): normalization, dropout, activation
derivatives, the Omega sliding-window aggregation, and the linear recurrence
scan. Kept here so `model.py` and `memory_layer.py` stay focused on module
definitions rather than low-level math.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax

from atlas_jax import kernels


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
    return cdf + x * pdf  # stays f32 — used in gradient chain


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


def linear_scan(h_init, gates, inputs):
    """Linear recurrence: h_t = gate_t * h_{t-1} + input_t.

    Uses fused Triton kernel when available (2-4× faster than associative scan),
    falls back to jax.lax.associative_scan otherwise.

    Args:
        h_init: (B, H, ...) initial state
        gates: (B, T, H) scalar gates per timestep
        inputs: (B, T, H, ...) per-timestep inputs

    Returns:
        h_all: (B, T, H, ...) all intermediate states
        h_final: (B, H, ...) final state
    """
    if kernels.HAS_TRITON_SCAN:
        return kernels.triton_linear_scan(h_init, gates, inputs)

    # Fallback: associative scan (O(log n) parallel depth)
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
