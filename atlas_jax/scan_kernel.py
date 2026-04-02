"""Optimized linear scan implementations for Atlas.

Provides:
1. associative_linear_scan: O(log n) parallel via jax.lax.associative_scan
2. sequential_linear_scan: Simple O(n) sequential via lax.scan with (B*H, DD) flattening

The associative scan is faster on GPU due to parallelism across timesteps.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax


def associative_linear_scan(h_init, gates, inputs):
    """Linear recurrence via associative scan (O(log n) parallel depth).

    The monoid is: (g1, x1) ⊕ (g2, x2) = (g1*g2, g2*x1 + x2)

    Args:
        h_init: (B, H, ...) initial state
        gates: (B, T, H) scalar gates per timestep
        inputs: (B, T, H, ...) per-timestep inputs

    Returns:
        h_all: (B, T, H, ...) all intermediate states
        h_final: (B, H, ...) final state
    """
    extra_dims = inputs.ndim - gates.ndim
    gates_expanded = gates
    for _ in range(extra_dims):
        gates_expanded = gates_expanded[..., jnp.newaxis]

    # Fold initial state into first position
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


def sequential_linear_scan(h_init, gates, inputs):
    """Linear recurrence via sequential lax.scan with flattened (B*H) batch.

    Simpler than associative scan, O(n) work, O(n) depth.
    Flattening (B, H) into one dimension gives XLA a larger parallel dimension.

    Args:
        h_init: (B, H, ...) initial state
        gates: (B, T, H) scalar gates
        inputs: (B, T, H, ...) per-timestep inputs

    Returns:
        h_all: (B, T, H, ...) all states
        h_final: (B, H, ...) final state
    """
    B, T, H = gates.shape
    state_shape = h_init.shape[2:]
    DD = 1
    for s in state_shape:
        DD *= s

    # Flatten (B, H) -> B*H for better GPU utilization
    h_flat = h_init.reshape(B * H, DD)
    inp_flat = inputs.reshape(B, T, H, DD).transpose(1, 0, 2, 3).reshape(T, B * H, DD)
    g_flat = gates.transpose(1, 0, 2).reshape(T, B * H)

    def step(h, gi):
        g, x = gi
        h_new = g[:, jnp.newaxis] * h + x
        return h_new, h_new

    h_final_flat, h_all_flat = lax.scan(step, h_flat, (g_flat, inp_flat))

    h_all = h_all_flat.reshape(T, B, H, *state_shape)
    # Transpose: (T, B, H, ...) -> (B, T, H, ...)
    perm = (1, 0, 2) + tuple(range(3, 3 + len(state_shape)))
    h_all = h_all.transpose(perm)
    h_final = h_final_flat.reshape(B, H, *state_shape)

    return h_all, h_final
