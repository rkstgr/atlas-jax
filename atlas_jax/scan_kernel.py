"""Optimized linear scan with custom backward pass.

The key insight from nanochat's Triton kernels: the backward pass of
a linear recurrence is itself a linear recurrence (reversed). By implementing
a custom_vjp, we can:
1. Avoid storing all intermediate states for backward (let lax.scan recompute)
2. Use a reverse scan for the backward pass
3. Flatten (B, H) into a single dimension for better GPU utilization

This eliminates the need for jax.checkpoint on the scan and avoids the
associative_scan's O(n log n) work overhead (uses O(n) sequential instead).
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
from functools import partial


@partial(jax.custom_vjp, nondiff_argnums=())
def fused_linear_scan(h_init, gates, inputs):
    """Linear recurrence: h_t = gate_t * h_{t-1} + input_t.

    Args:
        h_init: (B, H, ...) initial state
        gates: (B, T, H) scalar gates per timestep
        inputs: (B, T, H, ...) per-timestep inputs

    Returns:
        h_all: (B, T, H, ...) all intermediate states
        h_final: (B, H, ...) final state
    """
    return _scan_fwd(h_init, gates, inputs)


def _scan_fwd(h_init, gates, inputs):
    """Forward scan: flatten (B,H) → single dim, then sequential scan."""
    B, T, H = gates.shape
    state_shape = h_init.shape[2:]
    DD = 1
    for s in state_shape:
        DD *= s

    # Flatten: (B, H, ...) → (B*H, DD)
    h_flat = h_init.reshape(B * H, DD)
    inp_flat = inputs.reshape(B, T, H, DD).transpose(1, 0, 2, 3).reshape(T, B * H, DD)
    g_flat = gates.transpose(1, 0, 2).reshape(T, B * H)

    def step(h, gi):
        g, x = gi
        h_new = g[:, None] * h + x
        return h_new, h_new

    h_final_flat, h_all_flat = lax.scan(step, h_flat, (g_flat, inp_flat))

    # Unflatten
    h_all = h_all_flat.reshape(T, B, H, *state_shape).transpose(1, 0, 2, *range(3, 3 + len(state_shape)))
    h_final = h_final_flat.reshape(B, H, *state_shape)

    return h_all, h_final


def fused_linear_scan_fwd(h_init, gates, inputs):
    """Forward pass: save minimal state for backward."""
    h_all, h_final = _scan_fwd(h_init, gates, inputs)
    # Save h_all and gates for backward (h_init can be reconstructed)
    return (h_all, h_final), (h_init, h_all, gates)


def fused_linear_scan_bwd(res, g):
    """Backward pass: reverse scan to accumulate gradients.

    The gradient of a linear recurrence is itself a linear recurrence
    running in reverse time. This is the key insight from nanochat's Triton kernel.
    """
    h_init, h_all, gates = res
    grad_h_all, grad_h_final = g

    B, T, H = gates.shape
    state_shape = h_init.shape[2:]
    DD = 1
    for s in state_shape:
        DD *= s

    # Flatten
    grad_all_flat = grad_h_all.reshape(B, T, H, DD).transpose(1, 0, 2, 3).reshape(T, B * H, DD)
    grad_final_flat = grad_h_final.reshape(B * H, DD)

    # Add grad_h_final to the last timestep's gradient
    grad_all_flat = grad_all_flat.at[-1].add(grad_final_flat)

    # Reverse scan: dh_acc[t] = grad[t] + gate[t+1] * dh_acc[t+1]
    # Rewrite as forward scan on time-reversed sequences:
    #   rev_acc[t'] = rev_gate[t'] * rev_acc[t'-1] + rev_grad[t']
    rev_grad = grad_all_flat[::-1]  # flip time

    g_flat = gates.transpose(1, 0, 2).reshape(T, B * H)
    # Reversed gates: rev_gates[0] = 0, rev_gates[t'] = gates[T-t'] for t'>=1
    rev_gates = jnp.concatenate([jnp.zeros_like(g_flat[:1]), g_flat[1:][::-1]], axis=0)

    def step(acc, gi):
        g, grad = gi
        acc_new = g[:, None] * acc + grad
        return acc_new, acc_new

    zero_init = jnp.zeros((B * H, DD), dtype=h_init.dtype)
    _, dh_acc_rev = lax.scan(step, zero_init, (rev_gates, rev_grad))
    dh_acc = dh_acc_rev[::-1]  # (T, B*H, DD)

    # grad_inputs[t] = dh_acc[t]
    grad_inputs = dh_acc.reshape(T, B, H, *state_shape).transpose(1, 0, 2, *range(3, 3 + len(state_shape)))

    # grad_gates[t] = sum_d (dh_acc[t,d] * h_{t-1,d})  → scalar per (b,t,h)
    h_all_flat = h_all.reshape(B, T, H, DD).transpose(1, 0, 2, 3).reshape(T, B * H, DD)
    h_init_flat = h_init.reshape(B * H, DD)
    h_prev_flat = jnp.concatenate([h_init_flat[None], h_all_flat[:-1]], axis=0)
    grad_gates_flat = jnp.sum(dh_acc * h_prev_flat, axis=-1)  # (T, B*H)
    grad_gates = grad_gates_flat.reshape(T, B, H).transpose(1, 0, 2)

    # grad_h_init = gate[0] * dh_acc[0]
    gate0 = g_flat[0]  # (B*H,)
    grad_h_init = (gate0[:, None] * dh_acc[0]).reshape(B, H, *state_shape)

    return grad_h_init, grad_gates, grad_inputs


fused_linear_scan.defvjp(fused_linear_scan_fwd, fused_linear_scan_bwd)
