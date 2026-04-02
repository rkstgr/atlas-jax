"""Fused Triton linear scan kernel for JAX, via jax_triton.

Direct port of nanochat's atlas_kernels.py fused_linear_scan, adapted
to work with JAX arrays via jax_triton.triton_call.
"""

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
import jax_triton as jt
from functools import partial


@triton.jit
def _scan_fwd_kernel(
    h_init_ptr, gates_ptr, inputs_ptr, h_all_ptr,
    T: tl.constexpr,
    s_hi_0, s_hi_1,
    s_g_0, s_g_1, s_g_2,
    s_i_0, s_i_1, s_i_2,
    s_o_0, s_o_1, s_o_2,
    H: tl.constexpr,
    DD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Forward linear scan: h_t = gate_t * h_{t-1} + input_t.

    Each program handles BLOCK elements of the DD-dimensional state vector
    for one (batch, head) pair, iterating sequentially over T timesteps.
    """
    pid = tl.program_id(0)
    num_dd_blocks = tl.cdiv(DD, BLOCK)
    bh = pid // num_dd_blocks
    dd_block = pid % num_dd_blocks
    b = bh // H
    h = bh % H

    offs = dd_block * BLOCK + tl.arange(0, BLOCK)
    mask = offs < DD

    state = tl.load(h_init_ptr + b * s_hi_0 + h * s_hi_1 + offs, mask=mask, other=0.0)

    for t in range(T):
        gate = tl.load(gates_ptr + b * s_g_0 + t * s_g_1 + h * s_g_2)
        inp = tl.load(inputs_ptr + b * s_i_0 + t * s_i_1 + h * s_i_2 + offs, mask=mask, other=0.0)
        state = gate * state + inp
        tl.store(h_all_ptr + b * s_o_0 + t * s_o_1 + h * s_o_2 + offs, state, mask=mask)


def _run_triton_scan(h_init_flat, gates, inp_flat):
    """Run the Triton forward scan kernel.

    Args:
        h_init_flat: (B, H, DD) initial state
        gates: (B, T, H) scalar gates
        inp_flat: (B, T, H, DD) per-timestep inputs

    Returns:
        h_all_flat: (B, T, H, DD) all states
    """
    B, T, H, DD = inp_flat.shape
    BLOCK = min(1024, triton.next_power_of_2(DD))
    grid = (B * H * triton.cdiv(DD, BLOCK),)

    s_hi_0, s_hi_1 = H * DD, DD
    s_g_0, s_g_1, s_g_2 = T * H, H, 1
    s_i_0, s_i_1, s_i_2 = T * H * DD, H * DD, DD
    s_o_0, s_o_1, s_o_2 = T * H * DD, H * DD, DD

    return jt.triton_call(
        h_init_flat, gates, inp_flat,
        kernel=_scan_fwd_kernel,
        out_shape=jax.ShapeDtypeStruct((B, T, H, DD), inp_flat.dtype),
        grid=grid,
        T=T,
        s_hi_0=s_hi_0, s_hi_1=s_hi_1,
        s_g_0=s_g_0, s_g_1=s_g_1, s_g_2=s_g_2,
        s_i_0=s_i_0, s_i_1=s_i_1, s_i_2=s_i_2,
        s_o_0=s_o_0, s_o_1=s_o_1, s_o_2=s_o_2,
        H=H, DD=DD, BLOCK=BLOCK,
    )


@partial(jax.custom_vjp, nondiff_argnums=())
def triton_linear_scan(h_init, gates, inputs):
    """Fused linear scan with custom backward, both using Triton kernels.

    Args:
        h_init: (B, H, ...) initial state
        gates: (B, T, H) scalar gates per timestep
        inputs: (B, T, H, ...) per-timestep inputs

    Returns:
        h_all: (B, T, H, ...) all intermediate states
        h_final: (B, H, ...) final state
    """
    B, T, H = gates.shape
    state_shape = h_init.shape[2:]
    DD = 1
    for s in state_shape:
        DD *= s

    h_flat = h_init.reshape(B, H, DD)
    inp_flat = inputs.reshape(B, T, H, DD)

    h_all_flat = _run_triton_scan(h_flat, gates, inp_flat)

    h_all = h_all_flat.reshape(B, T, H, *state_shape)
    h_final = h_all[:, -1]
    return h_all, h_final


def _triton_scan_fwd(h_init, gates, inputs):
    h_all, h_final = triton_linear_scan(h_init, gates, inputs)
    return (h_all, h_final), (h_init, h_all, gates)


def _triton_scan_bwd(res, g):
    """Backward: reverse scan (also a linear recurrence).

    dh_acc[t] = grad[t] + gate[t+1] * dh_acc[t+1]
    Rewritten as forward scan on time-reversed sequences.
    """
    h_init, h_all, gates = res
    grad_h_all, grad_h_final = g

    B, T, H = gates.shape
    state_shape = h_init.shape[2:]
    DD = 1
    for s in state_shape:
        DD *= s

    # Combine direct and final-state gradients
    grad_all_flat = grad_h_all.reshape(B, T, H, DD)
    grad_all_flat = grad_all_flat.at[:, -1].add(grad_h_final.reshape(B, H, DD)[:, jnp.newaxis, :, :].squeeze(1))

    # Reverse scan: flip time, set first reversed gate to 0
    rev_grad = grad_all_flat[:, ::-1]  # (B, T, H, DD)
    rev_gates = jnp.concatenate([
        jnp.zeros_like(gates[:, :1]),
        gates[:, 1:][:, ::-1]
    ], axis=1)  # (B, T, H)

    zero_init = jnp.zeros((B, H, DD), dtype=h_init.dtype)
    dh_acc_rev = _run_triton_scan(zero_init, rev_gates, rev_grad)
    dh_acc = dh_acc_rev[:, ::-1]  # (B, T, H, DD)

    # grad_inputs = dh_acc
    grad_inputs = dh_acc.reshape(B, T, H, *state_shape)

    # grad_gates[t] = sum_d (dh_acc[t,d] * h_{t-1,d})
    h_all_flat = h_all.reshape(B, T, H, DD)
    h_init_flat = h_init.reshape(B, H, DD)
    h_prev = jnp.concatenate([h_init_flat[:, jnp.newaxis], h_all_flat[:, :-1]], axis=1)
    grad_gates = jnp.sum(dh_acc * h_prev, axis=-1)  # (B, T, H)

    # grad_h_init = gate[0] * dh_acc[0]
    grad_h_init = (gates[:, 0:1, :, jnp.newaxis] * dh_acc[:, 0:1]).squeeze(1).reshape(B, H, *state_shape)

    return grad_h_init, grad_gates, grad_inputs


triton_linear_scan.defvjp(_triton_scan_fwd, _triton_scan_bwd)
