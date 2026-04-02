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


def triton_linear_scan(h_init, gates, inputs):
    """Fused linear scan using Triton kernel via jax_triton.

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

    # Flatten state dimensions
    h_flat = h_init.reshape(B, H, DD)
    inp_flat = inputs.reshape(B, T, H, DD)

    BLOCK = min(1024, triton.next_power_of_2(DD))
    grid = (B * H * triton.cdiv(DD, BLOCK),)

    # Compute strides
    s_hi_0, s_hi_1 = H * DD, DD
    s_g_0, s_g_1, s_g_2 = T * H, H, 1
    s_i_0, s_i_1, s_i_2 = T * H * DD, H * DD, DD
    s_o_0, s_o_1, s_o_2 = T * H * DD, H * DD, DD

    h_all_flat = jt.triton_call(
        h_flat, gates, inp_flat,
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

    h_all = h_all_flat.reshape(B, T, H, *state_shape)
    h_final = h_all[:, -1]
    return h_all, h_final
