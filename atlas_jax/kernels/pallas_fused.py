"""Pallas fused chunk scan for Atlas deep memory (Mosaic GPU).

JAX-native replacement for fused_chunk.py (Triton). Same algorithm:
keeps carry state (W1, W2, S_W1, S_W2) in registers across all timesteps
within a chunk, eliminating HBM round-trips from lax.scan.

Mosaic GPU programming model:
- 1 Pallas thread = 1 CUDA warpgroup (128 lanes)
- Carry state lives in registers (RMEM) across fori_loop iterations
- All per-timestep data loaded from GMEM per iteration (no SMEM)
- PE coefficients baked as compile-time constants

Forward:  Single Pallas kernel per chunk (carry in GPU registers).
Backward: custom_vjp — recompute forward with regular JAX ops, then vjp.

Requires D == E (memory_expand=1).
"""

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from atlas_jax.memory_layer import (
    polar_express_ste, polar_express, POLAR_EXPRESS_COEFFS, DeepMemoryState,
)


# ---------------------------------------------------------------------------
# Pallas kernel (Mosaic GPU) — all data via GMEM refs
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _make_kernel(cs, D, ns_steps):
    """Build a Pallas kernel with static cs/D/ns_steps baked in.

    All inputs/outputs are GMEM refs (full arrays). Each grid cell uses
    program_id to index its (b,h) pair. PE coefficients baked as constants.
    """
    pe_abc = [(float(a), float(b), float(c))
              for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]]

    def kernel(
        # Input GMEM refs (full arrays, indexed by program_id)
        W1_ref,    # (BH, D, D)
        W2_ref,    # (BH, D, D)
        SW1_ref,   # (BH, D, D)
        SW2_ref,   # (BH, D, D)
        momW1_ref, # (BH, cs, D, D)
        momW2_ref, # (BH, cs, D, D)
        theta_ref, # (BH, cs)
        alpha_ref, # (BH, cs)
        q_ref,     # (BH, cs, D)
        # Output GMEM refs
        y_ref,     # (BH, cs, D)
        W1o_ref,   # (BH, D, D)
        W2o_ref,   # (BH, D, D)
        SW1o_ref,  # (BH, D, D)
        SW2o_ref,  # (BH, D, D)
    ):
        bh = pl.program_id(0)

        # Load carry from GMEM → registers (once)
        W1 = W1_ref[bh].astype(jnp.float32)
        W2 = W2_ref[bh].astype(jnp.float32)
        SW1 = SW1_ref[bh].astype(jnp.float32)
        SW2 = SW2_ref[bh].astype(jnp.float32)

        # Sequential scan — carry stays in registers, per-timestep
        # data loaded from GMEM each iteration (no dynamic_slice on
        # register arrays, no SMEM alignment constraints).
        def body(t, state):
            W1, W2, SW1, SW2 = state

            # Per-timestep loads from GMEM
            theta_t = theta_ref[bh, t].astype(jnp.float32)
            alpha_t = alpha_ref[bh, t].astype(jnp.float32)
            q_t = q_ref[bh, t].astype(jnp.float32)        # (D,)
            mW1 = momW1_ref[bh, t].astype(jnp.float32)    # (D, D)
            mW2 = momW2_ref[bh, t].astype(jnp.float32)

            # Momentum: S = theta * S + mom
            SW1 = theta_t * SW1 + mW1
            SW2 = theta_t * SW2 + mW2

            # PE(SW1): Frobenius normalize → Newton-Schulz
            # Use rsqrt (native GPU op) instead of sqrt (not in Mosaic GPU)
            frob_sq = jnp.sum(SW1 * SW1) + 1e-12
            frob = frob_sq * lax.rsqrt(frob_sq)  # sqrt(x) = x * rsqrt(x)
            X = SW1 / (frob * 1.01 + 1e-6)
            for a, b, c in pe_abc:
                # X @ X^T via dot_general — avoids transpose primitive
                A = lax.dot_general(X, X, (([1], [1]), ([], [])))
                B = b * A + c * jnp.dot(A, A)
                X = a * X + jnp.dot(B, X)
            SW1_orth = X

            # PE(SW2)
            frob_sq = jnp.sum(SW2 * SW2) + 1e-12
            frob = frob_sq * lax.rsqrt(frob_sq)
            X = SW2 / (frob * 1.01 + 1e-6)
            for a, b, c in pe_abc:
                A = lax.dot_general(X, X, (([1], [1]), ([], [])))
                B = b * A + c * jnp.dot(A, A)
                X = a * X + jnp.dot(B, X)
            SW2_orth = X

            # Memory: W = alpha * W + PE(S)
            W1 = alpha_t * W1 + SW1_orth
            W2 = alpha_t * W2 + SW2_orth

            # Output: y = q + W1 @ gelu(W2 @ q)
            h_q = jnp.dot(W2, q_t)
            g_q = h_q * 0.5 * (1.0 + lax.erf(h_q * 0.7071067811865476))
            y_t = q_t + jnp.dot(W1, g_q)

            # Write output directly to GMEM (no register accumulator)
            y_ref[bh, t] = y_t.astype(y_ref.dtype)

            return W1, W2, SW1, SW2

        W1, W2, SW1, SW2 = lax.fori_loop(
            0, cs, body, (W1, W2, SW1, SW2))

        # Write final carry to GMEM
        W1o_ref[bh] = W1.astype(W1o_ref.dtype)
        W2o_ref[bh] = W2.astype(W2o_ref.dtype)
        SW1o_ref[bh] = SW1.astype(SW1o_ref.dtype)
        SW2o_ref[bh] = SW2.astype(SW2o_ref.dtype)

    return kernel


# ---------------------------------------------------------------------------
# Pallas call wrapper
# ---------------------------------------------------------------------------

def _pallas_fused_fwd(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                       ns_steps):
    """Run fused forward via Pallas Mosaic GPU kernel."""
    B, H, D, E = W1.shape
    assert D == E, f"Pallas fused kernel requires D==E, got D={D}, E={E}"
    cs = momW1.shape[1]
    dtype = W1.dtype
    BH = B * H

    # Mosaic GPU requires every loaded/stored array to have total elements
    # that is a multiple of 128 (warpgroup size). For (D,) vectors this
    # means D must be a multiple of 128. Pad D → BD.
    BD = ((D + 127) // 128) * 128

    # Reshape (B, H, ...) → (B*H, ...) for grid=(B*H,)
    W1f = W1.reshape(BH, D, D)
    W2f = W2.reshape(BH, D, D)
    SW1f = SW1.reshape(BH, D, D)
    SW2f = SW2.reshape(BH, D, D)
    mW1f = momW1.transpose(0, 2, 1, 3, 4).reshape(BH, cs, D, D)
    mW2f = momW2.transpose(0, 2, 1, 3, 4).reshape(BH, cs, D, D)
    tf = theta.transpose(0, 2, 1).reshape(BH, cs)
    af = alpha.transpose(0, 2, 1).reshape(BH, cs)
    qf = q.transpose(0, 2, 1, 3).reshape(BH, cs, D)

    # Pad D → BD (zeros don't affect matmul correctness)
    if BD != D:
        p = BD - D
        W1f = jnp.pad(W1f, ((0, 0), (0, p), (0, p)))
        W2f = jnp.pad(W2f, ((0, 0), (0, p), (0, p)))
        SW1f = jnp.pad(SW1f, ((0, 0), (0, p), (0, p)))
        SW2f = jnp.pad(SW2f, ((0, 0), (0, p), (0, p)))
        mW1f = jnp.pad(mW1f, ((0, 0), (0, 0), (0, p), (0, p)))
        mW2f = jnp.pad(mW2f, ((0, 0), (0, 0), (0, p), (0, p)))
        qf = jnp.pad(qf, ((0, 0), (0, 0), (0, p)))

    # All GMEM — full array, trivial index_map, manual indexing via program_id
    def _gmem(shape):
        return pl.BlockSpec(block_shape=shape,
                            index_map=lambda i: (0,) * len(shape),
                            memory_space=plgpu.GMEM)

    carry_spec = _gmem((BH, BD, BD))
    mom_spec = _gmem((BH, cs, BD, BD))
    gate_spec = _gmem((BH, cs))
    qy_spec = _gmem((BH, cs, BD))

    kernel = _make_kernel(cs, BD, ns_steps)

    results = pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct((BH, cs, BD), dtype),
            jax.ShapeDtypeStruct((BH, BD, BD), dtype),
            jax.ShapeDtypeStruct((BH, BD, BD), dtype),
            jax.ShapeDtypeStruct((BH, BD, BD), dtype),
            jax.ShapeDtypeStruct((BH, BD, BD), dtype),
        ],
        in_specs=([carry_spec] * 4 + [mom_spec] * 2
                  + [gate_spec] * 2 + [qy_spec]),
        out_specs=[qy_spec] + [carry_spec] * 4,
        grid=(BH,),
    )(W1f, W2f, SW1f, SW2f, mW1f, mW2f, tf, af, qf)

    yf, W1of, W2of, SW1of, SW2of = results

    # Unpad + reshape (B*H, ...) → (B, H, ...)
    if BD != D:
        yf = yf[..., :D]
        W1of = W1of[:, :D, :D]
        W2of = W2of[:, :D, :D]
        SW1of = SW1of[:, :D, :D]
        SW2of = SW2of[:, :D, :D]

    y = yf.reshape(B, H, cs, D).transpose(0, 2, 1, 3)
    W1_out = W1of.reshape(B, H, D, D)
    W2_out = W2of.reshape(B, H, D, D)
    SW1_out = SW1of.reshape(B, H, D, D)
    SW2_out = SW2of.reshape(B, H, D, D)

    return y, W1_out, W2_out, SW1_out, SW2_out


# ---------------------------------------------------------------------------
# Pure JAX forward (for backward recomputation via vjp)
# ---------------------------------------------------------------------------

def _jax_linear_scan(init, gate, x):
    """Pure JAX linear scan: S[t] = gate[t] * S[t-1] + x[t]."""
    extra_dims = x.ndim - gate.ndim
    gate_exp = gate
    for _ in range(extra_dims):
        gate_exp = gate_exp[..., jnp.newaxis]

    def step(carry, inp):
        g, xi = inp
        new = g * carry + xi
        return new, new

    gate_t = jnp.moveaxis(gate_exp, 1, 0)
    x_t = jnp.moveaxis(x, 1, 0)
    final, all_t = jax.lax.scan(step, init, (gate_t, x_t))
    return jnp.moveaxis(all_t, 0, 1), final


def _regular_fwd(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                  ns_steps, pe_ste):
    """Same computation as Pallas kernel, using regular JAX ops."""
    _pe = polar_express_ste if pe_ste else polar_express

    chunk_SW1, SW1_f = _jax_linear_scan(SW1, theta, momW1)
    chunk_SW2, SW2_f = _jax_linear_scan(SW2, theta, momW2)

    SW1_orth = _pe(chunk_SW1, ns_steps)
    SW2_orth = _pe(chunk_SW2, ns_steps)

    W1_all, W1_f = _jax_linear_scan(W1, alpha, SW1_orth)
    W2_all, W2_f = _jax_linear_scan(W2, alpha, SW2_orth)

    h_q = jnp.einsum('bched,bchd->bche', W2_all, q)
    g_q = jax.nn.gelu(h_q)
    y = q + jnp.einsum('bchde,bche->bchd', W1_all, g_q)

    out_dtype = W1.dtype
    return (y.astype(out_dtype),
            DeepMemoryState(W1=W1_f.astype(out_dtype),
                            W2=W2_f.astype(out_dtype),
                            S_W1=SW1_f.astype(out_dtype),
                            S_W2=SW2_f.astype(out_dtype)))


# ---------------------------------------------------------------------------
# Public API: fused_chunk_scan with custom_vjp
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(9, 10))
def fused_chunk_scan(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                      ns_steps, pe_ste):
    """Fused chunk scan via Pallas Mosaic GPU."""
    out_dtype = W1.dtype
    y, W1o, W2o, SW1o, SW2o = _pallas_fused_fwd(
        W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q, ns_steps)
    return (y.astype(out_dtype),
            DeepMemoryState(W1=W1o.astype(out_dtype),
                            W2=W2o.astype(out_dtype),
                            S_W1=SW1o.astype(out_dtype),
                            S_W2=SW2o.astype(out_dtype)))


def _fused_scan_fwd(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                     ns_steps, pe_ste):
    result = fused_chunk_scan(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                               ns_steps, pe_ste)
    residuals = (W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q)
    return result, residuals


def _fused_scan_bwd(ns_steps, pe_ste, residuals, grads):
    W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q = residuals
    grad_y, grad_state = grads

    def fwd_for_vjp(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q):
        return _regular_fwd(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                            ns_steps, pe_ste)

    _, vjp_fn = jax.vjp(fwd_for_vjp, W1, W2, SW1, SW2, momW1, momW2,
                         theta, alpha, q)
    return vjp_fn((grad_y, grad_state))


fused_chunk_scan.defvjp(_fused_scan_fwd, _fused_scan_bwd)


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

def pallas_available():
    """Check if Pallas GPU backend is available."""
    try:
        from jax.experimental import pallas as _pl  # noqa: F401
        from jax.experimental.pallas import mosaic_gpu as _plgpu  # noqa: F401
        return True
    except ImportError:
        return False
