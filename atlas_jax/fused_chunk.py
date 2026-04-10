"""FlashATLAS: Fused chunk scan keeping carry state in GPU SRAM.

Same principle as FlashAttention applied to a different bottleneck:
- FlashAttention keeps the attention matrix in SRAM (avoids materializing [T,T] to HBM)
- FlashATLAS keeps the scan carry (W1, W2, S_W1, S_W2) in SRAM across timesteps

The bottleneck: jax.lax.scan compiles to an XLA while-loop that round-trips the
carry state through HBM on every iteration. With 4 matrices of (B,H,D,E) each,
that's 6+ MB copied per iteration × 32 chunks × 8 layers = thousands of small
MemcpyD2D operations, each dominated by kernel launch overhead (~10µs each).

The fix: a single Triton kernel processes all cs timesteps within a chunk,
keeping the carry in GPU registers. One kernel launch replaces ~20 separate
kernel launches (4 scans + PE fusions + einsum fusions).

Forward:  Triton kernel (carry in registers, single launch)
Backward: Recompute forward with regular JAX ops, then jax.vjp for gradients.
          This reuses the battle-tested Triton scan backward + PE STE backward.

Requires D == E (memory_expand=1). Falls back to regular ops otherwise.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp

from atlas_jax.polar_express import polar_express_ste, polar_express, POLAR_EXPRESS_COEFFS
from atlas_jax.state import DeepMemoryState

# Try to import Triton dependencies
try:
    import triton
    import triton.language as tl
    import jax_triton as jt
    from atlas_jax.triton_scan import triton_linear_scan
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _pe_coeffs_flat(ns_steps):
    """Pack PE coefficients as a flat (ns_steps * 3,) f32 array: [a0,b0,c0, a1,b1,c1, ...]."""
    coeffs = []
    for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
        coeffs.extend([a, b, c])
    return jnp.array(coeffs, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Triton forward kernel
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _fused_chunk_fwd_kernel(
        # --- Carry inputs: (B, H, D, E) contiguous ---
        W1_ptr, W2_ptr, SW1_ptr, SW2_ptr,
        # --- Per-timestep inputs ---
        momW1_ptr,      # (B, cs, H, D*E) flattened last two dims
        momW2_ptr,      # (B, cs, H, E*D) flattened last two dims
        theta_ptr,      # (B, cs, H)
        alpha_ptr,      # (B, cs, H)
        q_ptr,          # (B, cs, H, D)
        pe_coeffs_ptr,  # (ns_steps * 3,) flat f32
        # --- Single packed output buffer ---
        out_ptr,
        # --- Offsets within output buffer (element offsets, not bytes) ---
        y_off: tl.constexpr,
        w1o_off: tl.constexpr,
        w2o_off: tl.constexpr,
        sw1o_off: tl.constexpr,
        sw2o_off: tl.constexpr,
        # --- Dimensions ---
        cs: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        DE: tl.constexpr,       # D * E (= D * D when expand=1)
        ns_steps: tl.constexpr,
        BD: tl.constexpr,       # block size for D dim (padded, power of 2)
        BDE: tl.constexpr,      # block size for D*E (padded)
        # --- Strides (in elements) ---
        # carry: (B, H, DE) after reshape
        sc_b, sc_h,
        # mom: (B, cs, H, DE)
        sm_b, sm_t, sm_h,
        # gates: (B, cs, H)
        sg_b, sg_t,
        # q/y: (B, cs, H, D)
        sq_b, sq_t, sq_h,
    ):
        """Fused forward: momentum → PE → memory → output, carry stays in registers."""
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H

        # --- Load carry state into registers (upcast to f32) ---
        de_range = tl.arange(0, BDE)
        de_mask = de_range < DE
        c_off = b * sc_b + h * sc_h

        W1 = tl.load(W1_ptr + c_off + de_range, mask=de_mask, other=0.0).to(tl.float32)
        W2 = tl.load(W2_ptr + c_off + de_range, mask=de_mask, other=0.0).to(tl.float32)
        SW1 = tl.load(SW1_ptr + c_off + de_range, mask=de_mask, other=0.0).to(tl.float32)
        SW2 = tl.load(SW2_ptr + c_off + de_range, mask=de_mask, other=0.0).to(tl.float32)

        d_range = tl.arange(0, BD)
        d_mask = d_range < D

        # --- Sequential loop over timesteps ---
        for t in range(cs):
            # Load gates (scalars)
            g_off = b * sg_b + t * sg_t + h
            theta_t = tl.load(theta_ptr + g_off).to(tl.float32)
            alpha_t = tl.load(alpha_ptr + g_off).to(tl.float32)

            # Load momentum inputs (flat D*E vectors)
            m_off = b * sm_b + t * sm_t + h * sm_h
            mW1 = tl.load(momW1_ptr + m_off + de_range, mask=de_mask, other=0.0).to(tl.float32)
            mW2 = tl.load(momW2_ptr + m_off + de_range, mask=de_mask, other=0.0).to(tl.float32)

            # --- Momentum update: S = theta * S + mom ---
            SW1 = theta_t * SW1 + mW1
            SW2 = theta_t * SW2 + mW2

            # --- Polar Express for SW1 (operates on flat D*E vector) ---
            # Frobenius normalize
            frob_sq = tl.sum(SW1 * SW1)
            inv_frob = 1.0 / (tl.sqrt(frob_sq + 1e-12) * 1.01 + 1e-6)
            X1 = SW1 * inv_frob

            # Newton-Schulz iterations (flat vector form)
            # For D==E square matrices stored as flat (D*E,) vectors:
            #   A = X @ X^T (D,D), B = b*A + c*A^2, X = a*X + B @ X
            # We implement this with reshape tricks in the flat representation.
            # X reshaped to (D, D): X_mat[i,j] = X[i*D + j]
            # A = X_mat @ X_mat^T: A[i,j] = sum_k X_mat[i,k] * X_mat[j,k]
            #                             = sum_k X[i*D+k] * X[j*D+k]
            # This requires 2D operations. We use tl.reshape + tl.dot.
            X1_2d = tl.reshape(X1, (BD, BD))
            for step in range(ns_steps):
                a_c = tl.load(pe_coeffs_ptr + step * 3).to(tl.float32)
                b_c = tl.load(pe_coeffs_ptr + step * 3 + 1).to(tl.float32)
                c_c = tl.load(pe_coeffs_ptr + step * 3 + 2).to(tl.float32)
                A = tl.dot(X1_2d, tl.trans(X1_2d))          # (BD, BD)
                AA = tl.dot(A, A)                             # (BD, BD)
                B_mat = b_c * A + c_c * AA                    # (BD, BD)
                X1_2d = a_c * X1_2d + tl.dot(B_mat, X1_2d)  # (BD, BD)
            SW1_orth = tl.reshape(X1_2d, (BDE,))

            # --- Polar Express for SW2 ---
            frob_sq = tl.sum(SW2 * SW2)
            inv_frob = 1.0 / (tl.sqrt(frob_sq + 1e-12) * 1.01 + 1e-6)
            X2 = SW2 * inv_frob
            X2_2d = tl.reshape(X2, (BD, BD))
            for step in range(ns_steps):
                a_c = tl.load(pe_coeffs_ptr + step * 3).to(tl.float32)
                b_c = tl.load(pe_coeffs_ptr + step * 3 + 1).to(tl.float32)
                c_c = tl.load(pe_coeffs_ptr + step * 3 + 2).to(tl.float32)
                A = tl.dot(X2_2d, tl.trans(X2_2d))
                AA = tl.dot(A, A)
                B_mat = b_c * A + c_c * AA
                X2_2d = a_c * X2_2d + tl.dot(B_mat, X2_2d)
            SW2_orth = tl.reshape(X2_2d, (BDE,))

            # --- Memory update: W = alpha * W + PE(S) ---
            W1 = alpha_t * W1 + SW1_orth
            W2 = alpha_t * W2 + SW2_orth

            # --- Output: y = q + W1_mat @ gelu(W2_mat @ q) ---
            q_off = b * sq_b + t * sq_t + h * sq_h
            q_vec = tl.load(q_ptr + q_off + d_range, mask=d_mask, other=0.0).to(tl.float32)

            # W2 @ q: reshape W2 to (D, D), matrix-vector product
            W2_2d = tl.reshape(W2, (BD, BD))   # (D, D)
            # h_q[i] = sum_j W2_2d[i,j] * q[j]
            h_q = tl.sum(W2_2d * q_vec[None, :], axis=1)  # (BD,)

            # GELU: exact form x * 0.5 * (1 + erf(x / sqrt(2)))
            g_q = h_q * 0.5 * (1.0 + tl.erf(h_q * 0.7071067811865476))

            # W1 @ gelu: reshape W1 to (D, D)
            W1_2d = tl.reshape(W1, (BD, BD))
            y_vec = q_vec + tl.sum(W1_2d * g_q[None, :], axis=1)  # (BD,)

            # Store y[t] (bf16)
            y_out_off = y_off + b * sq_b + t * sq_t + h * sq_h
            tl.store(out_ptr + y_out_off + d_range, y_vec.to(tl.bfloat16), mask=d_mask)

        # --- Store final carry (bf16) ---
        tl.store(out_ptr + w1o_off + c_off + de_range, W1.to(tl.bfloat16), mask=de_mask)
        tl.store(out_ptr + w2o_off + c_off + de_range, W2.to(tl.bfloat16), mask=de_mask)
        tl.store(out_ptr + sw1o_off + c_off + de_range, SW1.to(tl.bfloat16), mask=de_mask)
        tl.store(out_ptr + sw2o_off + c_off + de_range, SW2.to(tl.bfloat16), mask=de_mask)


# ---------------------------------------------------------------------------
# JAX wrapper: calls Triton kernel, unpacks output
# ---------------------------------------------------------------------------

def _triton_fused_fwd(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                       pe_coeffs, ns_steps):
    """Run the fused forward Triton kernel.

    Args:
        W1, W2, SW1, SW2: (B, H, D, E) / (B, H, E, D) carry state
        momW1: (B, cs, H, D, E) pre-computed momentum input for W1
        momW2: (B, cs, H, E, D) pre-computed momentum input for W2
        theta, alpha: (B, cs, H) gates
        q: (B, cs, H, D) query vectors
        pe_coeffs: (ns_steps*3,) flat f32 PE coefficients
        ns_steps: number of PE iterations

    Returns:
        y: (B, cs, H, D) output
        W1_out, W2_out, SW1_out, SW2_out: updated carry
    """
    B, H_, D, E = W1.shape
    assert D == E, f"Fused kernel requires D==E, got D={D}, E={E}"
    cs = momW1.shape[1]
    H = H_
    DE = D * E

    # Pad D to next power of 2 (minimum 16 for tl.dot tensor core alignment)
    BD = max(triton.next_power_of_2(D), 16)
    BDE = BD * BD  # since D == E, BD == BE

    # Flatten last two dims of carry and mom for flat-vector kernel
    W1_flat = W1.reshape(B, H, DE)
    W2_flat = W2.reshape(B, H, DE)
    SW1_flat = SW1.reshape(B, H, DE)
    SW2_flat = SW2.reshape(B, H, DE)
    momW1_flat = momW1.reshape(B, cs, H, DE)
    momW2_flat = momW2.reshape(B, cs, H, DE)

    # Compute output buffer layout (element counts)
    y_size = B * cs * H * D
    carry_size = B * H * DE  # same for all 4 carry arrays
    total_size = y_size + 4 * carry_size

    # Strides for flat carry: (B, H, DE) contiguous
    sc_b = H * DE
    sc_h = DE

    # Strides for flat mom: (B, cs, H, DE) contiguous
    sm_b = cs * H * DE
    sm_t = H * DE
    sm_h = DE

    # Strides for gates: (B, cs, H) contiguous
    sg_b = cs * H
    sg_t = H

    # Strides for q/y: (B, cs, H, D) contiguous
    sq_b = cs * H * D
    sq_t = H * D
    sq_h = D

    # Output section offsets
    y_off = 0
    w1o_off = y_size
    w2o_off = y_size + carry_size
    sw1o_off = y_size + 2 * carry_size
    sw2o_off = y_size + 3 * carry_size

    grid = (B * H,)

    out_flat = jt.triton_call(
        W1_flat, W2_flat, SW1_flat, SW2_flat,
        momW1_flat, momW2_flat, theta, alpha, q,
        pe_coeffs,
        kernel=_fused_chunk_fwd_kernel,
        out_shape=jax.ShapeDtypeStruct((total_size,), jnp.bfloat16),
        grid=grid,
        num_warps=4,
        num_stages=1,
        # Offsets
        y_off=y_off, w1o_off=w1o_off, w2o_off=w2o_off,
        sw1o_off=sw1o_off, sw2o_off=sw2o_off,
        # Dimensions
        cs=cs, H=H, D=D, DE=DE, ns_steps=ns_steps, BD=BD, BDE=BDE,
        # Strides
        sc_b=sc_b, sc_h=sc_h,
        sm_b=sm_b, sm_t=sm_t, sm_h=sm_h,
        sg_b=sg_b, sg_t=sg_t,
        sq_b=sq_b, sq_t=sq_t, sq_h=sq_h,
    )

    # Unpack output
    y = out_flat[y_off:y_off + y_size].reshape(B, cs, H, D)
    W1_out = out_flat[w1o_off:w1o_off + carry_size].reshape(B, H, D, E)
    W2_out = out_flat[w2o_off:w2o_off + carry_size].reshape(B, H, E, D)
    SW1_out = out_flat[sw1o_off:sw1o_off + carry_size].reshape(B, H, D, E)
    SW2_out = out_flat[sw2o_off:sw2o_off + carry_size].reshape(B, H, E, D)

    return y, W1_out, W2_out, SW1_out, SW2_out


# ---------------------------------------------------------------------------
# Regular JAX forward (for backward recomputation)
# ---------------------------------------------------------------------------

def _regular_fwd(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                  ns_steps, pe_ste):
    """Same computation as the fused kernel, using regular JAX ops.

    Used in the backward pass: jax.vjp traces through this to get gradients.
    This reuses the existing Triton linear_scan + PE implementations.
    """
    _scan = triton_linear_scan if _HAS_TRITON else None
    if _scan is None:
        raise RuntimeError("Triton scan required for fused_chunk backward")

    # Must use JAX PE here (not Triton PE) because _regular_fwd runs inside
    # jax.vjp which needs to differentiate through it. Triton kernels via
    # jax_triton.triton_call are not automatically differentiable.
    _pe = polar_express_ste if pe_ste else polar_express

    # Momentum scans: S[t] = theta[t] * S[t-1] + mom[t]
    chunk_SW1, SW1_f = _scan(SW1, theta, momW1)
    chunk_SW2, SW2_f = _scan(SW2, theta, momW2)

    # Polar Express orthogonalization
    SW1_orth = _pe(chunk_SW1, ns_steps)
    SW2_orth = _pe(chunk_SW2, ns_steps)

    # Memory scans: W[t] = alpha[t] * W[t-1] + PE(S[t])
    W1_all, W1_f = _scan(W1, alpha, SW1_orth)
    W2_all, W2_f = _scan(W2, alpha, SW2_orth)

    # Output: y[t] = q[t] + W1[t] @ gelu(W2[t] @ q[t])
    h_q = jnp.einsum('bched,bchd->bche', W2_all, q)
    g_q = jax.nn.gelu(h_q)
    y = q + jnp.einsum('bchde,bche->bchd', W1_all, g_q)

    # Cast to match input dtype (the fused Triton kernel outputs bf16,
    # so the backward vjp must also work in the same dtype space)
    out_dtype = W1.dtype
    y = y.astype(out_dtype)
    state = DeepMemoryState(
        W1=W1_f.astype(out_dtype), W2=W2_f.astype(out_dtype),
        S_W1=SW1_f.astype(out_dtype), S_W2=SW2_f.astype(out_dtype))
    return y, state


# ---------------------------------------------------------------------------
# Public API: module-level custom_vjp with nondiff_argnums
# ---------------------------------------------------------------------------
# Uses nondiff_argnums for ns_steps/pe_ste (static Python ints/bools).
# This matches the triton_linear_scan pattern which works correctly under
# eqx.filter_value_and_grad. The previous factory+custom_vmap pattern broke
# VJP: custom_vmap wrapping custom_vjp from the outside prevented JAX from
# seeing the custom_vjp rules, causing a CustomVJPException about closed-over
# values and producing zero gradients (the cause of the training plateau).

@partial(jax.custom_vjp, nondiff_argnums=(9, 10))
def fused_chunk_scan(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                      ns_steps, pe_ste):
    """Fused chunk scan: momentum + PE + memory + output in one Triton kernel.

    This is the "FlashATLAS" kernel. Keeps W1, W2, S_W1, S_W2 in GPU registers
    across all cs timesteps, eliminating HBM round-trips.

    Args:
        W1:    (B, H, D, E) initial memory weight 1
        W2:    (B, H, E, D) initial memory weight 2
        SW1:   (B, H, D, E) initial momentum for W1
        SW2:   (B, H, E, D) initial momentum for W2
        momW1: (B, cs, H, D, E) pre-computed momentum input (-eta * omega_grad)
        momW2: (B, cs, H, E, D) pre-computed momentum input
        theta: (B, cs, H) momentum gate
        alpha: (B, cs, H) forget gate
        q:     (B, cs, H, D) query vectors
        ns_steps: (static int) number of PE iterations
        pe_ste:   (static bool) use straight-through estimator for PE

    Returns:
        y:     (B, cs, H, D) output
        state: DeepMemoryState with updated W1, W2, S_W1, S_W2
    """
    pe_coeffs = _pe_coeffs_flat(ns_steps)
    out_dtype = W1.dtype
    y, W1_out, W2_out, SW1_out, SW2_out = _triton_fused_fwd(
        W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
        pe_coeffs, ns_steps)
    y = y.astype(out_dtype)
    W1_out = W1_out.astype(out_dtype)
    W2_out = W2_out.astype(out_dtype)
    SW1_out = SW1_out.astype(out_dtype)
    SW2_out = SW2_out.astype(out_dtype)
    return y, DeepMemoryState(W1=W1_out, W2=W2_out, S_W1=SW1_out, S_W2=SW2_out)


def _fused_scan_fwd(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                     ns_steps, pe_ste):
    """custom_vjp forward: run fused Triton kernel, save inputs as residuals."""
    result = fused_chunk_scan(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                               ns_steps, pe_ste)
    residuals = (W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q)
    return result, residuals


def _fused_scan_bwd(ns_steps, pe_ste, residuals, grads):
    """custom_vjp backward: recompute with regular JAX ops, then vjp.

    Uses existing triton_linear_scan backward + PE STE backward + JAX
    autodiff for einsums. One extra forward recomputation, but with fast
    Triton scan kernels.
    """
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

def fused_chunk_available():
    """Check if the fused chunk kernel can be used."""
    return _HAS_TRITON
