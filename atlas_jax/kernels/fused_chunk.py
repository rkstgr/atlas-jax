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

Supports D != E (memory_expand > 1). W1 is (D, E) wide, W2 is (E, D) tall.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp

from atlas_jax.memory_layer import polar_express_ste, polar_express, POLAR_EXPRESS_COEFFS, DeepMemoryState

# Try to import Triton dependencies
try:
    import triton
    import triton.language as tl
    import jax_triton as jt
    from atlas_jax.kernels.triton_scan import triton_linear_scan
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
# Triton forward kernel — supports D != E (rectangular matrices)
# ---------------------------------------------------------------------------

if _HAS_TRITON:
    @triton.jit
    def _fused_chunk_fwd_kernel(
        # --- Carry inputs: flattened to (B, H, D*E) ---
        W1_ptr, W2_ptr, SW1_ptr, SW2_ptr,
        # --- Per-timestep inputs ---
        momW1_ptr,      # (B, cs, H, D*E) flattened
        momW2_ptr,      # (B, cs, H, E*D) flattened
        theta_ptr,      # (B, cs, H)
        alpha_ptr,      # (B, cs, H)
        q_ptr,          # (B, cs, H, D)
        pe_coeffs_ptr,  # (ns_steps * 3,) flat f32
        # --- Single packed output buffer ---
        out_ptr,
        # --- Offsets within output buffer ---
        y_off: tl.constexpr,
        w1o_off: tl.constexpr,
        w2o_off: tl.constexpr,
        sw1o_off: tl.constexpr,
        sw2o_off: tl.constexpr,
        # --- Dimensions ---
        cs: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        E: tl.constexpr,
        DE: tl.constexpr,       # D * E
        ns_steps: tl.constexpr,
        BD: tl.constexpr,       # block size for D (padded pow2)
        BE: tl.constexpr,       # block size for E (padded pow2)
        BDE: tl.constexpr,      # BD * BE (padded D*E)
        # --- Strides ---
        sc_b, sc_h,             # carry: (B, H, DE)
        sm_b, sm_t, sm_h,       # mom: (B, cs, H, DE)
        sg_b, sg_t,             # gates: (B, cs, H)
        sq_b, sq_t, sq_h,       # q/y: (B, cs, H, D)
    ):
        """Fused forward: momentum → PE → memory → output, carry in registers.

        W1 is (D, E) — wide when E > D. PE: A = X @ X^T, X_new = a*X + B @ X
        W2 is (E, D) — tall when E > D. PE: A = X^T @ X, X_new = a*X + X @ B
        """
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H

        # --- Load carry state into registers (f32) ---
        de_range = tl.arange(0, BDE)
        de_mask = de_range < DE
        c_off = b * sc_b + h * sc_h

        W1 = tl.load(W1_ptr + c_off + de_range, mask=de_mask, other=0.0).to(tl.float32)
        W2 = tl.load(W2_ptr + c_off + de_range, mask=de_mask, other=0.0).to(tl.float32)
        SW1 = tl.load(SW1_ptr + c_off + de_range, mask=de_mask, other=0.0).to(tl.float32)
        SW2 = tl.load(SW2_ptr + c_off + de_range, mask=de_mask, other=0.0).to(tl.float32)

        d_range = tl.arange(0, BD)
        d_mask = d_range < D
        e_range = tl.arange(0, BE)
        e_mask = e_range < E

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

            # --- Polar Express for SW1: (D, E) wide matrix ---
            # Wide/square: A = X @ X^T (BD, BD), X_new = a*X + B @ X
            frob_sq = tl.sum(SW1 * SW1)
            inv_frob = 1.0 / (tl.sqrt(frob_sq + 1e-12) * 1.01 + 1e-6)
            X1 = SW1 * inv_frob
            X1_2d = tl.reshape(X1, (BD, BE))  # (BD, BE) = (D, E) padded
            for step in range(ns_steps):
                a_c = tl.load(pe_coeffs_ptr + step * 3).to(tl.float32)
                b_c = tl.load(pe_coeffs_ptr + step * 3 + 1).to(tl.float32)
                c_c = tl.load(pe_coeffs_ptr + step * 3 + 2).to(tl.float32)
                A = tl.dot(X1_2d, tl.trans(X1_2d))          # (BD, BE) @ (BE, BD) = (BD, BD)
                AA = tl.dot(A, A)                             # (BD, BD)
                B_mat = b_c * A + c_c * AA                    # (BD, BD)
                X1_2d = a_c * X1_2d + tl.dot(B_mat, X1_2d)  # (BD, BD) @ (BD, BE) = (BD, BE)
            SW1_orth = tl.reshape(X1_2d, (BDE,))

            # --- Polar Express for SW2: (E, D) tall matrix ---
            # Tall: A = X^T @ X (BD, BD), X_new = a*X + X @ B
            frob_sq = tl.sum(SW2 * SW2)
            inv_frob = 1.0 / (tl.sqrt(frob_sq + 1e-12) * 1.01 + 1e-6)
            X2 = SW2 * inv_frob
            X2_2d = tl.reshape(X2, (BE, BD))  # (BE, BD) = (E, D) padded
            for step in range(ns_steps):
                a_c = tl.load(pe_coeffs_ptr + step * 3).to(tl.float32)
                b_c = tl.load(pe_coeffs_ptr + step * 3 + 1).to(tl.float32)
                c_c = tl.load(pe_coeffs_ptr + step * 3 + 2).to(tl.float32)
                A = tl.dot(tl.trans(X2_2d), X2_2d)          # (BD, BE) @ (BE, BD) = (BD, BD)
                AA = tl.dot(A, A)                             # (BD, BD)
                B_mat = b_c * A + c_c * AA                    # (BD, BD)
                X2_2d = a_c * X2_2d + tl.dot(X2_2d, B_mat)  # (BE, BD) @ (BD, BD) = (BE, BD)
            SW2_orth = tl.reshape(X2_2d, (BDE,))

            # --- Memory update: W = alpha * W + PE(S) ---
            W1 = alpha_t * W1 + SW1_orth
            W2 = alpha_t * W2 + SW2_orth

            # --- Output: y = q + W1 @ gelu(W2 @ q) ---
            q_off = b * sq_b + t * sq_t + h * sq_h
            q_vec = tl.load(q_ptr + q_off + d_range, mask=d_mask, other=0.0).to(tl.float32)

            # W2 @ q: (E, D) @ (D,) = (E,)
            W2_2d = tl.reshape(W2, (BE, BD))
            h_q = tl.sum(W2_2d * q_vec[None, :], axis=1)  # (BE,)

            # GELU
            g_q = h_q * 0.5 * (1.0 + tl.erf(h_q * 0.7071067811865476))

            # W1 @ gelu(h): (D, E) @ (E,) = (D,)
            W1_2d = tl.reshape(W1, (BD, BE))
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

    Supports D != E (memory_expand > 1). W1 is (B,H,D,E), W2 is (B,H,E,D).
    Both D and E must be powers of 2 (for tl.reshape alignment).
    """
    B, H_, D, E = W1.shape
    assert D & (D - 1) == 0, f"D must be power of 2, got {D}"
    assert E & (E - 1) == 0, f"E must be power of 2, got {E}"
    cs = momW1.shape[1]
    H = H_
    DE = D * E

    # Pad to next power of 2 (minimum 16 for tensor core alignment)
    BD = max(triton.next_power_of_2(D), 16)
    BE = max(triton.next_power_of_2(E), 16)
    BDE = BD * BE

    # Flatten last two dims for flat-vector kernel
    W1_flat = W1.reshape(B, H, DE)
    W2_flat = W2.reshape(B, H, DE)
    SW1_flat = SW1.reshape(B, H, DE)
    SW2_flat = SW2.reshape(B, H, DE)
    momW1_flat = momW1.reshape(B, cs, H, DE)
    momW2_flat = momW2.reshape(B, cs, H, DE)

    # Output buffer layout
    y_size = B * cs * H * D
    carry_size = B * H * DE
    total_size = y_size + 4 * carry_size

    # Strides
    sc_b, sc_h = H * DE, DE
    sm_b, sm_t, sm_h = cs * H * DE, H * DE, DE
    sg_b, sg_t = cs * H, H
    sq_b, sq_t, sq_h = cs * H * D, H * D, D

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
        cs=cs, H=H, D=D, E=E, DE=DE, ns_steps=ns_steps,
        BD=BD, BE=BE, BDE=BDE,
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

    out_dtype = W1.dtype
    y = y.astype(out_dtype)
    state = DeepMemoryState(
        W1=W1_f.astype(out_dtype), W2=W2_f.astype(out_dtype),
        S_W1=SW1_f.astype(out_dtype), S_W2=SW2_f.astype(out_dtype))
    return y, state


# ---------------------------------------------------------------------------
# Public API: module-level custom_vjp with nondiff_argnums
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(9, 10))
def fused_chunk_scan(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                      ns_steps, pe_ste):
    """Fused chunk scan: momentum + PE + memory + output in one Triton kernel.

    Supports D != E (memory_expand > 1). W1 is (D,E), W2 is (E,D).

    Args:
        W1:    (B, H, D, E) initial memory weight 1
        W2:    (B, H, E, D) initial memory weight 2
        SW1:   (B, H, D, E) initial momentum for W1
        SW2:   (B, H, E, D) initial momentum for W2
        momW1: (B, cs, H, D, E) pre-computed momentum input
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

def fused_chunk_available():
    """Check if the fused chunk kernel can be used."""
    return _HAS_TRITON
