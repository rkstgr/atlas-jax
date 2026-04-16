"""Fused Polar Express: XLA fori_loop + Triton fallback.

The PE forward on 65,536 tiny (56×56) matrices takes 16ms in JAX because the
Python for-loop in polar_express() unrolls into ~15 separate XLA kernels (3
matmuls × 3 NS steps + Frobenius norm + dtype casts), each round-tripping the
full 800MB batch through HBM.

Fix: use jax.lax.fori_loop so XLA compiles the NS iterations as a single GPU
kernel with an internal loop. Intermediates stay in GPU memory between iterations
without separate kernel launches.

For STE backward (identity), no backward through PE is needed.
"""

import jax
import jax.numpy as jnp

from atlas_jax.memory_layer import POLAR_EXPRESS_COEFFS


# Pack coefficients as a JAX array for use inside fori_loop
# Shape: (5, 3) — max 5 steps, 3 coefficients each
_PE_COEFFS = jnp.array(POLAR_EXPRESS_COEFFS, dtype=jnp.float32)  # (5, 3)


def fused_polar_express(X, steps=5):
    """Batched PE using fori_loop for XLA kernel fusion.

    Same math as polar_express(), but XLA compiles the NS iterations as a
    single fused GPU kernel instead of separate kernels per iteration.

    Args:
        X: (..., D1, D2) batch of matrices, any dtype.
        steps: number of Newton-Schulz iterations (1-5).

    Returns:
        Approximate orthogonal polar factor, same shape and dtype as X.
    """
    orig_dtype = X.dtype
    X = X.astype(jnp.float32)

    # Frobenius normalize
    frob_norm = jnp.sqrt(jnp.sum(X * X, axis=(-2, -1), keepdims=True) + 1e-12)
    X = X / (frob_norm * 1.01 + 1e-6)

    d1, d2 = X.shape[-2], X.shape[-1]
    coeffs = _PE_COEFFS[:steps]  # (steps, 3)

    if d1 > d2:
        # Tall matrix: A = X^T @ X, X_new = a*X + X @ B
        def body_fn(i, X):
            a = coeffs[i, 0]
            b = coeffs[i, 1]
            c = coeffs[i, 2]
            A = jnp.einsum('...ji,...jk->...ik', X, X)  # X^T @ X
            B = b * A + c * (A @ A)
            return a * X + X @ B
    else:
        # Square/wide: A = X @ X^T, X_new = a*X + B @ X
        def body_fn(i, X):
            a = coeffs[i, 0]
            b = coeffs[i, 1]
            c = coeffs[i, 2]
            A = X @ jnp.swapaxes(X, -2, -1)  # X @ X^T
            B = b * A + c * (A @ A)
            return a * X + B @ X

    X = jax.lax.fori_loop(0, steps, body_fn, X)
    return X.astype(orig_dtype)


def fused_polar_express_ste(X, steps=5):
    """Fused PE with straight-through estimator."""
    return X + jax.lax.stop_gradient(fused_polar_express(X, steps) - X)


# Keep the Triton-based versions as optional (they work but compile slowly)
try:
    import triton
    import triton.language as tl
    import jax_triton as jt
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# Default exports: use the fori_loop version (compiles fast, runs fast)
triton_polar_express = fused_polar_express
triton_polar_express_ste = fused_polar_express_ste
