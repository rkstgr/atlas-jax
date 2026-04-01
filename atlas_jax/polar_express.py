"""Polar Express orthogonalization (Newton-Schulz iteration).

Computes the nearest semi-orthogonal matrix to a batch of matrices using
5 steps of quintic Newton-Schulz iteration with optimized coefficients.

Reference: arXiv 2505.16932 (Polar Express Sign Method)
Used by Atlas as the internal memory optimizer (replaces standard GD).

Per-step iteration (for square/wide matrices):
  A = X @ X^T
  B = b*A + c*(A @ A)
  X_{i+1} = a*X + B @ X

Input is Frobenius-normalized before iteration.
"""

import jax
import jax.numpy as jnp

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
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
    # Frobenius-normalize each matrix (f32 for stability)
    orig_dtype = X.dtype
    X = X.astype(jnp.float32)
    frob_norm = jnp.sqrt(jnp.sum(X * X, axis=(-2, -1), keepdims=True) + 1e-12)
    X = X / (frob_norm * 1.01 + 1e-6)

    d1, d2 = X.shape[-2], X.shape[-1]

    if d1 > d2:
        # Tall matrix: use X^T @ X form
        for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
            A = jnp.einsum('...ji,...jk->...ik', X, X)  # X^T @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        # Square or wide matrix: use X @ X^T form
        for a, b, c in POLAR_EXPRESS_COEFFS[:steps]:
            A = X @ jnp.swapaxes(X, -2, -1)  # X @ X^T
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    return X.astype(orig_dtype)


def polar_express_ste(X, steps=5):
    """Polar Express with straight-through estimator for backward pass.

    Forward: full Newton-Schulz orthogonalization.
    Backward: pass gradients through unchanged (identity Jacobian approximation).

    Justified because PE is an internal optimizer step finding the nearest
    orthogonal matrix. Near-orthogonal matrices have Jacobian ~ identity.
    """
    return X + jax.lax.stop_gradient(polar_express(X, steps) - X)
