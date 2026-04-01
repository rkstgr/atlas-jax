"""Tests for Polar Express orthogonalization."""

import jax
import jax.numpy as jnp
import pytest

from atlas_jax.polar_express import polar_express, polar_express_ste


def test_identity_input():
    """PE of an identity matrix should stay close to identity (already orthogonal)."""
    X = jnp.eye(16, dtype=jnp.float32)[None]  # (1, 16, 16)
    Y = polar_express(X, steps=5)
    # Should be close to some orthogonal matrix (identity scaled)
    assert Y.shape == (1, 16, 16)
    # Y^T @ Y should be approximately identity (up to scaling)
    YtY = Y[0].T @ Y[0]
    diag = jnp.diag(YtY)
    # All diagonal elements should be similar (near-orthogonal)
    assert jnp.std(diag) < 0.1


def test_output_near_orthogonal():
    """PE output should satisfy X^T @ X ~ c * I for some scalar c."""
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (4, 32, 32))
    Y = polar_express(X, steps=5)

    for i in range(4):
        YtY = Y[i].T @ Y[i]
        # Off-diagonal elements should be much smaller than diagonal
        diag = jnp.diag(YtY)
        off_diag = YtY - jnp.diag(diag)
        assert jnp.max(jnp.abs(off_diag)) < 0.5 * jnp.mean(jnp.abs(diag))


def test_frobenius_normalization():
    """PE should handle matrices of varying scale."""
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (2, 16, 16)) * 100.0
    Y = polar_express(X, steps=5)
    # Output should still be reasonable magnitude
    assert jnp.all(jnp.isfinite(Y))
    assert jnp.max(jnp.abs(Y)) < 100.0


def test_tall_matrix():
    """PE should work for tall matrices using X^T @ X form."""
    key = jax.random.PRNGKey(1)
    X = jax.random.normal(key, (2, 32, 16))  # tall: 32 > 16
    Y = polar_express(X, steps=5)
    assert Y.shape == (2, 32, 16)
    assert jnp.all(jnp.isfinite(Y))


def test_wide_matrix():
    """PE should work for wide matrices using X @ X^T form."""
    key = jax.random.PRNGKey(2)
    X = jax.random.normal(key, (2, 16, 32))  # wide: 16 < 32
    Y = polar_express(X, steps=5)
    assert Y.shape == (2, 16, 32)
    assert jnp.all(jnp.isfinite(Y))


def test_ste_forward_matches():
    """STE variant should produce identical forward output."""
    key = jax.random.PRNGKey(3)
    X = jax.random.normal(key, (2, 16, 16))
    Y_pe = polar_express(X, steps=5)
    Y_ste = polar_express_ste(X, steps=5)
    assert jnp.allclose(Y_pe, Y_ste, atol=1e-5)


def test_ste_gradient_is_identity():
    """STE backward should pass gradients through unchanged."""
    key = jax.random.PRNGKey(4)
    X = jax.random.normal(key, (16, 16))

    # STE gradient: should be identity-like
    def ste_sum(x):
        return jnp.sum(polar_express_ste(x[None], steps=5))
    grad_ste = jax.grad(ste_sum)(X)

    # Should be close to all-ones (gradient of sum through identity)
    assert jnp.allclose(grad_ste, jnp.ones_like(X), atol=1e-5)


def test_batched_consistency():
    """Processing a batch should give same results as processing individually."""
    key = jax.random.PRNGKey(5)
    X = jax.random.normal(key, (4, 16, 16))
    Y_batch = polar_express(X, steps=5)
    for i in range(4):
        Y_single = polar_express(X[i:i+1], steps=5)
        assert jnp.allclose(Y_batch[i], Y_single[0], atol=1e-5)


def test_gradient_flows():
    """Gradients through polar_express should be finite."""
    key = jax.random.PRNGKey(6)
    X = jax.random.normal(key, (2, 16, 16))

    def f(x):
        return jnp.sum(polar_express(x, steps=5) ** 2)

    grad = jax.grad(f)(X)
    assert jnp.all(jnp.isfinite(grad))
