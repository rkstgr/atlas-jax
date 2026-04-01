"""Tests for Omega aggregation (sliding window)."""

import jax
import jax.numpy as jnp
import pytest

from atlas_jax.model import _omega_aggregate


def test_window_1_is_identity():
    """With omega_window=1, aggregation should just scale by gamma."""
    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, (2, 8, 4, 16))
    gamma = jnp.ones((2, 8, 4, 1))
    result = _omega_aggregate(u, gamma, omega_window=1)
    # cumsum with window=1 is just gamma * u (since window >= cs check returns cumsum,
    # but window=1 < cs=8, so it does the subtraction)
    # Actually window=1: result[t] = cum[t] - cum[t-1] = weighted[t] = gamma[t] * u[t]
    expected = gamma * u
    assert jnp.allclose(result, expected, atol=1e-6)


def test_full_window_is_cumsum():
    """With omega_window >= chunk_size, should return full cumsum."""
    key = jax.random.PRNGKey(1)
    cs = 8
    u = jax.random.normal(key, (2, cs, 4, 16))
    gamma = jnp.ones((2, cs, 4, 1))
    result = _omega_aggregate(u, gamma, omega_window=cs)
    expected = jnp.cumsum(u, axis=1)
    assert jnp.allclose(result, expected, atol=1e-5)


def test_matches_naive_reference():
    """Compare against naive O(n*w) implementation."""
    key = jax.random.PRNGKey(2)
    B, cs, H, D = 1, 8, 2, 4
    u = jax.random.normal(key, (B, cs, H, D))
    key2 = jax.random.PRNGKey(3)
    gamma = jax.nn.sigmoid(jax.random.normal(key2, (B, cs, H, 1)))
    w = 3

    # Naive reference
    expected = jnp.zeros_like(u)
    for t in range(cs):
        start = max(0, t - w + 1)
        for i in range(start, t + 1):
            expected = expected.at[:, t].add(gamma[:, i] * u[:, i])

    result = _omega_aggregate(u, gamma, omega_window=w)
    assert jnp.allclose(result, expected, atol=1e-5)


def test_gamma_zero_masks():
    """Zero gamma at position should exclude that position's contribution."""
    B, cs, H, D = 1, 4, 1, 2
    u = jnp.ones((B, cs, H, D))
    gamma = jnp.array([[[[1.0]], [[0.0]], [[1.0]], [[0.0]]]])  # (1, 4, 1, 1)
    result = _omega_aggregate(u, gamma, omega_window=4)

    # Position 0: gamma[0]*u[0] = 1
    # Position 1: gamma[0]*u[0] + gamma[1]*u[1] = 1 + 0 = 1
    # Position 2: ... = 1 + 0 + 1 = 2
    # Position 3: ... = 1 + 0 + 1 + 0 = 2
    expected_sums = jnp.array([1.0, 1.0, 2.0, 2.0])
    assert jnp.allclose(result[0, :, 0, 0], expected_sums, atol=1e-6)


def test_gradient_flows():
    """Gradients should flow through omega aggregate."""
    key = jax.random.PRNGKey(4)
    u = jax.random.normal(key, (2, 8, 4, 16))
    gamma = jnp.ones((2, 8, 4, 1))

    def f(u, gamma):
        return jnp.sum(_omega_aggregate(u, gamma, omega_window=4) ** 2)

    grad_u, grad_gamma = jax.grad(f, argnums=(0, 1))(u, gamma)
    assert jnp.all(jnp.isfinite(grad_u))
    assert jnp.all(jnp.isfinite(grad_gamma))
