"""Tests for Muon optimizer."""

import jax
import jax.numpy as jnp
import pytest

from atlas_jax.optim import muon, warmup_constant_warmdown_schedule


class TestMuon:
    def test_init_state(self):
        """Muon should initialize state for 2D params."""
        opt = muon(learning_rate=0.01)
        params = jnp.zeros((32, 64))
        state = opt.init(params)
        assert state.momentum.shape == (32, 64)

    def test_step_reduces_loss(self):
        """A Muon step on a simple quadratic should reduce loss."""
        key = jax.random.PRNGKey(0)
        params = jax.random.normal(key, (16, 16))
        target = jnp.eye(16)

        opt = muon(learning_rate=0.01)
        state = opt.init(params)

        def loss_fn(p):
            return jnp.sum((p - target) ** 2)

        loss_before = loss_fn(params)
        grad = jax.grad(loss_fn)(params)
        updates, new_state = opt.update(grad, state, params)
        new_params = params + updates
        loss_after = loss_fn(new_params)

        assert loss_after < loss_before

    def test_output_finite(self):
        """Muon updates should be finite."""
        key = jax.random.PRNGKey(1)
        params = jax.random.normal(key, (32, 16))
        grad = jax.random.normal(jax.random.PRNGKey(2), (32, 16))

        opt = muon(learning_rate=0.01)
        state = opt.init(params)
        updates, _ = opt.update(grad, state, params)
        assert jnp.all(jnp.isfinite(updates))

    def test_wide_matrix(self):
        """Muon should handle wide matrices (rows < cols)."""
        key = jax.random.PRNGKey(3)
        params = jax.random.normal(key, (16, 64))
        grad = jax.random.normal(jax.random.PRNGKey(4), (16, 64))

        opt = muon(learning_rate=0.01)
        state = opt.init(params)
        updates, _ = opt.update(grad, state, params)
        assert jnp.all(jnp.isfinite(updates))
        assert updates.shape == (16, 64)


class TestSchedule:
    def test_warmup(self):
        schedule = warmup_constant_warmdown_schedule(
            peak_lr=1.0, warmup_steps=100, total_steps=1000, warmdown_steps=100)
        assert schedule(0) == 0.0
        assert abs(schedule(50) - 0.5) < 0.01
        assert abs(schedule(100) - 1.0) < 0.01

    def test_constant_phase(self):
        schedule = warmup_constant_warmdown_schedule(
            peak_lr=1.0, warmup_steps=100, total_steps=1000, warmdown_steps=100)
        assert abs(schedule(500) - 1.0) < 0.01

    def test_warmdown(self):
        schedule = warmup_constant_warmdown_schedule(
            peak_lr=1.0, warmup_steps=100, total_steps=1000, warmdown_steps=100)
        assert abs(schedule(950) - 0.5) < 0.01
        assert abs(schedule(1000) - 0.0) < 0.01
