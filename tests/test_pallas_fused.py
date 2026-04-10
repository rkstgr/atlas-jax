"""Tests for Pallas fused chunk scan (Mosaic GPU).

Verifies that the Pallas kernel produces the same output as the
reference JAX implementation (_regular_fwd).
"""

import jax
import jax.numpy as jnp
import pytest

from atlas_jax.pallas_fused import (
    _regular_fwd, _pallas_fused_fwd, fused_chunk_scan,
    pallas_available,
)
from atlas_jax.state import DeepMemoryState


# Skip all tests if no GPU
pytestmark = pytest.mark.skipif(
    not any(d.platform == 'gpu' for d in jax.devices()),
    reason="Pallas GPU kernel requires a GPU",
)


def _make_inputs(B=2, H=4, D=64, cs=64, key=None):
    """Create random test inputs matching fused_chunk_scan API."""
    if key is None:
        key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 9)
    E = D  # memory_expand=1

    W1 = jax.random.normal(keys[0], (B, H, D, E), dtype=jnp.float32) * 0.1
    W2 = jax.random.normal(keys[1], (B, H, E, D), dtype=jnp.float32) * 0.1
    SW1 = jax.random.normal(keys[2], (B, H, D, E), dtype=jnp.float32) * 0.01
    SW2 = jax.random.normal(keys[3], (B, H, E, D), dtype=jnp.float32) * 0.01
    momW1 = jax.random.normal(keys[4], (B, cs, H, D, E), dtype=jnp.float32) * 0.1
    momW2 = jax.random.normal(keys[5], (B, cs, H, E, D), dtype=jnp.float32) * 0.1
    theta = jax.random.uniform(keys[6], (B, cs, H), dtype=jnp.float32, minval=0.8, maxval=0.99)
    alpha = jax.random.uniform(keys[7], (B, cs, H), dtype=jnp.float32, minval=0.8, maxval=0.99)
    q = jax.random.normal(keys[8], (B, cs, H, D), dtype=jnp.float32) * 0.1

    return W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q


class TestPallasFusedForward:
    """Test that Pallas kernel matches reference JAX forward."""

    def test_d64_cs64(self):
        """D=64, cs=64 — training config (n_embd=512, n_head=8)."""
        inputs = _make_inputs(B=1, H=2, D=64, cs=64)
        W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q = inputs
        ns_steps = 3

        y_ref, state_ref = _regular_fwd(
            W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
            ns_steps=ns_steps, pe_ste=True)

        y_pal, W1o, W2o, SW1o, SW2o = _pallas_fused_fwd(
            W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
            ns_steps)

        assert y_pal.shape == y_ref.shape
        assert jnp.allclose(y_pal, y_ref, atol=1e-4, rtol=1e-3), \
            f"y max diff: {jnp.max(jnp.abs(y_pal - y_ref))}"
        assert jnp.allclose(W1o, state_ref.W1, atol=1e-4, rtol=1e-3)
        assert jnp.allclose(SW1o, state_ref.S_W1, atol=1e-4, rtol=1e-3)

    def test_d96_cs64(self):
        """D=96, cs=64 — profile config (n_embd=384, n_head=4)."""
        inputs = _make_inputs(B=1, H=2, D=96, cs=64)
        W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q = inputs
        ns_steps = 3

        y_ref, state_ref = _regular_fwd(
            W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
            ns_steps=ns_steps, pe_ste=True)

        y_pal, W1o, W2o, SW1o, SW2o = _pallas_fused_fwd(
            W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
            ns_steps)

        assert y_pal.shape == y_ref.shape == (1, 64, 2, 96)
        assert jnp.allclose(y_pal, y_ref, atol=1e-3, rtol=1e-3), \
            f"y max diff: {jnp.max(jnp.abs(y_pal - y_ref))}"

    def test_bf16(self):
        """bf16 inputs (kernel computes in f32 internally)."""
        inputs = _make_inputs(B=1, H=2, D=64, cs=64)
        inputs_bf16 = tuple(x.astype(jnp.bfloat16) for x in inputs)
        W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q = inputs_bf16
        ns_steps = 3

        y_ref, state_ref = _regular_fwd(
            W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
            ns_steps=ns_steps, pe_ste=True)

        y_pal, W1o, W2o, SW1o, SW2o = _pallas_fused_fwd(
            W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
            ns_steps)

        assert y_pal.dtype == jnp.bfloat16
        assert jnp.allclose(y_pal, y_ref, atol=0.05, rtol=0.05), \
            f"y max diff: {jnp.max(jnp.abs(y_pal.astype(jnp.float32) - y_ref.astype(jnp.float32)))}"


class TestCustomVJP:
    """Test that custom_vjp backward produces valid gradients."""

    def test_gradient_flow(self):
        """Verify gradients are non-zero and finite."""
        inputs = _make_inputs(B=1, H=2, D=64, cs=64)
        W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q = inputs

        def loss_fn(W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q):
            y, state = fused_chunk_scan(
                W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q,
                ns_steps=3, pe_ste=True)
            return jnp.sum(y ** 2) + jnp.sum(state.W1 ** 2)

        grads = jax.grad(loss_fn, argnums=range(9))(
            W1, W2, SW1, SW2, momW1, momW2, theta, alpha, q)

        for name, g in zip(
            ['W1', 'W2', 'SW1', 'SW2', 'momW1', 'momW2', 'theta', 'alpha', 'q'],
            grads,
        ):
            assert jnp.all(jnp.isfinite(g)), f"grad {name} has non-finite values"
            assert jnp.any(g != 0), f"grad {name} is all zeros"
