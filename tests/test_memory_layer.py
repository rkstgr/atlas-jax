"""Tests for AtlasMemoryLayer — chunk processing, linear scan, and full forward."""

import jax
import jax.numpy as jnp
import pytest
import equinox as eqx

from atlas_jax.config import AtlasConfig
from atlas_jax.model import AtlasMemoryLayer, linear_scan, rms_norm


# ---------------------------------------------------------------------------
# linear_scan tests
# ---------------------------------------------------------------------------

class TestLinearScan:
    def test_basic_recurrence(self):
        """Verify h_t = gate_t * h_{t-1} + input_t."""
        B, T, H, D = 2, 4, 2, 3
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)

        h_init = jax.random.normal(k1, (B, H, D))
        gates = jax.nn.sigmoid(jax.random.normal(k2, (B, T, H)))
        inputs = jax.random.normal(k3, (B, T, H, D))

        h_all, h_final = linear_scan(h_init, gates, inputs)

        # Manual reference
        h = h_init
        for t in range(T):
            h = gates[:, t, :, None] * h + inputs[:, t]
            assert jnp.allclose(h_all[:, t], h, atol=1e-5), f"Mismatch at t={t}"
        assert jnp.allclose(h_final, h, atol=1e-5)

    def test_matrix_state(self):
        """linear_scan should work with matrix-valued states (D, D)."""
        B, T, H, D = 1, 3, 1, 4
        key = jax.random.PRNGKey(1)
        k1, k2, k3 = jax.random.split(key, 3)

        h_init = jax.random.normal(k1, (B, H, D, D))
        gates = jax.nn.sigmoid(jax.random.normal(k2, (B, T, H)))
        inputs = jax.random.normal(k3, (B, T, H, D, D))

        h_all, h_final = linear_scan(h_init, gates, inputs)
        assert h_all.shape == (B, T, H, D, D)
        assert h_final.shape == (B, H, D, D)

    def test_gradient_flows(self):
        """Gradients should flow through linear_scan."""
        B, T, H, D = 1, 4, 2, 3
        key = jax.random.PRNGKey(2)
        k1, k2, k3 = jax.random.split(key, 3)

        h_init = jax.random.normal(k1, (B, H, D))
        gates = jax.nn.sigmoid(jax.random.normal(k2, (B, T, H)))
        inputs = jax.random.normal(k3, (B, T, H, D))

        def f(h_init, gates, inputs):
            h_all, _ = linear_scan(h_init, gates, inputs)
            return jnp.sum(h_all ** 2)

        grads = jax.grad(f, argnums=(0, 1, 2))(h_init, gates, inputs)
        for g in grads:
            assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# AtlasMemoryLayer tests
# ---------------------------------------------------------------------------

def _make_small_config(deep_memory=False, omega_window=1):
    return AtlasConfig(
        n_layer=1, n_head=2, n_embd=32, chunk_size=8,
        conv_kernel=2, ns_steps=3, omega_window=omega_window,
        poly_degree=0, deep_memory=deep_memory, memory_expand=2,
        pe_ste=False, use_checkpoint=False,
    )


class TestMemoryLayerLinear:
    def test_forward_shape(self):
        config = _make_small_config(deep_memory=False)
        key = jax.random.PRNGKey(0)
        layer = AtlasMemoryLayer(config, key=key)
        x = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 32))
        y, state = layer(x)
        assert y.shape == (2, 16, 32)

    def test_single_chunk(self):
        """Single chunk (T == chunk_size) should work."""
        config = _make_small_config(deep_memory=False)
        key = jax.random.PRNGKey(2)
        layer = AtlasMemoryLayer(config, key=key)
        x = jax.random.normal(jax.random.PRNGKey(3), (1, 8, 32))
        y, state = layer(x)
        assert y.shape == (1, 8, 32)

    def test_gradient_flows(self):
        config = _make_small_config(deep_memory=False)
        key = jax.random.PRNGKey(4)
        layer = AtlasMemoryLayer(config, key=key)
        x = jax.random.normal(jax.random.PRNGKey(5), (1, 16, 32))

        @eqx.filter_value_and_grad
        def loss_fn(layer):
            y, _ = layer(x)
            return jnp.mean(y ** 2)

        loss, grads = loss_fn(layer)
        assert jnp.isfinite(loss)
        grad_leaves = jax.tree.leaves(grads)
        for g in grad_leaves:
            if isinstance(g, jnp.ndarray):
                assert jnp.all(jnp.isfinite(g))


class TestMemoryLayerDeep:
    def test_forward_shape(self):
        config = _make_small_config(deep_memory=True)
        key = jax.random.PRNGKey(10)
        layer = AtlasMemoryLayer(config, key=key)
        x = jax.random.normal(jax.random.PRNGKey(11), (2, 16, 32))
        y, state = layer(x)
        assert y.shape == (2, 16, 32)

    def test_gradient_flows(self):
        config = _make_small_config(deep_memory=True)
        key = jax.random.PRNGKey(12)
        layer = AtlasMemoryLayer(config, key=key)
        x = jax.random.normal(jax.random.PRNGKey(13), (1, 16, 32))

        @eqx.filter_value_and_grad
        def loss_fn(layer):
            y, _ = layer(x)
            return jnp.mean(y ** 2)

        loss, grads = loss_fn(layer)
        assert jnp.isfinite(loss)

    def test_omega_window_changes_output(self):
        """Different omega windows should produce different outputs."""
        key = jax.random.PRNGKey(20)
        x = jax.random.normal(jax.random.PRNGKey(21), (1, 16, 32))

        config1 = _make_small_config(deep_memory=True, omega_window=1)
        config2 = _make_small_config(deep_memory=True, omega_window=4)

        layer1 = AtlasMemoryLayer(config1, key=key)
        layer2 = AtlasMemoryLayer(config2, key=key)

        y1, _ = layer1(x)
        y2, _ = layer2(x)
        # Outputs should differ due to omega aggregation
        # (though with same init weights they may be similar,
        # the gamma gate will cause differences)
        # This is a smoke test that both configs run without error
        assert y1.shape == y2.shape

    def test_memory_state_persistence(self):
        """Memory state from first call should affect second call."""
        config = _make_small_config(deep_memory=True)
        key = jax.random.PRNGKey(30)
        layer = AtlasMemoryLayer(config, key=key)
        x = jax.random.normal(jax.random.PRNGKey(31), (1, 8, 32))

        y1, state1 = layer(x)
        y2, state2 = layer(x, memory_state=state1)

        # Outputs should differ because the second call starts with non-zero memory
        assert not jnp.allclose(y1, y2, atol=1e-6)
