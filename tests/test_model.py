"""Tests for full Atlas model forward/backward."""

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas


def _small_config():
    return AtlasConfig(
        sequence_len=32, vocab_size=256, n_layer=2, n_head=2, n_embd=32,
        chunk_size=8, conv_kernel=2, ns_steps=3, omega_window=4,
        poly_degree=2, deep_memory=True, memory_expand=2,
        pe_ste=False, use_checkpoint=False,
    )


def test_forward_shape():
    config = _small_config()
    key = jax.random.PRNGKey(0)
    model = Atlas(config, key=key)

    idx = jax.random.randint(jax.random.PRNGKey(1), (2, 32), 0, 256)
    logits, states = model(idx)
    assert logits.shape == (2, 32, 256)
    assert len(states) == config.n_layer


def test_forward_finite():
    config = _small_config()
    model = Atlas(config, key=jax.random.PRNGKey(0))
    idx = jax.random.randint(jax.random.PRNGKey(1), (1, 16), 0, 256)
    logits, _ = model(idx)
    assert jnp.all(jnp.isfinite(logits))


def test_soft_capping():
    """Logits should be bounded by [-15, 15] due to soft capping."""
    config = _small_config()
    model = Atlas(config, key=jax.random.PRNGKey(0))
    idx = jax.random.randint(jax.random.PRNGKey(1), (1, 16), 0, 256)
    logits, _ = model(idx)
    assert jnp.max(jnp.abs(logits)) <= 15.0 + 1e-5


def test_backward():
    """Model should be differentiable end-to-end."""
    config = _small_config()
    model = Atlas(config, key=jax.random.PRNGKey(0))

    idx = jax.random.randint(jax.random.PRNGKey(1), (1, 16), 0, 256)
    targets = jax.random.randint(jax.random.PRNGKey(2), (1, 16), 0, 256)

    @eqx.filter_value_and_grad
    def loss_fn(model):
        logits, _ = model(idx)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        return -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])

    loss, grads = loss_fn(model)
    assert jnp.isfinite(loss)

    # Check at least some gradients are non-zero
    grad_leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    has_nonzero = any(jnp.any(g != 0) for g in grad_leaves)
    assert has_nonzero, "All gradients are zero"


def test_sequence_padding():
    """Model should handle sequences not divisible by chunk_size."""
    config = _small_config()
    model = Atlas(config, key=jax.random.PRNGKey(0))
    # 13 is not divisible by chunk_size=8
    idx = jax.random.randint(jax.random.PRNGKey(1), (1, 13), 0, 256)
    logits, _ = model(idx)
    assert logits.shape == (1, 13, 256)


def test_memory_state_inference():
    """Memory state should persist across calls for autoregressive inference."""
    config = _small_config()
    model = Atlas(config, key=jax.random.PRNGKey(0))

    # First call: full prompt
    idx1 = jax.random.randint(jax.random.PRNGKey(1), (1, 16), 0, 256)
    _, states = model(idx1)

    # Second call: single token with memory state
    idx2 = jax.random.randint(jax.random.PRNGKey(2), (1, 1), 0, 256)
    logits, new_states = model(idx2, memory_states=states)
    assert logits.shape == (1, 1, 256)
    assert jnp.all(jnp.isfinite(logits))


def test_param_count():
    """Sanity check parameter count is reasonable."""
    config = _small_config()
    model = Atlas(config, key=jax.random.PRNGKey(0))
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    # Small model should have between 10K and 1M params
    assert 10_000 < n_params < 1_000_000, f"Unexpected param count: {n_params}"
