"""Test fused Triton kernel with D != E (memory_expand > 1).

Verifies that the fused kernel produces the same output as the non-fused path
for both D==E and D!=E configurations.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_default_matmul_precision", "float32")

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas


def _test_fused_vs_nonfused(D, E, n_head=2, chunk_size=16, seq_len=64, batch=2):
    """Compare fused vs non-fused output for given D, E."""
    n_embd = n_head * D  # dim_head = D when dim_head is None

    cfg_fused = AtlasConfig(
        vocab_size=256, n_layer=1, n_head=n_head, n_embd=n_embd,
        chunk_size=chunk_size, ns_steps=3, omega_window=2, poly_degree=2,
        deep_memory=True, memory_expand=E // D, pe_ste=True,
        fused_chunk=True, dropout=0.0, gate_bias_init=0.0,
        max_lr=0.1, logit_softcap=0.0, stop_grad_chunks=True,
        geglu_ff=False, num_persist_mem_tokens=0)

    cfg_nonfused = AtlasConfig(
        vocab_size=256, n_layer=1, n_head=n_head, n_embd=n_embd,
        chunk_size=chunk_size, ns_steps=3, omega_window=2, poly_degree=2,
        deep_memory=True, memory_expand=E // D, pe_ste=True,
        fused_chunk=False, dropout=0.0, gate_bias_init=0.0,
        max_lr=0.1, logit_softcap=0.0, stop_grad_chunks=True,
        geglu_ff=False, num_persist_mem_tokens=0)

    key = jax.random.PRNGKey(42)
    model_fused = Atlas(cfg_fused, key=key, pad_vocab_size_to=1)
    model_nonfused = Atlas(cfg_nonfused, key=key, pad_vocab_size_to=1)

    # Verify they have same weights
    leaves_f = jax.tree.leaves(eqx.filter(model_fused, eqx.is_array))
    leaves_n = jax.tree.leaves(eqx.filter(model_nonfused, eqx.is_array))
    for lf, ln in zip(leaves_f, leaves_n):
        assert jnp.allclose(lf, ln), "Models have different weights"

    idx = jax.random.randint(key, (batch, seq_len), 0, 256)

    logits_f, _ = model_fused(idx)
    logits_n, _ = model_nonfused(idx)

    # Compare — fused uses bf16 internally so tolerance is higher
    max_err = float(jnp.max(jnp.abs(logits_f - logits_n)))
    rel_err = max_err / (float(jnp.max(jnp.abs(logits_n))) + 1e-10)
    return max_err, rel_err


def test_fused_d_eq_e():
    """D == E (expand=1): the original case, should still work."""
    max_err, rel_err = _test_fused_vs_nonfused(D=64, E=64)
    print(f"D=64, E=64: max_err={max_err:.2e}, rel_err={rel_err:.2e}")
    assert rel_err < 0.05, f"D==E fused vs nonfused too different: rel_err={rel_err:.2e}"


def test_fused_d_ne_e():
    """D != E (expand=2): the new case."""
    max_err, rel_err = _test_fused_vs_nonfused(D=64, E=128)
    print(f"D=64, E=128: max_err={max_err:.2e}, rel_err={rel_err:.2e}")
    assert rel_err < 0.05, f"D!=E fused vs nonfused too different: rel_err={rel_err:.2e}"


def test_fused_d_ne_e_small():
    """D != E with smaller dimensions."""
    max_err, rel_err = _test_fused_vs_nonfused(D=32, E=64)
    print(f"D=32, E=64: max_err={max_err:.2e}, rel_err={rel_err:.2e}")
    assert rel_err < 0.05, f"D!=E small fused vs nonfused too different: rel_err={rel_err:.2e}"


def test_fused_backward_d_ne_e():
    """Verify backward pass produces finite gradients with D != E."""
    cfg = AtlasConfig(
        vocab_size=256, n_layer=1, n_head=2, n_embd=128, dim_head=None,
        chunk_size=16, ns_steps=3, omega_window=2, poly_degree=2,
        deep_memory=True, memory_expand=2, pe_ste=True,
        fused_chunk=True, dropout=0.0, gate_bias_init=0.0,
        max_lr=0.1, logit_softcap=0.0, stop_grad_chunks=True,
        geglu_ff=False, num_persist_mem_tokens=0)

    key = jax.random.PRNGKey(42)
    model = Atlas(cfg, key=key, pad_vocab_size_to=1)
    idx = jax.random.randint(key, (2, 64), 0, 256)

    def loss_fn(m):
        logits, _ = m(idx)
        return jnp.mean(logits)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    gl = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    n_nan = sum(1 for g in gl if not jnp.all(jnp.isfinite(g)))
    print(f"Backward D!=E fused: loss={float(loss):.4f}, nan_grads={n_nan}/{len(gl)}")
    assert n_nan == 0, f"Backward produced {n_nan} NaN gradient leaves"


if __name__ == "__main__":
    print("Testing fused kernel D != E support...")
    test_fused_d_eq_e()
    print("PASS: D == E")
    test_fused_d_ne_e()
    print("PASS: D != E (64, 128)")
    test_fused_d_ne_e_small()
    print("PASS: D != E (32, 64)")
    test_fused_backward_d_ne_e()
    print("PASS: backward D != E")
    print("\nAll tests passed!")
