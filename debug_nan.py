"""Debug NaN — systematic bisection on H100."""
import jax
jax.config.update("jax_default_matmul_precision", "float32")
import jax.numpy as jnp
import equinox as eqx

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas

def test(desc, config, batch, seq):
    model = Atlas(config, key=jax.random.PRNGKey(42))
    idx = jax.random.randint(jax.random.PRNGKey(1), (batch, seq), 0, 32768)
    @eqx.filter_value_and_grad
    def loss_fn(model):
        logits, _ = model(idx)
        f = logits.reshape(-1, logits.shape[-1])
        t = idx.reshape(-1)
        lp = jax.nn.log_softmax(f, axis=-1)
        return -jnp.mean(lp[jnp.arange(t.shape[0]), t])
    try:
        loss, grads = loss_fn(model)
        n_nan = sum(1 for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array)) if bool(jnp.any(jnp.isnan(g))))
        total = len(jax.tree.leaves(eqx.filter(grads, eqx.is_array)))
        print(f"{desc}: loss={float(loss):.4f} nan={n_nan}/{total}")
    except Exception as e:
        print(f"{desc}: FAILED {type(e).__name__}: {str(e)[:80]}")

base = dict(n_head=8, n_embd=448, chunk_size=64, ns_steps=5, omega_window=16,
            poly_degree=3, deep_memory=True, memory_expand=4, pe_ste=True, use_checkpoint=True)

# Vary layers
for nl in [1, 2, 4, 8]:
    test(f"L={nl} B=1 T=512", AtlasConfig(n_layer=nl, **base), 1, 512)

# Vary seq len with 8 layers
for T in [64, 128, 256, 512, 1024, 2048]:
    test(f"L=8 B=1 T={T}", AtlasConfig(n_layer=8, **base), 1, T)

# Vary batch with 8 layers
for B in [1, 2, 4, 8]:
    test(f"L=8 B={B} T=512", AtlasConfig(n_layer=8, **base), B, 512)

# Test without deep memory
test("L=8 LINEAR B=1 T=2048", AtlasConfig(n_layer=8, n_head=8, n_embd=448, chunk_size=64,
    ns_steps=5, omega_window=16, poly_degree=3, deep_memory=False, pe_ste=True, use_checkpoint=True), 1, 2048)

print("\nDone.")
