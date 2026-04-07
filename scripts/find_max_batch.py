"""Find max batch size for 59M Atlas on 1×H100."""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

jax.config.update("jax_compilation_cache_dir", "/p/scratch/westai0047/nanochat/jax_cache")
jax.config.update("jax_default_matmul_precision", "high")

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas

config = AtlasConfig(
    sequence_len=1024, n_layer=8, n_head=8, n_embd=512,
    chunk_size=64, ns_steps=3, omega_window=16, poly_degree=3,
    deep_memory=True, memory_expand=1, pe_ste=True,
    use_checkpoint=True, fused_chunk=True,
)

optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-3, weight_decay=0.1))

# No donation — we reuse model/opt_state across batch sizes
@eqx.filter_jit
def train_step(model, opt_state, inputs, targets):
    def loss_fn(m):
        logits, _ = m(inputs)
        lf = logits.reshape(-1, logits.shape[-1])
        tf = targets.reshape(-1)
        return -jnp.mean(jax.nn.log_softmax(lf, axis=-1)[jnp.arange(tf.shape[0]), tf])
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, nos = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), nos, loss

key = jax.random.PRNGKey(0)
model = Atlas(config, key=key)
model = jax.tree.map(
    lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x,
    model, is_leaf=eqx.is_array)
n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
print(f"Params: {n_params:,}")

opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

for B in [16, 32, 48, 64, 96, 128]:
    try:
        inp = jnp.ones((B, 1024), dtype=jnp.int32)
        tgt = jnp.ones((B, 1024), dtype=jnp.int32)
        _, _, loss = train_step(model, opt_state, inp, tgt)
        float(loss)
        mem = jax.local_devices()[0].memory_stats()
        peak_gb = mem["peak_bytes_in_use"] / 1e9 if mem else -1
        print(f"B={B:>3}: OK | peak={peak_gb:.1f}GB / 80GB")
    except Exception as e:
        print(f"B={B:>3}: FAIL | {str(e)[:120]}")
        break
