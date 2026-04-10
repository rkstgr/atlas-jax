"""Verify fused_chunk_scan fix: gradients flow correctly."""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

jax.config.update("jax_default_matmul_precision", "high")

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas

def make_config(fused_chunk):
    return AtlasConfig(
        sequence_len=256, n_layer=2, n_head=8, n_embd=512,
        chunk_size=64, ns_steps=3, omega_window=16, poly_degree=3,
        deep_memory=True, memory_expand=1, pe_ste=True,
        use_checkpoint=True, fused_chunk=fused_chunk,
    )

def loss_fn(model, inputs, targets):
    logits, _ = model(inputs)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])

def to_bf16(model):
    def _cast(x):
        return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
    return jax.tree.map(_cast, model, is_leaf=eqx.is_array)

def grad_stats(grads):
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    total = sum(l.size for l in leaves)
    zero_count = sum(int(jnp.sum(l == 0)) for l in leaves)
    nan_count = sum(int(jnp.sum(jnp.isnan(l))) for l in leaves)
    inf_count = sum(int(jnp.sum(jnp.isinf(l))) for l in leaves)
    max_abs = max(float(jnp.max(jnp.abs(l))) for l in leaves)
    return f"params={total} zero={zero_count/total:.1%} nan={nan_count} inf={inf_count} max={max_abs:.2e}"

print(f"JAX {jax.__version__} | devices: {jax.devices()}")
print("=" * 70)

key = jax.random.PRNGKey(42)
B, T = 2, 256
key, k1, k2 = jax.random.split(key, 3)
inputs = jax.random.randint(k1, (B, T), 0, 32768)
targets = jax.random.randint(k2, (B, T), 0, 32768)

# ===== TEST 1: Eager grad — both paths =====
print("\n--- TEST 1: Eager grad (should work for BOTH now) ---")
key, mk = jax.random.split(key)
model_nf = Atlas(make_config(fused_chunk=False), key=mk)
model_f = Atlas(make_config(fused_chunk=True), key=mk)

loss_nf, grads_nf = eqx.filter_value_and_grad(loss_fn)(model_nf, inputs, targets)
print(f"  non-fused: loss={float(loss_nf):.4f}  grads: {grad_stats(grads_nf)}")

loss_f, grads_f = eqx.filter_value_and_grad(loss_fn)(model_f, inputs, targets)
print(f"  fused:     loss={float(loss_f):.4f}  grads: {grad_stats(grads_f)}")

# ===== TEST 2: Multi-step training =====
print("\n--- TEST 2: 10-step training (does loss decrease?) ---")

@eqx.filter_jit(donate='all')
def train_step(model, opt_state, optimizer, inputs, targets):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, targets)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss

for label, fused, use_bf16 in [
    ("non-fused f32", False, False),
    ("fused f32", True, False),
    ("non-fused bf16", False, True),
    ("fused bf16", True, True),
]:
    key, mk = jax.random.split(key)
    model = Atlas(make_config(fused_chunk=fused), key=mk)
    if use_bf16:
        model = to_bf16(model)

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=3e-3))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    losses = []
    for i in range(10):
        model, opt_state, loss = train_step(model, opt_state, opt, inputs, targets)
        losses.append(float(loss))

    delta = losses[9] - losses[0]
    status = 'OK' if delta < -0.01 else 'FLAT!'
    print(f"  {label:20s}: loss[0]={losses[0]:.4f}  loss[9]={losses[9]:.4f}  "
          f"delta={delta:+.4f}  {status}")

print("\nDone.")
