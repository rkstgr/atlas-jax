#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --job-name=profile100m
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Profile 100M Atlas on 1×H100: bf16, fused, single train step trace
set -euo pipefail
cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

TRACE_DIR="/p/scratch/westai0047/nanochat/traces/atlas_100m_bf16_$(date +%s)"
echo "==> GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "==> Trace dir: $TRACE_DIR"

python -c "
import jax, jax.numpy as jnp, equinox as eqx, optax, time
jax.config.update('jax_default_matmul_precision', 'high')
from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas

cfg = AtlasConfig(
    vocab_size=32768, n_layer=12, n_head=12, n_embd=768,
    memory_expand=1, chunk_size=64, ns_steps=5, omega_window=2,
    poly_degree=2, deep_memory=True, pe_ste=True,
    fused_chunk=True, dropout=0.0, gate_bias_init=0.0, max_lr=0.1,
    logit_softcap=0.0, stop_grad_chunks=True, geglu_ff=True,
    num_persist_mem_tokens=4, use_checkpoint=True)

key = jax.random.PRNGKey(42)
model = Atlas(cfg, key=key)
model = jax.tree.map(
    lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x,
    model, is_leaf=eqx.is_array)
n = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
print(f'Model: {n/1e6:.1f}M params, bf16, fused')

opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-4))
opt_state = opt.init(eqx.filter(model, eqx.is_array))

B, T = 8, 1024
idx = jax.random.randint(key, (B, T), 0, cfg.vocab_size)
targets = jnp.roll(idx, -1, axis=1)

def _upcast(g):
    return jax.tree.map(
        lambda x: x.astype(jnp.float32) if eqx.is_array(x) and x.dtype == jnp.bfloat16 else x,
        g, is_leaf=eqx.is_array)

@eqx.filter_jit
def train_step(model, opt_state, x, t):
    def loss_fn(m):
        logits, _ = m(x)
        lf = logits.reshape(-1, logits.shape[-1]).astype(jnp.float32)
        tf = t.reshape(-1)
        return -jnp.mean(jax.nn.log_softmax(lf, axis=-1)[jnp.arange(tf.shape[0]), tf])
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    grads = _upcast(grads)
    updates, new_opt = opt.update(
        eqx.filter(grads, eqx.is_array), opt_state,
        eqx.filter(model, eqx.is_array))
    return eqx.apply_updates(model, updates), new_opt, loss

# Warmup (3 steps)
print('Warming up...')
for i in range(3):
    model, opt_state, loss = train_step(model, opt_state, idx, targets)
    print(f'  warmup {i}: loss={float(loss):.4f}')

# Profile 1 step
print(f'Profiling to {\"$TRACE_DIR\"}...')
with jax.profiler.trace('$TRACE_DIR'):
    model, opt_state, loss = train_step(model, opt_state, idx, targets)
    jax.block_until_ready(jax.tree.leaves(eqx.filter(model, eqx.is_array)))

print(f'Done. loss={float(loss):.4f}')
print(f'View: tensorboard --logdir=$TRACE_DIR')
print(f'  or: download and open in perfetto.dev')
"
echo "==> Trace at: $TRACE_DIR"
echo "==> Done"
