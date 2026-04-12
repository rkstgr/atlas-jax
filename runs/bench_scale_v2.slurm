#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=bench_v2
#SBATCH --array=0-7
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%A_%a.err

set -euo pipefail
cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate
echo "==> Task $SLURM_ARRAY_TASK_ID | GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

python -c "
import jax, jax.numpy as jnp, equinox as eqx, optax, time, sys
jax.config.update('jax_default_matmul_precision', 'float32')
from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas

tid = int(sys.argv[1])
SEQ = 1024

# (label, n_layer, n_embd, n_head, expand, fused, batch)
# Batch sizes chosen conservatively based on previous OOM results
configs = [
    ('100M', 12, 768, 12, 1, False, 48),
    ('100M', 12, 768, 12, 1, True,  40),
    ('100M', 12, 768, 12, 2, False, 44),
    ('100M', 12, 768, 12, 2, True,  32),
    ('1.2B', 24, 2048, 32, 1, False, 4),
    ('1.2B', 24, 2048, 32, 1, True,  4),
    ('1.2B', 24, 2048, 32, 2, False, 4),
    ('1.2B', 24, 2048, 32, 2, True,  2),
]

label, n_layer, n_embd, n_head, expand, fused, batch = configs[tid]
tag = 'FUSED' if fused else 'NOFUSE'

cfg = AtlasConfig(
    vocab_size=256, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    memory_expand=expand, chunk_size=64, ns_steps=5, omega_window=2,
    poly_degree=2, deep_memory=True, pe_ste=True,
    fused_chunk=fused, dropout=0.0, gate_bias_init=0.0, max_lr=0.1,
    logit_softcap=0.0, stop_grad_chunks=True, geglu_ff=True,
    num_persist_mem_tokens=4, use_checkpoint=True)

key = jax.random.PRNGKey(42)
model = Atlas(cfg, key=key, pad_vocab_size_to=1)
model = jax.tree.map(
    lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x,
    model, is_leaf=eqx.is_array)
n = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))

opt = optax.adamw(1e-4)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

idx = jax.random.randint(key, (batch, SEQ), 0, 256)
targets = jnp.roll(idx, -1, axis=1)

@eqx.filter_jit
def step(model, opt_state, x, t):
    def loss_fn(m):
        logits, _ = m(x)
        lf = logits.reshape(-1, logits.shape[-1]).astype(jnp.float32)
        tf = t.reshape(-1)
        return -jnp.mean(jax.nn.log_softmax(lf, axis=-1)[jnp.arange(tf.shape[0]), tf])
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    grads = jax.tree.map(lambda g: g.astype(jnp.float32) if eqx.is_array(g) and g.dtype==jnp.bfloat16 else g, grads, is_leaf=eqx.is_array)
    updates, new_opt = opt.update(eqx.filter(grads, eqx.is_array), opt_state, eqx.filter(model, eqx.is_array))
    return eqx.apply_updates(model, updates), new_opt, loss

# Warmup (3 steps)
for _ in range(3):
    model, opt_state, loss = step(model, opt_state, idx, targets)
    float(loss)

# Benchmark (10 steps)
times = []
for _ in range(10):
    t0 = time.time()
    model, opt_state, loss = step(model, opt_state, idx, targets)
    float(loss)
    times.append(time.time() - t0)

avg = sum(times) / len(times)
mem = jax.devices()[0].memory_stats()
peak_gb = mem['peak_bytes_in_use'] / 1e9
tps = batch * SEQ / avg

print(f'RESULT | {label} E={expand} {tag} | B={batch} | {avg*1000:.0f}ms | {peak_gb:.1f}GB | {n/1e6:.0f}M | {tps:.0f} tok/s')
" $SLURM_ARRAY_TASK_ID
