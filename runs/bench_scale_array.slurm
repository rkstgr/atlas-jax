#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --job-name=bench_arr
#SBATCH --array=0-7
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%A_%a.err

# Array indices:
# 0: 100M, expand=1, non-fused
# 1: 100M, expand=1, fused
# 2: 100M, expand=2, non-fused
# 3: 100M, expand=2, fused
# 4: 1.2B, expand=1, non-fused
# 5: 1.2B, expand=1, fused
# 6: 1.2B, expand=2, non-fused
# 7: 1.2B, expand=2, fused

set -euo pipefail
cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Task $SLURM_ARRAY_TASK_ID | GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

python -c "
import jax, jax.numpy as jnp, equinox as eqx, optax, time, gc, sys
jax.config.update('jax_default_matmul_precision', 'float32')
from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas

task_id = int(sys.argv[1])
SEQ = 1024

# Configs: (label, n_layer, n_embd, n_head, expand, fused)
configs = [
    ('100M', 12, 768, 12, 1, False),
    ('100M', 12, 768, 12, 1, True),
    ('100M', 12, 768, 12, 2, False),
    ('100M', 12, 768, 12, 2, True),
    ('1.2B', 24, 2048, 32, 1, False),
    ('1.2B', 24, 2048, 32, 1, True),
    ('1.2B', 24, 2048, 32, 2, False),
    ('1.2B', 24, 2048, 32, 2, True),
]

label, n_layer, n_embd, n_head, expand, fused = configs[task_id]
tag = 'FUSED' if fused else 'NOFUSE'

def make_model(batch):
    cfg = AtlasConfig(
        vocab_size=256, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        memory_expand=expand, chunk_size=64, ns_steps=5, omega_window=2,
        poly_degree=2, deep_memory=True, pe_ste=True,
        fused_chunk=fused, dropout=0.0, gate_bias_init=0.0, max_lr=0.1,
        logit_softcap=0.0, stop_grad_chunks=True, geglu_ff=True,
        num_persist_mem_tokens=4, use_checkpoint=True)
    key = jax.random.PRNGKey(42)
    model = Atlas(cfg, key=key, pad_vocab_size_to=1)
    # Cast to bf16 for large models
    model = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x,
        model, is_leaf=eqx.is_array)
    n = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    return model, cfg, n

def try_batch(batch):
    try:
        model, cfg, n_params = make_model(batch)
        opt = optax.adamw(1e-4)
        opt_state = opt.init(eqx.filter(model, eqx.is_array))
        idx = jax.random.randint(jax.random.PRNGKey(0), (batch, SEQ), 0, 256)
        targets = jnp.roll(idx, -1, axis=1)

        @eqx.filter_jit
        def step(model, opt_state, x, t):
            def loss_fn(m):
                logits, _ = m(x)
                lf = logits.reshape(-1, logits.shape[-1]).astype(jnp.float32)
                tf = t.reshape(-1)
                return -jnp.mean(jax.nn.log_softmax(lf, axis=-1)[jnp.arange(tf.shape[0]), tf])
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            grads = jax.tree.map(lambda g: g.astype(jnp.float32) if eqx.is_array(g) and g.dtype == jnp.bfloat16 else g, grads, is_leaf=eqx.is_array)
            updates, new_opt = opt.update(eqx.filter(grads, eqx.is_array), opt_state, eqx.filter(model, eqx.is_array))
            return eqx.apply_updates(model, updates), new_opt, loss

        model, opt_state, loss = step(model, opt_state, idx, targets)
        float(loss)
        peak = jax.devices()[0].memory_stats()['peak_bytes_in_use'] / 1e9
        del model, opt_state, idx, targets
        jax.clear_caches(); gc.collect()
        return True, peak, n_params
    except Exception as e:
        jax.clear_caches(); gc.collect()
        return False, 0, 0

# Binary search for max batch size
print(f'{label} E={expand} {tag}: searching max batch...')
lo, hi, best_batch = 1, 128 if '100M' in label else 32, 0
while lo <= hi:
    mid = (lo + hi) // 2
    ok, peak, n_params = try_batch(mid)
    if ok:
        best_batch = mid
        lo = mid + 1
        print(f'  B={mid}: OK (peak={peak:.1f}GB)')
    else:
        hi = mid - 1
        print(f'  B={mid}: OOM')

if best_batch == 0:
    print(f'{label} E={expand} {tag}: FAILED - cannot fit batch=1')
    sys.exit(0)

# Benchmark at max batch
print(f'Benchmarking at B={best_batch}...')
model, cfg, n_params = make_model(best_batch)
opt = optax.adamw(1e-4)
opt_state = opt.init(eqx.filter(model, eqx.is_array))
idx = jax.random.randint(jax.random.PRNGKey(0), (best_batch, SEQ), 0, 256)
targets = jnp.roll(idx, -1, axis=1)

@eqx.filter_jit
def step(model, opt_state, x, t):
    def loss_fn(m):
        logits, _ = m(x)
        lf = logits.reshape(-1, logits.shape[-1]).astype(jnp.float32)
        tf = t.reshape(-1)
        return -jnp.mean(jax.nn.log_softmax(lf, axis=-1)[jnp.arange(tf.shape[0]), tf])
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    grads = jax.tree.map(lambda g: g.astype(jnp.float32) if eqx.is_array(g) and g.dtype == jnp.bfloat16 else g, grads, is_leaf=eqx.is_array)
    updates, new_opt = opt.update(eqx.filter(grads, eqx.is_array), opt_state, eqx.filter(model, eqx.is_array))
    return eqx.apply_updates(model, updates), new_opt, loss

# Warmup
for _ in range(3):
    model, opt_state, loss = step(model, opt_state, idx, targets)
    float(loss)

# Time 10 steps
times = []
for _ in range(10):
    t0 = time.time()
    model, opt_state, loss = step(model, opt_state, idx, targets)
    float(loss)
    times.append(time.time() - t0)

avg = sum(times) / len(times)
mem = jax.devices()[0].memory_stats()
peak_gb = mem['peak_bytes_in_use'] / 1e9
tps = best_batch * SEQ / avg

print(f'')
print(f'RESULT | {label} E={expand} {tag} | B={best_batch} | {avg*1000:.0f}ms/step | {peak_gb:.1f}GB | {n_params/1e6:.0f}M params | {tps:.0f} tok/s')
" $SLURM_ARRAY_TASK_ID
