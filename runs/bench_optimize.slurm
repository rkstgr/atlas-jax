#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --job-name=atlas_bench_optimize
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Optimization benchmark: test chunk sizes, ns_steps, omega, no-ckpt with unrolled+TF32
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "==> Start: $(date)"

# Phase 1: Test unrolled + TF32 as baseline, then sweep chunk sizes and hyperparams
echo "==> Phase 1: Optimization sweep..."
srun python scripts/benchmark.py --configs \
    unrolled_tf32 \
    cs128_unrolled_tf32 \
    cs256_unrolled_tf32 \
    cs512_unrolled_tf32 \
    unrolled_tf32_nockpt_b16 \
    unrolled_tf32_ns2 \
    unrolled_tf32_ns1 \
    unrolled_tf32_omega1 \
    donate

# Phase 2: Capture JAX profiler trace for the unrolled+TF32 config
echo ""
echo "==> Phase 2: Capturing JAX profiler trace..."
TRACE_DIR="/p/scratch/westai0047/nanochat/logs/atlas_trace_optimize"
srun python -c "
import jax, jax.numpy as jnp, equinox as eqx, time
from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas

jax.config.update('jax_default_matmul_precision', 'high')

config = AtlasConfig(
    sequence_len=2048, n_layer=8, n_head=8, n_embd=448,
    omega_window=16, poly_degree=3, deep_memory=True,
    memory_expand=1, ns_steps=3, pe_ste=True,
    use_checkpoint=True, fused_chunk=True,
)
model = Atlas(config, key=jax.random.PRNGKey(0))
model = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x, model, is_leaf=eqx.is_array)
idx = jax.random.randint(jax.random.PRNGKey(0), (32, 2048), 0, config.vocab_size)

@eqx.filter_jit
def step(model, idx):
    def loss_fn(m):
        logits, _ = m(idx)
        return jnp.mean(logits)
    return eqx.filter_value_and_grad(loss_fn)(model)

# Warmup
out = step(model, idx)
jax.block_until_ready(out)
out = step(model, idx)
jax.block_until_ready(out)

# Profile
with jax.profiler.trace('$TRACE_DIR'):
    for _ in range(3):
        out = step(model, idx)
        jax.block_until_ready(out)
print(f'Trace saved to $TRACE_DIR')
"

echo "==> Done: $(date)"
