#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=atlas_bench_baseline
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Baseline benchmark: measure current performance on H100
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "==> JAX version: $(python -c 'import jax; print(jax.__version__)')"
echo "==> Start: $(date)"

# Run baseline + key single-variable comparisons
srun python scripts/benchmark.py --configs \
    baseline \
    baseline_tf32 \
    unrolled \
    unrolled_tf32 \
    cs128 \
    cs256 \
    donate \
    nocheckpoint \
    nocheckpoint_tf32

echo "==> Done: $(date)"
