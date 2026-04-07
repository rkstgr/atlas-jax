#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=atlas_bench_batched
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Benchmark: batched W1/W2 PE + scans optimization
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "==> Start: $(date)"

# Run tests first to verify correctness
echo "==> Running tests..."
python -m pytest tests/ -x -q --tb=short 2>&1 || {
    echo "TESTS FAILED"
    exit 1
}

# Compare baseline vs batched (batched is now the default when D==E)
echo ""
echo "==> Benchmark comparison..."
srun python -m atlas_jax.benchmark --configs \
    baseline \
    baseline_tf32 \
    cs256 \
    nocheckpoint \
    nocheckpoint_tf32

echo "==> Done: $(date)"
