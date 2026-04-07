#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --job-name=triton_pe
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Measure Triton PE speedup: isolated PE + full model comparison
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Step 1: Smoke test
echo "==> Smoke tests..."
python -m pytest tests/test_fused_chunk.py -v --tb=short 2>&1 || { echo "FAILED"; exit 1; }

# Step 2: PE microbenchmark (shows JAX PE vs Triton PE)
echo ""
echo "==> PE microbenchmark (TF32)..."
srun python -m atlas_jax.profile_pe --matmul-precision high

# Step 3: Full model comparison
echo ""
echo "==> Full model profile..."
srun python -m atlas_jax.profile_fused --preset h100

echo "==> Done $(date)"
