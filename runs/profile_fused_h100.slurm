#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=flash_atlas_profile
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Profile FlashATLAS: baseline vs unrolled+fused on H100
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "==> Start time: $(date)"

# Step 1: Run correctness tests
echo ""
echo "==> Running fused chunk correctness tests..."
python -m pytest tests/test_fused_chunk.py -v --tb=short 2>&1 || {
    echo "TESTS FAILED — aborting profile"
    exit 1
}

# Step 2: Profile comparison (all configs)
echo ""
echo "==> Running profile comparison..."
TRACE_DIR="/p/scratch/westai0047/nanochat/logs/flash_atlas_trace"
srun python -m atlas_jax.profile_fused --preset h100 \
    --trace-dir "$TRACE_DIR"

echo ""
echo "==> Done at $(date)"
