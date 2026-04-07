#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=flash_atlas_v2
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Profile FlashATLAS v2: f32 vs TF32 vs TF32+no-checkpoint
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "==> Start time: $(date)"

# Quick test that fused chunk still works
echo "==> Smoke test..."
python -m pytest tests/test_fused_chunk.py -v --tb=short 2>&1 || {
    echo "TESTS FAILED"
    exit 1
}

# Profile: baseline vs unrolled+f32 vs unrolled+TF32 vs unrolled+TF32+nockpt
echo ""
echo "==> Profile comparison..."
TRACE_DIR="/p/scratch/westai0047/nanochat/logs/flash_atlas_trace_v2"
srun python -m atlas_jax.profile_fused --preset h100 \
    --trace-dir "$TRACE_DIR"

echo ""
echo "==> Done at $(date)"
