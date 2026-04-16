#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=pe_micro
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo ""
echo "=== TF32 precision ==="
srun python scripts/profile_pe.py --matmul-precision high

echo ""
echo "=== f32 precision ==="
srun python scripts/profile_pe.py --matmul-precision float32

echo "==> Done $(date)"
