#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=atlas_profile
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Profile Atlas on H100: component-level timing + full model MFU
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "==> Start time: $(date)"

# Run the profiler with H100 preset, save JSON + JAX trace
TRACE_DIR="/p/scratch/westai0047/nanochat/logs/atlas_trace_h100"
srun python -m atlas_jax.profile --preset h100 \
    --json /p/scratch/westai0047/nanochat/logs/atlas_profile_h100.json \
    --trace "$TRACE_DIR"

echo "==> Done at $(date)"
