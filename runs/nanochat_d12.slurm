#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --job-name=nanochat_d12
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NANOCHAT_BASE_DIR="/p/scratch/westai0047/nanochat/nanochat_cache"
export HF_HOME="/p/scratch/westai0047/huggingface"
export WANDB_MODE=offline
mkdir -p $NANOCHAT_BASE_DIR

cd /p/project1/westai0047/nanochat && source .venv/bin/activate

echo "==> GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x4"
echo "==> Start: $(date)"

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1

echo "==> Done: $(date)"
