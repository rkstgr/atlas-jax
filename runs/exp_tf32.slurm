#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=atlas_tf
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Test TF32 precision with bf16 model (may be faster, risk of NaN)
set -euo pipefail

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

PROJ_DIR="/p/project1/westai0047/atlas-jax"
DATA_DIR="/p/scratch/westai0047/nanochat/base_data_climbmix"
TOK_DIR="/p/scratch/westai0047/nanochat/tokenizer"

echo "==> Node: $SLURM_JOB_NODELIST | Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd "$PROJ_DIR"
source .venv/bin/activate

echo "==> Starting TF32 experiment..."
srun python -m atlas_jax.train \
    --n-layer=6 \
    --n-head=4 \
    --n-embd=512 \
    --memory-expand=1 \
    --no-deep-memory \
    --poly-degree=3 \
    --omega-window=1 \
    --chunk-size=64 \
    --ns-steps=1 \
    --pe-ste \
    --seq-len=1024 \
    --batch-size=32 \
    --lr=5e-3 \
    --weight-decay=0.1 \
    --warmup-steps=100 \
    --total-steps=10000 \
    --time-budget=420 \
    --eval-every=200 \
    --eval-steps=10 \
    --gpu-peak-tflops=989.4 \
    --data-dir="$DATA_DIR" \
    --tokenizer-dir="$TOK_DIR" \
    --seed=42 \
    --matmul-precision=high

echo "==> Done at $(date)"
