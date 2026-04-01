#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=atlas_5L
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Slightly larger: 5L 320d — balance capacity vs throughput
set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

PROJ_DIR="/p/project1/westai0047/atlas-jax"
DATA_DIR="/p/scratch/westai0047/nanochat/base_data_climbmix"
TOK_DIR="/p/scratch/westai0047/nanochat/tokenizer"

cd "$PROJ_DIR" && source .venv/bin/activate

srun python -m atlas_jax.train \
    --n-layer=5 --n-head=4 --n-embd=320 \
    --memory-expand=1 --no-deep-memory \
    --poly-degree=3 --omega-window=1 --chunk-size=64 \
    --ns-steps=1 --pe-ste --seq-len=1024 \
    --batch-size=64 --lr=5e-3 --weight-decay=0.1 \
    --warmup-steps=50 --total-steps=10000 --time-budget=420 \
    --eval-every=200 --eval-steps=10 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir="$DATA_DIR" --tokenizer-dir="$TOK_DIR" --seed=42

echo "==> Done at $(date)"
