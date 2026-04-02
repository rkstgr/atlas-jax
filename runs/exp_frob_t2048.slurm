#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=atlas_t2k
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Longer sequence (T=2048) for better language modeling
set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"
cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate
srun python -m atlas_jax.train \
    --n-layer=4 --n-head=4 --n-embd=256 \
    --memory-expand=1 --no-deep-memory \
    --poly-degree=3 --omega-window=1 --chunk-size=64 \
    --ns-steps=0 --pe-ste --seq-len=2048 \
    --batch-size=64 --lr=3e-2 --weight-decay=0.1 \
    --warmup-steps=30 --total-steps=10000 --time-budget=420 \
    --eval-every=100 --eval-steps=10 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer --seed=42
