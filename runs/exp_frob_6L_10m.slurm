#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --job-name=atlas_6t
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"
cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate
# 6 layers 384d with 10min — will larger model beat 4L with more tokens?
srun python -m atlas_jax.train \
    --n-layer=6 --n-head=4 --n-embd=384 \
    --memory-expand=1 --no-deep-memory \
    --poly-degree=3 --omega-window=1 --chunk-size=64 \
    --ns-steps=0 --pe-ste --seq-len=1024 \
    --batch-size=64 --lr=1e-2 --weight-decay=0.1 \
    --warmup-steps=50 --total-steps=10000 --time-budget=600 \
    --eval-every=500 --eval-steps=5 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer --seed=42
