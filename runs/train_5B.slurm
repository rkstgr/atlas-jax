#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --job-name=atlas_5B
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Long training: best config (4GPU B=256 lr=2e-2), up to 5.6B tokens or 0.72 val_bpb
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"
cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

srun python -m atlas_jax.train \
    --n-layer=4 --n-head=4 --n-embd=256 \
    --memory-expand=1 --no-deep-memory \
    --poly-degree=3 --omega-window=1 --chunk-size=64 \
    --ns-steps=0 --pe-ste --seq-len=1024 \
    --batch-size=256 --lr=2e-2 --weight-decay=0.1 \
    --warmup-steps=200 --total-steps=100000 \
    --max-tokens=5.6e9 --target-bpb=0.72 \
    --eval-every=500 --eval-steps=20 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer --seed=42

echo "==> Done at $(date)"
