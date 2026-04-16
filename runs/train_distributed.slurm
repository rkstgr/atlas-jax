#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=atlas_dist_a
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Option A: jax.distributed — 4×H100, 1 process per GPU
# Each srun task gets 1 GPU. jax.distributed.initialize() connects them.
# shard_map provides named axis for lax.pmean (NCCL all-reduce).

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"
export MASTER_ADDR=$(hostname)

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Option A: jax.distributed on $(hostname), $SLURM_NTASKS tasks"
nvidia-smi --query-gpu=name,memory.total --format=csv

srun python scripts/train_distributed.py \
    --n-layer=8 --n-head=8 --n-embd=512 \
    --memory-expand=1 --deep-memory \
    --poly-degree=3 --omega-window=16 --chunk-size=64 \
    --ns-steps=3 --pe-ste --fused-chunk \
    --seq-len=1024 \
    --batch-size=128 --lr=3e-3 --weight-decay=0.1 \
    --warmup-steps=50 --total-steps=200 \
    --eval-every=50 --eval-steps=10 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer \
    --seed=42

echo "==> Done at $(date)"
