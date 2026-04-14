#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --job-name=atlas100m
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# 100M Atlas, 4×H100, bf16, fused kernel, expand=1, 500M tokens
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Nodes: $SLURM_JOB_NODELIST | Tasks: $SLURM_NTASKS | Start: $(date)"
srun --ntasks=1 nvidia-smi --query-gpu=name,memory.total --format=csv

srun python -m atlas_jax.train_distributed \
    --n-layer=12 --n-head=12 --n-embd=768 \
    --memory-expand=1 --deep-memory \
    --poly-degree=2 --omega-window=2 --chunk-size=64 \
    --ns-steps=5 --pe-ste --fused-chunk \
    --dropout=0.0 --gate-bias-init=0.0 \
    --seq-len=1024 \
    --batch-size=32 \
    --optimizer=adamw \
    --lr=3e-4 --weight-decay=0.01 \
    --warmup-steps=500 --warmdown-steps=0 \
    --total-steps=20000 \
    --max-tokens=5e8 \
    --eval-every=500 --eval-steps=20 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer \
    --ckpt-dir=/p/scratch/westai0047/nanochat/checkpoints/atlas_100m_bf16 \
    --ckpt-every=2000 \
    --seed=42

echo "==> Done at $(date)"
