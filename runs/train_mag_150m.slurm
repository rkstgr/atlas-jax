#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=mag150m
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# MAG 150M: Match nanochat D12 (151M params, 1.32B tokens, ClimbMix)
# 12L, 768d, 12H dh=64, expand=1, memory on even layers, seq=2048
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
    --model=mag \
    --n-layer=12 --n-head=12 --n-embd=768 \
    --memory-expand=1 --deep-memory \
    --memory-layers=1,3,5,7,9,11 \
    --window-size=64 \
    --poly-degree=2 --omega-window=2 --chunk-size=64 \
    --ns-steps=5 --pe-ste \
    --dropout=0.0 --gate-bias-init=0.0 --stop-grad-chunks \
    --seq-len=2048 \
    --batch-size=32 \
    --optimizer=adamw \
    --lr=1e-4 --weight-decay=0.01 \
    --warmup-steps=500 --warmdown-steps=0 \
    --total-steps=20160 \
    --max-tokens=1.32e9 \
    --eval-every=1000 --eval-steps=20 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer \
    --ckpt-dir=/p/scratch/westai0047/nanochat/checkpoints/mag_150m \
    --ckpt-every=5000 \
    --seed=42

echo "==> Done at $(date)"
