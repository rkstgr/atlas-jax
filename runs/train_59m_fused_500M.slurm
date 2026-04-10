#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL,CUDA_VISIBLE_DEVICES=0
#SBATCH --time=06:00:00
#SBATCH --job-name=atlas_59m_500M
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# ==============================================================================
# Atlas-JAX 59M — fused Triton kernel + bf16, 1×H100, 500M tokens
# Rerun of 14630747 with fixed custom_vjp + f32 grad upcast
# Config: n_embd=512, n_head=8, L=8, expand=1, pe_ste, fused_chunk
# ==============================================================================

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Node: $SLURM_JOB_NODELIST | Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "==> Starting Atlas 59M fused bf16 training (500M tokens)..."
python -m atlas_jax.train \
    --n-layer=8 --n-head=8 --n-embd=512 \
    --memory-expand=1 --deep-memory \
    --poly-degree=3 --omega-window=16 --chunk-size=64 \
    --ns-steps=3 --pe-ste --fused-chunk \
    --seq-len=1024 \
    --batch-size=64 --lr=3e-3 --weight-decay=0.1 \
    --warmup-steps=100 --total-steps=5000 \
    --max-tokens=5e8 \
    --eval-every=100 --eval-steps=20 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer \
    --seed=42

echo "==> Done at $(date)"
