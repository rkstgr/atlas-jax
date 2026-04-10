#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL,CUDA_VISIBLE_DEVICES=0
#SBATCH --time=00:15:00
#SBATCH --job-name=atlas_59m_fix
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# ==============================================================================
# Atlas-JAX 59M — verify fused chunk + bf16 fix (1×H100, ~10min)
# Config: n_embd=512, n_head=8, L=8, expand=1, pe_ste, fused_chunk, bf16
# Expect: loss decreasing from ~10.57 → well below 10.50 over ~2000 steps
# ==============================================================================

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Node: $SLURM_JOB_NODELIST | Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "==> Starting Atlas 59M fused bf16 training (fixed)..."
python -m atlas_jax.train \
    --n-layer=8 --n-head=8 --n-embd=512 \
    --memory-expand=1 --deep-memory \
    --poly-degree=3 --omega-window=16 --chunk-size=64 \
    --ns-steps=3 --pe-ste --fused-chunk \
    --seq-len=1024 \
    --batch-size=4 --lr=3e-3 --weight-decay=0.1 \
    --warmup-steps=200 --total-steps=5000 \
    --time-budget=600 \
    --eval-every=200 --eval-steps=20 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer \
    --seed=42

echo "==> Done at $(date)"
