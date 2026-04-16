#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=atlas_59m_muon
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# ==============================================================================
# Atlas-JAX 59M — 16×H100, Muon+AdamW hybrid optimizer
# Paper-aligned hyperparameters: Muon for 2D matrices, AdamW for rest
# Global batch: 512 tokens/step (paper uses 0.5M), 500M tokens total
# ==============================================================================

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Nodes: $SLURM_JOB_NODELIST | Tasks: $SLURM_NTASKS | Start: $(date)"
echo "==> MASTER_ADDR: $MASTER_ADDR"
srun --ntasks=1 nvidia-smi --query-gpu=name,memory.total --format=csv

# Paper uses 0.5M tokens/batch. With 16 GPUs × 32/GPU × 1024 seq = 524,288 ≈ 0.5M
# 500M tokens / 524,288 = 954 steps
echo "==> Starting Atlas 59M Muon training (16×H100, 500M tokens)..."
srun python scripts/train_distributed.py \
    --n-layer=8 --n-head=8 --n-embd=512 \
    --memory-expand=1 --deep-memory \
    --poly-degree=3 --omega-window=16 --chunk-size=64 \
    --ns-steps=3 --pe-ste --fused-chunk \
    --seq-len=1024 \
    --batch-size=512 \
    --optimizer=muon \
    --lr=0.02 --embedding-lr=0.004 --lm-head-lr=0.004 --scalar-lr=0.05 \
    --weight-decay=0.0 \
    --muon-momentum=0.95 --muon-ns-steps=5 \
    --warmup-steps=40 --warmdown-steps=0 \
    --total-steps=5000 \
    --max-tokens=5e8 \
    --eval-every=100 --eval-steps=20 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer \
    --seed=42

echo "==> Done at $(date)"
