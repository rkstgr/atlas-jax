#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --job-name=atlas_paper
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# ==============================================================================
# Atlas-JAX 170M — Paper-faithful config (arXiv 2505.23735)
#
# Architecture: 12 layers, 768 dim, 12 heads (D=64), expand=1, deep memory
#               poly_degree=2, ns_steps=5, omega_window=16, chunk_size=64
#               dropout=0.1, gate_bias=-2.0, layer norm at chunk boundaries
#               fused Triton kernel (D==E=64, 12.8x faster)
#
# Training:    AdamW lr=1e-4, cosine decay, warmup=4000, wd=0.01, grad_clip=1.0
#              seq_len=2048, batch=64 (16/GPU), ~500M tokens initial test
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

echo "==> Starting Atlas 170M paper-faithful training (4×H100, fused kernel)..."
srun python scripts/train_distributed.py \
    --n-layer=12 --n-head=12 --n-embd=768 \
    --memory-expand=1 --deep-memory \
    --poly-degree=2 --omega-window=16 --chunk-size=64 \
    --ns-steps=5 --pe-ste --fused-chunk \
    --dropout=0.1 --gate-bias-init=-2.0 \
    --seq-len=2048 \
    --batch-size=64 \
    --optimizer=adamw \
    --lr=1e-4 --weight-decay=0.01 \
    --warmup-steps=4000 --warmdown-steps=0 \
    --total-steps=50000 \
    --max-tokens=5e9 \
    --eval-every=500 --eval-steps=20 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer \
    --ckpt-dir=/p/scratch/westai0047/nanochat/checkpoints/atlas_170m_paper \
    --ckpt-every=1000 \
    --seed=42

echo "==> Done at $(date)"
