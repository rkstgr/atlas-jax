#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=atlas_59m_16gpu
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# ==============================================================================
# Atlas-JAX 59M — 16×H100 distributed (jax.distributed + shard_map)
# Same model config as 1-GPU 500M run, 16× data parallelism
# Global batch: 1024 (64/GPU), 500M tokens
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

echo "==> Starting Atlas 59M distributed training (16×H100, 500M tokens)..."
srun python -m atlas_jax.train_distributed \
    --n-layer=8 --n-head=8 --n-embd=512 \
    --memory-expand=1 --deep-memory \
    --poly-degree=3 --omega-window=16 --chunk-size=64 \
    --ns-steps=3 --pe-ste --fused-chunk \
    --seq-len=1024 \
    --batch-size=1024 --lr=3e-3 --weight-decay=0.1 \
    --warmup-steps=100 --total-steps=5000 \
    --max-tokens=5e8 \
    --eval-every=100 --eval-steps=20 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer \
    --seed=42

echo "==> Done at $(date)"
