#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --job-name=atlas_4gpu_test
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Quick test: 4×H100 data-parallel via SPMD sharding (no pmap, no vmap)

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Node: $SLURM_JOB_NODELIST | Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import jax; print(f'JAX {jax.__version__} | {jax.devices()}')"

echo "==> Starting 4-GPU SPMD test..."
python -m atlas_jax.train \
    --n-layer=8 --n-head=8 --n-embd=512 \
    --memory-expand=1 --deep-memory \
    --poly-degree=3 --omega-window=16 --chunk-size=64 \
    --ns-steps=3 --pe-ste --fused-chunk \
    --seq-len=1024 \
    --batch-size=256 --lr=3e-3 --weight-decay=0.1 \
    --warmup-steps=10 --total-steps=100 \
    --eval-every=25 --eval-steps=10 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer \
    --seed=42

echo "==> Done at $(date)"
