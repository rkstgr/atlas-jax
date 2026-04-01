#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=atlas_jax_50m
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# ==============================================================================
# Atlas-JAX 50M param validation run on 1×H100
# ==============================================================================

set -euo pipefail

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Tiktoken needs a cache dir
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

PROJ_DIR="/p/project1/westai0047/atlas-jax"
DATA_DIR="/p/scratch/westai0047/nanochat/base_data_climbmix"
TOK_DIR="/p/scratch/westai0047/nanochat/tokenizer"

echo "==> Node: $SLURM_JOB_NODELIST | Start: $(date)"
echo "==> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd "$PROJ_DIR"
source .venv/bin/activate

echo "==> JAX version check:"
python -c "import jax; print(f'JAX {jax.__version__} | {jax.devices()}')"

echo "==> Starting Atlas-JAX 50M training..."
srun python -m atlas_jax.train \
    --n-layer=8 \
    --n-head=8 \
    --n-embd=448 \
    --memory-expand=4 \
    --deep-memory \
    --poly-degree=3 \
    --omega-window=16 \
    --chunk-size=64 \
    --ns-steps=5 \
    --pe-ste \
    --seq-len=2048 \
    --batch-size=8 \
    --lr=3e-3 \
    --weight-decay=0.1 \
    --warmup-steps=200 \
    --total-steps=2000 \
    --eval-every=200 \
    --eval-steps=20 \
    --gpu-peak-tflops=989.4 \
    --data-dir="$DATA_DIR" \
    --tokenizer-dir="$TOK_DIR" \
    --seed=42

echo "==> Done at $(date)"
