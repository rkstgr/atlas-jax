#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --job-name=atlas_bench
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Full model training benchmark: paper-faithful, all optimizations
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

# Config: 8L, D=96 (H=4, embd=384), deep memory, expand=1, pe_ste, ns=3
# Best perf settings: bf16, TF32, B=32, checkpoint
echo "=== Benchmark: Paper-faithful Atlas with all optimizations ==="
srun python -m atlas_jax.train \
    --n-layer=8 --n-head=4 --n-embd=384 \
    --memory-expand=1 --deep-memory \
    --poly-degree=3 --omega-window=16 --chunk-size=64 \
    --ns-steps=3 --pe-ste --seq-len=2048 \
    --batch-size=32 --lr=3e-3 --weight-decay=0.1 \
    --warmup-steps=100 --total-steps=10000 --time-budget=420 \
    --eval-every=200 --eval-steps=10 \
    --gpu-peak-tflops=989.4 --matmul-precision=high \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer --seed=42

echo "==> Done at $(date)"
