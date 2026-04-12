#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=jax_enwik8
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# ==============================================================================
# Atlas-JAX enwik8 comparison — 10M config matching atlas_pytorch
# ==============================================================================

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "==> Start: $(date)"

echo "=== Full alignment: ResidualNorm + persist_mem=4 + retrieve_gate, stop_grad=True ==="
python -m atlas_jax.train_enwik8 \
    --dim=320 --depth=8 --heads=4 --dim-head=64 \
    --memory-expand=2 --poly-degree=2 --omega-window=2 \
    --chunk-size=64 --ns-steps=5 \
    --batch-size=4 --grad-accum=4 --seq-len=512 \
    --lr=2e-4 --weight-decay=0.01 --grad-clip=0.5 \
    --num-batches=5000 --validate-every=100 --eval-steps=10 \
    --stop-grad-chunks

echo "==> Done: $(date)"
