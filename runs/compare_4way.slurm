#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=cmp4way
#SBATCH --array=0-3
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%A_%a.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%A_%a.err

# 4-way MAG comparison: PyTorch vs JAX (f32/fused/bf16)
# All use: dim=256, depth=8, heads=4, dim_head=64, window_size=64
# memory on layers (2,4,6,8) [PT 1-indexed] = (1,3,5,7) [JAX 0-indexed]
# enwik8 byte-level, batch=4, grad_accum=4, seq=512, 1000 steps
# AdamW lr=2e-4, wd=0.01, grad_clip=0.5

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

TASK=$SLURM_ARRAY_TASK_ID
echo "==> Task $TASK | GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

case $TASK in
0)
    echo "=== PyTorch MAG (f32, no-accelerated-scan) ==="
    cd /p/project1/westai0047/atlas-pytorch && source .venv/bin/activate
    export WANDB_MODE=offline
    python train_mag.py \
        --no-accelerated-scan \
        --omega-window 2 \
        --poly-mode elementwise \
        --poly-degree 2 \
        --dim 256 \
        --depth 8 \
        --heads 4 \
        --dim-head 64 \
        --batch-size 4 \
        --grad-accum 4 \
        --seq-len 512 \
        --num-batches 1000 \
        --validate-every 100 \
        --generate-every 99999 \
        --data-path ./data/enwik8.gz
    ;;
1)
    echo "=== JAX MAG (f32, nonfused) ==="
    cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate
    python scripts/train_enwik8.py \
        --model=mag \
        --dim=256 --depth=8 --heads=4 --dim-head=64 \
        --memory-expand=2 --poly-degree=2 --omega-window=2 \
        --chunk-size=64 --ns-steps=5 --window-size=64 \
        --memory-layers=1,3,5,7 \
        --batch-size=4 --grad-accum=4 --seq-len=512 \
        --lr=2e-4 --weight-decay=0.01 --grad-clip=0.5 \
        --num-batches=1000 --validate-every=100 --eval-steps=10 \
        --stop-grad-chunks
    ;;
2)
    echo "=== JAX MAG (f32, fused) ==="
    cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate
    python scripts/train_enwik8.py \
        --model=mag \
        --dim=256 --depth=8 --heads=4 --dim-head=64 \
        --memory-expand=2 --poly-degree=2 --omega-window=2 \
        --chunk-size=64 --ns-steps=5 --window-size=64 \
        --memory-layers=1,3,5,7 --fused-chunk \
        --batch-size=4 --grad-accum=4 --seq-len=512 \
        --lr=2e-4 --weight-decay=0.01 --grad-clip=0.5 \
        --num-batches=1000 --validate-every=100 --eval-steps=10 \
        --stop-grad-chunks
    ;;
3)
    echo "=== JAX MAG (bf16, fused) ==="
    cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate
    python scripts/train_enwik8.py \
        --model=mag \
        --dim=256 --depth=8 --heads=4 --dim-head=64 \
        --memory-expand=2 --poly-degree=2 --omega-window=2 \
        --chunk-size=64 --ns-steps=5 --window-size=64 \
        --memory-layers=1,3,5,7 --fused-chunk --bf16 \
        --batch-size=4 --grad-accum=4 --seq-len=512 \
        --lr=2e-4 --weight-decay=0.01 --grad-clip=0.5 \
        --num-batches=1000 --validate-every=100 --eval-steps=10 \
        --stop-grad-chunks
    ;;
esac

echo "==> Done: $(date)"
