#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --job-name=atlas_trace
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Profile 10 train steps with paper-exact config (expand=4, ns=5, full PE backward)
set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

TRACE_DIR="/p/scratch/westai0047/nanochat/traces/atlas_paper_${SLURM_JOB_ID}"

srun python scripts/profile_train.py \
    --trace-dir="$TRACE_DIR" \
    --batch-size=4 --seq-len=1024 \
    --warmup-steps=3 --trace-steps=10 \
    --matmul-precision=high

echo "==> Trace saved to $TRACE_DIR"
echo "==> Done at $(date)"
