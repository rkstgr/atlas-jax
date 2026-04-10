#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --job-name=atlas_distributed
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"
export MASTER_ADDR=$(hostname)

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate
echo "==> Testing jax.distributed on $(hostname), $SLURM_NTASKS tasks"
srun python tests/test_distributed.py
echo "==> Done at $(date)"
