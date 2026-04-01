#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=debug_nan
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd /p/project1/westai0047/atlas-jax
source .venv/bin/activate
srun python debug_nan.py
