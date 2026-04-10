#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=atlas_multigpu
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# Test 3 multi-GPU approaches with Triton kernels
# Each task gets 1 GPU (SLURM_LOCALID=0..3)

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR="/p/project1/westai0047/tiktoken_cache"

cd /p/project1/westai0047/atlas-jax && source .venv/bin/activate

echo "==> Node: $SLURM_JOB_NODELIST | Tasks: $SLURM_NTASKS | Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Option C: shared memory (no extra deps)
echo ""
echo "========== Option C: shared memory =========="
srun python tests/test_multi_gpu.py --method=shm

# Option A: jax.distributed
echo ""
echo "========== Option A: jax.distributed =========="
export MASTER_ADDR=$(hostname)
srun python tests/test_multi_gpu.py --method=distributed

# Option B: mpi4jax (will likely skip if not installed)
echo ""
echo "========== Option B: mpi4jax =========="
srun python tests/test_multi_gpu.py --method=mpi

echo "==> Done at $(date)"
