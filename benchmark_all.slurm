#!/usr/bin/env bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=atlas_benchmark
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/%x_%j.out
#SBATCH --error=/p/scratch/westai0047/nanochat/logs/%x_%j.err

# ==============================================================================
# Atlas Cross-Implementation Benchmark
#
# Runs 5 configurations sequentially on 1×H100 (single GPU):
#   1. pytorch-base  — MAG + OmegaNet, no accelerated scan
#   2. pytorch-fast  — MAG + OmegaNet, with Triton accelerated scan
#   3. rnn-base      — RNN Memory Transformer (closed-form updates)
#   4. jax-base      — JAX/Equinox Atlas, non-fused
#   5. jax-fused     — JAX/Equinox Atlas, fused Triton kernel (expand=1)
#
# Common config: dim=256, depth=4, heads=4, dim_head=64, omega=2, poly=2
# 2000 steps, batch=4, seq=512, grad_accum=4, f32 precision
# ==============================================================================

set -euo pipefail
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TIKTOKEN_CACHE_DIR=/p/project1/westai0047/tiktoken_cache

BASE=/p/project1/westai0047
RESULTS_DIR=${BASE}/benchmark_results/$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "Atlas Benchmark — $(date)"
echo "Results: ${RESULTS_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

# Activate venvs as needed per implementation
# PyTorch runs use atlas-pytorch/.venv, JAX runs use atlas-jax/.venv

# --- PyTorch runs ---
echo ""
echo "=== PyTorch venv ==="
source ${BASE}/atlas-pytorch/.venv/bin/activate 2>/dev/null || \
    source ${BASE}/atlas-jax/.venv/bin/activate  # fallback: shared venv

cd ${BASE}/atlas-jax
python benchmark_atlas.py \
    --results-dir "${RESULTS_DIR}" \
    --runs pytorch-base,pytorch-fast,rnn-base

deactivate 2>/dev/null || true

# --- JAX runs ---
echo ""
echo "=== JAX venv ==="
source ${BASE}/atlas-jax/.venv/bin/activate

cd ${BASE}/atlas-jax
python benchmark_atlas.py \
    --results-dir "${RESULTS_DIR}" \
    --runs jax-base,jax-fused

# --- Compare ---
echo ""
echo "=== Comparison ==="
python compare_results.py \
    --results-dir "${RESULTS_DIR}" \
    --plot "${RESULTS_DIR}/comparison.png"

deactivate 2>/dev/null || true

echo ""
echo "Benchmark complete: ${RESULTS_DIR}"
echo "Plot: ${RESULTS_DIR}/comparison.png"
