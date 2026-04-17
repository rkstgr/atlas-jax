# Atlas-JAX (minimal)

Minimal reference implementation of Atlas (arXiv 2505.23735) in JAX/Equinox.
Correctness-first; no fused kernels, no multi-GPU, no Muon. Optimizations are
intentionally deferred until the reference is known-good on both enwik8 and
climbmix.

## Project layout

```
atlas_jax/
├── __init__.py     # exports Atlas, AtlasConfig
├── model.py        # all model code in one file (~420 lines)
├── data.py         # enwik8 byte-level loader + climbmix parquet loader
├── tokenizer.py    # thin wrapper around nanochat's tiktoken pickle
└── train.py        # single-GPU training loop
docs/               # paper PDF + LaTeX source (ground truth)
pyproject.toml
```

## Environment

- `.venv/` managed by `uv` — `uv venv .venv && uv pip install -e .`
- Key deps: jax 0.9.2 + cuda12, equinox, optax, pyarrow, tiktoken
- Data:
  - enwik8: `/p/scratch/westai0047/nanochat/enwik8/enwik8` (100MB)
  - climbmix: `/p/scratch/westai0047/nanochat/base_data_climbmix/` (2001 parquet shards)
- Tokenizer: `/p/scratch/westai0047/nanochat/tokenizer/tokenizer.pkl`

## Running

```bash
source .venv/bin/activate

# Smoke test on enwik8 (byte-level, vocab=256, no tokenizer needed):
python -m atlas_jax.train \
    --dataset enwik8 --data-path ./enwik8 \
    --n-layer 4 --n-head 8 --n-embd 256 --seq-len 512 \
    --batch-size 8 --total-steps 1000 --eval-every 100

# Full run on climbmix (tiktoken, vocab=32768):
python -m atlas_jax.train \
    --dataset climbmix \
    --data-path /p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir /p/scratch/westai0047/nanochat/tokenizer \
    --n-layer 8 --n-head 8 --n-embd 448 --seq-len 1024 \
    --batch-size 8 --total-steps 2000 --eval-every 200
```

## Architecture summary

The model is a stack of Atlas blocks. Each block replaces attention with an
Atlas memory layer and keeps a standard GEGLU MLP:

```
x -> x + memory(rms_norm(x))   # Omega rule + momentum + PolarExpress + weight-decay
  -> x + mlp(rms_norm(x))      # GEGLU feedforward
```

Inside the memory layer, for each chunk of `chunk_size` tokens:

1. Forward with frozen `W`, compute per-token MSE error.
2. Analytical gradient `u_t` w.r.t. `W` (no autodiff inside the scan).
3. Omega aggregation: sliding-window sum of gradients, weighted by gate `gamma`.
4. Momentum scan: `S_t = theta_t * S_{t-1} - eta_t * G_t` (associative scan).
5. Polar Express orthogonalization of `S` (Newton-Schulz + STE).
6. Weight decay scan: `W_t = alpha_t * W_{t-1} + S'_t` (associative scan).
7. Retrieve with per-timestep weights: `y_t = M_t(q_t)`.

Between chunks the memory state is passed via `stop_gradient` so gradients
don't compound across chunk boundaries (paper: frozen boundary).

## Critical design decisions

These are non-obvious pitfalls from prior iterations. Changing them without
understanding why will break training.

### 1. `stop_gradient` on chunk scan carry (model.py)

Required — without it, gradients compound across chunks and explode to NaN at
6+ chunks. Paper: "Gradients are computed w.r.t. the last state of the previous
chunk." Each chunk treats the incoming memory state as a frozen constant.

### 2. PE Frobenius-norm epsilon

Use `jnp.sqrt(sum + 1e-12)`, not `jnp.sqrt(sum) + eps`. The latter produces
`inf` gradient when input is zero, which happens for the initial zero momentum
state.

### 3. Pre-chunked scan `xs`

Arrays (q, k, v, gates) are pre-chunked into `(n_chunks, B, cs, ...)` and
passed as `xs` to `lax.scan`, rather than closed over and sliced with
`dynamic_slice_in_dim`. Closing over differentiable arrays and dynamic-slicing
inside a checkpointed scan body produced wrong gradients in earlier versions.

### 4. Gate bias init at -2.0

`sigmoid(-2) ≈ 0.12` at init. Without this, gates start at 0.5 and gradients
explode — this is a reliable way to make training diverge. See `_init_weights`
in model.py. Paper Section 9 specifies this.

### 5. f32 matmul precision

`--matmul-precision float32` is the default. TF32 (the H100 default) caused
NaN in early experiments on this architecture. Can be relaxed to `high` once
the reference is validated.

### 6. `memory_expand=1` default

Paper uses 4. In pure JAX (no fused kernel), `expand>1` is ~3× slower per
layer because the MLP hidden dim grows. The reference impl defaults to 1; the
`--memory-expand` flag lets us try higher values on climbmix.

## Writing SLURM jobs

Target cluster: JUWELS / `dc-hwai` (partition `dc-hwai`, 4×H100 per node).
Template for a single-node single-GPU run:

```bash
#!/bin/bash
#SBATCH --account=westai0047
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/p/scratch/westai0047/nanochat/logs/atlas_%j.out

set -eu
cd /p/project1/westai0047/atlas-jax
source .venv/bin/activate

export TIKTOKEN_CACHE_DIR=/p/project1/westai0047/tiktoken_cache
export PYTHONUNBUFFERED=1
# Enable compilation cache to avoid recompiling across runs.
export JAX_COMPILATION_CACHE_DIR=/p/scratch/westai0047/nanochat/jax_cache

srun python -m atlas_jax.train \
    --dataset climbmix \
    --data-path /p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir /p/scratch/westai0047/nanochat/tokenizer \
    --n-layer 8 --n-head 8 --n-embd 448 --seq-len 1024 \
    --batch-size 8 --total-steps 5000 --eval-every 200 \
    --out-dir /p/scratch/westai0047/nanochat/out/atlas-jax-minimal
```

Rules of thumb:
- Always use `srun` in front of `python` (SLURM handles GPU pinning).
- `--gres=gpu:1` with `--nodes=1` for single-GPU runs.
- Logs go under `/p/scratch/...` (shared scratch); `$SLURM_JOB_ID` in the
  output path.
- Activate the venv inside the job, not on the login node.
- Export the JAX compilation cache dir — saves ~60s on recompiles.
- `#SBATCH --time` is hard-capped at 24h on `dc-hwai`; use checkpoint+resume
  for longer runs (train.py auto-resumes from `--out-dir`).

## Differences from the paper

| Aspect | Paper | Here | Reason |
|---|---|---|---|
| `memory_expand` | 4 | 1 (default, configurable) | Pure-JAX expand>1 is slow; validate correctness first |
| PE backward | Full Newton-Schulz grad | STE (stop_gradient) | Paper validates this; 60× faster backward |
| Outer optimizer | AdamW | AdamW (optax) | Same |
| Chunk boundary | Frozen | `stop_gradient` on carry | Same |
| Gate init | bias=-2 | bias=-2.0 | Same |
| Dropout | 0.1 | 0.1 (hardcoded) | Same |
| Grad clip | max-norm 1.0 | `clip_by_global_norm(1.0)` | Same |
| Layer norm at chunk boundary | Mentioned | Not implemented | Prior RMS-norm attempt was destructive |
| ResidualNorm inside memory MLP | Optional | Not implemented | Keep reference simple |

## Reference

- Paper: `docs/papers/atlas_2505.23735.pdf`, LaTeX source: `docs/atlas-tex-source/`
- PE coefficients: arXiv 2505.16932 (Polar Express Sign Method)
