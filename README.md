# Atlas-JAX

Minimal reference implementation of **Atlas** ([arXiv 2505.23735](https://arxiv.org/abs/2505.23735))
in JAX/Equinox. Correctness-first, single-file model, no fused kernels.

See `CLAUDE.md` for architecture notes, critical design decisions, and SLURM
templates. See `docs/papers/` for the paper.

## Install

```bash
uv sync
uv pip install -e ".[cuda]"      # JAX CUDA wheels
```

## Run

```bash
# enwik8 byte-level (vocab=256)
python -m atlas_jax.train \
    --dataset enwik8 --data-path $SCRATCH/atlas-jax/enwik8 \
    --n-layer 4 --n-head 8 --n-embd 256 --seq-len 512 \
    --batch-size 8 --total-steps 1000 --eval-every 100
```

## Comparison vs `atlas-pytorch` (lucidrains impl)

Same config: `dim=256, depth=4, heads=8, head_dim=32, seq_len=512, batch=8`,
LR warmup 10 → peak 1e-3 → cosine to 1e-5, 100 steps, fp32, 1× Quadro RTX 8000.

| metric | **JAX** | **PyTorch** |
|---|---|---|
| params | 3.4M | 3.5M |
| step time | **450 ms** (9.1k tok/s) | 825 ms (5.0k tok/s) |
| peak VRAM | **7.95 GB** | 28.9 GB |
| train bpb @ 100 | **3.23** | 3.39 |
| val bpb @ 100 | **3.31** | 3.45 |

JAX is ~1.8× faster per step and ~3.6× less VRAM at matched config. Learning
curves overlap closely; residual val-bpb gap (~4%) is likely from architectural
detail differences (paper-minimal vs lucidrains) rather than a bug on either
side.
