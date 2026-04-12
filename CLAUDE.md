# Atlas-JAX

Atlas architecture (arXiv 2505.23735) ported to JAX/Equinox. Sibling project to nanochat's PyTorch Atlas.

## Project Layout

```
atlas_jax/
├── config.py          # AtlasConfig dataclass
├── polar_express.py   # PE orthogonalization + STE
├── state.py           # MemoryState NamedTuples + init helpers
├── model.py           # ShortConv, AtlasMemoryLayer, MLP, Block, Atlas
├── optim.py           # Muon optax transform (not yet used in training)
├── data.py            # Parquet loading, BOS-aligned best-fit packing
├── tokenizer.py       # Loads nanochat's tiktoken tokenizer.pkl directly
└── train.py           # CLI training loop with MFU/BPB reporting
tests/                 # 38 tests, all passing
runs/                  # SLURM scripts for JUWELS H100
```

## Environment

- **venv**: `.venv/` managed by `uv` — `uv venv .venv && uv pip install -e .`
- **Key deps**: jax 0.9.2 + cuda12, equinox 0.13.6, optax 0.2.8, tiktoken 0.12.0
- **Data**: `/p/scratch/westai0047/nanochat/base_data_climbmix/` (2001 parquet shards)
- **Tokenizer**: `/p/scratch/westai0047/nanochat/tokenizer/tokenizer.pkl`
- **Logs**: `/p/scratch/westai0047/nanochat/logs/`

## Critical Design Decisions (don't change without understanding why)

### 1. `stop_gradient` on chunk scan carry (model.py)

The `lax.scan` over chunks uses `jax.lax.stop_gradient(carry)` on the memory state.
This is **required** — without it, gradients compound across chunks and explode to NaN
at 6+ chunks (T >= 384 with chunk_size=64). This matches the paper: "Gradients are
computed w.r.t. the last state of the previous chunk" — each chunk treats the incoming
memory state as a frozen constant.

### 2. PE Frobenius norm epsilon (polar_express.py)

Uses `jnp.sqrt(sum + 1e-12)` not `jnp.sqrt(sum) + eps`. The latter produces `inf`
gradient when input is zero (which happens for the initial zero momentum state).

### 3. Pre-chunked scan xs (model.py)

Arrays (q, k, v, gates) are pre-chunked into `(n_chunks, B, cs, ...)` and passed as
`xs` to `lax.scan`, rather than closed over and sliced with `dynamic_slice_in_dim`.
Closing over differentiable arrays + dynamic slicing inside a checkpointed scan body
can produce incorrect gradients in some JAX/XLA configurations.

### 4. Direct matmul instead of vmap(Linear)

Uses `x @ weight.T` instead of `jax.vmap(eqx.nn.Linear)(x)` for projections. Avoids
vmap tracing overhead and is equivalent for bias-free linear layers.

### 5. Weight initialization (_init_block_weights in model.py)

Per paper (Section 9):
- `c_proj` output projections: GPT-2 style scaled init (std = 0.02 / sqrt(2*n_layer))
- Gate projections: **zero weights + bias at -2.0** (sigmoid(-2) ≈ 0.12 initially)
  This is critical — without the -2.0 bias, gates start at 0.5 and gradients explode.
- Q/K/V projections: Xavier uniform (std = 1/sqrt(fan_in))
- `lm_head` initialized small (std=0.02)

### 6. Dropout (model.py)

Paper specifies dropout=0.1 after memory layer output and MLP output. Applied during
training only (pass `dropout_key` to `model(idx, dropout_key=key)`; omit for eval).

### 7. f32 matmul precision (train.py)

`jax.config.update("jax_default_matmul_precision", "float32")` is set in train.py.
TF32 (H100 default) caused NaN in early experiments. Can be relaxed to "high" or
removed once bf16 training is properly implemented.

## Differences from Paper

| Aspect | Paper | Our Implementation | Reason |
|--------|-------|--------------------|--------|
| memory_expand | 4 | 1 (configurable) | Fused Triton kernel requires D==E; expand=1 is 12.8× faster |
| ns_steps | 5 | 5 (configurable) | Matches paper now |
| poly_degree | 2 | 2 (configurable) | Matches paper now |
| PE backward | Full | STE (stop_gradient) | 62× faster backward; paper validates this |
| Outer optimizer | AdamW | AdamW (optax) with grad clip 1.0 | Matches paper; Muon also available |
| Chunk gradient | Frozen boundary | stop_gradient on carry | Matches paper exactly |
| Gate init | bias=-2 (sigmoid≈0.12) | bias=-2.0 | Matches paper |
| Dropout | 0.1 | 0.1 (configurable) | Matches paper |
| Gradient clipping | max norm 1.0 | clip_by_global_norm(1.0) | Matches paper |
| Layer norm at chunk boundary | Paper mentions it | Not implemented | RMS norm on memory state was destructive; removed |

## First Training Results (2026-04-01)

**Config**: 48.8M params, L=8, D=448, H=8, deep memory (expand=1), poly_degree=3,
omega_window=16, ns_steps=3, pe_ste=True

**Run**: RTX 8000 (Quadro, 46GB, f32 only), B=4, T=1024, lr=3e-3, AdamW, 2000 steps

| Metric | Value |
|--------|-------|
| Final val_bpb | 4.5853 |
| Final val_loss | 10.4884 |
| MFU | 2.28% |
| Step time | 3.36s |
| Throughput | 1,218 tok/s |
| Total tokens | 8.2M |

Loss was still decreasing at step 2000. Model needs orders of magnitude more tokens
to converge (paper trains 15B-100B tokens).

## Performance Profile (RTX 8000, f32)

Per-layer fwd+bwd at B=2, T=2048:

| memory_expand | Time/layer | Relative |
|---------------|-----------|----------|
| 1 (D=56, E=56) | 460ms | 1.0× |
| 2 (D=56, E=112) | 737ms | 1.6× |
| 4 (D=56, E=224) | 1454ms | 3.2× |

Bottleneck is PE + einsum operations within each chunk, NOT the scan iteration count.
`linear_scan` (sequential or associative) is <1% of total time.

## Known Issues / TODOs

1. **Muon optimizer not wired up** — `optim.py` has the Muon transform but `train.py`
   uses plain AdamW. Need `optax.multi_transform` with param grouping (2D matrices →
   Muon, embeddings/scalars → AdamW).

2. **No bf16 training** — PE internally uses f32 (good), but everything else should run
   in bf16 for ~2× throughput on H100 tensor cores. Need to cast model to bf16 and
   keep master weights in f32.

3. **Low MFU** — 2.28% (RTX 8000) / ~7% (H100 est). The sequential chunk scan is
   inherently serial. Potential improvements:
   - Pallas kernels for the inner scan + PE (like nanochat's Triton kernels)
   - Larger chunk_size to reduce overhead
   - Scan-over-layers for the 8 transformer blocks

4. **H100 validation pending** — JUWELS dc-hwai was in maintenance on 2026-04-01.
   The code should work on H100 now (stop_gradient fix + f32 precision).

5. **Layer norm at chunk boundaries** — Paper specifies this but it's not implemented.

6. **Gradient accumulation** — Not implemented. Would allow larger effective batch size
   on memory-constrained GPUs.

7. **Multi-GPU (data parallel)** — Phase 2 per the plan. Use `jax.sharding.Mesh` with
   batch dim sharded, model replicated. XLA handles all-reduce automatically.

## Running

```bash
# Tests
cd /p/project1/westai0047/atlas-jax
source .venv/bin/activate
python -m pytest tests/ -v

# Training (local GPU)
CUDA_VISIBLE_DEVICES=0 TIKTOKEN_CACHE_DIR=/p/project1/westai0047/tiktoken_cache \
PYTHONUNBUFFERED=1 python -m atlas_jax.train \
    --n-layer=8 --n-head=8 --n-embd=448 --memory-expand=1 --deep-memory \
    --poly-degree=3 --omega-window=16 --chunk-size=64 --ns-steps=3 --pe-ste \
    --seq-len=1024 --batch-size=4 --lr=3e-3 --weight-decay=0.1 \
    --warmup-steps=200 --total-steps=2000 --eval-every=200 --eval-steps=20 \
    --gpu-peak-tflops=32.6 \
    --data-dir=/p/scratch/westai0047/nanochat/base_data_climbmix \
    --tokenizer-dir=/p/scratch/westai0047/nanochat/tokenizer

# SLURM (H100)
sbatch runs/train_50m.slurm
```

## Reference

- Paper: arXiv 2505.23735 (Atlas)
- Implementation details: `/p/project1/westai0047/nanochat/atlas-implementation-details.md`
- PyTorch reference: `/p/project1/westai0047/nanochat/nanochat/atlas.py` (use as structural guide, not ground truth)
- PE coefficients: arXiv 2505.16932 (Polar Express)
