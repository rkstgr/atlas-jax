# Atlas Cross-Implementation Benchmark — Handover

## What this is

A controlled benchmark comparing three Atlas implementations on enwik8 (character-level LM):

| Repo | Framework | Path |
|------|-----------|------|
| atlas-pytorch | PyTorch (lucidrains) | `../atlas-pytorch` |
| atlas-rnn | PyTorch (closed-form RNN) | `../atlas-rnn` |
| atlas-jax | JAX/Equinox | `./` (this repo) |

All three are on branch `benchmark/cross-impl-comparison`.

## How to run

```bash
# Smoke test (15 steps, ~2 min on H100)
source .venv/bin/activate
srun --account=westai0047 python benchmark_atlas.py --results-dir /tmp/smoke --num-batches 15

# Full benchmark (2000 steps, ~1h on H100)
srun --account=westai0047 python benchmark_atlas.py --results-dir results/full

# Compare results
python compare_results.py --results-dir results/full
```

Or via SLURM: `sbatch benchmark_all.slurm`

## Current state (2026-04-16)

All 4 configs pass the smoke test:

| Run | Architecture | Params | BPB@15 | tok/s | GPU Mem |
|-----|-------------|--------|--------|-------|---------|
| pytorch-base | AtlasLMM (vmap(grad), no accel scan) | 3.6M | 5.61 | 6,230 | 61 GB |
| rnn-base | RNN LMM (closed-form updates) | 3.3M | 6.28 | 24,271 | 14 GB |
| jax-base | Atlas LMM (lax.scan, chunk=1) | 3.6M | 6.86 | 3,155 | 11 GB |
| jax-fused | Atlas LMM (fused Triton, chunk=1) | 3.4M | 6.86 | 5,468 | 6 GB |

`pytorch-fast` (with `accelerated_scan` warpscan kernel) is disabled — crashes with illegal memory access at batch>=8. Upstream bug in the `accelerated_scan` package.

## Aligned config

All runs use identical hyperparameters:
- dim=256, depth=4, heads=4, dim_head=64
- omega_window=2, poly_degree=2, poly_mode=elementwise
- batch_size=8, seq_len=512, grad_accum=1
- lr=2e-4, weight_decay=0.01, grad_clip=0.5, AdamW betas=(0.9, 0.99)
- persist_mem_tokens=0, chunk_size=1, no causal conv
- f32 precision (TF32 disabled in PyTorch, jax_default_matmul_precision=float32)

## Key files

| File | Purpose |
|------|---------|
| `benchmark_atlas.py` | Unified runner — launches each config as subprocess with correct venv |
| `compare_results.py` | Reads JSONL results, prints table, generates matplotlib plot |
| `benchmark_all.slurm` | SLURM wrapper for full benchmark on H100 |
| `scripts/train_enwik8.py` | JAX training script (patched with JSONL + new flags) |
| `../atlas-pytorch/train_atlas.py` | PyTorch AtlasLMM (patched with JSONL + --persist-mem) |
| `../atlas-rnn/train_rnn_transformer.py` | RNN transformer (rewritten: step-based + JSONL) |

Each training script writes one JSONL line per step to `--metrics-file`:
```json
{"type": "config", "impl": "jax", "n_params": 3574856, ...}
{"type": "step", "step": 11, "loss": 4.75, "bpb": 6.85, "tok_s": 9328, "peak_mem_mb": 2872, ...}
{"type": "eval", "step": 100, "val_loss": 4.12, "val_bpb": 5.94}
{"type": "summary", "total_time_s": 120.5, "total_tokens": 8192000}
```

## Known issue: PyTorch converges faster

After alignment (same config, chunk_size=1, no conv, no persist tokens), PyTorch still converges ~1.2 BPB faster at step 15. The root cause is **where ResidualNorm is applied**:

- **PyTorch**: ResidualNorm wraps the MemoryMLP *inside* the per-sample gradient computation. The MLP forward is `norm(MLP(x)) + x`, and gradients are computed w.r.t. this normalized output. This stabilizes the gradient signal.
- **JAX**: ResidualNorm is applied *after* the chunk scan, on the final retrieved output. The internal MLP forward (`W2 @ gelu(W1 @ k)`) is not normalized. We added `--residual-norm` but it only wraps the output, not the internal prediction.

To close this gap, the JAX `_process_chunk_deep` function in `memory_layer.py` (line ~570) would need to apply LayerNorm inside the forward prediction:
```python
# Current:  y_pred = k_c + W1 @ gelu(W2 @ k_c)
# Needed:   y_pred = layer_norm(k_c + W1 @ gelu(W2 @ k_c)) * (gamma+1) + k_c
```
And the same normalization in the gradient computation and in the retrieval path. This is non-trivial because the analytical gradients (lines ~580-590) would need to account for the LayerNorm Jacobian.

## Other performance notes

- chunk_size=1 makes JAX **3x slower** than chunk_size=64 because XLA serializes 512 loop iterations instead of 8. For production use, chunk_size=64 is correct (9K→24K tok/s with fused kernel).
- PyTorch uses 61GB GPU memory vs JAX's 6-11GB for the same model — PyTorch's `vmap(grad)` materializes per-sample gradient tensors.
- RNN is fastest (24K tok/s) because closed-form updates avoid both vmap and scan overhead.
- Each repo has its own venv. The benchmark runner uses absolute Python paths (`atlas-pytorch/.venv/bin/python`, etc.) so no venv activation is needed during the run.

## What's left to do

1. **Full 2000-step benchmark** — run `sbatch benchmark_all.slurm` and check convergence curves
2. **Close the ResidualNorm gap** — implement internal ResidualNorm in JAX's `_process_chunk_deep` (hard: need analytical gradient of LayerNorm)
3. **Re-enable pytorch-fast** — debug the `accelerated_scan` crash or wait for upstream fix
4. **Add RNN CUDA kernel** — `OmegaRNNMemoryCell` supports `use_cuda=True` but `RNNMemoryTransformer` doesn't plumb it through. Would need to add `use_cuda` kwarg to block constructors.
5. **chunk_size sensitivity** — run JAX at chunk_size=1,8,64 to measure the convergence vs throughput tradeoff
