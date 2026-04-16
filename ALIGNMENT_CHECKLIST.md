# Atlas JAX vs PyTorch — Alignment Checklist

Comprehensive list of every known difference between `atlas-jax` and `atlas-pytorch`
that could affect convergence. Organized by severity.

Status legend: `[x]` fixed, `[ ]` open, `[~]` partial/uncertain

---

## A. Confirmed bugs (already fixed this session)

- [x] **LR schedule**: JAX had warmup+cosine decay, PyTorch uses flat constant LR.
      Fixed in `scripts/train_enwik8.py` — now flat LR.
- [x] **AdamW β2**: JAX used optax default 0.999, PyTorch uses 0.99.
      Fixed — now `b1=0.9, b2=0.99`.

---

## B. Algorithmic / structural (high impact)

### B1. Omega aggregation is dead at chunk_size=1
- **PyTorch**: computes ALL per-token gradients first (vmap), THEN runs omega
  sliding window over the full sequence (512 positions). Omega even carries a
  buffer across sequences.
- **JAX**: omega runs WITHIN each chunk. At chunk_size=1, the window has 1 token — no-op.
- **Fix**: Either (a) increase chunk_size to match omega_window, or (b) restructure
  JAX to aggregate omega across chunks (would require a buffer in the carry).

### B2. Processing order: omega vs momentum
- **PyTorch**: `gradients → momentum_scan → omega_aggregate → decay_scan`
- **JAX**: `gradients → omega_aggregate → momentum_scan → PE → decay_scan`
- Omega operates on post-momentum surprises in PyTorch, on raw gradients in JAX.
- **Fix**: Reorder JAX to match PyTorch (move omega after momentum scan).

### B3. Stale vs online weights for gradient computation
- **PyTorch**: ALL tokens in the sequence compute errors against W_init (initial weights).
  `weights_for_surprise = repeat(weights, 'b ... -> b n ...', n=num_chunks)` (line 616)
- **JAX**: each chunk sees W updated by all previous chunks (online learning via carry).
- At chunk_size=1, JAX is fully online (512 sequential updates). At chunk_size=512,
  JAX matches PyTorch (one chunk, all tokens see same W).
- **Impact**: PyTorch gets more gradient signal early in training; JAX may be better
  late (context-adapted weights). Cannot be "fixed" without restructuring — it's a
  design choice.

### B4. Stop-gradient on carry
- **PyTorch**: outer gradients flow through the full weight trajectory (assoc_scan is
  differentiable, no Hessian in the chain because of stale weights → stable).
- **JAX**: `stop_gradient(carry)` blocks outer gradient flow between chunks (required
  because online weights introduce Hessian terms that compound and explode).
- **Impact**: PyTorch's outer optimizer sees richer gradient information.
- **Fix**: None without switching to stale-weight approach (B3).

---

## C. Architecture differences (medium impact)

### C1. Learned affine in RMSNorm (Q/K)
- **PyTorch**: `MultiheadRMSNorm` has learned `gamma` parameter: `rms_norm(x) * (γ+1)`.
  Applied to Q and K after projection.
- **JAX**: `rms_norm(x)` — plain normalization, no learned affine.
- **Params**: 2 × H × D = 512 per layer.
- **Fix**: Add learned gamma to JAX's Q/K norm.

### C2. Learned store/retrieve norms
- **PyTorch**: `nn.RMSNorm(dim)` with learned weight on input before store and retrieve paths.
- **JAX**: no separate store/retrieve norms.
- **Params**: 2 × dim = 512 per layer.
- **Fix**: Add learned RMSNorm before Q/K/V projection and before retrieval.

### C3. QKV depthwise convolutions (kernel=1)
- **PyTorch**: `nn.Conv1d(dim, dim, kernel_size=1, groups=dim)` on Q, K, V sequences.
  Effectively a learned per-channel scale.
- **JAX**: no convolutions (benchmark passes `--conv-kernel 0`).
- **Params**: 3 × dim = 768 per layer.
- **Fix**: Add or ensure JAX conv matches (kernel_size=1 case).

### C4. ResidualNorm gamma updated by inner vs outer optimizer
- **PyTorch**: The LayerNorm gamma inside ResidualNorm is a `memory_model_parameter` —
  it's updated by `vmap(grad(forward_and_loss))` as part of the memory system.
  Per-head: shape (H, D), updated per-token via the inner optimizer.
- **JAX**: `ln_gamma` is a model parameter updated only by AdamW (outer optimizer).
- **Impact**: PyTorch's gamma adapts per-sequence; JAX's only per training step.
- **Fix**: Include ln_gamma in the analytical gradient and carry it through the scan
  (requires extending DeepMemoryState).

### C5. Retrieve gate
- **PyTorch**: `LinearNoBias(dim, heads) → Sigmoid`, applied per-head to memory output.
- **JAX**: `LinearNoBias(dim, heads) → Sigmoid`, same.
- **Status**: Should match. Verify init.

### C6. Output projection / combine_heads
- **PyTorch**: `combine_heads = Linear(dim, dim)` — a separate learned projection after
  merging heads.
- **JAX**: `c_proj = Linear(dim, dim)` — same purpose.
- **Status**: Should match. Verify init.

---

## D. Initialization differences (medium impact)

### D1. Gate bias init
- **PyTorch**: `init_adaptive_step_bias`, `init_momentum_bias`, `init_decay_bias` — each
  gate has separate configurable bias init. Defaults not explicit (likely 0).
- **JAX**: all gates use `gate_bias_init` (single value, default 0.0). Paper uses -2.0
  for memory gates.
- **Check**: Are the PyTorch benchmark gate biases 0.0? Does the benchmark pass any
  `--init-*-bias` flags?

### D2. Memory MLP weight init
- **PyTorch**: `MemoryMLP` uses `nn.init.xavier_uniform_` for both weight matrices.
  With `per_head_learned_parameters=True`, template is repeated per head.
- **JAX**: `W1_init`, `W2_init` use explicit Xavier uniform bounds:
  `bound = sqrt(6/(D+E))`, uniform in [-bound, bound].
- **Check**: Numerically equivalent? Xavier uniform in PyTorch uses the same formula.

### D3. Projection weight init
- **PyTorch**: default `nn.Linear` init (Kaiming uniform).
- **JAX**: Q/K/V use Xavier uniform (`std = 1/sqrt(fan_in)`), c_proj uses
  GPT-2 scaled init (`std = 0.02 / sqrt(2*n_layer)`).
- **Impact**: Different init distributions for the same layers.

### D4. Embedding / lm_head init
- **PyTorch**: default `nn.Embedding` init (normal, std=1).
- **JAX**: `std=0.02` for both wte and lm_head.
- **Impact**: Large init std in PyTorch could affect early dynamics.

---

## E. Hyperparameter alignment (check benchmark config)

### E1. Omega gate
- **PyTorch**: `use_omega_gate` flag + learned `_omega_gate_linear`. The benchmark
  passes `--use-omega-gate` ... or does it? Need to check.
- **JAX**: gamma gate (`gate_gamma`) always created when `omega_window > 1`.
- **Check**: Is the omega gate architecture the same?

### E2. Momentum order
- **PyTorch**: `momentum_order=1` (first-order momentum only).
- **JAX**: momentum is the `theta` gate controlling the linear scan on S.
  No explicit "momentum order" — single scan.
- **Check**: Are these equivalent? PyTorch has separate `momentum` and `decay`
  scans. JAX has `momentum_scan` (S) and `memory_scan` (W). Same structure?

### E3. Poly features
- **PyTorch** (elementwise, degree=2): `out = x + x²` (range 1..degree+1).
- **JAX** (degree=2): `result = coeff[0]*x + coeff[1]*x²` with coeffs=[1,1].
- **Status**: Mathematically identical. ✓

### E4. QK norm ordering
- **PyTorch**: Q → split_heads → q_norm → poly; K → poly → split_heads → k_norm.
- **JAX**: Q → split_heads → rms_norm → poly; K → split_heads → poly → rms_norm.
- **Check**: Order of poly vs norm vs head splitting. Element-wise poly shouldn't
  care about head splitting. But norm-then-poly vs poly-then-norm is different.

### E5. Feedforward
- **PyTorch**: `RMSNorm → Linear(dim, dim_inner*2) → GEGLU → Linear(dim_inner, dim)`.
  Uses `F.silu` gating (SiLU = Swish).
- **JAX**: same structure, configurable. `geglu_ff=True` in benchmark.
- **Check**: Is `dim_inner` computed identically? PyTorch uses `int(dim * 4 * 2/3)`,
  JAX uses `int(n_embd * 4 * 2/3)`.

### E6. Dropout
- **PyTorch**: dropout=0.0 in benchmark (disabled).
- **JAX**: dropout=0.0 in benchmark (disabled).
- **Status**: Match. ✓

---

## F. Training loop differences (low impact for short runs)

### F1. Gradient accumulation semantics
- **PyTorch**: accumulates gradients over sub-steps, one optimizer step per outer step.
- **JAX**: calls `train_step` per sub-step (each is a full optimizer update).
- **Status**: Benchmark uses `grad_accum=1`, so no difference. ✓

### F2. Warmup compilation step
- **JAX**: runs one real training step during JIT compilation before the main loop.
  Model sees one extra batch of data.
- **PyTorch**: no warmup step.
- **Impact**: Negligible (1 extra step out of 2000).

### F3. Data loading
- Both load enwik8 and sample random windows. Different RNG streams.
- **Impact**: Different data ordering, but same distribution. Noise, not bias.

---

## G. Suggested experiment plan

**Phase 1 — Quick wins (fix and rerun)**
1. Fix LR + betas ✓ (done)
2. Set chunk_size=64, omega_window=2 in JAX benchmark (enables omega)
3. Reorder omega/momentum in JAX to match PyTorch

**Phase 2 — Architecture alignment**
4. Add learned affine to Q/K RMSNorm
5. Add store/retrieve norms
6. Match weight initialization (D3, D4)

**Phase 3 — Deep structural**
7. Evaluate stale-vs-online weight impact at chunk_size=64
   (at cs=64 the difference is small: 8 stale windows vs fully online)
8. Evaluate stop_gradient impact at chunk_size=64
   (only 8 boundaries, less information loss)
9. Include ln_gamma in analytical gradient path (C4)

**Expected outcome**: Phase 1 should close most of the remaining gap.
chunk_size=64 alone re-enables omega AND reduces the stale/online difference
(64 tokens share stale W within each chunk, close to PyTorch's behavior).
