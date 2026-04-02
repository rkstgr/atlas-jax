# ATLAS Implementation Plan (JAX/Flax)

**Goal:** Go from ~1% MFU naive implementation to 25-40% MFU production-quality ATLAS on H100, in incremental stages with correctness and performance validation at every step.

**Philosophy:** Each phase produces a working, tested model. Never move to the next phase until the current one passes all its tests. Performance bugs compound — catch them early.

---

## Phase 0: Baseline Measurement & Reference Implementation

**What:** Establish ground truth numbers before touching anything. You need something to compare against.

**Tasks:**

- Profile the current JAX implementation with `jax.profiler` and export a trace to TensorBoard. Identify where time is actually spent — is it the MLP forward/backward? The momentum scan? Memory transfers? Don't guess.
- Record baseline numbers on a small, reproducible task:
  - Sequence length 2048, batch size 4, d_model 512, 4 layers
  - Tokens/second, peak HBM usage, step time breakdown
  - MFU at this exact config (compute theoretical FLOPs per step, divide by observed wall-clock × peak FLOPS)
- Write a minimal reference Transformer baseline at the same parameter count using standard FlashAttention. This gives you an MFU ceiling to calibrate against.

**Tests:**

| Test | What it validates | Pass criteria |
|------|-------------------|---------------|
| Profile trace analysis | Where time goes | Identify top-3 bottlenecks by wall-clock time |
| FLOP counting | MFU calculation is correct | Manual FLOP count matches `flax.nn.tabulate` or similar |
| Transformer baseline MFU | Hardware is working | 30-50% MFU for vanilla Transformer at same scale |
| Loss curve sanity | Current impl learns | Loss decreases monotonically on tiny dataset (overfit test) |

**Deliverable:** A spreadsheet with baseline numbers and a ranked list of bottlenecks. Do not proceed until you can explain where 99% of your time is going.

---

## Phase 1: Correct Chunked Computation (Correctness First, Speed Later)

**What:** Restructure the computation into chunks, but still in pure JAX (no custom kernels). The goal is to get the right answer with the chunked approach before optimizing it.

**Tasks:**

- Implement the chunked forward pass: split sequence into chunks of size `c`, process each chunk as a batched MLP forward `[c, d] → [c, d]`
- Implement the chunked Omega-rule gradient: compute loss over all `c` tokens in the window, backprop through the MLP to get the gradient of M's weights w.r.t. the combined loss
- Implement chunk-to-chunk recurrence: update M at chunk boundaries, carry state forward
- Use `jax.lax.scan` for the cross-chunk recurrence (not a Python for loop — this is critical for XLA compilation)

**Key implementation detail:** The Omega loss within a chunk is:
```
L = Σ_{i=1}^{c} γ_i · ||M(φ(k_i)) - v_i||²
```
The gradient `∇_M L` is a sum of per-token gradients. In JAX, use `jax.vmap` over the `c` tokens for the forward pass, then `jax.grad` on the summed loss. This gives you the batched computation for free.

**Tests:**

| Test | What it validates | Pass criteria |
|------|-------------------|---------------|
| Chunk-1 equivalence | Chunked code with c=1 matches token-by-token | Max absolute difference < 1e-5 across all outputs for a 512-token sequence |
| Gradient correctness | Omega-rule gradient is right | Compare `jax.grad` of chunked loss against finite differences (ε=1e-4), relative error < 1e-3 |
| State continuity | Chunk boundaries don't corrupt memory | Process seq as [one big chunk] vs [many small chunks] with c=1 — outputs match to 1e-5 |
| Determinism | Same input → same output | Run 5 times, all outputs bitwise identical (control PRNG) |
| Variable sequence length | Handles non-divisible lengths | Test seq_len=100 with c=64 (last chunk is 36 tokens) — no crash, correct output |

**Performance test (informational only, don't optimize yet):**
- Record step time. Expect 2-5× improvement over token-by-token just from `lax.scan` + `vmap` replacing Python loops.
- If MFU is still <2%, something is structurally wrong with the computation graph — inspect the XLA HLO.

**Deliverable:** A chunked implementation that produces identical outputs to the naive version (up to float precision).

---

## Phase 2: Efficient Momentum via Parallel Scan

**What:** Replace the sequential momentum accumulation with a parallel prefix scan within each chunk.

**Tasks:**

- The momentum recurrence `S_t = θ_t · S_{t-1} + G_t` is a linear scan. Implement it using `jax.lax.associative_scan` with the operator:
  ```python
  def binary_op(a, b):
      # a, b are tuples of (decay_product, accumulated_value)
      return (a[0] * b[0], a[1] * b[0] + b[1])
  ```
- Verify that this produces identical results to the sequential scan
- Carry the final momentum state across chunk boundaries in the `lax.scan` state

**Why this matters:** The momentum computation is on the critical path between "compute all gradients" and "apply Newton-Schulz." If it's sequential over `c` steps, it serializes the entire pipeline even though the gradients are already computed in parallel.

**Tests:**

| Test | What it validates | Pass criteria |
|------|-------------------|---------------|
| Scan equivalence | Parallel scan matches sequential | Max diff < 1e-5 against sequential loop for c=64, 128 |
| Numerical stability | No overflow/underflow in scan | Run with float32 and bfloat16, check for NaN/Inf on 10K random inputs |
| Cross-chunk continuity | Momentum carries across chunks | Compare: 1 chunk of 128 vs 2 chunks of 64 — final momentum matches |
| Scaling test | scan works at target chunk sizes | Test c = 32, 64, 128, 256 — all pass equivalence |

**Performance test:**
- Benchmark the momentum computation alone (isolate it). With `associative_scan` it should be 5-10× faster than sequential for c=64.
- Measure overall step time improvement. Expect modest gains (5-15%) since momentum isn't usually the biggest bottleneck.

---

## Phase 3: Newton-Schulz (Muon) Integration

**What:** Add the Newton-Schulz orthogonalization step that turns vanilla momentum into Muon.

**Tasks:**

- Implement Newton-Schulz iteration:
  ```python
  def newton_schulz(M, steps=5):
      a, b, c = (3.4445, -4.7750, 2.0315)  # optimal coefficients
      X = M / jnp.linalg.norm(M)
      for _ in range(steps):
          A = X @ X.T
          X = X @ (a * jnp.eye(d) + b * A + c * A @ A)
      return X
  ```
- Batch this across all `c` momentum matrices in the chunk using `jax.vmap`
- This should be the easiest phase — Newton-Schulz is pure matmuls with no data dependencies between timesteps

**Tests:**

| Test | What it validates | Pass criteria |
|------|-------------------|---------------|
| Orthogonality | NS output is approximately orthogonal | `||X^T X - I||_F < 1e-3` after k=5 steps |
| Singular value equalization | All singular values ≈ 1 | Max/min singular value ratio < 1.1 after k=5 |
| Gradient flow | Gradients flow through NS | `jax.grad` through NS doesn't produce NaN; gradient norm is reasonable |
| k ablation | More steps → better orthogonality | Test k = 1, 3, 5, 7: orthogonality error decreases monotonically |
| bfloat16 stability | Works in reduced precision | Same tests pass in bfloat16 (with relaxed tolerances: ortho error < 1e-2) |

**Performance test:**
- Profile NS in isolation: for `c=64` matrices of shape `[d, d]` with d=1024, the vmapped NS should take < 1ms on H100 (it's ~15 batched matmuls).
- If it's significantly slower, check that `vmap` is actually batching (inspect XLA HLO) rather than sequentializing.

---

## Phase 4: Memory Architecture — Deep MLP with Residual Connections

**What:** Ensure the memory module M is properly structured as a 2-layer MLP with residual connections, and optionally gated (ATLAS++).

**Tasks:**

- Base memory: `M(x) = x + W₁ · σ(W₂ · x)` with `W₁ ∈ [d, 4d]`, `W₂ ∈ [4d, d]` (or reversed depending on convention). The residual connection is important — without it, gradient descent on M's weights tends to be unstable.
- Gated variant (ATLAS++): `M(x) = x + W₁ · (σ(W₂ · x) ⊙ W₃ · x)` — SwiGLU-style gating
- The per-step gradient of the Omega loss w.r.t. `W₁, W₂` (and `W₃` for gated) must be computed efficiently. In JAX, `jax.grad` on the vmapped forward should handle this.

**Tests:**

| Test | What it validates | Pass criteria |
|------|-------------------|---------------|
| Capacity test | Deep memory stores more than linear | Train M to memorize N random (k,v) pairs. Linear memory fails at N≈d, deep memory should handle N≈2-4d |
| Residual gradient flow | Gradients are well-behaved | Gradient norm of M's weights stays within 1e-4 to 1e2 over 1000 update steps |
| Overfit test | ATLAS with deep M overfits small data | Train on 100 sentences, reach <0.1 perplexity |
| Gated vs ungated | Gating doesn't break anything | Both variants pass all above tests; gated should have slightly higher capacity |

**Performance test:**
- The MLP forward and backward should be the largest matmuls in the system. Profile to confirm they dominate compute (>50% of step time). If they don't, something else is bottlenecking.
- Measure MFU. At this point you should be at **5-15% MFU** with all components working together in pure JAX.

---

## Phase 5: XLA Optimization — Getting to 15-25% MFU

**What:** Optimize the JAX computation graph without writing custom kernels. This is about helping the compiler help you.

**Tasks:**

- **Constant folding and JIT boundaries:** Ensure `jax.jit` wraps the entire training step, not individual components. Multiple JIT boundaries cause recompilation and synchronization.
- **Precision strategy:** Use bfloat16 for the MLP forward/backward (this doubles throughput on tensor cores). Keep the momentum accumulation and Newton-Schulz in float32 for stability. JAX's `jax.custom_vjp` lets you control precision per operation.
- **Memory layout:** Ensure tensors are contiguous in the right dimensions. For batched matmuls `[batch, c, d] × [d, d]`, the `c` and `d` dimensions should be contiguous (C-order). Transpose operations are surprisingly expensive.
- **Remat (gradient checkpointing):** The within-chunk computation stores `c` activations for the backward pass. For large `c` or `d`, this can blow up HBM. Use `jax.checkpoint` on the per-chunk computation to trade recomputation for memory.
- **Donate buffers:** Use `jax.jit(donate_argnums=...)` to allow JAX to reuse memory buffers for the memory state across chunks, avoiding unnecessary allocations.

**Tests:**

| Test | What it validates | Pass criteria |
|------|-------------------|---------------|
| Numerical equivalence | Optimizations don't change results | float32 outputs match Phase 4 to 1e-5; bfloat16 outputs match to 1e-2 |
| No recompilation | JIT is stable | `jax.jit` compiles once and subsequent steps show no compilation time |
| HBM usage | Memory is under control | Peak HBM fits within 80GB for target config (1.3B params, seq_len 8K, batch 8) |
| XLA HLO inspection | Compiler is doing what you expect | `jax.make_jaxpr` → verify matmuls are fused, no unnecessary transposes |

**Performance tests:**
- Step time comparison against Phase 4 (same config). Expect 2-4× speedup from precision + compilation improvements.
- MFU should be **10-20%** at this point.
- Run a scaling test: measure step time at d=512, 1024, 2048. Verify that MFU increases with d (larger matmuls → better utilization).

---

## Phase 6: Custom Kernels (Pallas) — Getting to 25-40% MFU

**What:** Write fused kernels for the operations XLA can't optimize well on its own.

**Tasks (in priority order):**

1. **Fused chunk kernel:** The single highest-impact kernel. Fuse the within-chunk MLP forward, loss computation, and backward into one Pallas kernel that keeps intermediate activations in SRAM rather than writing them to HBM. This eliminates the memory bandwidth bottleneck for the inner loop.

2. **Fused momentum + Newton-Schulz:** Combine the associative scan with the NS iterations. The scan output flows directly into NS without a round-trip to HBM.

3. **Memory-efficient gating:** If using ATLAS++ (gated memory), the SwiGLU computation can be fused with the matmul that precedes it — same trick as in FlashAttention's fused softmax.

**Realistic expectations:** Writing Pallas kernels for this is a significant engineering effort (weeks, not days). The fused chunk kernel alone is probably 500+ lines of Pallas code. Start with kernel #1 and measure before deciding if #2 and #3 are worth the effort.

**Tests:**

| Test | What it validates | Pass criteria |
|------|-------------------|---------------|
| Kernel correctness | Custom kernel matches JAX reference | Max diff < 1e-4 against Phase 5 output on 100 random inputs |
| Numerical stability sweep | Kernel handles edge cases | Test with: very large values, very small values, zeros, near-orthogonal keys, highly correlated keys |
| Gradient correctness | Custom backward kernel is right | Compare against `jax.grad` on reference implementation, relative error < 1e-3 |
| Benchmark in isolation | Kernel performance is good | Each kernel should achieve >50% of theoretical peak throughput for its specific matmul sizes |

**Performance tests:**
- MFU should reach **20-35%** with the fused chunk kernel alone.
- Profile again: is the bottleneck now the cross-chunk recurrence? If so, you're approaching the architecture's inherent serial fraction.
- Run Amdahl's law analysis: given `c` parallel steps per 1 sequential step, your theoretical MFU ceiling is `c/(c+overhead)` × peak. Calculate this for your `c` values.

---

## Phase 7: End-to-End Training Validation

**What:** Verify that the optimized implementation actually trains a good model, not just a fast one.

**Tasks:**

- Train a 360M parameter ATLAS model on a standard dataset (FineWeb, SlimPajama, or similar) for a meaningful number of tokens (at least 10B)
- Compare against a Transformer++ baseline at the same parameter count and token budget
- Evaluate on the paper's benchmark suite: Wikitext perplexity, HellaSwag, PIQA, ARC

**Tests:**

| Test | What it validates | Pass criteria |
|------|-------------------|---------------|
| Loss curve health | Model trains stably | No loss spikes >2× baseline, smooth convergence |
| Perplexity comparison | ATLAS matches/beats Transformer | Within 5% of Transformer++ perplexity at same compute budget |
| Long context eval | Memory module actually helps | S-NIAH accuracy > Transformer++ at 4K+ context |
| Ablation: Omega rule | c>1 beats c=1 | Train with c=1 (Titans-like) and c=64: c=64 should have lower perplexity |
| Ablation: Muon vs SGD | Newton-Schulz helps | Train with and without NS step: NS variant should have lower perplexity, especially on recall tasks |
| Ablation: deep vs linear memory | Depth helps | Linear M vs 2-layer MLP: MLP should win on capacity-sensitive tasks |

**Performance targets for 360M model on single H100:**
- Throughput: >50K tokens/second (comparable to efficient Transformer implementations at this scale)
- MFU: 25-35%
- HBM usage: <40GB (leaving room for larger batch sizes)

---

## Summary: Expected MFU Progression

| Phase | What changes | Expected MFU |
|-------|-------------|--------------|
| 0 | Baseline (current) | ~1% |
| 1 | Chunked computation + lax.scan | 2-5% |
| 2 | Parallel scan for momentum | 3-7% |
| 3 | Newton-Schulz integration | 4-8% |
| 4 | Proper deep memory MLP | 5-15% |
| 5 | XLA optimization + mixed precision | 10-20% |
| 6 | Custom Pallas kernels | 20-35% |
| 7 | End-to-end validation | 25-35% (sustained) |

The biggest jumps are Phase 1 (eliminating Python loops), Phase 5 (helping the compiler), and Phase 6 (custom kernels). Each roughly doubles your MFU.

---

## Appendix: Quick Reference for Common Pitfalls

**JAX-specific:**
- Never use Python `for` loops over sequence steps — always `lax.scan` or `lax.fori_loop`
- `jax.vmap` over the chunk dimension is your friend — it turns small matmuls into big ones
- Watch for unintended materialization of large intermediates — `jax.checkpoint` aggressively
- `jnp.einsum` often generates better XLA than manual reshapes + matmuls

**Numerical:**
- Newton-Schulz is sensitive to the initial scaling `X₀ = S / ||S||`. If `||S||` varies wildly across chunks, normalize per-chunk
- bfloat16 has only 7 bits of mantissa — the Omega loss sum over `c` tokens can lose precision. Accumulate in float32
- The decay terms `α_t, θ_t` should be in float32 even if everything else is bfloat16

**Debugging:**
- If loss goes NaN, first suspect: Newton-Schulz on a near-zero momentum matrix. Add epsilon to the norm
- If MFU is mysteriously low, profile for "memcpy" operations in the trace — these indicate unnecessary HBM round-trips
- If scaling to multiple GPUs, the memory state `M` needs to be replicated (not sharded) since every token in the sequence reads/writes it
