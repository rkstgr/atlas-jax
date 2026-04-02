# Implementation Audit: Atlas-JAX vs Paper

Systematic comparison of arXiv 2505.23735 (Atlas) against our Equinox implementation.

---

## Summary

- **14** components audited
- **3** critical discrepancies
- **4** important discrepancies
- **4** minor discrepancies
- **3** items matching correctly

---

## Component-by-Component Audit

### 1. Memory Module Architecture

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| Linear memory: M @ x | M in R^{d_v x d_k} | `model.py:269` einsum `bhvk,bchk->bchv` | Yes |
| Deep MLP: x + W1 @ GELU(W2 @ x) | 2-layer MLP, residual, d_h=256 | `model.py:304-306` | Yes |
| Memory expansion factor | 4 (d_h=256 for d_k=64) | `config.py:19` default=4, running with 1 | **Mismatch** (see below) |
| Activation function | Not specified explicitly | GELU (`model.py:305`) | Assumed OK |

**Note on expand=1**: CLAUDE.md documents running with expand=1 as "3.2x faster; expand=1 deep still beats linear per paper." This is a conscious performance trade-off, not a bug.

### 2. Omega Rule

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| Loss: sum gamma_i \|\|M(k_i)-v_i\|\|^2 | Eq. 9 | `model.py:307` err = y_pred - v_c | Yes |
| Sliding window aggregation | sum over [t-w+1, t] | `model.py:51-74` `_omega_aggregate` via cumsum | Yes |
| Factor of 2 in gradient | 2 * d/dM \|\|...\|\|^2 | `model.py:310,316` u_W1/u_W2 multiply by 2.0 | Yes |
| Context gates gamma | Input-dependent, per-token | `model.py:319-323` | Yes |
| omega_window default | Varies by experiment | `config.py:16` default=16 | OK |

**Omega aggregation correctness**: The cumsum approach in `_omega_aggregate` correctly computes backward-looking sliding windows. The subtraction logic at line 70-73 handles the boundary correctly.

### 3. Deep Memory Gradients

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| grad W1: 2 * err outer GELU(W2@k) | Analytical | `model.py:310` einsum `bchd,bche->bchde` | Yes |
| grad W2: chain through GELU' | 2 * (W1^T @ err * GELU'(h)) outer k | `model.py:313-316` | Yes |
| GELU derivative | Phi(x) + x*phi(x) | `model.py:44-48` `_gelu_derivative` | Yes |

The analytical gradients are correctly derived and match the paper's formulation.

### 4. Momentum + Orthogonalization

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| S_t = theta * S_{t-1} - eta * G_t | Eq. 12-13 | `model.py:330-331` mom_input = -(eta * u) | Yes |
| M_t = alpha * M_{t-1} + PE(S_t) | Eq. 12 | `model.py:337` linear_scan with alpha | Yes |
| Both via linear_scan | Parallel scan | `model.py:335-337` | Yes |
| Separate S for W1 and W2 | Implied (separate params) | `model.py:340-341` S_W1, S_W2 | Yes |

### 5. Polar Express / Newton-Schulz

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| NS-5 (5 iterations) | Section 5, Appendix | `polar_express.py:30-61` | Yes |
| Quintic polynomial per step | Degree 5, per-step optimal | `polar_express.py:21-27` POLAR_EXPRESS_COEFFS | Yes |
| Frobenius normalization | X / \|\|X\|\|_F | `polar_express.py:43-44` | Yes |
| Safety factor 1.01 | PE paper: safety margin | `polar_express.py:44` `frob_norm * 1.01` | Yes |
| Tall vs wide dispatch | X^T@X for tall, X@X^T for wide | `polar_express.py:48-59` | Yes |
| STE variant | Forward=PE, backward=identity | `polar_express.py:64-73` | Yes |
| f32 upcast for stability | Required | `polar_express.py:42` | Yes |
| Running with ns_steps=3 | Paper says 5 | `train.py:90` default=3 | **Mismatch** |

**Note on ns_steps=3**: CLAUDE.md documents this as "1.5x faster PE forward; nanochat PyTorch found this works." Conscious trade-off.

### 6. Input-Dependent Gates

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| alpha, eta, theta, gamma | All via sigmoid | `model.py:370-376` | Yes |
| Per-head scalar | Not per-dimension | `model.py:236-238` Linear(C, H) | Yes |
| Gate bias init at -2 | sigmoid(-2) ~ 0.12 | **NOT implemented** | **CRITICAL** |
| gamma only when omega_window > 1 | Implied | `model.py:240-242` | Yes |

### 7. Chunk-Parallel Computation

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| stop_gradient on carry | Frozen boundary | `model.py:428` `jax.lax.stop_gradient(carry)` | Yes |
| Pre-chunked xs to lax.scan | Required for correct grads | `model.py:411-425` | Yes |
| Gradient checkpointing | Per-chunk | `model.py:438` `jax.checkpoint` | Yes |
| Layer norm at chunk boundaries | Paper specifies | **NOT implemented** | **CRITICAL** |

### 8. Short Causal Convolution

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| Depthwise 1D conv | Per-channel, causal | `model.py:129-159` ShortConv | Yes |
| Kernel size 4 | Standard | `config.py:14` conv_kernel=4 | Yes |
| Applied to Q, K, V | After projection, before heads | `model.py:357-359` | Yes |

### 9. Block Structure

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| Pre-norm residual | x + sublayer(norm(x)) | `model.py:486-488` | Yes |
| RMS Norm | Used throughout | `model.py:38-41` | Yes |
| Memory + MLP per block | Sequential | `model.py:476-489` Block | Yes |

### 10. Q/K Normalization

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| RMS norm on Q, K | Mentioned contextually | `model.py:362` rms_norm(q), rms_norm(k) | Likely correct |

### 11. Weight Initialization

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| Linear layers | Xavier uniform | Equinox default (Lecun normal) | **IMPORTANT** |
| Output projections | Not specified | Zero (`model.py:534-538`) | Our choice |
| Gate projections | Bias = -2 | Small weights, no bias (`model.py:540-544`) | **CRITICAL** |
| lm_head | Not specified | 0.02 std normal (`model.py:528-529`) | Our choice |
| Polynomial coeffs | a_i = 1/i! | `model.py:246` 1/factorial(i) | Yes |

### 12. MLP (Feed-Forward)

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| Expansion 4x | Standard | `model.py:461-462` 4 * n_embd | Yes |
| GELU activation | Standard | `model.py:467` | Yes |
| No bias | Common practice | `model.py:461-462` use_bias=False | Yes |

### 13. Outer Optimizer

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| AdamW | Paper uses AdamW | `train.py:162-165` AdamW | Yes |
| LR 1e-4 | For Atlas variants | `train.py:94` default=3e-3 | **IMPORTANT** |
| Cosine schedule | With warmup | `train.py:155-161` warmup_cosine_decay | Yes |
| Warmup 4000 steps | Paper default | `train.py:97` default=200 | **IMPORTANT** |
| Weight decay 0.01 | Paper default | `train.py:95` default=0.1 | **IMPORTANT** |
| Grad clip norm 1.0 | Paper default | `train.py:163` clip_by_global_norm(1.0) | Yes |

### 14. Miscellaneous

| Aspect | Paper | Code | Match? |
|--------|-------|------|--------|
| Soft logit capping | NOT in Atlas paper | `model.py:584-585` 15.0 * tanh | **Minor** |
| Dropout 0.1 | Paper uses | NOT implemented | **Minor** |
| Embedding RMS norm | Not specified | `model.py:569` rms_norm after embed | Our choice |
| W2 init [I; 0] | Not specified in paper | `state.py:48-51` and `model.py:383-384` | Our choice |

---

## Critical Discrepancies

### C1: Gate Bias Initialization (CRITICAL)

**Paper**: Gate projections have bias initialized at -2, giving sigmoid(-2) ~ 0.12 as initial gate value.

**Code** (`model.py:236-238`): Gates are `eqx.nn.Linear(C, H, use_bias=False)` with small weight init (std=0.01).

**Impact**: With zero-mean small weights, initial sigmoid is ~0.5 (not ~0.12). This means:
- alpha (forget) starts at 0.5 instead of 0.12 -- memory decays faster initially
- eta (lr) starts at 0.5 instead of 0.12 -- larger initial learning rate
- theta (momentum) starts at 0.5 instead of 0.12 -- more initial momentum

This could significantly affect training dynamics, especially early on.

**Fix**: Add `use_bias=True` to gate projections and initialize bias to -2.0.

### C2: Layer Norm at Chunk Boundaries (CRITICAL)

**Paper**: Specifies applying layer normalization to memory state at chunk boundaries.

**Code**: Not implemented. CLAUDE.md notes this as a known gap.

**Impact**: Without normalization, memory matrix norms can grow across chunks, potentially causing instability for long sequences. May not matter for short sequences or with stop_gradient.

**Fix**: Apply RMS norm (or layer norm) to memory state components (M, S or W1, W2, S_W1, S_W2) at the start of each chunk, after stop_gradient.

### C3: NS Steps Default (CRITICAL for paper-matching)

**Paper**: Uses NS-5 (5 iterations). Code default in config is 5, but train.py default is 3.

**Impact**: Fewer iterations means less-orthogonal momentum, which the paper found matters for quality. However, CLAUDE.md documents this as a conscious perf trade-off validated by nanochat.

**Fix**: Use ns_steps=5 for paper-matching experiments. Keep 3 as a speed option.

---

## Important Discrepancies

### I1: Weight Initialization (IMPORTANT)

**Paper**: Xavier uniform for linear layers.
**Code**: Equinox default (Lecun normal = normal(0, 1/sqrt(fan_in))).

Xavier uniform = uniform(-sqrt(6/(in+out)), sqrt(6/(in+out))).

These are similar but not identical distributions. Could affect convergence speed.

### I2: Learning Rate (IMPORTANT)

**Paper**: LR = 1e-4 for Atlas.
**Code**: Default LR = 3e-3 (30x higher).

This is likely intentional for the smaller model (48.8M vs paper's larger model), but should be noted.

### I3: Warmup Steps (IMPORTANT)

**Paper**: 4000 steps warmup.
**Code**: Default 200 steps.

Again, may be appropriate for smaller-scale experiments but doesn't match paper.

### I4: Weight Decay (IMPORTANT)

**Paper**: 0.01.
**Code**: Default 0.1 (10x higher).

---

## Minor Discrepancies

### M1: Soft Logit Capping

15.0 * tanh(logits/15.0) is not mentioned in the Atlas paper. This is borrowed from Gemma. It bounds logits to [-15, 15] which can help stability but is not part of the Atlas architecture.

### M2: Dropout

Paper uses dropout 0.1. Our implementation has no dropout. Minor for pre-training but could matter for overfitting on small datasets.

### M3: Memory Expansion Factor

Paper default is 4 (d_h=256 for d_k=64). Code has been running with 1. This is documented and intentional.

### M4: Polynomial Degree

Paper uses degree 2. Code default is 3. Higher degree = more capacity but more compute. Both are valid choices.

---

## Test Coverage Gaps

| Gap | Risk | Priority |
|-----|------|----------|
| No test for gate bias initialization effect | Could mask C1 bug | High |
| No test for chunk boundary layer norm | Can't verify C2 fix | High |
| No test for omega_window > chunk_size | Edge case | Medium |
| No test for deep memory W1/W2 gradient correctness vs finite diff | Could have subtle bugs | High |
| No test for checkpoint vs no-checkpoint numerical equivalence | Could mask gradient bugs | Medium |
| No test for polynomial feature mapping gradient flow | Feature may be broken | Medium |
| No test for very long sequences (>10 chunks) | Stability issues may hide | Medium |
| No end-to-end loss convergence test (overfit tiny dataset) | Correctness signal | High |
| No test comparing full PE vs STE on loss quality | Can't validate STE trade-off | Low |
| No test for Muon optimizer integration | optim.py is dead code | Low |

---

## What's Left to Do (Prioritized)

### Must Fix (before serious training)

1. **Gate bias initialization** -- add bias=-2 to gate projections
2. **Layer norm at chunk boundaries** -- add to chunk_body in model.py
3. **Gradient correctness test** -- finite-difference check on deep memory gradients
4. **Overfit test** -- verify model can memorize a tiny dataset

### Should Fix (for paper-matching experiments)

5. **Xavier uniform initialization** -- match paper's init scheme
6. **Training hyperparameters** -- provide paper-matching defaults (LR=1e-4, warmup=4000, wd=0.01)
7. **Wire up Muon outer optimizer** -- optim.py has the code, just needs multi_transform

### Nice to Have

8. **Dropout** -- add 0.1 dropout matching paper
9. **Remove soft logit capping** -- or make it configurable
10. **Add layer norm epsilon config** -- paper uses 1e-6, our rms_norm uses 1e-6 (matches)
