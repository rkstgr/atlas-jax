# Atlas: Technical Paper Reference

Extracted from arXiv 2505.23735 (Atlas) and arXiv 2505.16932 (Polar Express).

---

## 1. Architecture Overview

Atlas is a memory-augmented language model. Each layer consists of:

```
Input x
  |
  v
[RMS Norm] --> [Memory Layer] --> residual add
  |                                    |
  v                                    v
[RMS Norm] --> [MLP]           --> residual add
                                       |
                                       v
                                   Output x
```

The memory layer replaces the attention mechanism. Memory is a per-head
MLP (or linear matrix) that is updated at every token via an internal
optimizer (gradient descent with momentum + Newton-Schulz orthogonalization).

**Atlas vs Atlas++**: Atlas++ adds SwiGLU-style gating to the memory MLP.
The base Atlas uses a standard 2-layer MLP with GELU and residual.

---

## 2. Memory Module

### Linear Memory (L_M = 1)

Memory is a matrix M in R^{d_v x d_k}. Output: y_t = M_t @ q_t.

**Capacity** (Proposition 1): Can store O(d_k) independent key-value pairs.

### Deep MLP Memory (L_M >= 2)

Memory is a 2-layer MLP with residual connection:

```
M(x) = x + W1 @ sigma(W2 @ x)
```

Where:
- W1 in R^{d_v x d_h}, W2 in R^{d_h x d_k}
- sigma = GELU activation
- Paper default: d_h = 256 for d_k = 64 (expansion factor = 4)

**Capacity** (Theorem 1): Can store at least O(d_k * d_v) and at most
O(d_k * d_v * sum_i min{d_h^(j)}_{j>=i} * d_h^{(j+1)}) pairs.

### Polynomial Feature Mapping

```
phi_p(x) = [x^beta]_{|beta| <= p}
```

For the element-wise implementation used in practice:

```
phi(x) = a_1 * x + a_2 * x^2 + ... + a_p * x^p
```

Coefficients initialized as a_i = 1/i! (Taylor approximation of exp).

**Capacity** (Proposition 2): With degree-p mapping, stores O(d_k^p) pairs.

---

## 3. Omega Rule (Learning Rule)

### Loss Function (Eq. 9)

For each token t, the memory optimizes over a sliding window of size c:

```
L_t = sum_{i=t-c+1}^{t} gamma_i^(t) * ||M(phi(k_i)) - v_i||_2^2
```

Where:
- c: context window (chunk size) -- c=1 is online/Delta rule, c=inf is global
- gamma_i^(t) in [0,1]: input-dependent context gates (enable selective forgetting)
- phi: polynomial feature mapping

### Memory Update (Eq. 10)

General form with gradient descent:

```
M_t = alpha_t * M_{t-1} - nabla sum_{i=t-c+1}^{t} gamma_i^(t) * ||M(phi(k_i)) - v_i||_2^2
```

### Linear Memory Gradient (Eq. 11)

For linear M (M(x) = M @ x), the gradient has closed form:

```
u_t = 2 * (M @ phi(k_t) - v_t) @ phi(k_t)^T
```

This is an outer product (rank-1 update per token).

Full linear update:

```
M_t = (diag(alpha_t) - sum gamma_i * phi(k_i) @ phi(k_i)^T) * M_{t-1}
      - sum gamma_i * v_i @ phi(k_i)^T
```

### Deep Memory Gradient

For M(x) = x + W1 @ GELU(W2 @ x):

**Gradient w.r.t. W1:**
```
u_W1 = 2 * (M(phi(k_t)) - v_t) @ GELU(W2 @ phi(k_t))^T
```

**Gradient w.r.t. W2:**
```
u_W2 = 2 * (W1^T @ err * GELU'(W2 @ phi(k_t))) @ phi(k_t)^T
```

Where err = M(phi(k_t)) - v_t, and GELU' is the exact GELU derivative:
```
GELU'(x) = Phi(x) + x * phi(x)
```
with Phi = standard normal CDF, phi = standard normal PDF.

### Omega Aggregation (Sliding Window)

When omega_window > 1, per-token gradients are aggregated:

```
G_t = sum_{i=max(0, t-w+1)}^{t} gamma_i * u_i
```

Efficiently computed via prefix sum (O(n) instead of O(n*w)).

---

## 4. Momentum + Orthogonalization (Eq. 12-13)

The full per-token update sequence:

```
1. Compute gradient:  u_t = nabla_M ||M(phi(k_t)) - v_t||^2
2. Omega aggregate:   G_t = sum_{window} gamma_i * u_i
3. Momentum:          S_t = theta_t * S_{t-1} - eta_t * G_t
4. Orthogonalize:     S'_t = NS-5(S_t)       [Newton-Schulz, 5 iterations]
5. Memory update:     M_t = alpha_t * M_{t-1} + S'_t
6. Output:            y_t = M_t(q_t)
```

Where:
- S_t: momentum accumulator (same shape as M's parameters)
- NS-5: 5 iterations of Newton-Schulz polar approximation
- The momentum recurrence S_t = theta * S_{t-1} - eta * G_t is a linear scan

---

## 5. Polar Express (Newton-Schulz Orthogonalization)

### Classical Newton-Schulz (Degree 3)

```
X_0 = M / ||M||_F
X_{t+1} = (3/2) * X_t - (1/2) * X_t @ X_t^T @ X_t
```

Polynomial: p(x) = (3/2)x - (1/2)x^3. Quadratic convergence.

### Newton-Schulz Degree 5

```
p(x) = (15x - 10x^3 + 3x^5) / 8
```

Cubic convergence (approximately 2x faster than degree 3).

### Polar Express (Optimal Polynomials)

Instead of fixed polynomials, Polar Express uses per-step optimized polynomials:

```
X_0 = M / ||M||_F
X_{t+1} = p_t(X_t)
```

Where each p_t minimizes the worst-case error on the current spectral interval:

```
p_t = argmin_{p in P_d^odd} max_{x in [l_t, u_t]} |1 - p(x)|
```

The intervals shrink: l_{t+1} = p_t(l_t), u_{t+1} = 2 - l_{t+1}.

**Convergence** (Theorem 4.3): For degree d = 2q+1:
```
||polar(M) - X_T||_2 <= |1 - l^2|^{(q+1)^T}
```

- Degree 3 (q=1): quadratic convergence
- Degree 5 (q=2): cubic convergence — exponent grows as 3^T

### GPU-Efficient Evaluation

Odd monomials computed as:
```
M^{2q+1} = M @ (M^T @ M)^q
```

For degree-5 polynomial p(x) = a_0*x + a_1*x^3 + a_2*x^5:
```
A = X^T @ X           (for tall) or X @ X^T (for square/wide)
B = b*A + c*(A @ A)
X_{new} = a*X + B @ X (for square/wide) or a*X + X @ B (for tall)
```

Cost: ~3 matrix-matrix products per iteration.

### Precomputed Coefficients

For 5 steps with safety_factor=2e-2, cushion=2:

| Step | a | b | c |
|------|---|---|---|
| 1 | 8.1566 | -22.4833 | 15.8788 |
| 2 | 4.0429 | -2.8089 | 0.5000 |
| 3 | 3.8917 | -2.7725 | 0.5061 |
| 4 | 3.2858 | -2.3681 | 0.4645 |
| 5 | 2.3465 | -1.7098 | 0.4232 |

### Frobenius Normalization

```
X = X / (||X||_F * (1 + safety_margin) + eps)
```

The safety margin (1.01 = 1% extra) ensures singular values start below 1,
within the convergence basin. eps = 1e-6 for numerical safety.

### STE (Straight-Through Estimator)

Forward: full Newton-Schulz orthogonalization.
Backward: identity Jacobian (gradients pass through unchanged).

Justified because PE is an internal optimizer finding the nearest orthogonal
matrix. Near-orthogonal inputs have Jacobian approximately equal to identity.
62x faster backward than full Jacobian computation.

---

## 6. Input-Dependent Gates

All gates are produced by linear projections followed by sigmoid:

```
alpha_t = sigmoid(W_alpha @ x_t)    # forget gate (memory decay)
eta_t   = sigmoid(W_eta @ x_t)      # learning rate gate
theta_t = sigmoid(W_theta @ x_t)    # momentum gate
gamma_t = sigmoid(W_gamma @ x_t)    # context window gate (omega rule)
```

- Gates produce per-head scalar values (not per-dimension)
- Paper: gate projections have bias initialized at -2, giving initial sigmoid ~ 0.12
- gamma is only needed when omega_window > 1

---

## 7. Chunk-Parallel Training

### Chunking

Sequence of length L is divided into ceil(L/c) chunks of size c.
Each chunk is processed independently, enabling parallelism.

### Gradient Flow at Boundaries

- Memory state (M, S) at chunk boundaries uses **stop_gradient**
- Each chunk treats incoming memory state as a frozen constant
- Gradients flow through Q, K, V, gates within a chunk but NOT through
  the memory state from previous chunks
- This matches the paper: "Gradients are computed w.r.t. the last state
  of the previous chunk"

### Layer Norm at Chunk Boundaries

The paper specifies applying layer normalization to the memory state at
chunk boundaries to stabilize training. This normalizes the memory matrices
before they enter the next chunk's computation.

### Gradient Checkpointing

Each chunk's computation can be checkpointed (recompute activations on
backward pass) to trade compute for memory. Essential for long sequences.

---

## 8. Short Causal Convolution

Applied to Q, K, V after linear projection, before multi-head reshape.
Depthwise 1D convolution with kernel size 4, causal (left-padded).
Provides local temporal context to supplement the memory mechanism.

---

## 9. Training Details (from Paper Experiments)

### Model Configuration (Section 6)

| Parameter | Value |
|-----------|-------|
| Hidden size (d_model) | 768 |
| Num layers | 12 |
| Num heads | 12 |
| Head dimension | 64 |
| MLP expansion | 4x (intermediate = 3072) |
| Memory hidden dim | 256 (expansion = 4) |
| Polynomial degree | 2 |
| NS iterations | 5 |
| Chunk size / omega window | varies by experiment |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Outer optimizer | AdamW |
| Learning rate | 1e-4 (for Atlas variants) |
| LR schedule | Cosine decay with warmup |
| Warmup steps | 4000 |
| Batch size | 64 |
| Sequence length | 2048 |
| Token budget | ~100B |
| Weight decay | 0.01 |
| Gradient clipping | max norm 1.0 |
| Dropout | 0.1 |
| Layer norm epsilon | 1e-6 |

### Weight Initialization

| Component | Method |
|-----------|--------|
| Linear layers | Xavier uniform |
| Biases | Zero |
| LayerNorm weight | Ones |
| LayerNorm bias | Zeros |
| Gate projection bias | -2 (sigmoid ~ 0.12 initially) |
| Polynomial coefficients | a_i = 1/i! |

---

## 10. Ablation Results (Section 6.6)

### Key Findings

- **Deep vs linear memory**: Deep memory improves ~3-5% on long-context tasks
- **Polynomial degree**: Degree 2 improves ~2-3% over degree 1
- **Optimizer**: Muon (NS-5) > Momentum > SGD for memory updates
- **Window size**: Optimal typically 2-4x (sequence_length / num_layers)
- **Memory capacity**: Scales quadratically with polynomial degree
- **Training cost**: Linear in window size within chunk framework

### Scaling Patterns

- Performance scales with sqrt(d) for deep memory depth
- Capacity approximately quadratic in polynomial degree

---

## 11. Benchmark Results

### Language Modeling

- Outer optimizer: AdamW with LR 1e-4
- Evaluated on standard perplexity benchmarks

### Needle-in-a-Haystack (S-NIAH)

- Context length: 4K-128K tokens
- Atlas maintains >95% accuracy at 128K
- Outperforms Transformers at all lengths
- Outperforms Titans by ~15-20% at 64K+

### BABILong

- Context length: up to 10M tokens
- Window size: 4096
- Atlas achieves +80% accuracy improvement at 10M context
- Accuracy maintained above 90% with proper windowing

---

## 12. Key Theoretical Results

### Memory Capacity Hierarchy

1. **Linear matrix memory**: O(d_k) pairs
2. **Deep MLP memory (2+ layers)**: O(d_k * d_v) to O(d_k * d_v * product of hidden dims)
3. **With polynomial features degree p**: O(d_k^p) pairs

### Omega Rule Properties

- c=1: reduces to online/Delta rule (Titans)
- c=inf: global optimization (expensive, requires caching all tokens)
- Finite c: trades off capacity vs compute linearly

### Polar Express Convergence

For degree-5 polynomials after T iterations:
```
error <= |1 - l^2|^{3^T}
```

Converges super-exponentially (the exponent itself grows exponentially).

---

## 13. Connection to Other Models

| c | Optimizer | Memory | Model |
|---|-----------|--------|-------|
| 1 | GD | Linear | DeltaNet |
| 1 | GD | Deep | Titans (MAC) |
| >1 | GD | Linear | SWLA |
| >1 | GD | Deep | OmegaNet (DoT) |
| >1 | Muon (NS-5) | Deep | **Atlas** |
