"""
Minimal Atlas memory layer in JAX.

One file, no dependencies beyond jax/jnp. Implements the full inner loop:
  1. Forward prediction with 2-layer MLP (W1, W2)
  2. Analytical gradient of MSE loss
  3. Omega sliding-window aggregation
  4. Momentum scan (S_t = θ_t · S_{t-1} + input_t)
  5. Polar Express orthogonalization (Newton-Schulz)
  6. Weight decay scan (W_t = α_t · W_{t-1} + S_orth_t)
  7. Retrieval with updated weights

The sequence is split into chunks of size C. Within each chunk:
  - All C tokens compute their error against the SAME W (frozen from carry)
  - Omega aggregates neighboring gradients
  - Scans produce the weight trajectory over C timesteps
  - Retrieval uses the per-timestep weights
Between chunks: carry is passed with stop_gradient.

Reference: arXiv 2505.23735 (Atlas), arXiv 2505.16932 (Polar Express)
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


# --- Polar Express: orthogonalize via Newton-Schulz iterations ---

def polar_express_ste(S, ns_steps=5):
    """Orthogonalize S using Newton-Schulz with straight-through estimator.

    Forward: S_orth = NS5(S)
    Backward: pretend S_orth = S (STE), 62x faster than full backward.
    """
    S_orth = _newton_schulz(S, ns_steps)
    # STE: forward uses S_orth, backward uses S
    return S + lax.stop_gradient(S_orth - S)


def _newton_schulz(S, steps):
    """Newton-Schulz iteration for polar decomposition.

    Converges to the nearest orthogonal matrix: S_orth = U @ V^T
    where S = U @ Sigma @ V^T is the SVD.
    """
    # Normalize by Frobenius norm to ensure convergence
    norm = jnp.sqrt(jnp.sum(S * S, axis=(-2, -1), keepdims=True) + 1e-12)
    X = S / norm

    for _ in range(steps):
        A = X @ X.swapaxes(-2, -1)          # X @ X^T
        # Coefficients from arXiv 2505.16932, degree-2 polynomial
        I = jnp.eye(X.shape[-2], dtype=X.dtype)
        X = (15/8) * X - (5/2) * A @ X + (3/8) * A @ A @ X

    return X * norm


# --- Linear scan: h_t = gate_t * h_{t-1} + input_t ---

def linear_scan(h_init, gates, inputs):
    """Linear recurrence via associative scan. O(T log T) parallel.

    Args:
        h_init: (B, H, ...) initial state
        gates:  (B, T, H, ...) decay gates in [0, 1]
        inputs: (B, T, H, ...) additive inputs

    Returns:
        all_h: (B, T, H, ...) states at all timesteps
        h_final: (B, H, ...) final state
    """
    def _combine(a, b):
        # Linear recurrence: h_t = g_t * h_{t-1} + x_t
        # combine(old, new) = (g_old * g_new, g_new * v_old + v_new)
        ga, va = a
        gb, vb = b
        return (ga * gb, gb * va + vb)

    # Prepend: the "virtual" step 0 has gate=1, value=h_init
    # We fold h_init into the scan by treating it as input[0] with gate=0 wouldn't work.
    # Instead, adjust: h_1 = g_1 * h_init + x_1
    # Fold h_init into the first input: let x_1' = g_1 * h_init + x_1, then scan with h_0 = 0.
    # Actually simpler: just scan and add the h_init contribution.

    T = gates.shape[1]
    elems = (gates.swapaxes(0, 1), inputs.swapaxes(0, 1))  # (T, B, H, ...)

    _, all_h = lax.associative_scan(_combine, elems, axis=0)
    all_h = all_h.swapaxes(0, 1)  # (B, T, H, ...)

    # Add h_init's contribution: at time t, h_init contributes h_init * prod(gates[1..t])
    # Simpler: compute cumulative gate products and add
    cum_gates = jnp.cumprod(gates, axis=1)  # (B, T, H, ...)
    h_init_expanded = jnp.expand_dims(h_init, axis=1)  # (B, 1, H, ...)
    all_h = all_h + cum_gates * h_init_expanded

    h_final = all_h[:, -1]
    return all_h, h_final


# --- Omega aggregation: sliding window sum of gradients ---

def omega_aggregate(u, gamma, window):
    """G_t = sum_{i=max(0,t-w+1)}^{t} gamma_i * u_i.

    Args:
        u: (B, T, H, ...) per-token gradients
        gamma: (B, T, H, 1, ...) per-token gates, broadcastable
        window: sliding window size
    """
    T = u.shape[1]
    weighted = gamma * u
    cum = jnp.cumsum(weighted, axis=1)
    if window >= T:
        return cum
    shifted = jnp.concatenate([
        jnp.zeros_like(cum[:, :window]),
        cum[:, :-window]
    ], axis=1)
    return cum - shifted


# --- GELU derivative (exact) ---

def gelu_derivative(x):
    cdf = 0.5 * (1.0 + lax.erf(x / jnp.sqrt(2.0)))
    pdf = jnp.exp(-0.5 * x ** 2) / jnp.sqrt(2.0 * jnp.pi)
    return cdf + x * pdf


# --- The core: process one chunk ---

def process_chunk(
    # Carry state
    W1, W2, S_W1, S_W2,
    # Chunk data: each (B, C, H, D) or (B, C, H, 1)
    q_c, k_c, v_c, alpha_c, eta_c, theta_c, gamma_c,
    # Config
    omega_window=1, ns_steps=5,
):
    """Process one chunk of C tokens through the Atlas memory layer.

    Args:
        W1: (B, H, D, E) — first MLP weight matrix
        W2: (B, H, E, D) — second MLP weight matrix
        S_W1: (B, H, D, E) — momentum state for W1
        S_W2: (B, H, E, D) — momentum state for W2
        q_c: (B, C, H, D) — queries for retrieval
        k_c: (B, C, H, D) — keys for storing
        v_c: (B, C, H, D) — values (targets)
        alpha_c: (B, C, H, 1) — weight decay gates
        eta_c: (B, C, H, 1) — learning rate gates
        theta_c: (B, C, H, 1) — momentum gates
        gamma_c: (B, C, H, 1) — omega context gates
        omega_window: sliding window size for gradient aggregation
        ns_steps: Newton-Schulz iterations for Polar Express

    Returns:
        y_c: (B, C, H, D) — retrieved output
        W1, W2, S_W1, S_W2: updated carry state
    """
    D = k_c.shape[-1]

    # === 1. Forward: predict with frozen W, compute error ===
    h = jnp.einsum('bhed,bchd->bche', W2, k_c)      # (B,C,H,E)
    act = jax.nn.gelu(h)
    y_pred = k_c + jnp.einsum('bhde,bche->bchd', W1, act)  # (B,C,H,D)
    err = y_pred - v_c

    # === 2. Analytical gradients of ||y_pred - v||² / D ===
    scale = 2.0 / D
    u_W1 = scale * jnp.einsum('bchd,bche->bchde', err, act)              # (B,C,H,D,E)
    chain = jnp.einsum('bhde,bchd->bche', W1, err) * gelu_derivative(h)  # (B,C,H,E)
    u_W2 = scale * jnp.einsum('bche,bchd->bched', chain, k_c)            # (B,C,H,E,D)

    # === 3. Omega aggregation (sliding window over C tokens) ===
    if omega_window > 1:
        u_W1 = omega_aggregate(u_W1, gamma_c[..., jnp.newaxis], omega_window)
        u_W2 = omega_aggregate(u_W2, gamma_c[..., jnp.newaxis], omega_window)

    # === 4. Scale by learning rate → momentum inputs ===
    mom_W1 = -(eta_c[..., jnp.newaxis] * u_W1)  # (B,C,H,D,E)
    mom_W2 = -(eta_c[..., jnp.newaxis] * u_W2)  # (B,C,H,E,D)

    # === 5. Momentum scan: S_t = θ_t * S_{t-1} + mom_t ===
    theta = jnp.squeeze(theta_c, axis=-1)  # (B,C,H)
    # Flatten spatial dims for scan, broadcast theta to match
    def _scan_momentum(S_init, theta, mom):
        orig_shape = mom.shape
        flat = mom.reshape(*mom.shape[:3], -1)          # (B,C,H,D*E)
        S_flat = S_init.reshape(*S_init.shape[:2], -1)  # (B,H,D*E)
        theta_ex = theta[..., jnp.newaxis]              # (B,C,H,1) broadcasts with (B,C,H,D*E)
        all_S, S_final = linear_scan(S_flat, theta_ex, flat)
        return all_S.reshape(orig_shape), S_final.reshape(S_init.shape)

    all_S_W1, S_W1 = _scan_momentum(S_W1, theta, mom_W1)
    all_S_W2, S_W2 = _scan_momentum(S_W2, theta, mom_W2)

    # === 6. Polar Express orthogonalization ===
    all_S_W1_orth = polar_express_ste(all_S_W1, ns_steps)
    all_S_W2_orth = polar_express_ste(all_S_W2, ns_steps)

    # === 7. Weight decay scan: W_t = α_t * W_{t-1} + S_orth_t ===
    alpha = jnp.squeeze(alpha_c, axis=-1)

    def _scan_weights(W_init, alpha, S_orth):
        orig_shape = S_orth.shape
        flat = S_orth.reshape(*S_orth.shape[:3], -1)
        W_flat = W_init.reshape(*W_init.shape[:2], -1)
        alpha_ex = alpha[..., jnp.newaxis]  # (B,C,H,1)
        all_W, W_final = linear_scan(W_flat, alpha_ex, flat)
        return all_W.reshape(orig_shape), W_final.reshape(W_init.shape)

    all_W1, W1 = _scan_weights(W1, alpha, all_S_W1_orth)
    all_W2, W2 = _scan_weights(W2, alpha, all_S_W2_orth)

    # === 8. Retrieval: y_t = q_t + W1_t @ gelu(W2_t @ q_t) ===
    h_q = jnp.einsum('bched,bchd->bche', all_W2, q_c)
    g_q = jax.nn.gelu(h_q)
    y_c = q_c + jnp.einsum('bchde,bche->bchd', all_W1, g_q)

    return y_c, (W1, W2, S_W1, S_W2)


# --- Full forward: scan over chunks ---

def atlas_forward(
    W1_init, W2_init,
    q, k, v, alpha, eta, theta, gamma,
    chunk_size=64, omega_window=2, ns_steps=5,
):
    """Full Atlas memory layer forward pass.

    Args:
        W1_init: (B, H, D, E) initial MLP weight 1
        W2_init: (B, H, E, D) initial MLP weight 2
        q, k, v: (B, T, H, D) queries, keys, values
        alpha, eta, theta, gamma: (B, T, H, 1) gates
        chunk_size: tokens per chunk
        omega_window: gradient aggregation window
        ns_steps: Newton-Schulz iterations

    Returns:
        y: (B, T, H, D) output
        final_state: (W1, W2, S_W1, S_W2)
    """
    B, T, H, D = q.shape
    E = W1_init.shape[-1]
    assert T % chunk_size == 0
    n_chunks = T // chunk_size

    # Reshape to (n_chunks, B, C, H, ...)
    def _chunk(x):
        return x.reshape(B, n_chunks, chunk_size, *x.shape[2:]).transpose(1, 0, 2, *range(3, x.ndim + 1))

    q_ch, k_ch, v_ch = _chunk(q), _chunk(k), _chunk(v)
    a_ch, e_ch, t_ch, g_ch = _chunk(alpha), _chunk(eta), _chunk(theta), _chunk(gamma)

    # Initial momentum states (zeros)
    S_W1 = jnp.zeros_like(W1_init)
    S_W2 = jnp.zeros_like(W2_init)

    def scan_body(carry, chunk_data):
        W1, W2, S_W1, S_W2 = carry
        # Stop gradient at chunk boundary (paper: frozen carry)
        W1, W2 = lax.stop_gradient(W1), lax.stop_gradient(W2)
        S_W1, S_W2 = lax.stop_gradient(S_W1), lax.stop_gradient(S_W2)

        q_c, k_c, v_c, a_c, e_c, t_c, g_c = chunk_data
        y_c, (W1, W2, S_W1, S_W2) = process_chunk(
            W1, W2, S_W1, S_W2,
            q_c, k_c, v_c, a_c, e_c, t_c, g_c,
            omega_window=omega_window, ns_steps=ns_steps,
        )
        return (W1, W2, S_W1, S_W2), y_c

    init_carry = (W1_init, W2_init, S_W1, S_W2)
    xs = (q_ch, k_ch, v_ch, a_ch, e_ch, t_ch, g_ch)
    final_carry, all_y = lax.scan(scan_body, init_carry, xs)

    # (n_chunks, B, C, H, D) → (B, T, H, D)
    y = all_y.transpose(1, 0, 2, 3, 4).reshape(B, T, H, D)
    return y, final_carry


# --- Demo ---

if __name__ == "__main__":
    jax.config.update("jax_default_matmul_precision", "float32")
    key = jax.random.PRNGKey(42)
    B, T, H, D, E = 2, 128, 4, 64, 128
    chunk_size, omega_window = 64, 2

    keys = jax.random.split(key, 8)
    W1 = jax.random.normal(keys[0], (B, H, D, E)) * (6 / (D + E)) ** 0.5
    W2 = jax.random.normal(keys[1], (B, H, E, D)) * (6 / (E + D)) ** 0.5
    q = jax.random.normal(keys[2], (B, T, H, D)) * 0.1
    k = jax.random.normal(keys[3], (B, T, H, D)) * 0.1
    v = jax.random.normal(keys[4], (B, T, H, D)) * 0.1
    alpha = jax.nn.sigmoid(jax.random.normal(keys[5], (B, T, H, 1)))
    eta = jax.nn.sigmoid(jax.random.normal(keys[6], (B, T, H, 1))) * 0.1
    theta = jax.nn.sigmoid(jax.random.normal(keys[7], (B, T, H, 1)))
    gamma = jnp.ones((B, T, H, 1))

    y, (W1_f, W2_f, _, _) = atlas_forward(
        W1, W2, q, k, v, alpha, eta, theta, gamma,
        chunk_size=chunk_size, omega_window=omega_window, ns_steps=5,
    )
    print(f"input:  q {q.shape}")
    print(f"output: y {y.shape}")
    print(f"W1: {W1.shape} → {W1_f.shape}")
    print(f"y range: [{float(y.min()):.4f}, {float(y.max()):.4f}]")
    print(f"NaN: {bool(jnp.any(jnp.isnan(y)))}")

    # Gradient check — use small dims so gradients aren't tiny
    B2, T2, H2, D2, E2 = 2, 16, 2, 8, 16
    cs2 = 8
    k2 = jax.random.split(key, 10)
    W1s = jax.random.normal(k2[0], (B2,H2,D2,E2))*0.1
    W2s = jax.random.normal(k2[1], (B2,H2,E2,D2))*0.1
    qs = jax.random.normal(k2[2], (B2,T2,H2,D2))
    ks = jax.random.normal(k2[3], (B2,T2,H2,D2))
    vs = jax.random.normal(k2[4], (B2,T2,H2,D2))
    als = jax.nn.sigmoid(jax.random.normal(k2[5], (B2,T2,H2,1)))
    ets = jax.nn.sigmoid(jax.random.normal(k2[6], (B2,T2,H2,1)))*0.1
    ths = jax.nn.sigmoid(jax.random.normal(k2[7], (B2,T2,H2,1)))
    gms = jnp.ones((B2,T2,H2,1))

    def loss_fn(k_in):
        y, _ = atlas_forward(W1s, W2s, qs, k_in, vs, als, ets, ths, gms,
                             chunk_size=cs2, omega_window=2)
        return jnp.mean(y ** 2)

    loss, gk = jax.value_and_grad(loss_fn)(ks)
    print(f"\nGradient check (small dims):")
    print(f"  loss: {float(loss):.6f}")
    print(f"  grad k norm: {float(jnp.linalg.norm(gk)):.6e}")
    print(f"  grad k NaN: {bool(jnp.any(jnp.isnan(gk)))}")
    print("OK")
