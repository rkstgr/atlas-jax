"""Atlas memory layer: ShortConv + AtlasMemoryLayer.

Split out from model.py so the full-model file (MLP, Block, Atlas, init helpers)
stays readable. See model.py for the module docstring describing the overall
architecture.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx

from atlas_jax import kernels
from atlas_jax.config import AtlasConfig
from atlas_jax.ops import (
    rms_norm,
    _gelu_derivative,
    _omega_aggregate,
    linear_scan,
)
from atlas_jax.polar_express import polar_express, polar_express_ste
from atlas_jax.state import LinearMemoryState, DeepMemoryState


# ---------------------------------------------------------------------------
# ShortConv: causal depthwise 1D convolution
# ---------------------------------------------------------------------------

class ShortConv(eqx.Module):
    """Causal depthwise 1D convolution (per Titans / Based convention)."""
    weight: jax.Array   # (D, 1, K)
    bias: jax.Array     # (D,)
    kernel_size: int = eqx.field(static=True)

    def __init__(self, dim, kernel_size=4, *, key):
        self.kernel_size = kernel_size
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (dim, 1, kernel_size)) * 0.02
        self.bias = jnp.zeros(dim)

    def __call__(self, x):
        """x: (B, T, D) -> (B, T, D), causal conv over time dim."""
        B, T, D = x.shape
        # Transpose to (B, D, T) for conv
        x = jnp.transpose(x, (0, 2, 1))  # (B, D, T)
        # Causal left-padding
        x = jnp.pad(x, ((0, 0), (0, 0), (self.kernel_size - 1, 0)))
        # Depthwise conv1d: each channel independently
        # JAX conv_general_dilated with feature_group_count=D
        x = lax.conv_general_dilated(
            x,                                          # (B, D, T+K-1)
            self.weight.astype(x.dtype),                # (D, 1, K)
            window_strides=(1,),
            padding='VALID',
            dimension_numbers=('NCW', 'OIW', 'NCW'),   # batch=N, channel=C, spatial=W
            feature_group_count=D,
        )  # (B, D, T)
        x = x + self.bias.astype(x.dtype)[:, jnp.newaxis]
        return jnp.transpose(x, (0, 2, 1))  # (B, T, D)


# ---------------------------------------------------------------------------
# AtlasMemoryLayer
# ---------------------------------------------------------------------------

class AtlasMemoryLayer(eqx.Module):
    """Multi-head memory layer with Omega rule + Polar Express update.

    Paper formulation (per head, per token):
      1. Gradient:     u_t = 2 * d/dM ||M(phi(k_t)) - v_t||^2
      2. Omega agg:    G_t = sum_{i in window} gamma_i * u_i
      3. Momentum:     S_t = theta_t * S_{t-1} - eta_t * G_t
      4. Orthogonalize: S'_t = PolarExpress(S_t)
      5. Memory:       M_t = alpha_t * M_{t-1} + S'_t
      6. Output:       y_t = M_t(q_t)
    """
    # Projections
    c_q: eqx.nn.Linear
    c_k: eqx.nn.Linear
    c_v: eqx.nn.Linear
    c_proj: eqx.nn.Linear

    # Short causal convolutions
    conv_q: ShortConv
    conv_k: ShortConv
    conv_v: ShortConv

    # Gates
    gate_alpha: eqx.nn.Linear
    gate_eta: eqx.nn.Linear
    gate_theta: eqx.nn.Linear
    gate_gamma: eqx.nn.Linear | None

    # Polynomial feature mapping coefficients
    poly_coeffs: jax.Array | None

    # ResidualNorm gamma: LayerNorm(MLP(x)) * (gamma+1) + x (per-head, init=0)
    ln_gamma: jax.Array  # (H, D)

    # Learnable initial memory weights (Xavier init, like PyTorch)
    W1_init: jax.Array  # (H, D, E)
    W2_init: jax.Array  # (H, E, D)

    # Retrieve gate: per-head sigmoid gate on memory output (PyTorch: when heads > 1)
    retrieve_gate: eqx.nn.Linear | None

    # Static config
    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    expand_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    ns_steps: int = eqx.field(static=True)
    omega_window: int = eqx.field(static=True)
    poly_degree: int = eqx.field(static=True)
    deep_memory: bool = eqx.field(static=True)
    pe_ste: bool = eqx.field(static=True)
    use_checkpoint: bool = eqx.field(static=True)
    fused_chunk: bool = eqx.field(static=True)
    max_lr: float = eqx.field(static=True)
    stop_grad_chunks: bool = eqx.field(static=True)

    def __init__(self, config: AtlasConfig, *, key):
        keys = jax.random.split(key, 14)
        H = config.n_head
        D = config.dim_head if config.dim_head is not None else config.n_embd // H
        E = config.memory_expand * D if config.deep_memory else D
        dim_inner = H * D  # may differ from n_embd when dim_head is set

        self.n_head = H
        self.head_dim = D
        self.expand_dim = E
        self.chunk_size = config.chunk_size
        self.ns_steps = config.ns_steps
        self.omega_window = config.omega_window
        self.poly_degree = config.poly_degree
        self.deep_memory = config.deep_memory
        self.pe_ste = config.pe_ste
        self.use_checkpoint = config.use_checkpoint
        self.max_lr = config.max_lr
        self.stop_grad_chunks = config.stop_grad_chunks
        _d_is_pow2 = (D & (D - 1) == 0)
        _e_is_pow2 = (E & (E - 1) == 0)
        self.fused_chunk = config.fused_chunk and kernels.HAS_FUSED_CHUNK and _d_is_pow2 and _e_is_pow2
        if config.fused_chunk and kernels.HAS_FUSED_CHUNK and not (_d_is_pow2 and _e_is_pow2):
            import warnings
            warnings.warn(
                f"Fused chunk kernel disabled: D={D}, E={E} must both be powers of 2. "
                f"Falling back to regular scan ops."
            )

        C = config.n_embd
        # Q/K/V project from n_embd to dim_inner (= heads * dim_head)
        self.c_q = eqx.nn.Linear(C, dim_inner, use_bias=False, key=keys[0])
        self.c_k = eqx.nn.Linear(C, dim_inner, use_bias=False, key=keys[1])
        self.c_v = eqx.nn.Linear(C, dim_inner, use_bias=False, key=keys[2])
        # Output projection: dim_inner back to n_embd
        self.c_proj = eqx.nn.Linear(dim_inner, C, use_bias=False, key=keys[3])

        self.conv_q = ShortConv(dim_inner, config.conv_kernel, key=keys[4])
        self.conv_k = ShortConv(dim_inner, config.conv_kernel, key=keys[5])
        self.conv_v = ShortConv(dim_inner, config.conv_kernel, key=keys[6])

        self.gate_alpha = eqx.nn.Linear(C, H, use_bias=True, key=keys[7])
        self.gate_eta = eqx.nn.Linear(C, H, use_bias=True, key=keys[8])
        self.gate_theta = eqx.nn.Linear(C, H, use_bias=True, key=keys[9])

        if self.omega_window > 1:
            self.gate_gamma = eqx.nn.Linear(C, H, use_bias=True, key=keys[10])
        else:
            self.gate_gamma = None

        if self.poly_degree > 0:
            # All-ones coefficients: phi(x) = x + x^2 + ... + x^p
            # Matches PyTorch elementwise mode (no 1/i! Taylor scaling)
            coeffs = jnp.ones(self.poly_degree)
            self.poly_coeffs = coeffs
        else:
            self.poly_coeffs = None

        # ResidualNorm gamma (per head): LayerNorm(MLP(x)) * (gamma+1) + x
        # Init=0 matches PyTorch (gamma starts at 0, so initially scale=1)
        # But we also need the outer optimizer to learn gamma, so it stays as zeros
        self.ln_gamma = jnp.zeros((H, D))

        # Learnable initial memory weights (Xavier uniform, matching PyTorch)
        # PyTorch clones learned MLP params per batch*head at init. We store
        # per-head initial weights that get broadcast to (B, H, D, E).
        k_w1, k_w2 = jax.random.split(keys[11])
        bound1 = (6.0 / (D + E)) ** 0.5
        bound2 = (6.0 / (E + D)) ** 0.5
        self.W1_init = jax.random.uniform(k_w1, (H, D, E), minval=-bound1, maxval=bound1)
        self.W2_init = jax.random.uniform(k_w2, (H, E, D), minval=-bound2, maxval=bound2)

        # Retrieve gate: per-head sigmoid on memory output (PyTorch has this when heads > 1)
        if H > 1:
            self.retrieve_gate = eqx.nn.Linear(C, H, use_bias=False, key=keys[12])
        else:
            self.retrieve_gate = None

    def _poly_features(self, x):
        """Element-wise polynomial: phi(x) = sum_{i=1}^{p} a_i * x^i."""
        result = self.poly_coeffs[0] * x
        x_pow = x
        for i in range(1, self.poly_degree):
            x_pow = x_pow * x
            result = result + self.poly_coeffs[i] * x_pow
        return result

    def _process_chunk_linear(self, M, S, q_c, k_c, v_c, a_c, e_c, t_c, g_c):
        """Process one chunk with linear memory.

        Linear memory: M is a matrix, M(x) = M @ x (no residual/activation).
        Gradient: u_t = 2 * (M @ k_t - v_t) outer k_t^T
        """
        _pe = polar_express_ste if self.pe_ste else polar_express

        # Parallel: per-position gradients w.r.t. frozen M (f32 for bf16 stability)
        pred = jnp.einsum('bhvk,bchk->bchv', M, k_c)
        err = pred - v_c
        D = k_c.shape[-1]
        orig_dtype = err.dtype
        err_f = err.astype(jnp.float32)
        k_f = k_c.astype(jnp.float32)
        u = ((2.0 / D) * jnp.einsum('bchv,bchk->bchvk', err_f, k_f)).astype(orig_dtype)

        # Omega rule
        if self.omega_window > 1 and g_c is not None:
            u = _omega_aggregate(u, g_c[..., jnp.newaxis], self.omega_window)

        # Momentum scan: S_t = theta_t * S_{t-1} - eta_t * u_t
        theta = jnp.squeeze(t_c, axis=-1)         # (B, cs, H)
        mom_input = -(e_c[..., jnp.newaxis] * u)  # (B, cs, H, D, D)
        chunk_S, S = linear_scan(S, theta, mom_input)

        # Polar Express orthogonalization
        chunk_S_orth = _pe(chunk_S, self.ns_steps)

        # Memory scan: M_t = alpha_t * M_{t-1} + PE(S_t)
        alpha = jnp.squeeze(a_c, axis=-1)          # (B, cs, H)
        M_all, M = linear_scan(M, alpha, chunk_S_orth)

        # Output: y_t = M_t @ q_t
        y_c = jnp.einsum('bchvk,bchk->bchv', M_all, q_c)

        return y_c, LinearMemoryState(M=M, S=S)

    def _process_chunk_deep(self, state, q_c, k_c, v_c, a_c, e_c, t_c, g_c):
        """Process one chunk with deep MLP memory.

        Memory: M(x) = x + W1 @ GELU(W2 @ x) with residual connection.
        Gradients computed analytically w.r.t. W1 and W2 with 1/D scaling
        (matching PyTorch MSE mean(dim=-1)).

        NOTE: PyTorch wraps memory in ResidualNorm(LayerNorm + gamma). Our analytical
        approach uses plain residual. The ResidualNorm analytical gradients were
        verified correct but produce worse training dynamics, likely due to
        interaction with PE orthogonalization. Keeping plain residual for now.
        """
        W1, W2, S_W1, S_W2 = state.W1, state.W2, state.S_W1, state.S_W2
        if kernels.HAS_TRITON_PE:
            _pe = kernels.triton_polar_express_ste if self.pe_ste else kernels.triton_polar_express
        else:
            _pe = polar_express_ste if self.pe_ste else polar_express

        # Forward through frozen MLP memory
        with jax.named_scope("mem_fwd"):
            h = jnp.einsum('bhed,bchd->bche', W2, k_c)          # (B, cs, H, E)
            act = jax.nn.gelu(h)                                   # (B, cs, H, E)
            y_pred = k_c + jnp.einsum('bhde,bche->bchd', W1, act) # (B, cs, H, D)
            err = y_pred - v_c                                      # (B, cs, H, D)

        # Gradient computation in f32 for bf16 numerical stability
        with jax.named_scope("mem_grad"):
            orig_dtype = err.dtype
            err_f = err.astype(jnp.float32)
            act_f = act.astype(jnp.float32)
            D = k_c.shape[-1]
            scale = 2.0 / D
            u_W1 = (scale * jnp.einsum('bchd,bche->bchde', err_f, act_f)).astype(orig_dtype)
            # Gradient w.r.t. W2: chain through GELU' and W1^T
            gelu_prime = _gelu_derivative(h)                                     # f32 (always)
            w1t_err = jnp.einsum('bhde,bchd->bche', W1.astype(jnp.float32), err_f)
            chain = w1t_err * gelu_prime
            u_W2 = (scale * jnp.einsum('bche,bchd->bched', chain, k_c.astype(jnp.float32))).astype(orig_dtype)

        # Omega rule
        with jax.named_scope("omega"):
            if self.omega_window > 1 and g_c is not None:
                g_w1 = g_c[..., jnp.newaxis]  # broadcast for (B, cs, H, D, E)
                g_w2 = g_c[..., jnp.newaxis]  # broadcast for (B, cs, H, E, D)
                u_W1 = _omega_aggregate(u_W1, g_w1, self.omega_window)
                u_W2 = _omega_aggregate(u_W2, g_w2, self.omega_window)

        theta = jnp.squeeze(t_c, axis=-1)  # (B, cs, H)
        alpha = jnp.squeeze(a_c, axis=-1)  # (B, cs, H)

        # Fused momentum -> PE -> memory scan for both W1 and W2
        # Process both weight matrices with shared gates
        mom_W1 = -(e_c[..., jnp.newaxis] * u_W1)
        mom_W2 = -(e_c[..., jnp.newaxis] * u_W2)

        if self.fused_chunk:
            # FlashATLAS: single Triton kernel keeps carry in SRAM
            with jax.named_scope("flash_atlas"):
                y_c, new_state = kernels.fused_chunk_scan(
                    W1, W2, S_W1, S_W2,
                    mom_W1, mom_W2, theta, alpha, q_c,
                    self.ns_steps, self.pe_ste)
            return y_c, new_state

        # Batched path: when D==E, stack W1/W2 to halve kernel launches
        # (2 momentum scans -> 1, 2 PE calls -> 1, 2 memory scans -> 1)
        D, E = self.head_dim, self.expand_dim
        if D == E:
            with jax.named_scope("batched_momentum_scan"):
                # Stack: (B, H, D, D) x2 -> (B, H, 2, D, D) -> flatten to (B, H, 2*D*D)
                S_cat = jnp.concatenate(
                    [S_W1.reshape(*S_W1.shape[:2], -1),
                     S_W2.reshape(*S_W2.shape[:2], -1)], axis=-1)
                mom_cat = jnp.concatenate(
                    [mom_W1.reshape(*mom_W1.shape[:3], -1),
                     mom_W2.reshape(*mom_W2.shape[:3], -1)], axis=-1)
                chunk_S_cat, S_final_cat = linear_scan(S_cat, theta, mom_cat)
                DD = D * E
                chunk_S_W1 = chunk_S_cat[..., :DD].reshape(*chunk_S_cat.shape[:3], D, E)
                chunk_S_W2 = chunk_S_cat[..., DD:].reshape(*chunk_S_cat.shape[:3], E, D)
                S_W1 = S_final_cat[..., :DD].reshape(*S_final_cat.shape[:2], D, E)
                S_W2 = S_final_cat[..., DD:].reshape(*S_final_cat.shape[:2], E, D)

            with jax.named_scope("batched_polar_express"):
                # Stack along dim 3: (B, cs, H, D, D) x2 -> (B, cs, H, 2, D, D)
                # PE operates on last 2 dims, batch dims are transparent
                stacked_S = jnp.stack([chunk_S_W1, chunk_S_W2], axis=3)
                stacked_orth = _pe(stacked_S, self.ns_steps)
                chunk_S_W1_orth = stacked_orth[:, :, :, 0]
                chunk_S_W2_orth = stacked_orth[:, :, :, 1]

            with jax.named_scope("batched_memory_scan"):
                W_cat = jnp.concatenate(
                    [W1.reshape(*W1.shape[:2], -1),
                     W2.reshape(*W2.shape[:2], -1)], axis=-1)
                orth_cat = jnp.concatenate(
                    [chunk_S_W1_orth.reshape(*chunk_S_W1_orth.shape[:3], -1),
                     chunk_S_W2_orth.reshape(*chunk_S_W2_orth.shape[:3], -1)], axis=-1)
                W_all_cat, W_final_cat = linear_scan(W_cat, alpha, orth_cat)
                W1_all = W_all_cat[..., :DD].reshape(*W_all_cat.shape[:3], D, E)
                W2_all = W_all_cat[..., DD:].reshape(*W_all_cat.shape[:3], E, D)
                W1 = W_final_cat[..., :DD].reshape(*W_final_cat.shape[:2], D, E)
                W2 = W_final_cat[..., DD:].reshape(*W_final_cat.shape[:2], E, D)
        else:
            # Fallback: separate scan + PE + scan for each weight matrix
            def _fused_scan(S_init, W_init, theta, alpha, mom_input, name):
                with jax.named_scope(f"{name}/momentum_scan"):
                    chunk_S, S_final = linear_scan(S_init, theta, mom_input)
                with jax.named_scope(f"{name}/polar_express"):
                    chunk_S_orth = _pe(chunk_S, self.ns_steps)
                with jax.named_scope(f"{name}/memory_scan"):
                    W_all, W_final = linear_scan(W_init, alpha, chunk_S_orth)
                return W_all, W_final, S_final

            with jax.named_scope("fused_W1"):
                W1_all, W1, S_W1 = _fused_scan(S_W1, W1, theta, alpha, mom_W1, "W1")
            with jax.named_scope("fused_W2"):
                W2_all, W2, S_W2 = _fused_scan(S_W2, W2, theta, alpha, mom_W2, "W2")

        # Output: y_t = q_t + W1_t @ GELU(W2_t @ q_t)
        with jax.named_scope("mem_output"):
            h_q = jnp.einsum('bched,bchd->bche', W2_all, q_c)
            g_q = jax.nn.gelu(h_q)
            y_c = q_c + jnp.einsum('bchde,bche->bchd', W1_all, g_q)

        return y_c, DeepMemoryState(W1=W1, W2=W2, S_W1=S_W1, S_W2=S_W2)

    def __call__(self, x, memory_state=None):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim
        E = self.expand_dim
        cs = self.chunk_size
        dim_inner = H * D  # may differ from C when dim_head is set

        # Project (direct matmul, no vmap overhead) + short causal conv + multi-head reshape
        with jax.named_scope("qkv_proj"):
            q = self.conv_q((x @ self.c_q.weight.T).reshape(B, T, dim_inner)).reshape(B, T, H, D)
            k = self.conv_k((x @ self.c_k.weight.T).reshape(B, T, dim_inner)).reshape(B, T, H, D)
            v = self.conv_v((x @ self.c_v.weight.T).reshape(B, T, dim_inner)).reshape(B, T, H, D)

        # Polynomial features + RMS norm
        # PyTorch order: queries = norm(q) then poly(q), keys = poly(k) then norm(k)
        with jax.named_scope("qk_norm_poly"):
            if self.poly_degree > 0:
                q = rms_norm(q)
                q = self._poly_features(q)
                k = self._poly_features(k)
                k = rms_norm(k)
            else:
                q, k = rms_norm(q), rms_norm(k)

        # Input-dependent gates via sigmoid (direct matmul + bias)
        # Paper: bias initialized at -2, so sigmoid(-2) ≈ 0.12 initially
        with jax.named_scope("gates"):
            def _gate(layer, x):
                out = x @ layer.weight.T  # (B, T, H)
                if layer.bias is not None:
                    out = out + layer.bias
                return jax.nn.sigmoid(out.reshape(B, T, H, 1))

            alpha = _gate(self.gate_alpha, x)
            eta = _gate(self.gate_eta, x) * self.max_lr  # PyTorch: sigmoid * max_lr (default 0.1)
            theta = _gate(self.gate_theta, x)

            gamma = None
            if self.omega_window > 1 and self.gate_gamma is not None:
                gamma = _gate(self.gate_gamma, x)

        # Initialize memory state if needed
        if memory_state is None:
            if self.deep_memory:
                # Clone learned initial weights per batch (matching PyTorch's
                # repeat of memory_model_parameter_dict per batch*head)
                W1 = jnp.broadcast_to(
                    self.W1_init[jnp.newaxis].astype(x.dtype), (B, H, D, E))
                W2 = jnp.broadcast_to(
                    self.W2_init[jnp.newaxis].astype(x.dtype), (B, H, E, D))
                S_W1 = jnp.zeros((B, H, D, E), dtype=x.dtype)
                S_W2 = jnp.zeros((B, H, E, D), dtype=x.dtype)
                memory_state = DeepMemoryState(W1=W1, W2=W2, S_W1=S_W1, S_W2=S_W2)
            else:
                M = jnp.zeros((B, H, D, D), dtype=x.dtype)
                S = jnp.zeros((B, H, D, D), dtype=x.dtype)
                memory_state = LinearMemoryState(M=M, S=S)

        # Pad sequence to multiple of chunk_size
        T_orig = T
        if T % cs != 0:
            pad = cs - T % cs
            q = jnp.pad(q, ((0, 0), (0, pad), (0, 0), (0, 0)))
            k = jnp.pad(k, ((0, 0), (0, pad), (0, 0), (0, 0)))
            v = jnp.pad(v, ((0, 0), (0, pad), (0, 0), (0, 0)))
            # alpha=1 carries memory unchanged, eta=0 no update, theta=0 kill momentum
            alpha = jnp.pad(alpha, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=1.0)
            eta = jnp.pad(eta, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=0.0)
            theta = jnp.pad(theta, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=0.0)
            if gamma is not None:
                gamma = jnp.pad(gamma, ((0, 0), (0, pad), (0, 0), (0, 0)), constant_values=0.0)
            T = q.shape[1]

        n_chunks = T // cs

        def chunk_body(carry, chunk_data):
            # NOTE: PyTorch uses full gradient flow across chunks via associative scan.
            # Paper specifies stop_gradient (frozen boundary).
            # stop_grad_chunks=True -> paper mode, False -> PyTorch mode.
            mem_state = jax.lax.stop_gradient(carry) if self.stop_grad_chunks else carry
            q_c, k_c, v_c, a_c, e_c, t_c, g_c = chunk_data
            if self.deep_memory:
                y_c, new_state = self._process_chunk_deep(
                    mem_state, q_c, k_c, v_c, a_c, e_c, t_c, g_c)
            else:
                y_c, new_state = self._process_chunk_linear(
                    mem_state.M, mem_state.S, q_c, k_c, v_c, a_c, e_c, t_c, g_c)
            return new_state, y_c

        process_fn = jax.checkpoint(chunk_body) if self.use_checkpoint else chunk_body

        if self.fused_chunk:
            # Unrolled Python for-loop with transpose-free chunking.
            # Keep (B, n_chunks, cs, ...) layout and index with [:, i] —
            # avoids the (n_chunks, B, cs, ...) transpose that costs ~255ms.
            def _chunk_bt(x):
                return x.reshape(B, n_chunks, cs, *x.shape[2:])

            q_bt = _chunk_bt(q)
            k_bt = _chunk_bt(k)
            v_bt = _chunk_bt(v)
            a_bt = _chunk_bt(alpha)
            e_bt = _chunk_bt(eta)
            t_bt = _chunk_bt(theta)
            g_bt = _chunk_bt(gamma) if gamma is not None else jnp.zeros_like(a_bt)

            ys = []
            state = memory_state
            for i in range(n_chunks):
                chunk_data = (q_bt[:, i], k_bt[:, i], v_bt[:, i],
                              a_bt[:, i], e_bt[:, i], t_bt[:, i], g_bt[:, i])
                state, y_c = process_fn(state, chunk_data)
                ys.append(y_c)
            final_state = state
            all_y = jnp.stack(ys, axis=1)  # (B, n_chunks, cs, H, D) — no transpose
        else:
            # lax.scan path: needs (n_chunks, B, cs, ...) for the scan axis
            def _chunk(x):
                return x.reshape(B, n_chunks, cs, *x.shape[2:]).transpose(1, 0, 2, *range(3, x.ndim + 1))

            q_chunks = _chunk(q)
            k_chunks = _chunk(k)
            v_chunks = _chunk(v)
            a_chunks = _chunk(alpha)
            e_chunks = _chunk(eta)
            t_chunks = _chunk(theta)
            g_chunks = _chunk(gamma) if gamma is not None else jnp.zeros_like(a_chunks)

            xs = (q_chunks, k_chunks, v_chunks, a_chunks, e_chunks, t_chunks, g_chunks)
            final_state, all_y = lax.scan(process_fn, memory_state, xs)

        # all_y -> (B, T, H, D)
        if self.fused_chunk:
            # (B, n_chunks, cs, H, D) — already B-first, just reshape
            y = all_y.reshape(B, T, H, D)
        else:
            # (n_chunks, B, cs, H, D) — needs transpose
            y = jnp.transpose(all_y, (1, 0, 2, 3, 4)).reshape(B, T, H, D)

        # Trim padding
        y = y[:, :T_orig]  # (B, T_orig, H, D)

        # Retrieve gate: per-head sigmoid gating (PyTorch: when heads > 1)
        if self.retrieve_gate is not None:
            gate = jax.nn.sigmoid(x @ self.retrieve_gate.weight.T)  # (B, T_orig, H)
            y = y * gate[..., jnp.newaxis]  # (B, T_orig, H, D) * (B, T_orig, H, 1)

        # Project back to residual stream
        y = y.reshape(B, T_orig, -1)
        y = (y @ self.c_proj.weight.T).reshape(B, T_orig, C)

        return y, final_state
