"""Microbenchmark: isolate PE and chunk-body costs.

Measures each component of _process_chunk_deep separately to find
exactly where the 4.4s step time goes.
"""

import time
import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial

from atlas_jax.config import AtlasConfig
from atlas_jax.model import AtlasMemoryLayer, linear_scan, rms_norm, _gelu_derivative, _omega_aggregate
from atlas_jax.memory_layer import polar_express, polar_express_ste, POLAR_EXPRESS_COEFFS
from atlas_jax.memory_layer import DeepMemoryState


def time_fn(fn, *args, warmup=3, repeats=10):
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return min(times) * 1000, out  # ms


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--matmul-precision", default="high")
    args = parser.parse_args()
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    # Match the H100 profile config
    B, T, H, D, E = 32, 2048, 8, 56, 56
    cs = 256
    n_chunks = T // cs
    ns_steps = 3
    C = H * D  # 448

    print(f"Config: B={B}, T={T}, H={H}, D={D}, E={E}, cs={cs}, ns_steps={ns_steps}")
    print(f"Precision: {args.matmul_precision}")
    print(f"Chunks per layer: {n_chunks}, Layers: 8")
    print(f"Batch of matrices per PE call: {B * cs * H} = {B*cs*H}")
    print("=" * 70)

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 20)

    # --- Component tensors (bf16 like training) ---
    x = jax.random.normal(keys[0], (B, cs, C), dtype=jnp.bfloat16) * 0.1
    W1 = jax.random.normal(keys[1], (B, H, D, E), dtype=jnp.bfloat16) * 0.1
    W2 = jax.random.normal(keys[2], (B, H, E, D), dtype=jnp.bfloat16) * 0.1
    SW1 = jnp.zeros((B, H, D, E), dtype=jnp.bfloat16)
    SW2 = jnp.zeros((B, H, E, D), dtype=jnp.bfloat16)
    k_c = jax.random.normal(keys[3], (B, cs, H, D), dtype=jnp.bfloat16) * 0.1
    v_c = jax.random.normal(keys[4], (B, cs, H, D), dtype=jnp.bfloat16) * 0.1
    q_c = jax.random.normal(keys[5], (B, cs, H, D), dtype=jnp.bfloat16) * 0.1
    theta = jnp.full((B, cs, H), 0.9, dtype=jnp.bfloat16)
    alpha = jnp.full((B, cs, H), 0.95, dtype=jnp.bfloat16)
    eta = jnp.full((B, cs, H, 1), 0.1, dtype=jnp.bfloat16)
    mom_W1 = jax.random.normal(keys[6], (B, cs, H, D, E), dtype=jnp.bfloat16) * 0.01
    mom_W2 = jax.random.normal(keys[7], (B, cs, H, E, D), dtype=jnp.bfloat16) * 0.01

    # For PE: chunk_S shape = (B, cs, H, D, E) — the momentum after scan
    chunk_S = jax.random.normal(keys[8], (B, cs, H, D, E), dtype=jnp.bfloat16) * 0.01

    # --- Projection weights (for the q/k/v matmul) ---
    W_proj = jax.random.normal(keys[9], (C, C), dtype=jnp.bfloat16) * 0.02

    # =========================================================
    # 1. PE forward only
    # =========================================================
    @jax.jit
    def pe_fwd(X):
        return polar_express(X, ns_steps)

    @jax.jit
    def pe_ste_fwd(X):
        return polar_express_ste(X, ns_steps)

    ms, _ = time_fn(pe_fwd, chunk_S)
    print(f"PE forward (3 NS steps, {B*cs*H} matrices of {D}x{E}):")
    print(f"  polar_express:       {ms:8.2f} ms")

    ms, _ = time_fn(pe_ste_fwd, chunk_S)
    print(f"  polar_express_ste:   {ms:8.2f} ms")

    # PE fwd+bwd (what happens in backward recompute)
    @jax.jit
    def pe_ste_fwd_bwd(X):
        return jax.grad(lambda X: jnp.sum(polar_express_ste(X, ns_steps)))(X)

    ms, _ = time_fn(pe_ste_fwd_bwd, chunk_S)
    print(f"  PE STE fwd+bwd:      {ms:8.2f} ms")

    # Triton fused PE
    try:
        from atlas_jax.kernels.triton_pe import triton_polar_express, triton_polar_express_ste

        @jax.jit
        def triton_pe_fwd(X):
            return triton_polar_express(X, ns_steps)

        @jax.jit
        def triton_pe_ste_fwd(X):
            return triton_polar_express_ste(X, ns_steps)

        @jax.jit
        def triton_pe_ste_fwd_bwd(X):
            return jax.grad(lambda X: jnp.sum(triton_polar_express_ste(X, ns_steps)))(X)

        ms, _ = time_fn(triton_pe_fwd, chunk_S)
        print(f"  TRITON PE fwd:       {ms:8.2f} ms")
        ms, _ = time_fn(triton_pe_ste_fwd, chunk_S)
        print(f"  TRITON PE STE fwd:   {ms:8.2f} ms")
        ms, _ = time_fn(triton_pe_ste_fwd_bwd, chunk_S)
        print(f"  TRITON PE STE f+b:   {ms:8.2f} ms")
    except Exception as e:
        print(f"  TRITON PE: FAILED ({e})")

    # =========================================================
    # 2. Linear scan (Triton)
    # =========================================================
    @jax.jit
    def scan_fwd(h, g, inp):
        return linear_scan(h, g, inp)

    ms, _ = time_fn(scan_fwd, SW1, theta, mom_W1)
    print(f"\nLinear scan (Triton, cs={cs}, state {D}x{E}):")
    print(f"  scan forward:        {ms:8.2f} ms")

    @jax.jit
    def scan_fwd_bwd(h, g, inp):
        return jax.grad(lambda h, g, i: jnp.sum(linear_scan(h, g, i)[0]),
                        argnums=(0, 1, 2))(h, g, inp)

    ms, _ = time_fn(scan_fwd_bwd, SW1, theta, mom_W1)
    print(f"  scan fwd+bwd:        {ms:8.2f} ms")

    # =========================================================
    # 3. Memory MLP forward (einsums on frozen M)
    # =========================================================
    @jax.jit
    def mem_fwd(W1, W2, k_c, v_c):
        h = jnp.einsum('bhed,bchd->bche', W2, k_c)
        act = jax.nn.gelu(h)
        y_pred = k_c + jnp.einsum('bhde,bche->bchd', W1, act)
        err = y_pred - v_c
        return err, act, h

    ms, _ = time_fn(mem_fwd, W1, W2, k_c, v_c)
    print(f"\nMemory MLP forward (einsums on frozen M):")
    print(f"  mem_fwd:             {ms:8.2f} ms")

    # =========================================================
    # 4. Gradient computation (analytical einsums)
    # =========================================================
    err, act, h = mem_fwd(W1, W2, k_c, v_c)

    @jax.jit
    def grad_compute(W1, err, act, h, k_c):
        u_W1 = 2.0 * jnp.einsum('bchd,bche->bchde', err, act)
        gelu_prime = _gelu_derivative(h)
        w1t_err = jnp.einsum('bhde,bchd->bche', W1, err)
        chain = w1t_err * gelu_prime
        u_W2 = 2.0 * jnp.einsum('bche,bchd->bched', chain, k_c)
        return u_W1, u_W2

    ms, _ = time_fn(grad_compute, W1, err, act, h, k_c)
    print(f"\nGradient computation (analytical einsums):")
    print(f"  grad_compute:        {ms:8.2f} ms")

    # =========================================================
    # 5. Output retrieval (einsums with per-token M)
    # =========================================================
    W1_all = jax.random.normal(keys[10], (B, cs, H, D, E), dtype=jnp.bfloat16) * 0.01
    W2_all = jax.random.normal(keys[11], (B, cs, H, E, D), dtype=jnp.bfloat16) * 0.01

    @jax.jit
    def output_retrieval(W1_all, W2_all, q_c):
        h_q = jnp.einsum('bched,bchd->bche', W2_all, q_c)
        g_q = jax.nn.gelu(h_q)
        y_c = q_c + jnp.einsum('bchde,bche->bchd', W1_all, g_q)
        return y_c

    ms, _ = time_fn(output_retrieval, W1_all, W2_all, q_c)
    print(f"\nOutput retrieval (einsums with per-token M):")
    print(f"  output_retrieval:    {ms:8.2f} ms")

    # =========================================================
    # 6. Projection matmuls (q/k/v/gates — "the fast 1%")
    # =========================================================
    x_flat = jax.random.normal(keys[12], (B * T, C), dtype=jnp.bfloat16) * 0.1

    @jax.jit
    def proj_matmul(x, W):
        return x @ W.T

    ms, _ = time_fn(proj_matmul, x_flat, W_proj)
    print(f"\nProjection matmul ({B*T}x{C} @ {C}x{C}):")
    print(f"  single proj:         {ms:8.2f} ms")
    print(f"  x8 (q,k,v,4 gates): {ms*8:8.2f} ms")

    # =========================================================
    # 7. Full chunk body fwd+bwd (one chunk, one layer)
    # =========================================================
    config = AtlasConfig(
        sequence_len=T, n_layer=1, n_head=H, n_embd=C,
        chunk_size=cs, omega_window=16, poly_degree=3,
        deep_memory=True, memory_expand=1, ns_steps=ns_steps,
        pe_ste=True, use_checkpoint=False, fused_chunk=False,
    )
    layer = AtlasMemoryLayer(config, key=keys[13])
    def _to_bf16(x):
        return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
    layer = jax.tree.map(_to_bf16, layer, is_leaf=eqx.is_array)
    x_layer = jax.random.normal(keys[14], (B, T, C), dtype=jnp.bfloat16)

    @eqx.filter_jit
    def layer_fwd_bwd(layer, x):
        return eqx.filter_value_and_grad(lambda l, x: jnp.sum(l(x)[0]))(layer, x)

    ms, _ = time_fn(layer_fwd_bwd, layer, x_layer)
    print(f"\nFull memory layer fwd+bwd (1 layer, {n_chunks} chunks):")
    print(f"  layer_fwd_bwd:       {ms:8.2f} ms")
    print(f"  x8 layers:           {ms*8:8.2f} ms")

    # =========================================================
    # Summary: reconstruct the step time
    # =========================================================
    print("\n" + "=" * 70)
    print("COST MODEL (per step, 8 layers, estimated)")
    print("=" * 70)
    # Note: these are isolated component times; real step has overlap/fusion


if __name__ == "__main__":
    main()
