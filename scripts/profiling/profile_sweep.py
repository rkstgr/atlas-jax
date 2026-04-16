"""Sweep key knobs to find path to >10% MFU.

Tests: dtype, checkpointing, chunk_size, batch_size on a single block fwd+bwd.
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Block


def time_fn(fn, *args, warmup=2, repeats=5, **kwargs):
    for _ in range(warmup):
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return min(times), out


def profile_config(B, T, n_embd, n_head, chunk_size, ns_steps, memory_expand,
                   deep_memory, pe_ste, omega_window, use_checkpoint, dtype_str,
                   matmul_precision):
    jax.config.update("jax_default_matmul_precision", matmul_precision)

    H = n_head
    D = n_embd // H
    E = memory_expand * D if deep_memory else D
    n_chunks = T // chunk_size

    config = AtlasConfig(
        sequence_len=T, n_layer=1, n_head=n_head, n_embd=n_embd,
        chunk_size=chunk_size, ns_steps=ns_steps, omega_window=omega_window,
        poly_degree=3, deep_memory=deep_memory, memory_expand=memory_expand,
        pe_ste=pe_ste, use_checkpoint=use_checkpoint,
    )

    key = jax.random.PRNGKey(0)
    block = Block(config, key=key)

    if dtype_str == 'bf16':
        def to_bf16(x):
            return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
        block = jax.tree.map(to_bf16, block, is_leaf=eqx.is_array)
        x = jax.random.normal(key, (B, T, n_embd), dtype=jnp.bfloat16)
    else:
        x = jax.random.normal(key, (B, T, n_embd))

    @eqx.filter_jit
    def block_fwd_bwd(block, x):
        def f(block, x):
            y, _ = block(x)
            return jnp.sum(y)
        return eqx.filter_value_and_grad(f)(block, x)

    try:
        t, _ = time_fn(block_fwd_bwd, block, x)
        tokens = B * T
        tps = tokens / t

        # Approximate FLOPs for this config (1 block)
        if deep_memory:
            elem_flops = H * (D * E + E * D) * 5
            ns_flops = 2 * 3 * ns_steps * 2 * H * max(D, E) ** 3
        else:
            elem_flops = H * D * D * 5
            ns_flops = 3 * ns_steps * 2 * H * D * D * D

        n_params_block = 4 * n_embd * n_embd + 4 * n_embd * n_embd  # QKV+proj + MLP
        flops_per_token = 6 * n_params_block + elem_flops + ns_flops
        mfu = (flops_per_token * tps) / (989.4e12) * 100

        return t, tps, mfu
    except Exception as e:
        return None, None, None


def main():
    print("Profiling single-block fwd+bwd on H100 (with associative scan)")
    print("=" * 100)
    print(f"{'Config':<60} {'ms':>8} {'tok/s':>10} {'MFU%':>8}")
    print("-" * 100)

    configs = [
        # (label, B, T, embd, heads, cs, ns, expand, deep, ste, omega, chk, dtype, precision)
        # Core comparisons
        ("bf16 chk B=8",                  8, 2048, 448, 8, 64,  3, 1, True,  True,  16, True,  'bf16', 'float32'),
        ("bf16 chk TF32 B=8",            8, 2048, 448, 8, 64,  3, 1, True,  True,  16, True,  'bf16', 'high'),
        ("bf16 chk TF32 B=16",          16, 2048, 448, 8, 64,  3, 1, True,  True,  16, True,  'bf16', 'high'),
        ("bf16 chk TF32 B=32",          32, 2048, 448, 8, 64,  3, 1, True,  True,  16, True,  'bf16', 'high'),
        ("bf16 chk TF32 B=64",          64, 2048, 448, 8, 64,  3, 1, True,  True,  16, True,  'bf16', 'high'),
        # Chunk size
        ("bf16 chk TF32 B=32 cs=128",   32, 2048, 448, 8, 128, 3, 1, True,  True,  16, True,  'bf16', 'high'),
        ("bf16 chk TF32 B=32 cs=256",   32, 2048, 448, 8, 256, 3, 1, True,  True,  16, True,  'bf16', 'high'),
        # No checkpoint
        ("bf16 no-chk TF32 B=8",         8, 2048, 448, 8, 64,  3, 1, True,  True,  16, False, 'bf16', 'high'),
        ("bf16 no-chk TF32 B=16",       16, 2048, 448, 8, 64,  3, 1, True,  True,  16, False, 'bf16', 'high'),
        ("bf16 no-chk TF32 B=32",       32, 2048, 448, 8, 64,  3, 1, True,  True,  16, False, 'bf16', 'high'),
        # Linear memory
        ("linear bf16 chk TF32 B=32",   32, 2048, 448, 8, 64,  3, 1, False, True,  16, True,  'bf16', 'high'),
        ("linear bf16 chk TF32 B=64",   64, 2048, 448, 8, 64,  3, 1, False, True,  16, True,  'bf16', 'high'),
    ]

    for label, B, T, embd, heads, cs, ns, expand, deep, ste, omega, chk, dtype, prec in configs:
        t, tps, mfu = profile_config(B, T, embd, heads, cs, ns, expand, deep, ste, omega, chk, dtype, prec)
        if t is not None:
            print(f"{label:<60} {t*1000:8.1f} {tps:10.0f} {mfu:8.2f}")
        else:
            print(f"{label:<60} {'OOM':>8}")

    print("=" * 100)

    # === Component breakdown for best config ===
    print("\n\nComponent breakdown: bf16 chk TF32 B=32")
    print("-" * 60)
    jax.config.update("jax_default_matmul_precision", "high")

    from atlas_jax.model import AtlasMemoryLayer, MLP, linear_scan, rms_norm
    from atlas_jax.memory_layer import polar_express_ste

    B, T = 32, 2048
    cfg = AtlasConfig(
        sequence_len=T, n_layer=1, n_head=8, n_embd=448,
        chunk_size=64, ns_steps=3, omega_window=16, poly_degree=3,
        deep_memory=True, memory_expand=1, pe_ste=True, use_checkpoint=True,
    )
    H, D, E, cs = 8, 56, 56, 64

    key = jax.random.PRNGKey(0)

    # PE
    X_pe = jax.random.normal(key, (B, cs, H, D, E), dtype=jnp.bfloat16)
    @jax.jit
    def pe_ste_fwd_bwd(X):
        def f(X): return jnp.sum(polar_express_ste(X, steps=3))
        return jax.grad(f)(X)
    t, _ = time_fn(pe_ste_fwd_bwd, X_pe)
    print(f"  PE STE fwd+bwd (per chunk): {t*1000:8.2f} ms")

    # Associative scan
    h_init = jax.random.normal(key, (B, H, D, E), dtype=jnp.bfloat16)
    gates = jax.random.uniform(key, (B, cs, H), dtype=jnp.bfloat16)
    inputs_s = jax.random.normal(key, (B, cs, H, D, E), dtype=jnp.bfloat16)
    @jax.jit
    def scan_fwd_bwd(h, g, i):
        def f(h, g, i):
            out, _ = linear_scan(h, g, i)
            return jnp.sum(out)
        return jax.grad(f, argnums=(0,1,2))(h, g, i)
    t, _ = time_fn(scan_fwd_bwd, h_init, gates, inputs_s)
    print(f"  Assoc scan fwd+bwd (cs=64): {t*1000:8.2f} ms  state=(B={B},H,D,E)")

    # Einsums
    W2 = jax.random.normal(key, (B, H, E, D), dtype=jnp.bfloat16)
    k_c = jax.random.normal(key, (B, cs, H, D), dtype=jnp.bfloat16)
    @jax.jit
    def einsum_fwd_bwd(W, k):
        def f(W, k): return jnp.sum(jnp.einsum('bhed,bchd->bche', W, k))
        return jax.grad(f, argnums=(0,1))(W, k)
    t, _ = time_fn(einsum_fwd_bwd, W2, k_c)
    print(f"  Einsum fwd+bwd (per call):  {t*1000:8.2f} ms")

    # Full layer
    layer = AtlasMemoryLayer(cfg, key=key)
    def to_bf16(x):
        return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
    layer = jax.tree.map(to_bf16, layer, is_leaf=eqx.is_array)
    x_layer = jax.random.normal(key, (B, T, 448), dtype=jnp.bfloat16)
    @eqx.filter_jit
    def layer_fwd_bwd(l, x):
        def f(l, x):
            y, _ = l(x)
            return jnp.sum(y)
        return eqx.filter_value_and_grad(f)(l, x)
    t, _ = time_fn(layer_fwd_bwd, layer, x_layer)
    print(f"  Full memory layer fwd+bwd:  {t*1000:8.2f} ms")

    # MLP
    mlp = MLP(cfg, key=key)
    mlp = jax.tree.map(to_bf16, mlp, is_leaf=eqx.is_array)
    @eqx.filter_jit
    def mlp_fwd_bwd(m, x):
        def f(m, x): return jnp.sum(m(x))
        return eqx.filter_value_and_grad(f)(m, x)
    t, _ = time_fn(mlp_fwd_bwd, mlp, x_layer)
    print(f"  MLP fwd+bwd:                {t*1000:8.2f} ms")

    # Projections (Q,K,V + output = 4 matmuls of (B,T,C) @ (C,C))
    W = jax.random.normal(key, (448, 448), dtype=jnp.bfloat16)
    @jax.jit
    def proj_fwd_bwd(x, W):
        def f(x, W): return jnp.sum(x @ W)
        return jax.grad(f, argnums=(0,1))(x, W)
    t, _ = time_fn(proj_fwd_bwd, x_layer, W)
    print(f"  Projection fwd+bwd (1 of 4):{t*1000:8.2f} ms")

    # Estimate: 32 chunks × (PE + 4 scans + ~7 einsums + omega agg + element-wise)
    print()
    print("Expected layer time = 32 chunks × (chunk_ops)")
    print("If layer time >> 32 × sum(component times), the overhead is XLA scheduling")


if __name__ == '__main__':
    main()
