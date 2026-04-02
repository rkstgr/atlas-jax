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


if __name__ == '__main__':
    main()
