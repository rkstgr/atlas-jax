"""Test single-chunk (cs=T) performance: no outer loop at all."""
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


def profile_config(B, T, chunk_size, use_checkpoint, label):
    jax.config.update("jax_default_matmul_precision", "high")

    config = AtlasConfig(
        sequence_len=T, n_layer=1, n_head=8, n_embd=448,
        chunk_size=chunk_size, ns_steps=3, omega_window=16, poly_degree=3,
        deep_memory=True, memory_expand=1, pe_ste=True,
        use_checkpoint=use_checkpoint,
    )
    H, D = 8, 56

    key = jax.random.PRNGKey(0)
    block = Block(config, key=key)

    def to_bf16(x):
        return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
    block = jax.tree.map(to_bf16, block, is_leaf=eqx.is_array)
    x = jax.random.normal(key, (B, T, 448), dtype=jnp.bfloat16)

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
        # FLOPs per token for MFU
        elem_flops = H * (D*D + D*D) * 5
        ns_flops = 2 * 3 * 3 * 2 * H * D ** 3
        n_params_block = 4 * 448 * 448 + 4 * 448 * 448
        flops_per_token = 6 * n_params_block + elem_flops + ns_flops
        mfu = (flops_per_token * tps) / (989.4e12) * 100
        print(f"  {label:<55} {t*1000:8.1f} ms  {tps:10.0f} tok/s  {mfu:6.2f}% MFU  chunks={T//chunk_size}")
    except Exception as e:
        print(f"  {label:<55} FAILED: {e}")


def main():
    print("Single-chunk profiling: cs=T eliminates outer loop entirely")
    print("=" * 100)

    for B in [8, 16, 32]:
        print(f"\n--- B={B} ---")
        for cs, chk, label in [
            (64,   True,  f"B={B} cs=64 chk (baseline)"),
            (128,  True,  f"B={B} cs=128 chk"),
            (256,  True,  f"B={B} cs=256 chk"),
            (512,  True,  f"B={B} cs=512 chk"),
            (1024, True,  f"B={B} cs=1024 chk"),
            (2048, True,  f"B={B} cs=2048 chk (SINGLE CHUNK)"),
            (2048, False, f"B={B} cs=2048 no-chk (SINGLE CHUNK)"),
            (64,   False, f"B={B} cs=64 no-chk"),
        ]:
            profile_config(B, 2048, cs, chk, label)

    print("\n" + "=" * 100)
    print("Done.")


if __name__ == '__main__':
    main()
