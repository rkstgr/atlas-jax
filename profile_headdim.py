"""Profile with larger head dimensions for better tensor core utilization."""
import time
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_default_matmul_precision", "high")

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


def profile(B, T, n_embd, n_head, cs, ns, expand, deep, label):
    H = n_head
    D = n_embd // H
    E = expand * D if deep else D

    config = AtlasConfig(
        sequence_len=T, n_layer=1, n_head=n_head, n_embd=n_embd,
        chunk_size=cs, ns_steps=ns, omega_window=16, poly_degree=3,
        deep_memory=deep, memory_expand=expand, pe_ste=True, use_checkpoint=True,
    )

    key = jax.random.PRNGKey(0)
    block = Block(config, key=key)

    def to_bf16(x):
        return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
    block = jax.tree.map(to_bf16, block, is_leaf=eqx.is_array)
    x = jax.random.normal(key, (B, T, n_embd), dtype=jnp.bfloat16)

    @eqx.filter_jit
    def fwd_bwd(block, x):
        def f(block, x):
            y, _ = block(x)
            return jnp.sum(y)
        return eqx.filter_value_and_grad(f)(block, x)

    try:
        t, _ = time_fn(fwd_bwd, block, x)
        tokens = B * T
        tps = tokens / t

        if deep:
            elem_flops = H * (D * E + E * D) * 5
            ns_flops = 2 * 3 * ns * 2 * H * max(D, E) ** 3
        else:
            elem_flops = H * D * D * 5
            ns_flops = 3 * ns * 2 * H * D ** 3

        n_params_block = 4 * n_embd**2 + 4 * n_embd**2
        flops_per_token = 6 * n_params_block + elem_flops + ns_flops
        mfu = (flops_per_token * tps) / (989.4e12) * 100

        print(f"  {label:<50} D={D:>3} {t*1000:8.1f}ms {tps:8.0f} tok/s {mfu:6.2f}% MFU  FLOPs/tok={flops_per_token/1e6:.0f}M")
    except Exception as e:
        print(f"  {label:<50} FAILED: {type(e).__name__}")


def main():
    print("Head dimension sweep: larger D → better tensor core utilization")
    print("=" * 110)

    T = 2048

    configs = [
        # (B, embd, heads, cs, ns, expand, deep, label)
        # Paper-like: H=8, D=56
        (32, 448, 8, 64, 3, 1, True, "paper: 448d H=8 D=56"),
        # Wider heads: D=96
        (32, 384, 4, 64, 3, 1, True, "wide: 384d H=4 D=96"),
        # Very wide: D=128
        (32, 384, 3, 64, 3, 1, True, "vwide: 384d H=3 D=128"),
        (16, 512, 4, 64, 3, 1, True, "vwide: 512d H=4 D=128"),
        # D=192
        (16, 768, 4, 64, 3, 1, True, "huge: 768d H=4 D=192"),
        (8, 768, 4, 64, 3, 1, True, "huge: 768d H=4 D=192 B=8"),
        # D=256
        (8, 768, 3, 64, 3, 1, True, "max: 768d H=3 D=256"),
        (4, 1024, 4, 64, 3, 1, True, "max: 1024d H=4 D=256"),
        # Linear memory for comparison
        (32, 448, 8, 64, 3, 1, False, "linear: 448d H=8 D=56"),
        (32, 384, 4, 64, 3, 1, False, "linear wide: 384d H=4 D=96"),
    ]

    for B, embd, heads, cs, ns, expand, deep, label in configs:
        profile(B, T, embd, heads, cs, ns, expand, deep, f"B={B} {label}")

    print("=" * 110)


if __name__ == '__main__':
    main()
