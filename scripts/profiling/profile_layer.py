"""Profile individual components of the Atlas memory layer.

Runs forward+backward of a single AtlasMemoryLayer with paper-faithful config
and reports time breakdown per operation.
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_default_matmul_precision", "float32")

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas, AtlasMemoryLayer, Block, MLP, rms_norm
from atlas_jax.memory_layer import polar_express, polar_express_ste


def time_fn(fn, *args, warmup=2, repeats=5, **kwargs):
    """Time a JIT-compiled function."""
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


def main():
    # Paper-faithful config (expand=1 deep, as validated in paper)
    B, T = 8, 2048
    config = AtlasConfig(
        sequence_len=T, n_layer=8, n_head=8, n_embd=448,
        chunk_size=64, ns_steps=3, omega_window=16, poly_degree=3,
        deep_memory=True, memory_expand=1, pe_ste=True,
    )
    H = config.n_head
    D = config.n_embd // H  # 56
    E = config.memory_expand * D  # 56
    cs = config.chunk_size  # 64
    n_chunks = T // cs  # 32

    print(f"Config: B={B}, T={T}, H={H}, D={D}, E={E}, cs={cs}, n_chunks={n_chunks}")
    print(f"ns_steps={config.ns_steps}, omega_window={config.omega_window}, poly_degree={config.poly_degree}")
    print(f"deep_memory={config.deep_memory}, memory_expand={config.memory_expand}, pe_ste={config.pe_ste}")
    print("=" * 80)

    key = jax.random.PRNGKey(0)

    # === 1. Profile PE (Polar Express) ===
    print("\n--- Polar Express ---")
    X_pe = jax.random.normal(key, (B, cs, H, D, E), dtype=jnp.float32)

    @jax.jit
    def pe_fwd(X):
        return polar_express(X, steps=3)

    @jax.jit
    def pe_ste_fwd(X):
        return polar_express_ste(X, steps=3)

    @jax.jit
    def pe_fwd_bwd(X):
        def f(X):
            return jnp.sum(polar_express(X, steps=3))
        return jax.grad(f)(X)

    @jax.jit
    def pe_ste_fwd_bwd(X):
        def f(X):
            return jnp.sum(polar_express_ste(X, steps=3))
        return jax.grad(f)(X)

    t, _ = time_fn(pe_fwd, X_pe)
    print(f"  PE forward (3 steps):     {t*1000:8.2f} ms  shape={X_pe.shape}")
    t, _ = time_fn(pe_ste_fwd, X_pe)
    print(f"  PE STE forward (3 steps): {t*1000:8.2f} ms")
    t, _ = time_fn(pe_fwd_bwd, X_pe)
    print(f"  PE full fwd+bwd:          {t*1000:8.2f} ms")
    t, _ = time_fn(pe_ste_fwd_bwd, X_pe)
    print(f"  PE STE fwd+bwd:           {t*1000:8.2f} ms")

    # === 2. Profile linear_scan ===
    print("\n--- Linear Scan ---")
    from atlas_jax.model import linear_scan
    h_init = jax.random.normal(key, (B, H, D, E), dtype=jnp.float32)
    gates = jax.random.uniform(key, (B, cs, H), dtype=jnp.float32)
    inputs = jax.random.normal(key, (B, cs, H, D, E), dtype=jnp.float32)

    @jax.jit
    def scan_fwd(h, g, i):
        return linear_scan(h, g, i)

    @jax.jit
    def scan_fwd_bwd(h, g, i):
        def f(h, g, i):
            out, _ = linear_scan(h, g, i)
            return jnp.sum(out)
        return jax.grad(f, argnums=(0, 1, 2))(h, g, i)

    t, _ = time_fn(scan_fwd, h_init, gates, inputs)
    print(f"  scan forward (cs={cs}):    {t*1000:8.2f} ms  state={h_init.shape}")
    t, _ = time_fn(scan_fwd_bwd, h_init, gates, inputs)
    print(f"  scan fwd+bwd (cs={cs}):    {t*1000:8.2f} ms")

    # === 3. Profile einsum operations ===
    print("\n--- Einsums (per chunk) ---")
    W2 = jax.random.normal(key, (B, H, E, D))
    k_c = jax.random.normal(key, (B, cs, H, D))
    W1 = jax.random.normal(key, (B, H, D, E))
    err = jax.random.normal(key, (B, cs, H, D))
    act = jax.random.normal(key, (B, cs, H, E))

    @jax.jit
    def einsum_w2k(W2, k):
        return jnp.einsum('bhed,bchd->bche', W2, k)

    @jax.jit
    def einsum_outer(err, act):
        return jnp.einsum('bchd,bche->bchde', err, act)

    @jax.jit
    def einsum_w1act(W1, act):
        return jnp.einsum('bhde,bche->bchd', W1, act)

    t, _ = time_fn(einsum_w2k, W2, k_c)
    print(f"  W2 @ k (bhed,bchd->bche): {t*1000:8.2f} ms")
    t, _ = time_fn(einsum_outer, err, act)
    print(f"  outer (bchd,bche->bchde): {t*1000:8.2f} ms")
    t, _ = time_fn(einsum_w1act, W1, act)
    print(f"  W1 @ act (bhde,bche->bchd):{t*1000:8.2f} ms")

    # === 4. Profile full memory layer ===
    print("\n--- Full Memory Layer (1 layer, fwd only) ---")
    key, k1 = jax.random.split(key)
    layer = AtlasMemoryLayer(config, key=k1)
    x_layer = jax.random.normal(key, (B, T, config.n_embd))

    @eqx.filter_jit
    def layer_fwd(layer, x):
        return layer(x)

    @eqx.filter_jit
    def layer_fwd_bwd(layer, x):
        def f(layer, x):
            y, _ = layer(x)
            return jnp.sum(y)
        return eqx.filter_value_and_grad(f)(layer, x)

    t, _ = time_fn(layer_fwd, layer, x_layer)
    print(f"  layer forward:            {t*1000:8.2f} ms")
    t, _ = time_fn(layer_fwd_bwd, layer, x_layer)
    print(f"  layer fwd+bwd:            {t*1000:8.2f} ms")

    # === 5. Profile MLP ===
    print("\n--- MLP (1 layer) ---")
    key, k2 = jax.random.split(key)
    mlp = MLP(config, key=k2)

    @eqx.filter_jit
    def mlp_fwd_bwd(mlp, x):
        def f(mlp, x):
            return jnp.sum(mlp(x))
        return eqx.filter_value_and_grad(f)(mlp, x)

    t, _ = time_fn(mlp_fwd_bwd, mlp, x_layer)
    print(f"  MLP fwd+bwd:              {t*1000:8.2f} ms")

    # === 6. Profile full block ===
    print("\n--- Full Block (memory + MLP, fwd+bwd) ---")
    key, k3 = jax.random.split(key)
    block = Block(config, key=k3)

    @eqx.filter_jit
    def block_fwd_bwd(block, x):
        def f(block, x):
            y, _ = block(x)
            return jnp.sum(y)
        return eqx.filter_value_and_grad(f)(block, x)

    t, _ = time_fn(block_fwd_bwd, block, x_layer)
    print(f"  block fwd+bwd:            {t*1000:8.2f} ms")

    # === 7. Profile full model train step ===
    print("\n--- Full Model Train Step ---")
    key, k4 = jax.random.split(key)
    model = Atlas(config, key=k4)
    idx = jax.random.randint(key, (B, T), 0, config.vocab_size)

    @eqx.filter_jit
    def train_fwd_bwd(model, idx):
        def loss_fn(model):
            logits, _ = model(idx)
            return jnp.mean(logits)
        return eqx.filter_value_and_grad(loss_fn)(model)

    t, _ = time_fn(train_fwd_bwd, model, idx)
    tokens = B * T
    tps = tokens / t
    flops_per_token = 6 * 48e6 + 8 * (8 * (56*56 + 56*56) * 5 + 2*3*3*2*8*56**3)
    mfu = (flops_per_token * tps) / (989.4e12) * 100
    print(f"  full model fwd+bwd:       {t*1000:8.2f} ms  ({tokens} tokens)")
    print(f"  throughput:               {tps:8.0f} tok/s")
    print(f"  MFU (approx):             {mfu:8.2f}%")

    # === Summary ===
    print("\n" + "=" * 80)
    print("Profile complete. Key bottlenecks are the operations with highest ms.")


if __name__ == '__main__':
    main()
