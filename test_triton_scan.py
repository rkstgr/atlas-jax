"""Test Triton fused scan kernel on H100."""
import time
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "high")


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
    return min(times), out


def main():
    print(f"Devices: {jax.devices()}")
    print("=" * 80)

    from atlas_jax.model import linear_scan as assoc_scan
    from atlas_jax.triton_scan import triton_linear_scan

    for label, B, T, H, D, E in [
        ("small",  8, 64, 8, 56, 56),   # Paper config per chunk
        ("B=32",  32, 64, 8, 56, 56),
        ("D=96",  32, 64, 4, 96, 96),   # Wider heads
    ]:
        print(f"\n--- {label}: B={B}, T={T}, H={H}, D={D}, E={E} ---")
        key = jax.random.PRNGKey(0)
        h_init = jax.random.normal(key, (B, H, D, E)) * 0.1
        gates = jax.random.uniform(key, (B, T, H), minval=0.5, maxval=0.99)
        inputs = jax.random.normal(key, (B, T, H, D, E)) * 0.1

        # Correctness
        h_all_a, _ = assoc_scan(h_init, gates, inputs)
        h_all_t, _ = triton_linear_scan(h_init, gates, inputs)
        diff = float(jnp.max(jnp.abs(h_all_a - h_all_t)))
        print(f"  Correctness: max_diff={diff:.2e}")

        # Benchmark
        @jax.jit
        def run_a(h, g, i): return assoc_scan(h, g, i)
        @jax.jit
        def run_t(h, g, i): return triton_linear_scan(h, g, i)

        t_a, _ = time_fn(run_a, h_init, gates, inputs)
        t_t, _ = time_fn(run_t, h_init, gates, inputs)
        print(f"  Associative: {t_a*1000:.3f} ms")
        print(f"  Triton:      {t_t*1000:.3f} ms")
        print(f"  Speedup:     {t_a/t_t:.2f}x")

        # bf16
        h_bf = h_init.astype(jnp.bfloat16)
        g_bf = gates.astype(jnp.bfloat16)
        i_bf = inputs.astype(jnp.bfloat16)
        t_t_bf, _ = time_fn(run_t, h_bf, g_bf, i_bf)
        print(f"  Triton bf16: {t_t_bf*1000:.3f} ms")

    # Gradient test
    print("\n--- Gradient test (B=8 T=16) ---")
    B, T, H, D = 8, 16, 4, 8
    key = jax.random.PRNGKey(0)
    h = jax.random.normal(key, (B, H, D, D)) * 0.1
    g = jax.random.uniform(key, (B, T, H), minval=0.5, maxval=0.99)
    inp = jax.random.normal(key, (B, T, H, D, D)) * 0.1

    # Triton kernel is forward-only (no autograd). Check:
    try:
        def loss_t(h, g, i):
            a, _ = triton_linear_scan(h, g, i)
            return jnp.sum(a)
        grad_t = jax.grad(loss_t)(h, g, inp)
        print(f"  Triton grad: OK shape={grad_t.shape}")
    except Exception as e:
        print(f"  Triton grad: FAILED ({type(e).__name__}: {e})")
        print("  NOTE: Triton kernel is forward-only. Need custom_vjp for backward.")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
