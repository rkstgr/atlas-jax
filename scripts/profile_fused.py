"""Profile comparison: baseline (lax.scan) vs FlashATLAS (unrolled + fused kernel).

Measures step time, throughput, MFU for both code paths with various chunk sizes.

Usage:
    python scripts/profile_fused.py                    # auto-detect GPU
    python scripts/profile_fused.py --preset h100      # H100 config
    python scripts/profile_fused.py --trace-dir /path  # save JAX traces
"""

import argparse
import time
from dataclasses import asdict

import jax
import jax.numpy as jnp
import equinox as eqx

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas


GPU_PEAK_TFLOPS = {"H100": 989.4, "A100": 312.0, "RTX 8000": 32.6}

CONFIGS = {
    "baseline": dict(
        label="Baseline (lax.scan, cs=64, f32)",
        chunk_size=64, fused_chunk=False, matmul_precision="float32",
    ),
    "unrolled_f32": dict(
        label="Unrolled + fused (cs=256, f32)",
        chunk_size=256, fused_chunk=True, matmul_precision="float32",
    ),
    "unrolled_tf32": dict(
        label="Unrolled + fused (cs=256, TF32)",
        chunk_size=256, fused_chunk=True, matmul_precision="high",
    ),
    "unrolled_tf32_nockpt": dict(
        label="Unrolled + fused (cs=256, TF32, no ckpt)",
        chunk_size=256, fused_chunk=True, matmul_precision="high",
        use_checkpoint=False,
    ),
}


def detect_gpu():
    devices = jax.devices()
    d = devices[0]
    if d.platform != "gpu":
        return "cpu", None
    name = getattr(d, "device_kind", "unknown")
    peak = None
    for k, v in GPU_PEAK_TFLOPS.items():
        if k.lower() in name.lower():
            peak = v
    return name, peak


def estimate_flops_per_token(config, n_params, n_embed_params):
    H = config.n_head
    D = config.n_embd // H
    E = config.memory_expand * D if config.deep_memory else D
    if config.deep_memory:
        elementwise_flops = H * (D * E + E * D) * 5
        ns_flops = 2 * 3 * config.ns_steps * 2 * H * max(D, E) ** 3
    else:
        elementwise_flops = H * D * D * 5
        ns_flops = 3 * config.ns_steps * 2 * H * D * D * D
    memory_flops_per_token = (elementwise_flops + ns_flops) * config.n_layer
    return 6 * (n_params - n_embed_params) + memory_flops_per_token


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


def profile_config(name, cfg_overrides, base_kwargs, gpu_peak_tflops):
    """Profile a single configuration."""
    ckwargs = {**base_kwargs, **cfg_overrides}
    ckwargs.pop("label", None)
    B = ckwargs.pop("batch_size", 32)
    matmul_prec = ckwargs.pop("matmul_precision", "float32")
    jax.config.update("jax_default_matmul_precision", matmul_prec)
    config = AtlasConfig(**ckwargs)
    T = config.sequence_len

    key = jax.random.PRNGKey(0)
    model = Atlas(config, key=key)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    n_embed_params = model.wte.weight.size
    flops_per_token = estimate_flops_per_token(config, n_params, n_embed_params)

    # Cast to bf16
    def _to_bf16(x):
        return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
    model = jax.tree.map(_to_bf16, model, is_leaf=eqx.is_array)

    idx = jax.random.randint(key, (B, T), 0, config.vocab_size)

    @eqx.filter_jit
    def train_fwd_bwd(model, idx):
        def loss_fn(model):
            logits, _ = model(idx)
            return jnp.mean(logits)
        return eqx.filter_value_and_grad(loss_fn)(model)

    # Compilation
    print(f"  Compiling {name}...", end=" ", flush=True)
    t0 = time.perf_counter()
    out = train_fwd_bwd(model, idx)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    print(f"{compile_s:.1f}s")

    # Timing
    step_s, _ = time_fn(train_fwd_bwd, model, idx, warmup=2, repeats=5)
    tokens = B * T
    tps = tokens / step_s
    mfu = None
    if gpu_peak_tflops:
        mfu = (flops_per_token * tps) / (gpu_peak_tflops * 1e12) * 100

    # Memory
    peak_mb = None
    try:
        stats = jax.local_devices()[0].memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            peak_mb = stats["peak_bytes_in_use"] / 1e6
    except Exception:
        pass

    return {
        "name": name,
        "label": cfg_overrides.get("label", name),
        "compile_s": compile_s,
        "step_ms": step_s * 1000,
        "tok_s": tps,
        "mfu": mfu,
        "peak_mb": peak_mb,
        "n_chunks": T // cfg_overrides.get("chunk_size", 64),
        "chunk_size": cfg_overrides.get("chunk_size", 64),
        "fused": cfg_overrides.get("fused_chunk", False),
    }


def main():
    parser = argparse.ArgumentParser(description="FlashATLAS profiling comparison")
    parser.add_argument("--preset", choices=["h100", "small"], default="h100")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Which configs to run (default: all)")
    parser.add_argument("--matmul-precision", default="float32")
    parser.add_argument("--trace-dir", default=None,
                        help="Save JAX profiler traces to this dir")
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    gpu_name, gpu_peak = detect_gpu()
    print(f"JAX {jax.__version__} | {gpu_name} | Peak: {gpu_peak} TFLOPS")

    if args.preset == "h100":
        base = dict(
            sequence_len=2048, n_layer=8, n_head=8, n_embd=448,
            omega_window=16, poly_degree=3, deep_memory=True,
            memory_expand=1, ns_steps=3, pe_ste=True,
            use_checkpoint=True, batch_size=32,
        )
    else:
        base = dict(
            sequence_len=512, n_layer=4, n_head=4, n_embd=128,
            omega_window=8, poly_degree=2, deep_memory=True,
            memory_expand=1, ns_steps=3, pe_ste=True,
            use_checkpoint=True, batch_size=4,
        )

    configs_to_run = args.configs or list(CONFIGS.keys())

    print(f"\nBase config: L={base['n_layer']}, D={base['n_embd']}, H={base['n_head']}, "
          f"B={base['batch_size']}, T={base['sequence_len']}")
    print("=" * 80)

    results = []
    for name in configs_to_run:
        if name not in CONFIGS:
            print(f"Unknown config: {name}, skipping")
            continue
        cfg = CONFIGS[name]
        print(f"\n--- {cfg['label']} ---")
        try:
            r = profile_config(name, cfg, base, gpu_peak)
            results.append(r)
            print(f"  Step: {r['step_ms']:.1f} ms | {r['tok_s']:.0f} tok/s"
                  + (f" | MFU: {r['mfu']:.2f}%" if r['mfu'] else "")
                  + (f" | Peak: {r['peak_mb']:.0f} MB" if r['peak_mb'] else ""))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        baseline_ms = results[0]["step_ms"] if results else 1
        print(f"{'Config':<35} {'Step (ms)':>10} {'tok/s':>10} {'MFU%':>8} {'Speedup':>8} {'Chunks':>7}")
        print("-" * 80)
        for r in results:
            speedup = baseline_ms / r["step_ms"]
            mfu_str = f"{r['mfu']:.2f}" if r['mfu'] else "N/A"
            print(f"{r['label']:<35} {r['step_ms']:>10.1f} {r['tok_s']:>10.0f} "
                  f"{mfu_str:>8} {speedup:>7.2f}x {r['n_chunks']:>7}")

    # JAX trace (for the best config)
    if args.trace_dir and results:
        best = min(results, key=lambda r: r["step_ms"])
        print(f"\nCapturing trace for best config: {best['name']}...")
        cfg = CONFIGS[best["name"]]
        ckwargs = {**base, **cfg}
        ckwargs.pop("label", None)
        ckwargs.pop("batch_size", None)
        config = AtlasConfig(**ckwargs)
        B = base["batch_size"]

        model = Atlas(config, key=jax.random.PRNGKey(0))
        def _to_bf16(x):
            return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
        model = jax.tree.map(_to_bf16, model, is_leaf=eqx.is_array)
        idx = jax.random.randint(jax.random.PRNGKey(0), (B, config.sequence_len), 0, config.vocab_size)

        @eqx.filter_jit
        def step(model, idx):
            def loss_fn(m):
                logits, _ = m(idx)
                return jnp.mean(logits)
            return eqx.filter_value_and_grad(loss_fn)(model)

        out = step(model, idx)
        jax.block_until_ready(out)

        with jax.profiler.trace(args.trace_dir):
            out = step(model, idx)
            jax.block_until_ready(out)
        print(f"Trace saved to {args.trace_dir}")


if __name__ == "__main__":
    main()
