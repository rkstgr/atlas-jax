"""Focused benchmark: measure fwd+bwd step time with controlled configurations.

Each configuration is tested independently to isolate the effect of single changes.
Reports step time, tokens/s, MFU, and peak memory.

Usage:
    python scripts/benchmark.py                     # run all configs
    python scripts/benchmark.py --configs baseline   # run just baseline
    python scripts/benchmark.py --configs baseline batched_pe  # compare two
"""

import argparse
import gc
import time

import jax
import jax.numpy as jnp
import equinox as eqx

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas


GPU_PEAK_TFLOPS = {"H100": 989.4, "A100": 312.0, "RTX 8000": 32.6}


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


def time_fn(fn, *args, warmup=3, repeats=10):
    """Time a function, return median and all times."""
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    return median, times


def profile_config(name, config_kwargs, batch_size, matmul_precision, gpu_peak_tflops,
                   donate_buffers=False):
    """Profile a single configuration: compile + measure step time."""
    jax.config.update("jax_default_matmul_precision", matmul_precision)

    config = AtlasConfig(**config_kwargs)
    T = config.sequence_len
    B = batch_size

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

    if donate_buffers:
        @eqx.filter_jit(donate='warn')
        def train_fwd_bwd(model, idx):
            def loss_fn(model):
                logits, _ = model(idx)
                return jnp.mean(logits)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            return loss, grads
    else:
        @eqx.filter_jit
        def train_fwd_bwd(model, idx):
            def loss_fn(model):
                logits, _ = model(idx)
                return jnp.mean(logits)
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            return loss, grads

    # Compilation
    print(f"  Compiling {name}...", end=" ", flush=True)
    t0 = time.perf_counter()
    out = train_fwd_bwd(model, idx)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    print(f"{compile_s:.1f}s")

    # Timing
    median_s, all_times = time_fn(train_fwd_bwd, model, idx, warmup=3, repeats=10)
    tokens = B * T
    tps = tokens / median_s
    mfu = (flops_per_token * tps) / (gpu_peak_tflops * 1e12) * 100 if gpu_peak_tflops else None

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
        "compile_s": compile_s,
        "step_ms": median_s * 1000,
        "min_ms": min(all_times) * 1000,
        "max_ms": max(all_times) * 1000,
        "tok_s": tps,
        "mfu": mfu,
        "peak_mb": peak_mb,
        "n_params": n_params,
        "flops_per_token": flops_per_token,
        "matmul_precision": matmul_precision,
        "chunk_size": config.chunk_size,
        "batch_size": batch_size,
    }


# Base config matching the training setup
BASE_CONFIG = dict(
    sequence_len=2048, n_layer=8, n_head=8, n_embd=448,
    omega_window=16, poly_degree=3, deep_memory=True,
    memory_expand=1, ns_steps=3, pe_ste=True,
    use_checkpoint=True, fused_chunk=False,
)

# Each config is: (config_overrides, batch_size, matmul_precision, donate_buffers, description)
CONFIGS = {
    "baseline": (
        {},
        32, "float32", False,
        "Baseline: lax.scan, cs=64, f32, checkpoint",
    ),
    "baseline_tf32": (
        {},
        32, "high", False,
        "TF32 matmul precision (everything else same)",
    ),
    "unrolled": (
        {"fused_chunk": True},
        32, "float32", False,
        "Unrolled Python loop (fused_chunk=True), cs=64, f32",
    ),
    "unrolled_tf32": (
        {"fused_chunk": True},
        32, "high", False,
        "Unrolled + TF32",
    ),
    "cs128": (
        {"chunk_size": 128},
        32, "float32", False,
        "chunk_size=128 (vs 64 baseline)",
    ),
    "cs256": (
        {"chunk_size": 256},
        32, "float32", False,
        "chunk_size=256",
    ),
    "cs128_unrolled": (
        {"chunk_size": 128, "fused_chunk": True},
        32, "float32", False,
        "cs=128, unrolled",
    ),
    "cs256_unrolled": (
        {"chunk_size": 256, "fused_chunk": True},
        32, "float32", False,
        "cs=256, unrolled",
    ),
    "donate": (
        {},
        32, "float32", True,
        "Buffer donation (donate='warn')",
    ),
    "nocheckpoint": (
        {"use_checkpoint": False},
        32, "float32", False,
        "No gradient checkpointing",
    ),
    "nocheckpoint_tf32": (
        {"use_checkpoint": False},
        32, "high", False,
        "No checkpoint + TF32",
    ),
    "cs128_unrolled_tf32": (
        {"chunk_size": 128, "fused_chunk": True},
        32, "high", False,
        "cs=128, unrolled, TF32",
    ),
    "cs256_unrolled_tf32": (
        {"chunk_size": 256, "fused_chunk": True},
        32, "high", False,
        "cs=256, unrolled, TF32",
    ),
    "cs512_unrolled_tf32": (
        {"chunk_size": 512, "fused_chunk": True},
        32, "high", False,
        "cs=512, unrolled, TF32",
    ),
    "unrolled_tf32_nockpt_b16": (
        {"fused_chunk": True, "use_checkpoint": False},
        16, "high", False,
        "Unrolled, TF32, no ckpt, B=16",
    ),
    "unrolled_tf32_ns2": (
        {"fused_chunk": True, "ns_steps": 2},
        32, "high", False,
        "Unrolled, TF32, ns_steps=2 (vs 3)",
    ),
    "unrolled_tf32_ns1": (
        {"fused_chunk": True, "ns_steps": 1},
        32, "high", False,
        "Unrolled, TF32, ns_steps=1",
    ),
    "unrolled_tf32_omega1": (
        {"fused_chunk": True, "omega_window": 1},
        32, "high", False,
        "Unrolled, TF32, omega_window=1 (no omega)",
    ),
    "best_combo": (
        {"chunk_size": 256, "fused_chunk": True, "use_checkpoint": False},
        16, "high", True,
        "Best combo: cs=256, unrolled, TF32, no ckpt, B=16",
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Atlas-JAX step benchmark")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Which configs to run (default: all)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for all configs")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    gpu_name, gpu_peak = detect_gpu()
    print(f"JAX {jax.__version__} | {gpu_name} | Peak: {gpu_peak} TFLOPS")
    print(f"Devices: {jax.devices()}")

    configs_to_run = args.configs or list(CONFIGS.keys())

    print(f"\nBase: L=8, D=448, H=8, B=32, T=2048, expand=1, ns=3, pe_ste=True")
    print("=" * 100)

    results = []
    for name in configs_to_run:
        if name not in CONFIGS:
            print(f"Unknown config: {name}, skipping")
            continue
        overrides, bs, precision, donate, desc = CONFIGS[name]
        if args.batch_size is not None:
            bs = args.batch_size
        config_kwargs = {**BASE_CONFIG, **overrides}
        print(f"\n--- {desc} ---")
        try:
            r = profile_config(name, config_kwargs, bs, precision, gpu_peak,
                               donate_buffers=donate)
            r["description"] = desc
            results.append(r)
            print(f"  Median: {r['step_ms']:.1f} ms | {r['tok_s']:.0f} tok/s"
                  + (f" | MFU: {r['mfu']:.2f}%" if r['mfu'] else "")
                  + (f" | Peak: {r['peak_mb']:.0f} MB" if r['peak_mb'] else "")
                  + f" | range: [{r['min_ms']:.1f}, {r['max_ms']:.1f}] ms")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

        # Force cleanup between configs
        gc.collect()

    # Summary table
    if len(results) > 1:
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        baseline_ms = results[0]["step_ms"] if results else 1
        header = f"{'Config':<45} {'Step(ms)':>9} {'tok/s':>9} {'MFU%':>7} {'Speedup':>8} {'Peak MB':>9}"
        print(header)
        print("-" * 100)
        for r in results:
            speedup = baseline_ms / r["step_ms"]
            mfu_str = f"{r['mfu']:.2f}" if r['mfu'] else "N/A"
            peak_str = f"{r['peak_mb']:.0f}" if r['peak_mb'] else "N/A"
            print(f"{r['description'][:45]:<45} {r['step_ms']:>9.1f} {r['tok_s']:>9.0f} "
                  f"{mfu_str:>7} {speedup:>7.2f}x {peak_str:>9}")


if __name__ == "__main__":
    main()
