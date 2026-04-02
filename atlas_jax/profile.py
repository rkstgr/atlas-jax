"""Atlas profiling baseline: component-level timing, MFU, memory usage.

Consolidates profile_layer.py, profile_sweep.py, profile_headdim.py,
profile_singlechunk.py into one structured tool.

Usage:
    python -m atlas_jax.profile                    # laptop defaults (CPU-safe)
    python -m atlas_jax.profile --preset h100      # H100 target config
    python -m atlas_jax.profile --json out.json    # structured JSON output
    python -m atlas_jax.profile --n-embd 256       # custom overrides
"""

import argparse
import json
import platform
import sys
import time

import jax
import jax.numpy as jnp
import equinox as eqx

from atlas_jax.config import AtlasConfig
from atlas_jax.model import (
    Atlas, AtlasMemoryLayer, Block, MLP, ShortConv,
    linear_scan, rms_norm,
)
from atlas_jax.polar_express import polar_express, polar_express_ste


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

GPU_PEAK_TFLOPS = {
    "H100": 989.4,
    "A100": 312.0,
    "RTX 8000": 32.6,
    "RTX 4090": 82.6,
    "RTX 3090": 35.6,
}


def detect_hardware():
    """Detect device type and estimate peak TFLOPS."""
    devices = jax.devices()
    device = devices[0]

    if device.platform == "gpu":
        name = getattr(device, "device_kind", "unknown GPU")
        # Try to match known GPUs
        peak = None
        for key, val in GPU_PEAK_TFLOPS.items():
            if key.lower() in name.lower():
                peak = val
                break
        return {"device": "gpu", "name": name, "peak_tflops": peak, "count": len(devices)}
    else:
        name = platform.processor() or platform.machine()
        return {"device": "cpu", "name": name, "peak_tflops": None, "count": 1}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "laptop": dict(
        n_layer=2, n_head=4, n_embd=128, chunk_size=32,
        omega_window=8, poly_degree=2, deep_memory=True,
        memory_expand=1, ns_steps=3, pe_ste=True,
        batch_size=1, seq_len=512,
    ),
    "h100": dict(
        n_layer=24, n_head=16, n_embd=1536, chunk_size=64,
        omega_window=16, poly_degree=3, deep_memory=True,
        memory_expand=4, ns_steps=5, pe_ste=True,
        batch_size=8, seq_len=2048,
    ),
    "small_gpu": dict(
        n_layer=8, n_head=8, n_embd=448, chunk_size=64,
        omega_window=16, poly_degree=3, deep_memory=True,
        memory_expand=1, ns_steps=3, pe_ste=True,
        batch_size=4, seq_len=1024,
    ),
}


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def time_fn(fn, *args, warmup=2, repeats=5, **kwargs):
    """Time a function, returning (min_time_seconds, output)."""
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


# ---------------------------------------------------------------------------
# FLOP counting
# ---------------------------------------------------------------------------

def count_flops(config, n_params, n_embed_params):
    """Detailed FLOP count per token (forward + backward = 6N + memory ops).

    Returns dict with per-category breakdown and total.
    """
    H = config.n_head
    D = config.n_embd // H
    E = config.memory_expand * D if config.deep_memory else D
    C = config.n_embd
    L = config.n_layer

    # Standard transformer-like FLOPs (6N for fwd+bwd)
    # N = params minus embeddings (embeddings are just lookups)
    standard_flops = 6 * (n_params - n_embed_params)

    # Per-layer memory operation FLOPs (per token)
    # These happen inside the memory layer and are NOT counted in 6N
    mem_flops_per_layer = 0

    if config.deep_memory:
        # Forward through memory MLP (per token, per head):
        #   W2 @ k: (E, D) @ (D,) = 2*E*D flops
        #   W1 @ act: (D, E) @ (E,) = 2*D*E flops
        #   err computation: D adds
        mlp_fwd = H * (2 * E * D + 2 * D * E)

        # Gradient computation (per token, per head):
        #   u_W1 = err outer act: D*E multiplies
        #   W1^T @ err: 2*E*D
        #   chain * k outer product: E*D
        #   u_W2 outer product: E*D
        grad_compute = H * (D * E + 2 * E * D + E * D + E * D)

        # Polar Express (per token, per head, per weight matrix):
        #   Each step: A = X @ X^T or X^T @ X: 2*max(D,E)^2 * min(D,E) flops
        #   B = b*A + c*(A@A): 2*max(D,E)^3 + max(D,E)^2
        #   X = a*X + B@X: 2*max(D,E)^2 * min(D,E) + max(D,E)*min(D,E)
        # Two weight matrices (W1: DxE, W2: ExD)
        pe_per_step = 2 * H * (2 * max(D, E)**2 * min(D, E) + 2 * max(D, E)**3)
        pe_flops = config.ns_steps * pe_per_step * 2  # x2 for W1 and W2

        mem_flops_per_layer = mlp_fwd + grad_compute + pe_flops
    else:
        # Linear memory: M @ k = 2*D*D per head
        # Gradient outer product: D*D per head
        # PE: ns_steps * (2*D^3 + 2*D^3) per head
        lin_fwd = H * 2 * D * D
        lin_grad = H * D * D
        pe_flops = config.ns_steps * H * 4 * D**3
        mem_flops_per_layer = lin_fwd + lin_grad + pe_flops

    memory_flops = mem_flops_per_layer * L

    # Projection FLOPs are already in the 6N count (they're model params)
    # MLP FLOPs are already in the 6N count

    total = standard_flops + memory_flops

    return {
        "standard_6N": standard_flops,
        "memory_ops_per_token": memory_flops,
        "total_per_token": total,
        "memory_fraction": memory_flops / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Component profiling
# ---------------------------------------------------------------------------

def profile_components(config, B, key):
    """Profile individual components, returning dict of timings in ms."""
    H = config.n_head
    D = config.n_embd // H
    E = config.memory_expand * D if config.deep_memory else D
    cs = config.chunk_size
    C = config.n_embd
    T = config.sequence_len

    results = {}
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # --- Polar Express ---
    X_pe = jax.random.normal(k1, (B, cs, H, D, E), dtype=jnp.float32)

    @jax.jit
    def pe_ste_fwd(X):
        return polar_express_ste(X, steps=config.ns_steps)

    @jax.jit
    def pe_ste_fwd_bwd(X):
        return jax.grad(lambda X: jnp.sum(polar_express_ste(X, config.ns_steps)))(X)

    @jax.jit
    def pe_full_fwd_bwd(X):
        return jax.grad(lambda X: jnp.sum(polar_express(X, config.ns_steps)))(X)

    t, _ = time_fn(pe_ste_fwd, X_pe)
    results["pe_ste_fwd_ms"] = t * 1000
    t, _ = time_fn(pe_ste_fwd_bwd, X_pe)
    results["pe_ste_fwd_bwd_ms"] = t * 1000
    t, _ = time_fn(pe_full_fwd_bwd, X_pe)
    results["pe_full_fwd_bwd_ms"] = t * 1000

    # --- Linear Scan ---
    h_init = jax.random.normal(k2, (B, H, D, E), dtype=jnp.float32)
    gates = jax.random.uniform(k2, (B, cs, H), dtype=jnp.float32)
    inputs = jax.random.normal(k2, (B, cs, H, D, E), dtype=jnp.float32)

    @jax.jit
    def scan_fwd(h, g, i):
        return linear_scan(h, g, i)

    @jax.jit
    def scan_fwd_bwd(h, g, i):
        return jax.grad(lambda h, g, i: jnp.sum(linear_scan(h, g, i)[0]),
                        argnums=(0, 1, 2))(h, g, i)

    t, _ = time_fn(scan_fwd, h_init, gates, inputs)
    results["scan_fwd_ms"] = t * 1000
    t, _ = time_fn(scan_fwd_bwd, h_init, gates, inputs)
    results["scan_fwd_bwd_ms"] = t * 1000

    # --- Full Memory Layer ---
    layer = AtlasMemoryLayer(config, key=k3)
    x_layer = jax.random.normal(k3, (B, T, C))

    @eqx.filter_jit
    def layer_fwd(layer, x):
        return layer(x)

    @eqx.filter_jit
    def layer_fwd_bwd(layer, x):
        return eqx.filter_value_and_grad(
            lambda l, x: jnp.sum(l(x)[0]))(layer, x)

    t, _ = time_fn(layer_fwd, layer, x_layer)
    results["memory_layer_fwd_ms"] = t * 1000
    t, _ = time_fn(layer_fwd_bwd, layer, x_layer)
    results["memory_layer_fwd_bwd_ms"] = t * 1000

    # --- MLP ---
    mlp = MLP(config, key=k4)

    @eqx.filter_jit
    def mlp_fwd_bwd(mlp, x):
        return eqx.filter_value_and_grad(
            lambda m, x: jnp.sum(m(x)))(mlp, x)

    t, _ = time_fn(mlp_fwd_bwd, mlp, x_layer)
    results["mlp_fwd_bwd_ms"] = t * 1000

    # --- Full Block ---
    block = Block(config, key=k5)

    @eqx.filter_jit
    def block_fwd_bwd(block, x):
        return eqx.filter_value_and_grad(
            lambda b, x: jnp.sum(b(x)[0]))(block, x)

    t, _ = time_fn(block_fwd_bwd, block, x_layer)
    results["block_fwd_bwd_ms"] = t * 1000

    return results


# ---------------------------------------------------------------------------
# Full model profiling
# ---------------------------------------------------------------------------

def profile_full_model(config, B, key, hardware):
    """Profile full model train step."""
    T = config.sequence_len

    model = Atlas(config, key=key)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    n_embed_params = model.wte.weight.size

    # Cast to bf16 if on GPU
    if hardware["device"] == "gpu":
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

    # Compilation time
    t0 = time.perf_counter()
    out = train_fwd_bwd(model, idx)
    jax.block_until_ready(out)
    compile_time = time.perf_counter() - t0

    # Steady-state timing
    t, _ = time_fn(train_fwd_bwd, model, idx, warmup=2, repeats=5)

    tokens = B * T
    tps = tokens / t
    flops = count_flops(config, n_params, n_embed_params)

    mfu = None
    if hardware["peak_tflops"] is not None:
        mfu = (flops["total_per_token"] * tps) / (hardware["peak_tflops"] * 1e12) * 100

    # Memory stats (GPU only)
    peak_memory_mb = None
    if hardware["device"] == "gpu":
        try:
            stats = jax.local_devices()[0].memory_stats()
            if stats and "peak_bytes_in_use" in stats:
                peak_memory_mb = stats["peak_bytes_in_use"] / 1e6
        except Exception:
            pass

    return {
        "compile_time_s": compile_time,
        "step_time_ms": t * 1000,
        "tokens_per_second": tps,
        "mfu_percent": mfu,
        "n_params_M": n_params / 1e6,
        "n_embed_params_M": n_embed_params / 1e6,
        "peak_memory_mb": peak_memory_mb,
        "flops": flops,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_human(config, B, hardware, components, model_results):
    """Format results as human-readable string."""
    lines = []
    lines.append("Atlas Profiling Results")
    lines.append("=" * 60)
    lines.append(f"Hardware: {hardware['name']} ({hardware['device']})"
                 + (f" | Peak: {hardware['peak_tflops']} TFLOPS" if hardware["peak_tflops"] else ""))
    lines.append(f"Config:  B={B}, T={config.sequence_len}, L={config.n_layer}, "
                 f"D={config.n_embd}, H={config.n_head}, cs={config.chunk_size}")
    lines.append(f"Memory:  deep={config.deep_memory}, expand={config.memory_expand}, "
                 f"ns={config.ns_steps}, omega_w={config.omega_window}, poly={config.poly_degree}")
    lines.append("")

    lines.append(f"Compilation: {model_results['compile_time_s']:.1f}s")
    lines.append(f"Parameters:  {model_results['n_params_M']:.1f}M")
    lines.append("")

    # Component breakdown
    lines.append("Component Breakdown (per chunk, 1 layer):")
    lines.append("-" * 60)

    block_total = components.get("block_fwd_bwd_ms", 1)
    items = [
        ("PE STE fwd+bwd", "pe_ste_fwd_bwd_ms"),
        ("PE full fwd+bwd", "pe_full_fwd_bwd_ms"),
        ("Linear scan fwd+bwd", "scan_fwd_bwd_ms"),
        ("Memory layer fwd+bwd", "memory_layer_fwd_bwd_ms"),
        ("MLP fwd+bwd", "mlp_fwd_bwd_ms"),
        ("Block fwd+bwd", "block_fwd_bwd_ms"),
    ]
    for label, key in items:
        val = components.get(key, 0)
        pct = val / block_total * 100 if block_total > 0 else 0
        lines.append(f"  {label:30s} {val:8.2f} ms  ({pct:5.1f}% of block)")

    lines.append("")
    lines.append("Full Model:")
    lines.append("-" * 60)
    lines.append(f"  Step time:       {model_results['step_time_ms']:8.1f} ms")
    lines.append(f"  Throughput:      {model_results['tokens_per_second']:8.0f} tok/s")
    if model_results["mfu_percent"] is not None:
        lines.append(f"  MFU:             {model_results['mfu_percent']:8.2f}%")
    else:
        lines.append(f"  MFU:             N/A (CPU)")

    if model_results["peak_memory_mb"] is not None:
        lines.append(f"  Peak memory:     {model_results['peak_memory_mb']:8.0f} MB")

    lines.append("")
    lines.append("FLOP Breakdown (per token, fwd+bwd):")
    lines.append("-" * 60)
    flops = model_results["flops"]
    lines.append(f"  Standard (6N):   {flops['standard_6N']:>12,}")
    lines.append(f"  Memory ops:      {flops['memory_ops_per_token']:>12,}")
    lines.append(f"  Total:           {flops['total_per_token']:>12,}")
    lines.append(f"  Memory fraction: {flops['memory_fraction']:8.1%}")

    # Identify bottlenecks
    lines.append("")
    lines.append("Top Bottlenecks:")
    lines.append("-" * 60)
    sortable = [(components.get(k, 0), label) for label, k in items[:5]]
    sortable.sort(reverse=True)
    for i, (val, label) in enumerate(sortable[:3], 1):
        lines.append(f"  {i}. {label} ({val:.2f} ms)")

    return "\n".join(lines)


def build_json(config, B, hardware, components, model_results):
    """Build structured JSON output."""
    from dataclasses import asdict
    return {
        "config": asdict(config),
        "batch_size": B,
        "hardware": hardware,
        "compilation_time_s": model_results["compile_time_s"],
        "components": components,
        "throughput": {
            "tokens_per_second": model_results["tokens_per_second"],
            "ms_per_step": model_results["step_time_ms"],
            "mfu_percent": model_results["mfu_percent"],
        },
        "memory": {
            "model_params_M": model_results["n_params_M"],
            "peak_memory_mb": model_results["peak_memory_mb"],
        },
        "flop_breakdown": model_results["flops"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Atlas profiling baseline")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                        help="Use a preset config (laptop, small_gpu, h100)")
    parser.add_argument("--json", type=str, default=None,
                        help="Write JSON results to this file")
    parser.add_argument("--gpu-peak-tflops", type=float, default=None,
                        help="Override GPU peak TFLOPS for MFU calculation")
    parser.add_argument("--matmul-precision", type=str, default="float32",
                        choices=["float32", "high", "default"])

    # Config overrides
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--omega-window", type=int, default=None)
    parser.add_argument("--poly-degree", type=int, default=None)
    parser.add_argument("--memory-expand", type=int, default=None)
    parser.add_argument("--ns-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    # Build config from preset + overrides
    preset = PRESETS.get(args.preset or "laptop", PRESETS["laptop"])
    B = args.batch_size or preset.get("batch_size", 1)
    T = args.seq_len or preset.get("seq_len", 512)

    config_kwargs = {}
    for field in ["n_layer", "n_head", "n_embd", "chunk_size", "omega_window",
                  "poly_degree", "memory_expand", "ns_steps"]:
        cli_val = getattr(args, field.replace("-", "_"), None)
        if cli_val is not None:
            config_kwargs[field] = cli_val
        elif field in preset:
            config_kwargs[field] = preset[field]

    for field in ["deep_memory", "pe_ste"]:
        if field in preset:
            config_kwargs[field] = preset[field]

    config = AtlasConfig(sequence_len=T, **config_kwargs)

    # Detect hardware
    hardware = detect_hardware()
    if args.gpu_peak_tflops is not None:
        hardware["peak_tflops"] = args.gpu_peak_tflops

    print(f"JAX {jax.__version__} | {hardware['device']}: {hardware['name']}")
    print(f"Profiling with B={B}, T={T}, preset={args.preset or 'laptop'}")
    print()

    key = jax.random.PRNGKey(args.seed)
    k1, k2 = jax.random.split(key)

    # Profile components
    print("Profiling components...")
    components = profile_components(config, B, k1)

    # Profile full model
    print("Profiling full model (includes compilation)...")
    model_results = profile_full_model(config, B, k2, hardware)

    # Output
    print()
    print(format_human(config, B, hardware, components, model_results))

    if args.json:
        data = build_json(config, B, hardware, components, model_results)
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nJSON results written to {args.json}")


if __name__ == "__main__":
    main()
