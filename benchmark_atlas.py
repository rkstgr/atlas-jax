#!/usr/bin/env python3
"""
Unified benchmark runner for Atlas implementations.

Launches training for each implementation/mode combination as a subprocess,
writing JSONL metrics to a results directory. Each run gets a fresh process
to avoid GPU memory fragmentation.

Usage:
    python benchmark_atlas.py --results-dir results/run1
    python benchmark_atlas.py --results-dir results/run1 --runs pytorch-base,jax-fused
    python benchmark_atlas.py --results-dir results/run1 --num-batches 50  # smoke test
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path("/p/project1/westai0047")

# Each repo has its own venv — use the correct Python for each
PYTHON = {
    "pytorch": str(BASE_DIR / "atlas-pytorch/.venv/bin/python"),
    "rnn":     str(BASE_DIR / "atlas-rnn/.venv/bin/python"),
    "jax":     str(BASE_DIR / "atlas-jax/.venv/bin/python"),
}

# Common config for fair comparison
COMMON = dict(
    dim=256,
    depth=4,
    heads=4,
    dim_head=64,
    omega_window=2,
    poly_degree=2,
    poly_mode="elementwise",
    batch_size=16,
    seq_len=512,
    grad_accum=1,
    lr=2e-4,
    weight_decay=0.01,
    grad_clip=0.5,
    num_batches=2000,
    validate_every=100,
    eval_steps=10,
    warmup_steps=10,
    seed=42,
)

# Per-run config overrides (PyTorch OOMs at batch=16 due to vmap(grad))
RUN_OVERRIDES = {
    "pytorch-base": {"batch_size": 4, "grad_accum": 4},
    "pytorch-fast": {"batch_size": 4, "grad_accum": 4},
}

# Run definitions: each has a name, working dir, command template
# All use Atlas/LMM architecture (pure memory, no attention) for fair comparison
RUNS = {
    "pytorch-base": {
        "impl": "pytorch",
        "mode": "base",
        "cwd": BASE_DIR / "atlas-pytorch",
        "cmd": [
            "__PYTHON__", "train_atlas.py",
            "--dim", "{dim}", "--depth", "{depth}",
            "--heads", "{heads}", "--dim-head", "{dim_head}",
            "--omega-window", "{omega_window}",
            "--poly-mode", "{poly_mode}", "--poly-degree", "{poly_degree}",
            "--batch-size", "{batch_size}", "--seq-len", "{seq_len}",
            "--grad-accum", "{grad_accum}",
            "--learning-rate", "{lr}",
            "--weight-decay", "{weight_decay}",
            "--grad-clip", "{grad_clip}",
            "--num-batches", "{num_batches}",
            "--validate-every", "{validate_every}",
            "--eval-steps", "{eval_steps}",
            "--warmup-steps", "{warmup_steps}",
            "--seed", "{seed}",
            "--no-accelerated-scan",  # base mode: no Triton scan
            "--force-f32",
            "--no-generate",
            "--metrics-file", "{metrics_file}",
        ],
    },
    "pytorch-fast": {
        "impl": "pytorch",
        "mode": "fast",
        "cwd": BASE_DIR / "atlas-pytorch",
        "cmd": [
            "__PYTHON__", "train_atlas.py",
            "--dim", "{dim}", "--depth", "{depth}",
            "--heads", "{heads}", "--dim-head", "{dim_head}",
            "--omega-window", "{omega_window}",
            "--poly-mode", "{poly_mode}", "--poly-degree", "{poly_degree}",
            "--batch-size", "{batch_size}", "--seq-len", "{seq_len}",
            "--grad-accum", "{grad_accum}",
            "--learning-rate", "{lr}",
            "--weight-decay", "{weight_decay}",
            "--grad-clip", "{grad_clip}",
            "--num-batches", "{num_batches}",
            "--validate-every", "{validate_every}",
            "--eval-steps", "{eval_steps}",
            "--warmup-steps", "{warmup_steps}",
            "--seed", "{seed}",
            # accelerated_scan ON (default)
            "--force-f32",
            "--no-generate",
            "--metrics-file", "{metrics_file}",
        ],
    },
    "rnn-base": {
        "impl": "rnn",
        "mode": "base",
        "cwd": BASE_DIR / "atlas-rnn",
        "cmd": [
            "__PYTHON__", "train_rnn_transformer.py",
            "--arch", "lmm",
            "--model", "omeganet",
            "--dim", "{dim}", "--depth", "{depth}",
            "--heads", "{heads}", "--dim-head", "{dim_head}",
            "--omega-window", "{omega_window}",
            "--poly-mode", "{poly_mode}", "--poly-degree", "{poly_degree}",
            "--batch-size", "{batch_size}", "--seq-len", "{seq_len}",
            "--grad-accum", "{grad_accum}",
            "--lr", "{lr}",
            "--weight-decay", "{weight_decay}",
            "--grad-clip", "{grad_clip}",
            "--num-batches", "{num_batches}",
            "--validate-every", "{validate_every}",
            "--eval-steps", "{eval_steps}",
            "--warmup-steps", "{warmup_steps}",
            "--seed", "{seed}",
            "--force-f32",
            "--metrics-file", "{metrics_file}",
        ],
    },
    "jax-base": {
        "impl": "jax",
        "mode": "base",
        "cwd": BASE_DIR / "atlas-jax",
        "cmd": [
            "__PYTHON__", "-m", "scripts.train_enwik8",
            "--model", "lmm",
            "--dim", "{dim}", "--depth", "{depth}",
            "--heads", "{heads}", "--dim-head", "{dim_head}",
            "--omega-window", "{omega_window}",
            "--poly-degree", "{poly_degree}",
            "--memory-expand", "2",
            "--batch-size", "{batch_size}", "--seq-len", "{seq_len}",
            "--grad-accum", "{grad_accum}",
            "--lr", "{lr}",
            "--weight-decay", "{weight_decay}",
            "--grad-clip", "{grad_clip}",
            "--num-batches", "{num_batches}",
            "--validate-every", "{validate_every}",
            "--eval-steps", "{eval_steps}",
            "--warmup-steps", "{warmup_steps}",
            "--seed", "{seed}",
            "--stop-grad-chunks",
            "--metrics-file", "{metrics_file}",
        ],
    },
    "jax-fused": {
        "impl": "jax",
        "mode": "fused",
        "cwd": BASE_DIR / "atlas-jax",
        "cmd": [
            "__PYTHON__", "-m", "scripts.train_enwik8",
            "--model", "lmm",
            "--dim", "{dim}", "--depth", "{depth}",
            "--heads", "{heads}", "--dim-head", "{dim_head}",
            "--omega-window", "{omega_window}",
            "--poly-degree", "{poly_degree}",
            "--memory-expand", "1",  # required for fused kernel
            "--fused-chunk",
            "--batch-size", "{batch_size}", "--seq-len", "{seq_len}",
            "--grad-accum", "{grad_accum}",
            "--lr", "{lr}",
            "--weight-decay", "{weight_decay}",
            "--grad-clip", "{grad_clip}",
            "--num-batches", "{num_batches}",
            "--validate-every", "{validate_every}",
            "--eval-steps", "{eval_steps}",
            "--warmup-steps", "{warmup_steps}",
            "--seed", "{seed}",
            "--stop-grad-chunks",
            "--metrics-file", "{metrics_file}",
        ],
    },
}


def run_benchmark(name, run_def, config, results_dir):
    """Run a single benchmark configuration as a subprocess."""
    metrics_file = results_dir / f"{name}.jsonl"
    run_config = {**config, **RUN_OVERRIDES.get(name, {})}
    config_with_file = {**run_config, "metrics_file": str(metrics_file)}

    # Format command with config values, substitute correct Python per impl
    impl = run_def["impl"]
    python_bin = PYTHON[impl]
    cmd = [str(c).format(**config_with_file).replace("__PYTHON__", python_bin)
           for c in run_def["cmd"]]
    cwd = run_def["cwd"]

    print(f"\n{'='*70}")
    print(f"  Running: {name} ({run_def['impl']}/{run_def['mode']})")
    print(f"  CWD: {cwd}")
    print(f"  Metrics: {metrics_file}")
    print(f"  Cmd: {' '.join(cmd[:6])}...")
    print(f"{'='*70}\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Avoid disk quota issues from Triton/CUDA kernel compilation in home dir
    env.setdefault("TMPDIR", "/tmp")
    env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
    if run_def["impl"] == "jax":
        env.setdefault("TIKTOKEN_CACHE_DIR", "/p/project1/westai0047/tiktoken_cache")

    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=str(cwd), env=env,
        stdout=sys.stdout, stderr=sys.stderr,
    )
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n  [{name}] {status} in {elapsed:.1f}s")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Atlas cross-implementation benchmark')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory for JSONL results')
    parser.add_argument('--runs', type=str, default=None,
                        help='Comma-separated run names (default: all)')
    parser.add_argument('--num-batches', type=int, default=None,
                        help='Override number of training steps')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--dim', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build config with overrides
    config = dict(COMMON)
    for key in ['num_batches', 'batch_size', 'seq_len', 'dim', 'depth']:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    # Select runs
    if args.runs:
        run_names = [r.strip() for r in args.runs.split(',')]
    else:
        run_names = list(RUNS.keys())

    for name in run_names:
        if name not in RUNS:
            print(f"Unknown run: {name}. Available: {list(RUNS.keys())}")
            sys.exit(1)

    # Save benchmark config
    meta = {
        "config": config,
        "runs": run_names,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (results_dir / "benchmark_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Atlas Cross-Implementation Benchmark")
    print(f"Results: {results_dir}")
    print(f"Runs: {run_names}")
    print(f"Config: dim={config['dim']}, depth={config['depth']}, "
          f"heads={config['heads']}, steps={config['num_batches']}")
    print(f"{'='*70}")

    results = {}
    for name in run_names:
        ok = run_benchmark(name, RUNS[name], config, results_dir)
        results[name] = "OK" if ok else "FAILED"

    print(f"\n{'='*70}")
    print("Summary:")
    for name, status in results.items():
        print(f"  {name:20s} {status}")
    print(f"{'='*70}")
    print(f"Results written to {results_dir}/")
    print(f"Run: python compare_results.py --results-dir {results_dir}")


if __name__ == "__main__":
    main()
