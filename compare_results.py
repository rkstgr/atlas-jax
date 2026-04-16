#!/usr/bin/env python3
"""
Compare Atlas benchmark results across implementations.

Reads JSONL files from a results directory and produces:
1. Summary table (loss, BPB, tok/s, memory, wall time)
2. Loss curve plot (matplotlib, saved as PNG)

Usage:
    python compare_results.py --results-dir results/run1
    python compare_results.py --results-dir results/run1 --plot loss_curves.png
"""

import argparse
import json
import sys
from pathlib import Path


def load_run(jsonl_path):
    """Load a JSONL metrics file, returning config, steps, evals, summary."""
    config = {}
    steps = []
    evals = []
    summary = {}

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rtype = record.get("type", "")
            if rtype == "config":
                config = record
            elif rtype == "step":
                steps.append(record)
            elif rtype == "eval":
                evals.append(record)
            elif rtype == "summary":
                summary = record

    return config, steps, evals, summary


def compute_stats(steps):
    """Compute aggregate statistics from step records."""
    if not steps:
        return {}

    losses = [s["loss"] for s in steps]
    bpbs = [s["bpb"] for s in steps]
    tok_s_vals = [s["tok_s"] for s in steps]
    step_times = [s["step_time_ms"] for s in steps]
    mems = [s.get("peak_mem_mb", 0) for s in steps]

    return {
        "final_loss": losses[-1],
        "final_bpb": bpbs[-1],
        "min_loss": min(losses),
        "min_bpb": min(bpbs),
        "mean_tok_s": sum(tok_s_vals) / len(tok_s_vals),
        "median_tok_s": sorted(tok_s_vals)[len(tok_s_vals) // 2],
        "mean_step_ms": sum(step_times) / len(step_times),
        "peak_mem_mb": max(mems) if mems else 0,
        "n_steps": len(steps),
    }


def print_table(runs_data):
    """Print a comparison table."""
    # Header
    cols = [
        ("Run", 20),
        ("Params", 10),
        ("Final Loss", 11),
        ("Final BPB", 10),
        ("Best BPB", 10),
        ("tok/s", 10),
        ("ms/step", 10),
        ("Mem (MB)", 10),
        ("Time (s)", 10),
    ]

    header = " | ".join(f"{name:{width}s}" for name, width in cols)
    sep = "-+-".join("-" * width for _, width in cols)

    print(f"\n{header}")
    print(sep)

    for name, (config, steps, evals, summary) in sorted(runs_data.items()):
        stats = compute_stats(steps)
        if not stats:
            print(f"{name:20s} | {'NO DATA':>10s}")
            continue

        n_params = config.get("n_params", 0)
        total_time = summary.get("total_time_s", 0)

        row = [
            f"{name:20s}",
            f"{n_params/1e6:>9.1f}M",
            f"{stats['final_loss']:>11.4f}",
            f"{stats['final_bpb']:>10.4f}",
            f"{stats['min_bpb']:>10.4f}",
            f"{stats['median_tok_s']:>10.0f}",
            f"{stats['mean_step_ms']:>10.0f}",
            f"{stats['peak_mem_mb']:>10.0f}",
            f"{total_time:>10.1f}",
        ]
        print(" | ".join(row))

    # Eval results
    print(f"\nValidation Results:")
    print(f"{'Run':20s} | {'Step':>6s} | {'Val Loss':>10s} | {'Val BPB':>10s}")
    print("-" * 55)
    for name, (config, steps, evals, summary) in sorted(runs_data.items()):
        if evals:
            last_eval = evals[-1]
            print(f"{name:20s} | {last_eval['step']:>6d} | "
                  f"{last_eval['val_loss']:>10.4f} | {last_eval['val_bpb']:>10.4f}")


def plot_loss_curves(runs_data, output_path):
    """Generate a loss curve comparison plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        'pytorch-base': '#1f77b4',
        'pytorch-fast': '#1f77b4',
        'rnn-base': '#ff7f0e',
        'jax-base': '#2ca02c',
        'jax-fused': '#2ca02c',
    }
    linestyles = {
        'pytorch-base': '-',
        'pytorch-fast': '--',
        'rnn-base': '-',
        'jax-base': '-',
        'jax-fused': '--',
    }

    # Plot 1: Loss curves
    ax = axes[0]
    for name, (config, steps, evals, summary) in sorted(runs_data.items()):
        if not steps:
            continue
        x = [s["step"] for s in steps]
        y = [s["loss"] for s in steps]
        # Subsample for cleaner plot
        if len(x) > 200:
            step_size = len(x) // 200
            x = x[::step_size]
            y = y[::step_size]
        ax.plot(x, y, label=name,
                color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'),
                alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss (nats)')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: BPB curves
    ax = axes[1]
    for name, (config, steps, evals, summary) in sorted(runs_data.items()):
        if not evals:
            continue
        x = [e["step"] for e in evals]
        y = [e["val_bpb"] for e in evals]
        ax.plot(x, y, 'o-', label=name,
                color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'),
                markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Validation BPB')
    ax.set_title('Validation BPB')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Throughput over time
    ax = axes[2]
    for name, (config, steps, evals, summary) in sorted(runs_data.items()):
        if not steps:
            continue
        x = [s["step"] for s in steps]
        y = [s["tok_s"] for s in steps]
        if len(x) > 200:
            step_size = len(x) // 200
            x = x[::step_size]
            y = y[::step_size]
        ax.plot(x, y, label=name,
                color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'),
                alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens/sec')
    ax.set_title('Throughput')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Atlas Cross-Implementation Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Atlas benchmark results')
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--plot', type=str, default=None,
                        help='Output path for loss curve plot (PNG)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    # Load all JSONL files
    runs_data = {}
    for jsonl_file in sorted(results_dir.glob("*.jsonl")):
        name = jsonl_file.stem
        config, steps, evals, summary = load_run(jsonl_file)
        runs_data[name] = (config, steps, evals, summary)
        print(f"Loaded {name}: {len(steps)} steps, {len(evals)} evals")

    if not runs_data:
        print("No JSONL files found")
        sys.exit(1)

    # Print comparison table
    print_table(runs_data)

    # Load and print benchmark meta if available
    meta_file = results_dir / "benchmark_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        print(f"\nBenchmark config: dim={meta['config'].get('dim')}, "
              f"depth={meta['config'].get('depth')}, "
              f"steps={meta['config'].get('num_batches')}")
        print(f"Timestamp: {meta.get('timestamp', 'unknown')}")

    # Plot
    plot_path = args.plot
    if plot_path is None:
        plot_path = str(results_dir / "comparison.png")
    plot_loss_curves(runs_data, plot_path)


if __name__ == "__main__":
    main()
