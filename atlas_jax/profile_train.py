"""Profile training steps of Atlas with fused Triton kernel.

Uses jax.profiler.trace to produce a Perfetto-compatible trace.
Fused config: memory_expand=1, ns_steps=3, pe_ste=True, fused_chunk=True.

Usage:
    python -m atlas_jax.profile_train                          # auto-detect GPU
    python -m atlas_jax.profile_train --trace-dir /tmp/trace   # custom output dir
"""

import argparse
import math
import os
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas
from atlas_jax.data import data_loader
from atlas_jax.tokenizer import get_tokenizer


@eqx.filter_jit(donate='warn')
def train_step(model, opt_state, optimizer, inputs, targets):
    def loss_fn(model):
        logits, _ = model(inputs)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        loss = -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])
        return loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


def main():
    parser = argparse.ArgumentParser(description='Profile Atlas training (paper-exact)')
    parser.add_argument('--trace-dir', type=str, default='/p/scratch/westai0047/nanochat/traces/atlas_fused',
                        help='Directory for JAX trace output')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (small for profiling)')
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--warmup-steps', type=int, default=3,
                        help='Warmup steps before tracing')
    parser.add_argument('--trace-steps', type=int, default=8,
                        help='Number of steps to trace')
    parser.add_argument('--matmul-precision', type=str, default='high',
                        choices=['float32', 'high', 'default'])
    parser.add_argument('--data-dir', type=str,
                        default='/p/scratch/westai0047/nanochat/base_data_climbmix')
    parser.add_argument('--tokenizer-dir', type=str,
                        default='/p/scratch/westai0047/nanochat/tokenizer')
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    print(f"JAX {jax.__version__} | devices: {jax.devices()} | precision: {args.matmul_precision}")

    # Fused Triton kernel config: requires memory_expand=1 (D==E)
    config = AtlasConfig(
        sequence_len=args.seq_len,
        n_layer=8,
        n_head=8,
        n_embd=512,
        chunk_size=64,
        ns_steps=3,
        omega_window=16,
        poly_degree=3,
        deep_memory=True,
        memory_expand=1,       # D==E required for fused Triton kernel
        pe_ste=True,           # STE backward (identity, ~free)
        use_checkpoint=True,
        fused_chunk=True,      # FlashATLAS Triton kernel
    )
    print(f"Fused kernel config: {config}")
    print(f"  memory_expand=1, ns_steps=3, pe_ste=True, fused_chunk=True")

    key = jax.random.PRNGKey(42)
    key, model_key = jax.random.split(key)
    model = Atlas(config, key=model_key)

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Parameters: {n_params:,}")

    # Cast to bf16
    def _to_bf16(x):
        return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
    model = jax.tree.map(_to_bf16, model, is_leaf=eqx.is_array)
    print("Model cast to bf16")

    # Optimizer
    lr = 3e-3
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=0.1),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Data
    tokenizer = get_tokenizer(args.tokenizer_dir)
    loader = data_loader(args.data_dir, tokenizer, args.batch_size, args.seq_len, split='train')
    tokens_per_step = args.batch_size * args.seq_len

    # --- Compilation ---
    print(f"\nCompiling (B={args.batch_size}, T={args.seq_len})...")
    t0 = time.time()
    inputs, targets = next(loader)
    model, opt_state, loss = train_step(model, opt_state, optimizer, inputs, targets)
    float(loss)  # block
    print(f"Compilation: {time.time() - t0:.1f}s | loss: {float(loss):.4f}")

    # --- Warmup ---
    print(f"\nWarmup ({args.warmup_steps} steps)...")
    for i in range(args.warmup_steps):
        inputs, targets = next(loader)
        model, opt_state, loss = train_step(model, opt_state, optimizer, inputs, targets)
        float(loss)
    print(f"Warmup done | loss: {float(loss):.4f}")

    # --- Traced steps ---
    os.makedirs(args.trace_dir, exist_ok=True)
    print(f"\nTracing {args.trace_steps} steps → {args.trace_dir}")

    with jax.profiler.trace(args.trace_dir):
        for i in range(args.trace_steps):
            t0 = time.time()
            inputs, targets = next(loader)
            model, opt_state, loss = train_step(model, opt_state, optimizer, inputs, targets)
            loss_val = float(loss)
            dt = time.time() - t0
            tps = tokens_per_step / dt
            print(f"  step {i+1:2d} | loss {loss_val:.4f} | {dt*1000:.0f}ms | {tps:.0f} tok/s")

    print(f"\nTrace saved to {args.trace_dir}")
    print("View with: https://ui.perfetto.dev/ (upload the .json.gz file)")
    print("Or: tensorboard --logdir", args.trace_dir)


if __name__ == '__main__':
    main()
