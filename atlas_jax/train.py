"""Training loop for Atlas-JAX.

Data-parallel training via shard_map:
- shard_map for multi-GPU (no vmap, so Triton kernels work)
- filter_jit for single-GPU
- eqx.filter_value_and_grad for differentiation
- MFU and BPB reporting
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from functools import partial
from dataclasses import asdict

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas
from atlas_jax.data import data_loader
from atlas_jax.tokenizer import get_tokenizer


# ============================================================================
# Metrics: loss, grad upcasting, FLOPs + BPB accounting
# ============================================================================

CHARS_PER_TOKEN = 3.3
BPB_FACTOR = 1.0 / (math.log(2) * CHARS_PER_TOKEN)

GPU_PEAK_TFLOPS = {"H100": 989.4, "RTX8000": 32.6}


def loss_fn(model, inputs, targets, *, dropout_key=None):
    """Standard next-token cross-entropy in nats."""
    logits, _ = model(inputs, dropout_key=dropout_key)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])


def upcast_grads(grads):
    """Cast bf16 gradients to f32 for numerically stable optimizer updates."""
    def _cast(x):
        if eqx.is_array(x) and x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
        return x
    return jax.tree.map(_cast, grads, is_leaf=eqx.is_array)


def estimate_flops_per_token(config, n_params, n_embed_params):
    """FLOPs per token for Atlas (forward + backward = 6N + memory ops)."""
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


# ============================================================================
# Checkpointing: atomic save + optional load
# ============================================================================

def _to_cpu(x):
    """Move a JAX array to CPU for serialization; pass through non-arrays."""
    if eqx.is_array(x):
        return jax.device_get(x)
    return x


def save_checkpoint(model, opt_state, step, ckpt_dir, rank=0):
    """Save model + optimizer state to `ckpt_dir`. No-op if `rank != 0`."""
    if rank != 0:
        return
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_cpu = jax.tree.map(_to_cpu, model, is_leaf=eqx.is_array)
    opt_cpu = jax.tree.map(_to_cpu, opt_state, is_leaf=eqx.is_array)

    tmp_model = ckpt_dir / "model.eqx.tmp"
    tmp_opt = ckpt_dir / "opt_state.eqx.tmp"
    tmp_meta = ckpt_dir / "meta.tmp"

    eqx.tree_serialise_leaves(str(tmp_model), model_cpu)
    eqx.tree_serialise_leaves(str(tmp_opt), opt_cpu)
    with open(tmp_meta, "w") as f:
        f.write(f"{step}\n")

    tmp_model.rename(ckpt_dir / "model.eqx")
    tmp_opt.rename(ckpt_dir / "opt_state.eqx")
    tmp_meta.rename(ckpt_dir / "meta.txt")
    print(f"[ckpt] Saved step {step} to {ckpt_dir}", flush=True)


def load_checkpoint(model, opt_state, ckpt_dir):
    """Load model + optimizer state from `ckpt_dir`.

    Returns `(model, opt_state, step)`. When no checkpoint is present, returns
    the inputs unchanged with `step == 0`.
    """
    ckpt_dir = Path(ckpt_dir)
    model_path = ckpt_dir / "model.eqx"
    opt_path = ckpt_dir / "opt_state.eqx"
    meta_path = ckpt_dir / "meta.txt"

    if not model_path.exists():
        return model, opt_state, 0

    model = eqx.tree_deserialise_leaves(str(model_path), model)
    opt_state = eqx.tree_deserialise_leaves(str(opt_path), opt_state)
    with open(meta_path) as f:
        step = int(f.read().strip())

    print(f"[ckpt] Resumed from step {step} at {ckpt_dir}", flush=True)
    return model, opt_state, step


# ============================================================================
# Train / eval steps
# ============================================================================

def make_train_step(mesh):
    """Create train step.

    For multi-GPU: uses shard_map to run per-device on local batch shards.
    For single-GPU: plain filter_jit (no shard_map overhead).
    """
    n_devices = len(mesh.devices) if mesh is not None else 1

    if n_devices > 1:
        def _train_body(model, opt_state, optimizer, inputs, targets):
            loss, grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=False)(model, inputs, targets)
            grads = upcast_grads(grads)
            grads = jax.lax.pmean(grads, axis_name='data')
            loss = jax.lax.pmean(loss, axis_name='data')
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss

        @eqx.filter_jit(donate='all')
        def train_step(model, opt_state, optimizer, inputs, targets):
            model_spec = jax.tree.map(lambda _: P(), eqx.filter(model, eqx.is_array))
            opt_spec = jax.tree.map(lambda _: P(), eqx.filter(opt_state, eqx.is_array))
            data_spec = P('data')

            in_specs = (model_spec, opt_spec, P(), data_spec, data_spec)
            out_specs = (model_spec, opt_spec, P())

            return shard_map(
                _train_body,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_rep=False,
            )(model, opt_state, optimizer, inputs, targets)

        return train_step
    else:
        @eqx.filter_jit(donate='all')
        def train_step(model, opt_state, optimizer, inputs, targets):
            loss, grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=False)(model, inputs, targets)
            grads = upcast_grads(grads)
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss

        return train_step


def make_eval_step(mesh):
    """Create eval step."""
    n_devices = len(mesh.devices) if mesh is not None else 1

    if n_devices > 1:
        def _eval_body(model, inputs, targets):
            loss = loss_fn(model, inputs, targets)
            return jax.lax.pmean(loss, axis_name='data')

        @eqx.filter_jit
        def eval_step(model, inputs, targets):
            model_spec = jax.tree.map(lambda _: P(), eqx.filter(model, eqx.is_array))
            in_specs = (model_spec, P('data'), P('data'))
            out_specs = P()

            return shard_map(
                _eval_body,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_rep=False,
            )(model, inputs, targets)

        return eval_step
    else:
        @eqx.filter_jit
        def eval_step(model, inputs, targets):
            return loss_fn(model, inputs, targets)

        return eval_step


def main():
    parser = argparse.ArgumentParser(description='Atlas-JAX Training')
    parser.add_argument('--n-layer', type=int, default=8)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-embd', type=int, default=448)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--omega-window', type=int, default=16)
    parser.add_argument('--poly-degree', type=int, default=3)
    parser.add_argument('--deep-memory', action='store_true', default=True)
    parser.add_argument('--no-deep-memory', dest='deep_memory', action='store_false')
    parser.add_argument('--memory-expand', type=int, default=1)
    parser.add_argument('--pe-ste', action='store_true', default=True)
    parser.add_argument('--no-pe-ste', dest='pe_ste', action='store_false')
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--ns-steps', type=int, default=3)
    parser.add_argument('--no-checkpoint', action='store_true', default=False)
    parser.add_argument('--fused-chunk', action='store_true', default=False,
                        help='FlashATLAS fused Triton kernel (requires memory-expand=1)')

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--total-steps', type=int, default=2000)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--eval-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-peak-tflops', type=float, default=989.4,
                        help='GPU peak bf16 TFLOPS for MFU calc (H100=989.4)')

    parser.add_argument('--matmul-precision', type=str, default='float32',
                        choices=['float32', 'high', 'default'])
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--tokenizer-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default='out/atlas-jax')
    parser.add_argument('--time-budget', type=int, default=0,
                        help='Training time budget in seconds (0 = use total-steps)')
    parser.add_argument('--max-tokens', type=float, default=0,
                        help='Max tokens to train on (0 = no limit). Accepts scientific notation e.g. 5.6e9')
    parser.add_argument('--target-bpb', type=float, default=0,
                        help='Stop early if val_bpb reaches this target (0 = disabled)')

    args = parser.parse_args()

    jax.config.update("jax_compilation_cache_dir", "/p/scratch/westai0047/nanochat/jax_cache")
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    n_devices = len(jax.devices())
    print(f"JAX {jax.__version__} | devices: {jax.devices()} | precision: {args.matmul_precision}")

    if n_devices > 1 and args.batch_size % n_devices != 0:
        raise ValueError(f"batch_size={args.batch_size} must be divisible by n_devices={n_devices}")

    if n_devices > 1:
        mesh = Mesh(jax.devices(), axis_names=('data',))
        data_sharding = NamedSharding(mesh, P('data'))
        replicate_sharding = NamedSharding(mesh, P())
    else:
        mesh = None

    key = jax.random.PRNGKey(args.seed)

    config = AtlasConfig(
        sequence_len=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        chunk_size=args.chunk_size,
        ns_steps=args.ns_steps,
        omega_window=args.omega_window,
        poly_degree=args.poly_degree,
        deep_memory=args.deep_memory,
        memory_expand=args.memory_expand,
        pe_ste=args.pe_ste,
        use_checkpoint=not args.no_checkpoint,
        fused_chunk=args.fused_chunk,
    )
    print(f"Config: {asdict(config)}")

    key, model_key = jax.random.split(key)
    model = Atlas(config, key=model_key)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    n_embed_params = model.wte.weight.size
    flops_per_token = estimate_flops_per_token(config, n_params, n_embed_params)
    print(f"Parameters: {n_params:,} | FLOPs/token: {flops_per_token:,.0f}")

    def _to_bf16(x):
        return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
    model = jax.tree.map(_to_bf16, model, is_leaf=eqx.is_array)
    print(f"Model cast to bf16")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.total_steps,
        end_value=args.lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    if n_devices > 1:
        def _replicate(x):
            return jax.device_put(x, replicate_sharding) if eqx.is_array(x) else x
        model = jax.tree.map(_replicate, model, is_leaf=eqx.is_array)
        opt_state = jax.tree.map(_replicate, opt_state, is_leaf=eqx.is_array)
        print(f"Data parallel (shard_map): batch {args.batch_size} across {n_devices} GPUs "
              f"({args.batch_size // n_devices}/GPU)")

    train_step = make_train_step(mesh)
    eval_step = make_eval_step(mesh)

    tokenizer = get_tokenizer(args.tokenizer_dir)
    train_loader = data_loader(
        args.data_dir, tokenizer, args.batch_size, args.seq_len, split='train')
    val_loader = data_loader(
        args.data_dir, tokenizer, args.batch_size, args.seq_len, split='val')

    tokens_per_step = args.batch_size * args.seq_len
    gpu_peak_flops = args.gpu_peak_tflops * 1e12 * n_devices

    if args.max_tokens > 0:
        max_steps = int(args.max_tokens / tokens_per_step)
        args.total_steps = max_steps
        print(f"Max tokens: {args.max_tokens:.2e} -> {max_steps} steps")
    if args.target_bpb > 0:
        print(f"Early stopping target: val_bpb <= {args.target_bpb}")

    print(f"Tokens/step: {tokens_per_step:,} | GPU peak: {args.gpu_peak_tflops} TFLOPS")
    use_time_budget = args.time_budget > 0
    if use_time_budget:
        print(f"Time budget: {args.time_budget}s")
    else:
        print(f"Training for {args.total_steps} steps")
    print("-" * 80)

    def shard_batch(x):
        if n_devices > 1:
            return jax.device_put(x, data_sharding)
        return x

    print("Compiling train step (first step will be slow)...")
    t_compile = time.time()
    inputs, targets = next(train_loader)
    inputs, targets = shard_batch(inputs), shard_batch(targets)
    model, opt_state, loss = train_step(model, opt_state, optimizer, inputs, targets)
    float(loss)
    compile_time = time.time() - t_compile
    print(f"Compilation done in {compile_time:.1f}s | initial loss: {float(loss):.4f}")
    print("-" * 80)

    training_start = time.time()
    step = 0
    step_times = []

    while True:
        step += 1
        if use_time_budget:
            if time.time() - training_start >= args.time_budget:
                break
        else:
            if step >= args.total_steps:
                break

        t0 = time.time()
        inputs, targets = next(train_loader)
        inputs, targets = shard_batch(inputs), shard_batch(targets)
        model, opt_state, loss = train_step(model, opt_state, optimizer, inputs, targets)
        loss_val = float(loss)
        dt = time.time() - t0
        step_times.append(dt)

        if step % 10 == 0 or step < 5:
            tps = tokens_per_step / dt
            mfu = (flops_per_token * tps) / gpu_peak_flops * 100
            bpb = loss_val * BPB_FACTOR
            elapsed = time.time() - training_start
            print(f"step {step:5d} | loss {loss_val:.4f} | bpb {bpb:.4f} | "
                  f"{dt*1000:.0f}ms | {tps:.0f} tok/s | MFU {mfu:.2f}% | {elapsed:.0f}s")

        if args.eval_every > 0 and step % args.eval_every == 0:
            val_losses = []
            for _ in range(args.eval_steps):
                val_inputs, val_targets = next(val_loader)
                val_inputs, val_targets = shard_batch(val_inputs), shard_batch(val_targets)
                val_loss = eval_step(model, val_inputs, val_targets)
                val_losses.append(float(val_loss))
            avg_val_loss = sum(val_losses) / len(val_losses)
            val_bpb = avg_val_loss * BPB_FACTOR
            total_tok = step * tokens_per_step
            print(f"  >>> EVAL | val_loss {avg_val_loss:.4f} | val_bpb {val_bpb:.4f} | tokens {total_tok/1e6:.0f}M")
            if args.target_bpb > 0 and val_bpb <= args.target_bpb:
                print(f"  >>> TARGET REACHED: val_bpb {val_bpb:.4f} <= {args.target_bpb}")
                break

    training_seconds = time.time() - training_start

    print("-" * 80)
    print("Final evaluation...")
    val_losses = []
    for _ in range(args.eval_steps):
        val_inputs, val_targets = next(val_loader)
        val_inputs, val_targets = shard_batch(val_inputs), shard_batch(val_targets)
        val_loss = eval_step(model, val_inputs, val_targets)
        val_losses.append(float(val_loss))
    avg_val_loss = sum(val_losses) / len(val_losses)
    val_bpb = avg_val_loss * BPB_FACTOR
    total_seconds = time.time() - training_start
    total_tokens = step * tokens_per_step

    warmup_skip = min(5, len(step_times))
    if len(step_times) > warmup_skip:
        avg_dt = sum(step_times[warmup_skip:]) / len(step_times[warmup_skip:])
        avg_tps = tokens_per_step / avg_dt
        avg_mfu = (flops_per_token * avg_tps) / gpu_peak_flops * 100
    else:
        avg_tps = total_tokens / max(training_seconds, 1e-6)
        avg_mfu = (flops_per_token * avg_tps) / gpu_peak_flops * 100

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"val_loss:         {avg_val_loss:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"mfu_percent:      {avg_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {n_params / 1e6:.1f}")
    print(f"tokens_per_sec:   {avg_tps:.0f}")
    print(f"ms_per_step:      {avg_dt * 1000:.1f}" if len(step_times) > warmup_skip else "ms_per_step:      0.0")
    print(f"FINAL | val_loss {avg_val_loss:.4f} | val_bpb {val_bpb:.4f}")
    print("Training complete.")


if __name__ == '__main__':
    main()
