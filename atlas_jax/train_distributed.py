"""Multi-process distributed training for Atlas-JAX (Option A).

Uses jax.distributed.initialize() for multi-process coordination.
Each SLURM task = 1 process = 1 GPU. Triton kernels run in single-device
context (no vmap/pmap), gradients are synced via shard_map + lax.pmean
over the multi-process mesh.

Launch: srun --ntasks=N --gres=gpu:N python -m atlas_jax.train_distributed [args]
"""

import os
import sys
import time
import math
import argparse
from functools import partial
from dataclasses import asdict

# --- Multi-process setup: MUST happen before any JAX computation ---
_LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", "0"))
_WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", "1"))
_COORDINATOR = os.environ.get("MASTER_ADDR", "localhost")

import jax

# Initialize distributed runtime BEFORE any JAX computation.
# local_device_ids tells JAX which GPU this process owns (don't use CUDA_VISIBLE_DEVICES).
if _WORLD_SIZE > 1:
    jax.distributed.initialize(
        coordinator_address=f"{_COORDINATOR}:29500",
        num_processes=_WORLD_SIZE,
        process_id=_LOCAL_RANK,
        local_device_ids=[_LOCAL_RANK],
    )

import jax.numpy as jnp
import equinox as eqx
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas
from atlas_jax.data import data_loader
from atlas_jax.tokenizer import get_tokenizer
from atlas_jax.train import estimate_flops_per_token


# ---------------------------------------------------------------------------
# Train / eval steps (multi-process via shard_map)
# ---------------------------------------------------------------------------

def _loss_fn(model, inputs, targets):
    logits, _ = model(inputs)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])


def _upcast_grads(grads):
    def _cast(x):
        if eqx.is_array(x) and x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
        return x
    return jax.tree.map(_cast, grads, is_leaf=eqx.is_array)


def make_train_step(mesh, optimizer, model, opt_state):
    """Train step using shard_map over multi-process mesh.

    Each process runs the body on its single local GPU. shard_map provides
    the 'data' named axis for lax.pmean (NCCL all-reduce between processes).
    Optimizer is closed over (not passed through shard_map) since it's not a JAX array.
    """
    n_devices = len(mesh.devices)

    if n_devices > 1:
        def _train_body(model, opt_state, inputs, targets):
            loss, grads = eqx.filter_value_and_grad(
                _loss_fn, has_aux=False)(model, inputs, targets)
            grads = _upcast_grads(grads)
            grads = jax.lax.pmean(grads, axis_name='data')
            loss = jax.lax.pmean(loss, axis_name='data')
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss

        model_spec = jax.tree.map(lambda _: P(), eqx.filter(model, eqx.is_array))
        opt_spec = jax.tree.map(lambda _: P(), eqx.filter(opt_state, eqx.is_array))

        @eqx.filter_jit(donate='all')
        def train_step(model, opt_state, inputs, targets):
            return shard_map(
                _train_body,
                mesh=mesh,
                in_specs=(model_spec, opt_spec, P('data'), P('data')),
                out_specs=(model_spec, opt_spec, P()),
                check_rep=False,
            )(model, opt_state, inputs, targets)

        return train_step
    else:
        @eqx.filter_jit(donate='all')
        def train_step(model, opt_state, inputs, targets):
            loss, grads = eqx.filter_value_and_grad(
                _loss_fn, has_aux=False)(model, inputs, targets)
            grads = _upcast_grads(grads)
            updates, new_opt_state = optimizer.update(grads, opt_state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss

        return train_step


def make_eval_step(mesh):
    n_devices = len(mesh.devices)

    if n_devices > 1:
        def _eval_body(model, inputs, targets):
            loss = _loss_fn(model, inputs, targets)
            return jax.lax.pmean(loss, axis_name='data')

        @eqx.filter_jit
        def eval_step(model, inputs, targets):
            model_spec = jax.tree.map(lambda _: P(), eqx.filter(model, eqx.is_array))
            return shard_map(
                _eval_body,
                mesh=mesh,
                in_specs=(model_spec, P('data'), P('data')),
                out_specs=P(),
                check_rep=False,
            )(model, inputs, targets)

        return eval_step
    else:
        @eqx.filter_jit
        def eval_step(model, inputs, targets):
            return _loss_fn(model, inputs, targets)

        return eval_step


def log(msg, rank=0):
    """Print only on rank 0."""
    if _LOCAL_RANK == rank:
        print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Atlas-JAX Distributed Training (Option A)')
    parser.add_argument('--n-layer', type=int, default=8)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-embd', type=int, default=512)
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
    parser.add_argument('--fused-chunk', action='store_true', default=False)

    parser.add_argument('--batch-size', type=int, default=32,
                        help='GLOBAL batch size (split across all GPUs)')
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--total-steps', type=int, default=2000)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--eval-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-peak-tflops', type=float, default=989.4)

    parser.add_argument('--matmul-precision', type=str, default='high',
                        choices=['float32', 'high', 'default'])
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--tokenizer-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default='out/atlas-jax')
    parser.add_argument('--time-budget', type=int, default=0)
    parser.add_argument('--max-tokens', type=float, default=0)
    parser.add_argument('--target-bpb', type=float, default=0)

    args = parser.parse_args()

    jax.config.update("jax_compilation_cache_dir", "/p/scratch/westai0047/nanochat/jax_cache")
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    n_devices = len(jax.devices())  # global device count across all processes
    local_batch = args.batch_size // n_devices

    log(f"JAX {jax.__version__} | global devices: {jax.devices()} | "
        f"local: {jax.local_devices()} | precision: {args.matmul_precision}")
    log(f"World size: {_WORLD_SIZE} | Rank: {_LOCAL_RANK} | "
        f"Global batch: {args.batch_size} | Per-GPU: {local_batch}")

    if args.batch_size % n_devices != 0:
        raise ValueError(f"batch_size={args.batch_size} must be divisible by n_devices={n_devices}")

    # Multi-process mesh: one device per process, all connected via NCCL
    mesh = Mesh(jax.devices(), axis_names=('data',))
    data_sharding = NamedSharding(mesh, P('data'))
    replicate_sharding = NamedSharding(mesh, P())

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
    log(f"Config: {asdict(config)}")

    key, model_key = jax.random.split(key)
    model = Atlas(config, key=model_key)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    n_embed_params = model.wte.weight.size
    flops_per_token = estimate_flops_per_token(config, n_params, n_embed_params)
    log(f"Parameters: {n_params:,} | FLOPs/token: {flops_per_token:,.0f}")

    # Cast to bf16
    def _to_bf16(x):
        return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
    model = jax.tree.map(_to_bf16, model, is_leaf=eqx.is_array)
    log("Model cast to bf16")

    CHARS_PER_TOKEN = 3.3
    BPB_FACTOR = 1.0 / (math.log(2) * CHARS_PER_TOKEN)

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

    # Replicate model and opt_state across all devices
    def _replicate(x):
        return jax.device_put(x, replicate_sharding) if eqx.is_array(x) else x
    model = jax.tree.map(_replicate, model, is_leaf=eqx.is_array)
    opt_state = jax.tree.map(_replicate, opt_state, is_leaf=eqx.is_array)

    train_step = make_train_step(mesh, optimizer, model, opt_state)
    eval_step = make_eval_step(mesh)

    # Data — each rank reads its own shard of parquet row groups
    tokenizer = get_tokenizer(args.tokenizer_dir)
    train_loader = data_loader(
        args.data_dir, tokenizer, local_batch, args.seq_len, split='train',
        rank=_LOCAL_RANK, world_size=_WORLD_SIZE)
    val_loader = data_loader(
        args.data_dir, tokenizer, local_batch, args.seq_len, split='val',
        rank=_LOCAL_RANK, world_size=_WORLD_SIZE)

    tokens_per_step = args.batch_size * args.seq_len  # global
    gpu_peak_flops = args.gpu_peak_tflops * 1e12 * n_devices

    if args.max_tokens > 0:
        max_steps = int(args.max_tokens / tokens_per_step)
        args.total_steps = max_steps
        log(f"Max tokens: {args.max_tokens:.2e} -> {max_steps} steps")
    if args.target_bpb > 0:
        log(f"Early stopping target: val_bpb <= {args.target_bpb}")

    log(f"Tokens/step: {tokens_per_step:,} | GPU peak: {args.gpu_peak_tflops} TFLOPS")
    use_time_budget = args.time_budget > 0
    if use_time_budget:
        log(f"Time budget: {args.time_budget}s")
    else:
        log(f"Training for {args.total_steps} steps")
    log("-" * 80)

    def make_global_batch(local_inputs, local_targets):
        """Each rank produces local_batch; shard_map expects global array
        sharded along batch dim across the mesh."""
        return (jax.device_put(local_inputs, data_sharding),
                jax.device_put(local_targets, data_sharding))

    # Warmup compilation
    log("Compiling train step (first step will be slow)...")
    t_compile = time.time()
    inputs, targets = next(train_loader)
    inputs, targets = make_global_batch(inputs, targets)
    model, opt_state, loss = train_step(model, opt_state, inputs, targets)
    float(loss)
    compile_time = time.time() - t_compile
    log(f"Compilation done in {compile_time:.1f}s | initial loss: {float(loss):.4f}")
    log("-" * 80)

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
        inputs, targets = make_global_batch(inputs, targets)
        model, opt_state, loss = train_step(model, opt_state, inputs, targets)
        loss_val = float(loss)
        dt = time.time() - t0
        step_times.append(dt)

        if step % 10 == 0 or step < 5:
            tps = tokens_per_step / dt
            mfu = (flops_per_token * tps) / gpu_peak_flops * 100
            bpb = loss_val * BPB_FACTOR
            elapsed = time.time() - training_start
            log(f"step {step:5d} | loss {loss_val:.4f} | bpb {bpb:.4f} | "
                f"{dt*1000:.0f}ms | {tps:.0f} tok/s | MFU {mfu:.2f}% | {elapsed:.0f}s")

        if args.eval_every > 0 and step % args.eval_every == 0:
            val_losses = []
            for _ in range(args.eval_steps):
                val_inputs, val_targets = next(val_loader)
                val_inputs, val_targets = make_global_batch(val_inputs, val_targets)
                val_loss = eval_step(model, val_inputs, val_targets)
                val_losses.append(float(val_loss))
            avg_val_loss = sum(val_losses) / len(val_losses)
            val_bpb = avg_val_loss * BPB_FACTOR
            total_tok = step * tokens_per_step
            log(f"  >>> EVAL | val_loss {avg_val_loss:.4f} | val_bpb {val_bpb:.4f} | tokens {total_tok/1e6:.0f}M")
            if args.target_bpb > 0 and val_bpb <= args.target_bpb:
                log(f"  >>> TARGET REACHED: val_bpb {val_bpb:.4f} <= {args.target_bpb}")
                break

    training_seconds = time.time() - training_start

    # Final eval
    log("-" * 80)
    log("Final evaluation...")
    val_losses = []
    for _ in range(args.eval_steps):
        val_inputs, val_targets = next(val_loader)
        val_inputs, val_targets = make_global_batch(val_inputs, val_targets)
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

    log("---")
    log(f"method:           jax.distributed (Option A)")
    log(f"world_size:       {_WORLD_SIZE}")
    log(f"val_bpb:          {val_bpb:.6f}")
    log(f"val_loss:         {avg_val_loss:.6f}")
    log(f"training_seconds: {training_seconds:.1f}")
    log(f"total_seconds:    {total_seconds:.1f}")
    log(f"mfu_percent:      {avg_mfu:.2f}")
    log(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    log(f"num_steps:        {step}")
    log(f"num_params_M:     {n_params / 1e6:.1f}")
    log(f"tokens_per_sec:   {avg_tps:.0f}")
    log(f"ms_per_step:      {avg_dt * 1000:.1f}" if len(step_times) > warmup_skip else "ms_per_step:      0.0")
    log(f"FINAL | val_loss {avg_val_loss:.4f} | val_bpb {val_bpb:.4f}")
    log("Training complete.")


if __name__ == '__main__':
    main()
