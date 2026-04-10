"""Test Option A: jax.distributed + shard_map for multi-GPU with Triton kernels.

Each SLURM task = 1 process = 1 GPU. jax.distributed connects them via NCCL.
shard_map provides named axis for lax.pmean gradient all-reduce.

Run: srun --ntasks=4 --gres=gpu:4 python tests/test_option_a.py
"""

import os
import sys
import time
import math

# --- Must happen before any JAX computation ---
local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
n_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
coordinator = os.environ.get("MASTER_ADDR", "localhost")

import jax
jax.distributed.initialize(
    coordinator_address=f"{coordinator}:29500",
    num_processes=n_tasks,
    process_id=local_rank,
    local_device_ids=[local_rank],
)

import jax.numpy as jnp
import equinox as eqx
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

jax.config.update("jax_default_matmul_precision", "high")

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas
from atlas_jax.data import data_loader
from atlas_jax.tokenizer import get_tokenizer


def log(msg):
    if local_rank == 0:
        print(msg, flush=True)


log(f"jax.distributed: {n_tasks} processes")
log(f"Global devices: {jax.devices()}")
log(f"Local devices: {jax.local_devices()}")

# --- Config (matches current 1-GPU run) ---
B_LOCAL = 64          # per-GPU batch size
B_TOTAL = B_LOCAL * n_tasks
SEQ_LEN = 1024
N_STEPS = 100
EVAL_EVERY = 50
EVAL_STEPS = 10

config = AtlasConfig(
    sequence_len=SEQ_LEN, n_layer=8, n_head=8, n_embd=512,
    chunk_size=64, ns_steps=3, omega_window=16, poly_degree=3,
    deep_memory=True, memory_expand=1, pe_ste=True,
    use_checkpoint=True, fused_chunk=True)

key = jax.random.PRNGKey(42)
model = Atlas(config, key=key)
model = jax.tree.map(
    lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x,
    model, is_leaf=eqx.is_array)

n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
log(f"Model: {n_params/1e6:.1f}M params, fused_chunk=True, bf16")
log(f"Batch: {B_TOTAL} total ({B_LOCAL}/GPU), seq_len={SEQ_LEN}")

# --- Optimizer ---
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-3, warmup_steps=50,
    decay_steps=N_STEPS, end_value=3e-5)
opt = optax.chain(optax.clip_by_global_norm(1.0),
                  optax.adamw(learning_rate=schedule, weight_decay=0.1))
opt_state = opt.init(eqx.filter(model, eqx.is_array))

# --- Mesh + sharding ---
mesh = Mesh(jax.devices(), axis_names=('data',))
replicate_sharding = NamedSharding(mesh, P())
data_sharding = NamedSharding(mesh, P('data'))

model = jax.tree.map(
    lambda x: jax.device_put(x, replicate_sharding) if eqx.is_array(x) else x,
    model, is_leaf=eqx.is_array)
opt_state = jax.tree.map(
    lambda x: jax.device_put(x, replicate_sharding) if eqx.is_array(x) else x,
    opt_state, is_leaf=eqx.is_array)


def loss_fn(model, inputs, targets):
    logits, _ = model(inputs)
    lf = logits.reshape(-1, logits.shape[-1])
    tf = targets.reshape(-1)
    lp = jax.nn.log_softmax(lf, axis=-1)
    return -jnp.mean(lp[jnp.arange(tf.shape[0]), tf])


def upcast_grads(grads):
    def _cast(x):
        if eqx.is_array(x) and x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
        return x
    return jax.tree.map(_cast, grads, is_leaf=eqx.is_array)


# Close over optimizer — can't pass functions through shard_map
def _train_body(model, opt_state, inputs, targets):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, targets)
    grads = upcast_grads(grads)
    grads = jax.lax.pmean(grads, axis_name='data')
    loss = jax.lax.pmean(loss, axis_name='data')
    updates, new_opt_state = opt.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


def _eval_body(model, inputs, targets):
    loss = loss_fn(model, inputs, targets)
    return jax.lax.pmean(loss, axis_name='data')


model_spec = jax.tree.map(lambda _: P(), eqx.filter(model, eqx.is_array))
opt_spec = jax.tree.map(lambda _: P(), eqx.filter(opt_state, eqx.is_array))


@eqx.filter_jit(donate='all')
def train_step(model, opt_state, inputs, targets):
    return shard_map(
        _train_body, mesh=mesh,
        in_specs=(model_spec, opt_spec, P('data'), P('data')),
        out_specs=(model_spec, opt_spec, P()),
        check_rep=False,
    )(model, opt_state, inputs, targets)


@eqx.filter_jit
def eval_step(model, inputs, targets):
    return shard_map(
        _eval_body, mesh=mesh,
        in_specs=(model_spec, P('data'), P('data')),
        out_specs=P(),
        check_rep=False,
    )(model, inputs, targets)


# --- Data: each rank loads its own shard ---
tokenizer = get_tokenizer("/p/scratch/westai0047/nanochat/tokenizer")
train_loader = data_loader(
    "/p/scratch/westai0047/nanochat/base_data_climbmix",
    tokenizer, B_LOCAL, SEQ_LEN, split='train',
    rank=local_rank, world_size=n_tasks)
val_loader = data_loader(
    "/p/scratch/westai0047/nanochat/base_data_climbmix",
    tokenizer, B_LOCAL, SEQ_LEN, split='val',
    rank=local_rank, world_size=n_tasks)


def shard_batch(inputs, targets):
    """Each rank has local (B_LOCAL, T) arrays — assemble into global sharded array."""
    local_device = jax.local_devices()[0]
    inp = jax.make_array_from_single_device_arrays(
        (B_TOTAL, SEQ_LEN),
        data_sharding,
        [jax.device_put(inputs, local_device)])
    tgt = jax.make_array_from_single_device_arrays(
        (B_TOTAL, SEQ_LEN),
        data_sharding,
        [jax.device_put(targets, local_device)])
    return inp, tgt


# --- Compilation warmup ---
log("\nCompiling (first step)...")
t_compile = time.time()
inputs, targets = next(train_loader)
inputs, targets = shard_batch(inputs, targets)
model, opt_state, loss = train_step(model, opt_state, inputs, targets)
float(loss)
log(f"Compilation: {time.time() - t_compile:.1f}s | initial loss: {float(loss):.4f}")

# --- Training ---
CHARS_PER_TOKEN = 3.3
BPB_FACTOR = 1.0 / (math.log(2) * CHARS_PER_TOKEN)
tokens_per_step = B_TOTAL * SEQ_LEN
gpu_peak_flops = 989.4e12 * n_tasks

from atlas_jax.train import estimate_flops_per_token
flops_per_token = estimate_flops_per_token(config, n_params, n_params - model.wte.weight.size)

log(f"\nTraining: {N_STEPS} steps, {tokens_per_step:,} tok/step")
log("-" * 80)

training_start = time.time()
step_times = []

for step in range(1, N_STEPS + 1):
    t0 = time.time()
    inputs, targets = next(train_loader)
    inputs, targets = shard_batch(inputs, targets)
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

    if EVAL_EVERY > 0 and step % EVAL_EVERY == 0:
        val_losses = []
        for _ in range(EVAL_STEPS):
            vi, vt = next(val_loader)
            vi, vt = shard_batch(vi, vt)
            val_losses.append(float(eval_step(model, vi, vt)))
        avg_vl = sum(val_losses) / len(val_losses)
        log(f"  >>> EVAL | val_loss {avg_vl:.4f} | val_bpb {avg_vl * BPB_FACTOR:.4f} "
            f"| tokens {step * tokens_per_step / 1e6:.0f}M")

# --- Summary ---
log("-" * 80)
skip = min(5, len(step_times))
steady = step_times[skip:]
avg_dt = sum(steady) / len(steady)
avg_tps = tokens_per_step / avg_dt
avg_mfu = (flops_per_token * avg_tps) / gpu_peak_flops * 100

log(f"Option A (jax.distributed + shard_map): DONE")
log(f"  GPUs:           {n_tasks}")
log(f"  Batch:          {B_TOTAL} ({B_LOCAL}/GPU)")
log(f"  ms/step:        {avg_dt * 1000:.0f}")
log(f"  tok/s:          {avg_tps:.0f}")
log(f"  MFU:            {avg_mfu:.2f}%")
log(f"  total_tokens:   {N_STEPS * tokens_per_step / 1e6:.1f}M")
