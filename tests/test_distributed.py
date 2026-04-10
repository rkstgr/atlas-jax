"""Test Option A: jax.distributed for multi-GPU with Triton kernels.

jax.distributed.initialize() MUST be called before any JAX operation.
Each SLURM task gets 1 GPU. After init, each process sees all GPUs but
computes on its local device. We use shard_map with the multi-process
mesh for gradient all-reduce.

Run: srun --ntasks=4 --gres=gpu:4 python tests/test_distributed.py
"""

import os
import sys

# Get rank info from SLURM BEFORE importing JAX
local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
n_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
coordinator = os.environ.get("MASTER_ADDR", "localhost")

# Set visible GPU BEFORE JAX init
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

# Initialize JAX distributed BEFORE any other JAX call
import jax
jax.distributed.initialize(
    coordinator_address=f"{coordinator}:29500",
    num_processes=n_tasks,
    process_id=local_rank,
)

import jax.numpy as jnp
import equinox as eqx
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.shard_map import shard_map

jax.config.update("jax_default_matmul_precision", "high")

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas

if local_rank == 0:
    print(f"jax.distributed initialized: {n_tasks} processes")
    print(f"Global devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")

# Create mesh over all devices (one per process)
mesh = Mesh(jax.devices(), axis_names=('data',))

config = AtlasConfig(
    sequence_len=1024, n_layer=8, n_head=8, n_embd=512,
    chunk_size=64, ns_steps=3, omega_window=16, poly_degree=3,
    deep_memory=True, memory_expand=1, pe_ste=True,
    use_checkpoint=True, fused_chunk=True)

key = jax.random.PRNGKey(42)
model = Atlas(config, key=key)
model = jax.tree.map(
    lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x,
    model, is_leaf=eqx.is_array)

if local_rank == 0:
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model: {n_params/1e6:.1f}M params, fused_chunk=True, bf16")

opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=3e-3))
opt_state = opt.init(eqx.filter(model, eqx.is_array))

# Replicate model across devices
replicate_sharding = NamedSharding(mesh, P())
data_sharding = NamedSharding(mesh, P('data'))

def _replicate(x):
    return jax.device_put(x, replicate_sharding) if eqx.is_array(x) else x
model = jax.tree.map(_replicate, model, is_leaf=eqx.is_array)
opt_state = jax.tree.map(_replicate, opt_state, is_leaf=eqx.is_array)


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


# Per-device train body for shard_map
def _train_body(model, opt_state, optimizer, inputs, targets):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, targets)
    grads = upcast_grads(grads)
    grads = jax.lax.pmean(grads, axis_name='data')
    loss = jax.lax.pmean(loss, axis_name='data')
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


# Build shard_map specs
model_spec = jax.tree.map(lambda _: P(), eqx.filter(model, eqx.is_array))
opt_spec = jax.tree.map(lambda _: P(), eqx.filter(opt_state, eqx.is_array))

@eqx.filter_jit(donate='all')
def train_step(model, opt_state, optimizer, inputs, targets):
    return shard_map(
        _train_body,
        mesh=mesh,
        in_specs=(model_spec, opt_spec, P(), P('data'), P('data')),
        out_specs=(model_spec, opt_spec, P()),
        check_rep=False,
    )(model, opt_state, optimizer, inputs, targets)


# Training loop
B_total = 32  # 8 per GPU
key = jax.random.PRNGKey(123)

if local_rank == 0:
    print(f"\nTraining: B={B_total} ({B_total//n_tasks}/GPU), 10 steps")
    print("-" * 40)

import time
for step in range(10):
    key, k1, k2 = jax.random.split(key, 3)
    inputs = jax.random.randint(k1, (B_total, 1024), 0, 32768)
    targets = jax.random.randint(k2, (B_total, 1024), 0, 32768)
    inputs = jax.device_put(inputs, data_sharding)
    targets = jax.device_put(targets, data_sharding)

    t0 = time.time()
    model, opt_state, loss = train_step(model, opt_state, opt, inputs, targets)
    loss_val = float(loss)
    dt = time.time() - t0

    if local_rank == 0:
        print(f"step {step:3d} | loss {loss_val:.4f} | {dt*1000:.0f}ms")

if local_rank == 0:
    print("\njax.distributed + shard_map: OK")
