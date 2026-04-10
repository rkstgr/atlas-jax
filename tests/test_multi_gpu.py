"""Test multi-GPU data parallelism — 3 approaches.

Run with:
  Option A (jax.distributed): srun --ntasks=4 --gres=gpu:4 python tests/test_multi_gpu.py --method=distributed
  Option B (mpi4jax):         srun --ntasks=4 --gres=gpu:4 python tests/test_multi_gpu.py --method=mpi
  Option C (shared memory):   srun --ntasks=4 --gres=gpu:4 python tests/test_multi_gpu.py --method=shm

Each process gets 1 GPU via SLURM. Runs 10 training steps with fused Triton kernel.
"""

import os
import sys
import time
import argparse

# Must set CUDA_VISIBLE_DEVICES BEFORE importing JAX
# SLURM sets SLURM_LOCALID (0-3 for 4 tasks on 1 node)
local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
n_tasks = int(os.environ.get("SLURM_NTASKS", "1"))
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

jax.config.update("jax_default_matmul_precision", "high")

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas


def make_model():
    config = AtlasConfig(
        sequence_len=1024, n_layer=8, n_head=8, n_embd=512,
        chunk_size=64, ns_steps=3, omega_window=16, poly_degree=3,
        deep_memory=True, memory_expand=1, pe_ste=True,
        use_checkpoint=True, fused_chunk=True)
    key = jax.random.PRNGKey(42)
    model = Atlas(config, key=key)
    # bf16
    model = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x,
        model, is_leaf=eqx.is_array)
    return model


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


# ============================================================
# Option A: jax.distributed
# ============================================================
def test_jax_distributed():
    """Multi-process JAX with jax.distributed.initialize().

    Each process sees 1 GPU. jax.distributed connects them.
    Use jax.make_array_from_single_device_arrays for gradient all-reduce.
    """
    print(f"[rank {local_rank}] Initializing jax.distributed...")

    coordinator_address = os.environ.get("MASTER_ADDR", "localhost") + ":29500"
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=n_tasks,
        process_id=local_rank,
    )

    print(f"[rank {local_rank}] JAX devices: {jax.devices()}, local: {jax.local_devices()}")

    model = make_model()
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=3e-3))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # Each rank gets different data shard
    key = jax.random.PRNGKey(42 + local_rank)
    B_local = 8  # per-GPU batch

    @eqx.filter_jit
    def local_grad(model, inputs, targets):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, targets)
        grads = upcast_grads(grads)
        return loss, grads

    for step in range(10):
        key, k1, k2 = jax.random.split(key, 3)
        inputs = jax.random.randint(k1, (B_local, 1024), 0, 32768)
        targets = jax.random.randint(k2, (B_local, 1024), 0, 32768)

        loss, grads = local_grad(model, inputs, targets)

        # All-reduce gradients (mean across ranks)
        grads = jax.tree.map(
            lambda g: jax.lax.pmean(g, axis_name='i') if eqx.is_array(g) else g,
            grads, is_leaf=eqx.is_array)
        # ^ This won't work without named axis. Use psum + divide instead:
        # Actually jax.distributed doesn't give us pmean easily in plain jit.
        # We need a different approach — skip this method.

        if local_rank == 0 and step % 5 == 0:
            print(f"  step {step} | loss {float(loss):.4f}")

    if local_rank == 0:
        print("Option A: jax.distributed — pmean not available in plain jit, SKIPPED")


# ============================================================
# Option B: mpi4jax
# ============================================================
def test_mpi4jax():
    """MPI-based gradient all-reduce using mpi4jax."""
    try:
        import mpi4jax
        from mpi4py import MPI
    except ImportError:
        if local_rank == 0:
            print("Option B: mpi4jax not installed, SKIPPED")
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"[rank {rank}] JAX device: {jax.devices()}")

    model = make_model()
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=3e-3))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    B_local = 8

    @eqx.filter_jit
    def train_step(model, opt_state, opt, inputs, targets):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, targets)
        grads = upcast_grads(grads)

        # All-reduce gradients via MPI (NCCL-backed)
        def _allreduce_mean(g):
            if not eqx.is_array(g):
                return g
            summed, _ = mpi4jax.allreduce(g, op=MPI.SUM, comm=comm)
            return summed / size
        grads = jax.tree.map(_allreduce_mean, grads, is_leaf=eqx.is_array)

        updates, new_opt_state = opt.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    key = jax.random.PRNGKey(42 + rank)
    for step in range(10):
        key, k1, k2 = jax.random.split(key, 3)
        inputs = jax.random.randint(k1, (B_local, 1024), 0, 32768)
        targets = jax.random.randint(k2, (B_local, 1024), 0, 32768)
        model, opt_state, loss = train_step(model, opt_state, opt, inputs, targets)
        if rank == 0 and step % 5 == 0:
            print(f"  step {step} | loss {float(loss):.4f}")

    if rank == 0:
        print("Option B: mpi4jax — OK")


# ============================================================
# Option C: Shared memory gradient averaging
# ============================================================
def test_shm():
    """Gradient averaging via /dev/shm files. Simple but CPU round-trip."""
    import numpy as np
    import tempfile

    shm_dir = f"/dev/shm/atlas_grad_sync_{os.environ.get('SLURM_JOB_ID', '0')}"
    os.makedirs(shm_dir, exist_ok=True)

    print(f"[rank {local_rank}] JAX device: {jax.devices()}, shm_dir: {shm_dir}")

    model = make_model()
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=3e-3))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    B_local = 8

    @eqx.filter_jit
    def compute_grads(model, inputs, targets):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, targets)
        grads = upcast_grads(grads)
        return loss, grads

    def barrier(step_id, phase):
        """Simple file-based barrier."""
        flag = os.path.join(shm_dir, f"step{step_id}_phase{phase}_rank{local_rank}")
        open(flag, 'w').close()
        # Wait for all ranks
        for r in range(n_tasks):
            target = os.path.join(shm_dir, f"step{step_id}_phase{phase}_rank{r}")
            while not os.path.exists(target):
                time.sleep(0.001)

    def allreduce_mean_shm(grads, step_id):
        """Write grads to shm, wait for all ranks, average."""
        flat_grads, treedef = jax.tree.flatten(eqx.filter(grads, eqx.is_array))

        # Write this rank's gradients
        arrays = [np.array(g) for g in flat_grads]
        grad_file = os.path.join(shm_dir, f"step{step_id}_rank{local_rank}.npz")
        np.savez(grad_file, *arrays)

        barrier(step_id, "write")

        # Read all ranks and average
        all_arrays = [arrays]  # start with our own
        for r in range(n_tasks):
            if r == local_rank:
                continue
            other_file = os.path.join(shm_dir, f"step{step_id}_rank{r}.npz")
            data = np.load(other_file)
            all_arrays.append([data[f"arr_{i}"] for i in range(len(arrays))])

        # Average
        avg_arrays = []
        for i in range(len(arrays)):
            stacked = np.stack([a[i] for a in all_arrays], axis=0)
            avg_arrays.append(jnp.array(stacked.mean(axis=0)))

        avg_grads = jax.tree.unflatten(treedef, avg_arrays)
        # Reconstruct full grad pytree (put averaged arrays back)
        return eqx.combine(avg_grads, eqx.filter(grads, lambda x: not eqx.is_array(x)))

    key = jax.random.PRNGKey(42 + local_rank)
    for step in range(10):
        key, k1, k2 = jax.random.split(key, 3)
        inputs = jax.random.randint(k1, (B_local, 1024), 0, 32768)
        targets = jax.random.randint(k2, (B_local, 1024), 0, 32768)

        loss, grads = compute_grads(model, inputs, targets)
        grads = allreduce_mean_shm(grads, step)

        updates, new_opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        opt_state = new_opt_state

        barrier(step, "done")

        if local_rank == 0 and step % 5 == 0:
            print(f"  step {step} | loss {float(loss):.4f}")

    # Cleanup
    barrier(99, "cleanup")
    if local_rank == 0:
        import shutil
        shutil.rmtree(shm_dir, ignore_errors=True)
        print("Option C: shm — OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["distributed", "mpi", "shm", "all"], default="all")
    args = parser.parse_args()

    if local_rank == 0:
        print(f"Testing multi-GPU: {n_tasks} tasks, method={args.method}")
        print(f"JAX {jax.__version__} | device: {jax.devices()}")
        print("=" * 60)

    if args.method in ("distributed", "all"):
        if local_rank == 0:
            print("\n--- Option A: jax.distributed ---")
        test_jax_distributed()

    if args.method in ("mpi", "all"):
        if local_rank == 0:
            print("\n--- Option B: mpi4jax ---")
        test_mpi4jax()

    if args.method in ("shm", "all"):
        if local_rank == 0:
            print("\n--- Option C: shared memory ---")
        test_shm()

    if local_rank == 0:
        print("\nDone.")
