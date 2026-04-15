"""Equinox-based checkpointing: atomic save + optional load.

Saves model + optimizer state as two separate `.eqx` files plus a small
`meta.txt` holding the step counter. Rank-guarded so multi-process training
only writes from rank 0. Writes go to `*.tmp` first, then rename atomically
so a crash mid-write can't leave a corrupt checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import jax
import equinox as eqx


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

    # Atomic rename on the same filesystem
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
