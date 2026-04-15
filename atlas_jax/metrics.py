"""Shared training utilities: loss, grad upcasting, FLOPs + BPB accounting.

Pulled out of `train.py` / `train_distributed.py` so the two scripts can share
a single source of truth. Nothing here is JIT-specific — each caller wraps
these in its own `filter_jit` / `shard_map` shell.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import equinox as eqx


# Characters per BPE token (nanochat convention for English text).
CHARS_PER_TOKEN = 3.3
# Multiply cross-entropy (nats) by this factor to get bits per byte.
BPB_FACTOR = 1.0 / (math.log(2) * CHARS_PER_TOKEN)

# H100 SXM bf16 peak TFLOPS (used as the default denominator in MFU).
GPU_PEAK_TFLOPS = {"H100": 989.4, "RTX8000": 32.6}


def loss_fn(model, inputs, targets, *, dropout_key=None):
    """Standard next-token cross-entropy in nats.

    `dropout_key` is forwarded to the model so the same helper works for both
    train (dropout on) and eval (dropout off) paths.
    """
    logits, _ = model(inputs, dropout_key=dropout_key)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])


def upcast_grads(grads):
    """Cast bf16 gradients to f32 for numerically stable optimizer updates.

    Without this, `optax.clip_by_global_norm` squares bf16 values, which
    overflows to inf for any gradient > 256 (since 256^2 > bf16 max).
    """
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
