"""Memory state containers and initialization helpers.

NamedTuples are native JAX pytrees — they compose with jax.lax.scan,
jax.checkpoint, and jit without any registration boilerplate.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from atlas_jax.config import AtlasConfig


class LinearMemoryState(NamedTuple):
    """State for linear (matrix-valued) memory per head."""
    M: jax.Array    # (B, H, D, D) memory matrix
    S: jax.Array    # (B, H, D, D) momentum matrix


class DeepMemoryState(NamedTuple):
    """State for deep MLP memory per head.

    Memory is a 2-layer MLP: M(x) = x + W1 @ GELU(W2 @ x)
    W2 is initialized to [I; 0] so GELU(W2 @ k) != 0 from step 1.
    """
    W1: jax.Array     # (B, H, D, E) first layer weights
    W2: jax.Array     # (B, H, E, D) second layer weights
    S_W1: jax.Array   # (B, H, D, E) momentum for W1
    S_W2: jax.Array   # (B, H, E, D) momentum for W2


def init_memory_state(config: AtlasConfig, batch_size: int, dtype=jnp.bfloat16):
    """Create a fresh zero-initialized memory state for all heads.

    Returns one state per layer (list of LinearMemoryState or DeepMemoryState).
    """
    H = config.n_head
    D = config.n_embd // config.n_head
    E = config.memory_expand * D if config.deep_memory else D
    B = batch_size

    states = []
    for _ in range(config.n_layer):
        if config.deep_memory:
            W1 = jnp.zeros((B, H, D, E), dtype=dtype)
            # W2 initialized to [I; 0] so GELU(W2 @ k) != 0
            W2 = jnp.zeros((B, H, E, D), dtype=dtype)
            eye = jnp.eye(min(E, D), dtype=dtype)
            # Broadcast eye into the top-left block of each (E, D) matrix
            W2 = W2.at[:, :, :min(E, D), :min(E, D)].set(eye)
            S_W1 = jnp.zeros((B, H, D, E), dtype=dtype)
            S_W2 = jnp.zeros((B, H, E, D), dtype=dtype)
            states.append(DeepMemoryState(W1=W1, W2=W2, S_W1=S_W1, S_W2=S_W2))
        else:
            M = jnp.zeros((B, H, D, D), dtype=dtype)
            S = jnp.zeros((B, H, D, D), dtype=dtype)
            states.append(LinearMemoryState(M=M, S=S))

    return states
