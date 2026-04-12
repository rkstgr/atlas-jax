"""MAG (Memory As Gate) Transformer.

Two parallel branches per layer:
1. Sliding window attention (short-term)
2. Atlas memory layer (long-term)
Combined via learned sigmoid gate: x = x + attn_out + gate * mem_out

Reference: atlas_pytorch/mag_transformer.py (lucidrains)
"""

import math

import jax
import jax.numpy as jnp
import equinox as eqx

from atlas_jax.config import AtlasConfig
from atlas_jax.model import AtlasMemoryLayer, MLP, rms_norm, _dropout
from atlas_jax.attention import SlidingWindowAttention


class MemoryGate(eqx.Module):
    """Learned gate: RMSNorm → Linear → Sigmoid. Element-wise gating on dim."""
    linear: eqx.nn.Linear

    def __init__(self, dim, *, key):
        self.linear = eqx.nn.Linear(dim, dim, use_bias=True, key=key)

    def __call__(self, x):
        """x: (B, T, C) → (B, T, C) in [0, 1]"""
        x = rms_norm(x)
        return jax.nn.sigmoid(x @ self.linear.weight.T + self.linear.bias)


class MAGBlock(eqx.Module):
    """Single MAG layer: attention + optional gated memory + feedforward."""
    attn: SlidingWindowAttention
    memory: AtlasMemoryLayer | None
    gate: MemoryGate | None
    mlp: MLP

    def __init__(self, config: AtlasConfig, has_memory: bool, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        dim_head = config.dim_head if config.dim_head is not None else config.n_embd // config.n_head

        self.attn = SlidingWindowAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            dim_head=dim_head,
            window_size=config.window_size,
            key=k1,
        )

        if has_memory:
            self.memory = AtlasMemoryLayer(config, key=k2)
            self.gate = MemoryGate(config.n_embd, key=k3)
        else:
            self.memory = None
            self.gate = None

        self.mlp = MLP(config, key=k4)

    def __call__(self, x, memory_state=None):
        # Attention branch
        attn_out = self.attn(x)

        # Memory branch (if this layer has memory)
        new_state = None
        if self.memory is not None:
            mem_out, new_state = self.memory(rms_norm(x), memory_state)
            g = self.gate(mem_out)
            x = x + attn_out + g * mem_out
        else:
            x = x + attn_out

        # Feedforward
        x = x + self.mlp(rms_norm(x))

        return x, new_state


class MAGTransformer(eqx.Module):
    """Memory As Gate Transformer.

    Combines sliding window attention with Atlas memory layers.
    Memory is selectively enabled on specific layers via neural_memory_layers.
    """
    wte: eqx.nn.Embedding
    blocks: list  # List of MAGBlock (not stacked — different layers have different structure)
    lm_head: eqx.nn.Linear
    config: AtlasConfig = eqx.field(static=True)
    memory_layer_mask: tuple = eqx.field(static=True)  # which layers have memory

    def __init__(self, config: AtlasConfig, *, key):
        keys = jax.random.split(key, config.n_layer + 3)

        self.config = config
        self.wte = eqx.nn.Embedding(config.vocab_size, config.n_embd, key=keys[0])

        # Determine which layers get memory
        if config.neural_memory_layers is not None:
            mem_layers = set(config.neural_memory_layers)
        else:
            mem_layers = set(range(config.n_layer))  # all layers

        self.memory_layer_mask = tuple(i in mem_layers for i in range(config.n_layer))

        # Create blocks (can't stack — heterogeneous structure)
        self.blocks = [
            MAGBlock(config, has_memory=self.memory_layer_mask[i], key=keys[i + 1])
            for i in range(config.n_layer)
        ]

        self.lm_head = eqx.nn.Linear(config.n_embd, config.vocab_size, use_bias=False, key=keys[-1])
        # Small init for lm_head
        init_key = jax.random.split(keys[-1])[0]
        self.lm_head = eqx.tree_at(
            lambda m: m.weight, self.lm_head,
            jax.random.normal(init_key, self.lm_head.weight.shape) * 0.02)

    def __call__(self, idx, memory_states=None, *, dropout_key=None):
        """Forward pass.

        Args:
            idx: (B, T) token indices
            memory_states: optional list of per-layer memory states

        Returns:
            logits: (B, T, vocab_size)
            new_memory_states: list of updated states (None for non-memory layers)
        """
        B, T = idx.shape
        cfg = self.config

        x = self.wte.weight[idx]  # (B, T, C)

        if memory_states is None:
            memory_states = [None] * cfg.n_layer

        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, memory_states[i])
            new_states.append(new_state)

        x = rms_norm(x)
        logits = x @ self.lm_head.weight.T
        logits = logits.astype(jnp.float32)

        return logits, new_states
