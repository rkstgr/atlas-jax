"""Atlas model: MLP, Block, Atlas, and sliding window attention.

Reference: arXiv 2505.23735 — Atlas: Learning to Optimally Memorize the Context at Test Time
"""

import math

import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx

from atlas_jax.config import AtlasConfig
from atlas_jax.memory_layer import (
    ShortConv,
    AtlasMemoryLayer,
    LinearMemoryState,
    DeepMemoryState,
    rms_norm,
    _dropout,
    _gelu_derivative,
    _omega_aggregate,
    linear_scan,
)

__all__ = [
    # Re-exports (kept for external callers that import from atlas_jax.model)
    "ShortConv",
    "AtlasMemoryLayer",
    "rms_norm",
    "_dropout",
    "_gelu_derivative",
    "_omega_aggregate",
    "linear_scan",
    # Attention
    "RotaryEmbedding",
    "SlidingWindowAttention",
    # Defined here
    "MLP",
    "Block",
    "Atlas",
]


# ============================================================================
# Sliding window causal attention with rotary position embeddings
# ============================================================================

class RotaryEmbedding(eqx.Module):
    """Rotary position embeddings (RoPE)."""
    inv_freq: jax.Array  # (dim_head // 2,)
    dim: int = eqx.field(static=True)

    def __init__(self, dim):
        self.dim = dim
        inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, seq_len):
        """Return (seq_len, dim) cos/sin embeddings."""
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, self.inv_freq)
        return jnp.cos(freqs), jnp.sin(freqs)


def _apply_rope(x, cos, sin):
    """Apply rotary embeddings to x: (B, H, T, D)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[jnp.newaxis, jnp.newaxis, :, :]
    sin = sin[jnp.newaxis, jnp.newaxis, :, :]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return jnp.concatenate([out1, out2], axis=-1)


class SlidingWindowAttention(eqx.Module):
    """Sliding window causal attention with RoPE."""
    norm: None
    to_qkv: eqx.nn.Linear
    to_out: eqx.nn.Linear
    rope: RotaryEmbedding

    n_head: int = eqx.field(static=True)
    dim_head: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    n_embd: int = eqx.field(static=True)

    def __init__(self, n_embd, n_head, dim_head, window_size, *, key):
        k1, k2 = jax.random.split(key)
        dim_inner = n_head * dim_head
        self.n_head = n_head
        self.dim_head = dim_head
        self.window_size = window_size
        self.n_embd = n_embd
        self.norm = None

        self.to_qkv = eqx.nn.Linear(n_embd, dim_inner * 3, use_bias=False, key=k1)
        self.to_out = eqx.nn.Linear(dim_inner, n_embd, use_bias=False, key=k2)
        self.rope = RotaryEmbedding(dim_head)

    def __call__(self, x):
        """x: (B, T, C) -> (B, T, C)"""
        B, T, C = x.shape
        H, D = self.n_head, self.dim_head

        x = rms_norm(x)

        qkv = x @ self.to_qkv.weight.T
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)

        cos, sin = self.rope(T)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        scale = 1.0 / math.sqrt(D)
        attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale

        q_pos = jnp.arange(T)[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
        k_pos = jnp.arange(T)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        causal_mask = q_pos >= k_pos
        window_mask = (q_pos - k_pos) <= self.window_size
        mask = causal_mask & window_mask

        attn = jnp.where(mask, attn, -1e9)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)

        out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)
        out = out @ self.to_out.weight.T

        return out


# ============================================================================
# MLP
# ============================================================================

class MLP(eqx.Module):
    """Feedforward MLP. Supports GEGLU (SiLU-gated, PyTorch default) or plain GELU."""
    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    geglu: bool = eqx.field(static=True)

    def __init__(self, config: AtlasConfig, *, key):
        k1, k2 = jax.random.split(key)
        self.geglu = config.geglu_ff
        if self.geglu:
            dim_inner = int(config.n_embd * 4 * 2 / 3)
            self.c_fc = eqx.nn.Linear(config.n_embd, dim_inner * 2, use_bias=False, key=k1)
            self.c_proj = eqx.nn.Linear(dim_inner, config.n_embd, use_bias=False, key=k2)
        else:
            self.c_fc = eqx.nn.Linear(config.n_embd, 4 * config.n_embd, use_bias=False, key=k1)
            self.c_proj = eqx.nn.Linear(4 * config.n_embd, config.n_embd, use_bias=False, key=k2)

    def __call__(self, x):
        """x: (B, T, C) -> (B, T, C)"""
        h = x @ self.c_fc.weight.T
        if self.geglu:
            v, gate = jnp.split(h, 2, axis=-1)
            h = jax.nn.silu(gate) * v
        else:
            h = jax.nn.gelu(h)
        return h @ self.c_proj.weight.T


# ============================================================================
# Block
# ============================================================================

class Block(eqx.Module):
    memory: AtlasMemoryLayer
    mlp: MLP
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, config: AtlasConfig, *, key):
        k1, k2 = jax.random.split(key)
        self.memory = AtlasMemoryLayer(config, key=k1)
        self.mlp = MLP(config, key=k2)
        self.dropout_rate = config.dropout

    def __call__(self, x, memory_state=None, *, dropout_key=None):
        mem_out, new_state = self.memory(rms_norm(x), memory_state)
        if dropout_key is not None and self.dropout_rate > 0:
            k1, k2 = jax.random.split(dropout_key)
            mem_out = _dropout(mem_out, self.dropout_rate, k1)
        x = x + mem_out
        mlp_out = self.mlp(rms_norm(x))
        if dropout_key is not None and self.dropout_rate > 0:
            mlp_out = _dropout(mlp_out, self.dropout_rate, k2)
        x = x + mlp_out
        return x, new_state


# ============================================================================
# Weight initialization + memory state helpers
# ============================================================================

def _init_block_weights(blocks_list, key, config=None):
    """Initialize block weights per paper specifications."""
    n = len(blocks_list)
    keys = jax.random.split(key, 6 * n + 10)
    ki = 0
    proj_std = 0.02 / math.sqrt(2 * n)
    gate_bias_init = config.gate_bias_init if config is not None else -2.0

    for i in range(n):
        w = blocks_list[i].memory.c_proj.weight
        blocks_list[i] = eqx.tree_at(
            lambda b: b.memory.c_proj.weight, blocks_list[i],
            jax.random.normal(keys[ki], w.shape) * proj_std)
        ki += 1
        w = blocks_list[i].mlp.c_proj.weight
        blocks_list[i] = eqx.tree_at(
            lambda b: b.mlp.c_proj.weight, blocks_list[i],
            jax.random.normal(keys[ki], w.shape) * proj_std)
        ki += 1

        C = blocks_list[i].memory.c_q.weight.shape[1]
        xavier_std = 1.0 / math.sqrt(C)
        for attr in ['c_q', 'c_k', 'c_v']:
            w = getattr(blocks_list[i].memory, attr).weight
            blocks_list[i] = eqx.tree_at(
                lambda b, a=attr: getattr(b.memory, a).weight, blocks_list[i],
                jax.random.uniform(keys[ki], w.shape, minval=-xavier_std, maxval=xavier_std))
            ki += 1

        for attr in ['gate_alpha', 'gate_eta', 'gate_theta']:
            gate = getattr(blocks_list[i].memory, attr)
            blocks_list[i] = eqx.tree_at(
                lambda b, a=attr: getattr(b.memory, a).weight, blocks_list[i],
                jnp.zeros_like(gate.weight))
            blocks_list[i] = eqx.tree_at(
                lambda b, a=attr: getattr(b.memory, a).bias, blocks_list[i],
                jnp.full_like(gate.bias, gate_bias_init))
        if blocks_list[i].memory.gate_gamma is not None:
            gate = blocks_list[i].memory.gate_gamma
            blocks_list[i] = eqx.tree_at(
                lambda b: b.memory.gate_gamma.weight, blocks_list[i],
                jnp.zeros_like(gate.weight))
            blocks_list[i] = eqx.tree_at(
                lambda b: b.memory.gate_gamma.bias, blocks_list[i],
                jnp.full_like(gate.bias, gate_bias_init))

    return blocks_list


def _make_initial_memory_states(n_layer, B, H, D, E, deep_memory, dtype):
    """Create stacked initial memory states with leading n_layer dim."""
    if deep_memory:
        W1 = jnp.zeros((n_layer, B, H, D, E), dtype=dtype)
        W2 = jnp.zeros((n_layer, B, H, E, D), dtype=dtype)
        eye = jnp.eye(min(E, D), dtype=dtype)
        W2 = W2.at[:, :, :, :min(E, D), :min(E, D)].set(eye)
        S_W1 = jnp.zeros((n_layer, B, H, D, E), dtype=dtype)
        S_W2 = jnp.zeros((n_layer, B, H, E, D), dtype=dtype)
        return DeepMemoryState(W1=W1, W2=W2, S_W1=S_W1, S_W2=S_W2)
    else:
        M = jnp.zeros((n_layer, B, H, D, D), dtype=dtype)
        S = jnp.zeros((n_layer, B, H, D, D), dtype=dtype)
        return LinearMemoryState(M=M, S=S)


# ============================================================================
# Atlas (full model)
# ============================================================================

class Atlas(eqx.Module):
    wte: eqx.nn.Embedding
    blocks: Block  # Stacked: each array leaf has leading n_layer dim (for lax.scan)
    lm_head: eqx.nn.Linear
    persist_mem: jax.Array | None  # (num_persist_mem_tokens, n_embd)
    config: AtlasConfig = eqx.field(static=True)
    padded_vocab_size: int = eqx.field(static=True)

    def __init__(self, config: AtlasConfig, *, key, pad_vocab_size_to=64):
        keys = jax.random.split(key, config.n_layer + 3)

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.padded_vocab_size = padded_vocab_size
        self.config = config

        self.wte = eqx.nn.Embedding(padded_vocab_size, config.n_embd, key=keys[0])

        if config.num_persist_mem_tokens > 0:
            pm_key = jax.random.split(keys[0])[0]
            self.persist_mem = jax.random.normal(
                pm_key, (config.num_persist_mem_tokens, config.n_embd)) * 0.02
        else:
            self.persist_mem = None

        blocks_list = [Block(config, key=keys[i + 1]) for i in range(config.n_layer)]
        blocks_list = _init_block_weights(blocks_list, keys[-2], config=config)
        self.blocks = jax.tree.map(lambda *xs: jnp.stack(xs), *blocks_list)

        self.lm_head = eqx.nn.Linear(config.n_embd, padded_vocab_size, use_bias=False, key=keys[-1])
        init_key = jax.random.split(keys[-2], 2)[0]
        self.lm_head = eqx.tree_at(
            lambda m: m.weight, self.lm_head,
            jax.random.normal(init_key, self.lm_head.weight.shape) * 0.02)

    def __call__(self, idx, memory_states=None, *, dropout_key=None):
        B, T = idx.shape
        cfg = self.config
        H = cfg.n_head
        D = cfg.dim_head if cfg.dim_head is not None else cfg.n_embd // H
        E = cfg.memory_expand * D if cfg.deep_memory else D
        n_layer = cfg.n_layer
        drop_rate = cfg.dropout

        x = self.wte.weight[idx]

        n_persist = cfg.num_persist_mem_tokens
        if self.persist_mem is not None and n_persist > 0:
            pm = jnp.broadcast_to(
                self.persist_mem[jnp.newaxis], (B, n_persist, cfg.n_embd))
            x = jnp.concatenate([pm, x], axis=1)

        if memory_states is None:
            stacked_states = _make_initial_memory_states(
                n_layer, B, H, D, E, cfg.deep_memory, x.dtype)
        else:
            stacked_states = jax.tree.map(lambda *xs: jnp.stack(xs), *memory_states)

        if dropout_key is not None and drop_rate > 0:
            drop_keys = jax.random.split(dropout_key, n_layer)
        else:
            drop_keys = None

        use_dropout = dropout_key is not None and drop_rate > 0

        def scan_fn(x, layer_data):
            if use_dropout:
                block, mem_state, dk = layer_data
                k1, k2 = jax.random.split(dk)
            else:
                block, mem_state = layer_data
            mem_out, new_state = block.memory(rms_norm(x), mem_state)
            if use_dropout:
                mem_out = _dropout(mem_out, drop_rate, k1)
            x = x + mem_out
            mlp_out = block.mlp(rms_norm(x))
            if use_dropout:
                mlp_out = _dropout(mlp_out, drop_rate, k2)
            x = x + mlp_out
            return x, new_state

        if use_dropout:
            scan_data = (self.blocks, stacked_states, drop_keys)
        else:
            scan_data = (self.blocks, stacked_states)
        x, new_states_stacked = lax.scan(scan_fn, x, scan_data)

        x = rms_norm(x)

        if self.persist_mem is not None and n_persist > 0:
            x = x[:, n_persist:]

        logits = x @ self.lm_head.weight.T
        logits = logits[..., :cfg.vocab_size]
        logits = logits.astype(jnp.float32)
        if cfg.logit_softcap > 0:
            softcap = cfg.logit_softcap
            logits = softcap * jnp.tanh(logits / softcap)

        new_states = [
            jax.tree.map(lambda s: s[i], new_states_stacked)
            for i in range(n_layer)
        ]

        return logits, new_states
