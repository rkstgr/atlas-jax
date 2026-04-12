"""Sliding window causal attention with rotary position embeddings.

Used by the MAG (Memory As Gate) architecture alongside the Atlas memory layer.
"""

import math

import jax
import jax.numpy as jnp
import equinox as eqx


def _rms_norm(x):
    """RMS normalization in f32."""
    dtype = x.dtype
    x = x.astype(jnp.float32)
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(ms + 1e-6)).astype(dtype)


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
        freqs = jnp.outer(t, self.inv_freq)  # (seq_len, dim//2)
        return jnp.cos(freqs), jnp.sin(freqs)


def _apply_rope(x, cos, sin):
    """Apply rotary embeddings to x: (B, H, T, D)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    # cos, sin are (T, D//2) — broadcast over B, H
    cos = cos[jnp.newaxis, jnp.newaxis, :, :]  # (1, 1, T, D//2)
    sin = sin[jnp.newaxis, jnp.newaxis, :, :]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return jnp.concatenate([out1, out2], axis=-1)


class SlidingWindowAttention(eqx.Module):
    """Sliding window causal attention with RoPE.

    Each position attends to at most `window_size` previous positions (+ itself).
    Optional persistent memory tokens prepended to K/V.
    """
    norm: None  # placeholder — we use functional rms_norm
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
        """x: (B, T, C) → (B, T, C)"""
        B, T, C = x.shape
        H, D = self.n_head, self.dim_head

        x = _rms_norm(x)

        # QKV projection + split heads
        qkv = x @ self.to_qkv.weight.T  # (B, T, 3*dim_inner)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, T, dim_inner)
        q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)  # (B, H, T, D)
        k = k.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, D).transpose(0, 2, 1, 3)

        # RoPE
        cos, sin = self.rope(T)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Scaled dot-product attention with causal + sliding window mask
        scale = 1.0 / math.sqrt(D)
        attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale  # (B, H, T, T)

        # Causal mask: q_pos >= k_pos
        q_pos = jnp.arange(T)[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
        k_pos = jnp.arange(T)[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        causal_mask = q_pos >= k_pos
        window_mask = (q_pos - k_pos) <= self.window_size
        mask = causal_mask & window_mask

        attn = jnp.where(mask, attn, -1e9)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)

        # Weighted sum
        out = jnp.einsum('bhqk,bhkd->bhqd', attn, v)  # (B, H, T, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H * D)  # (B, T, dim_inner)
        out = out @ self.to_out.weight.T  # (B, T, C)

        return out
