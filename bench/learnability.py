"""Paper §5.3 learnability baseline.

Train a small MLP in online fashion on (i_j, o_j) pairs across five
input→output mapping types. This is the paper's reference baseline: what does
an ideal per-step optimizer achieve on these tasks?

Not an Atlas-specific test. Verifies:
- MLP + optax AdamW integration on the same primitives used by `atlas_jax`.
- The five synthetic tasks have the expected difficulty ordering.
- Loss drops below 1 eventually (paper's qualitative pass bar).

Usage:
    python bench/learnability.py                    # all tasks, default config
    python bench/learnability.py --task low_rank    # single task
    python bench/learnability.py --layers 3 --expand 4

Exits 0 if every requested task's final-window mean loss < 1, else 1.
"""

import argparse
import math
import sys
import time

import jax
import jax.numpy as jnp
import jax.lax as lax
import equinox as eqx
import optax


# =============================================================================
# Model: a plain MLP (the "memory" M in the paper).
# =============================================================================

def make_mlp(d, expand, depth, key):
    """depth = number of hidden layers; width = d * expand. GELU activations."""
    return eqx.nn.MLP(
        in_size=d, out_size=d, width_size=d * expand,
        depth=depth, activation=jax.nn.gelu, key=key,
    )


# =============================================================================
# Task generators. Each returns (inputs, targets) each of shape (T, d).
# =============================================================================

def task_low_rank(T, d, key, k=32):
    """o = W^T i, W = XY, W has rank k."""
    kx, ky, ki = jax.random.split(key, 3)
    X = jax.random.normal(kx, (d, k)) / math.sqrt(k)
    Y = jax.random.normal(ky, (k, d)) / math.sqrt(d)
    W = X @ Y
    inputs = jax.random.normal(ki, (T, d))
    targets = inputs @ W
    return inputs, targets


def task_mlp(T, d, key):
    """o = M_target(i), M_target a random single-hidden-layer GELU MLP."""
    kt, ki = jax.random.split(key)
    M_tgt = make_mlp(d, expand=1, depth=1, key=kt)
    inputs = jax.random.normal(ki, (T, d))
    targets = jax.vmap(M_tgt)(inputs)
    return inputs, targets


def _causal_attn(q, k, v, window=None):
    """Single-head causal softmax attention with optional left-sliding window."""
    T, d = q.shape
    scores = (q @ k.T) / math.sqrt(d)
    causal = jnp.tril(jnp.ones((T, T), dtype=bool))
    if window is not None:
        idx = jnp.arange(T)
        win = (idx[:, None] - idx[None, :]) < window
        causal = causal & win
    scores = jnp.where(causal, scores, -jnp.inf)
    return jax.nn.softmax(scores, axis=-1) @ v


def _attn_outputs(T, d, key, window=None):
    """Shared helper: returns (raw_inputs, attn_output, M_target_applied)."""
    kq, kk, kv, km, ki = jax.random.split(key, 5)
    Wq = jax.random.normal(kq, (d, d)) / math.sqrt(d)
    Wk = jax.random.normal(kk, (d, d)) / math.sqrt(d)
    Wv = jax.random.normal(kv, (d, d)) / math.sqrt(d)
    M_tgt = make_mlp(d, expand=1, depth=1, key=km)
    raw = jax.random.normal(ki, (T, d))
    o_prime = _causal_attn(raw @ Wq, raw @ Wk, raw @ Wv, window=window)
    o = jax.vmap(M_tgt)(o_prime)
    return raw, o_prime, o


def task_attn_mlp(T, d, key, window=None):
    """Learn  i -> M(attn(i)).  Input: raw i. Target: MLP(attn_output)."""
    raw, _, o = _attn_outputs(T, d, key, window=window)
    return raw, o


def task_attn_io(T, d, key, window=None):
    """Learn  o' -> M(o').  Input: attention output. Target: MLP(attn_output).

    Easier than attn_mlp because the input is already attention-transformed —
    the memory only has to learn the pointwise MLP.
    """
    _, o_prime, o = _attn_outputs(T, d, key, window=window)
    return o_prime, o


TASKS = {
    "low_rank": lambda T, d, key: task_low_rank(T, d, key),
    "mlp":      lambda T, d, key: task_mlp(T, d, key),
    "attn_mlp": lambda T, d, key: task_attn_mlp(T, d, key, window=None),
    "attn_io":  lambda T, d, key: task_attn_io(T, d, key, window=None),
    "swa_mlp":  lambda T, d, key: task_attn_mlp(T, d, key, window=512),
}


# =============================================================================
# Online training loop: one AdamW step per (i_j, o_j) pair.
# =============================================================================

def online_train(M, inputs, targets, lr):
    params, static = eqx.partition(M, eqx.is_array)
    optimizer = optax.adamw(learning_rate=lr)
    opt_state = optimizer.init(params)

    def step(carry, xs):
        params, opt_state = carry
        i_j, o_j = xs

        def loss_fn(p):
            M_ = eqx.combine(p, static)
            pred = M_(i_j)
            return jnp.sum((pred - o_j) ** 2) / (jnp.sum(o_j ** 2) + 1e-12)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return (params, opt_state), loss

    (params_fin, _), losses = lax.scan(step, (params, opt_state), (inputs, targets))
    return losses, eqx.combine(params_fin, static)


# =============================================================================
# Entry point.
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=list(TASKS.keys()) + ['all'], default='all')
    p.add_argument('--layers', type=int, default=2,
                   help='Number of hidden layers in M (paper: 2 or 3).')
    p.add_argument('--expand', type=int, default=1,
                   help='Width expansion: hidden = d * expand (paper: 1 or 4).')
    p.add_argument('--lr', type=float, default=5e-4,
                   help='AdamW learning rate (paper figure filename: adamw_lr_p0005).')
    p.add_argument('--seq-len', type=int, default=2048)
    p.add_argument('--d', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--window-frac', type=float, default=0.0625,
                   help='Last N fraction of losses averaged for pass/fail (default: 128/2048).')
    p.add_argument('--threshold', type=float, default=1.0,
                   help='Pass if mean final-window loss < threshold.')
    args = p.parse_args()

    jax.config.update('jax_default_matmul_precision', 'float32')
    print(f'JAX {jax.__version__} | devices: {jax.devices()}')
    print(f'Config: d={args.d} layers={args.layers} expand={args.expand} '
          f'T={args.seq_len} lr={args.lr}')
    print('-' * 72)

    tasks = list(TASKS.keys()) if args.task == 'all' else [args.task]
    all_pass = True
    window = max(1, int(args.seq_len * args.window_frac))

    for task in tasks:
        key = jax.random.PRNGKey(args.seed)
        kt, km = jax.random.split(key)
        inputs, targets = TASKS[task](args.seq_len, args.d, kt)
        M = make_mlp(args.d, expand=args.expand, depth=args.layers, key=km)

        t0 = time.time()
        losses, _ = online_train(M, inputs, targets, args.lr)
        losses.block_until_ready()
        dt = time.time() - t0

        early = float(jnp.mean(losses[:window]))
        final = float(jnp.mean(losses[-window:]))
        status = 'PASS' if final < args.threshold else 'FAIL'
        print(f'[learnability:{task:9s}] early={early:6.3f} final={final:6.3f} '
              f'threshold={args.threshold} {status} ({dt:.1f}s)')
        all_pass = all_pass and (final < args.threshold)

    print('-' * 72)
    print('OVERALL: ' + ('PASS' if all_pass else 'FAIL'))
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
