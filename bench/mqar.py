"""Multi-Query Associative Recall (Zoology / Arora 2023) — atlas-jax bench.

Faithful port of zoology/data/multiquery_ar.py data generator: each sequence
freshly samples its own KV map (in-context recall, not weight memorization),
queries embedded at biased positions in the filler region, loss masked to
query positions only.

This is the most discriminating sub-GPU-hour bench for the memory mechanism:
attention / GatedDeltaNet / DeltaNet ≈ 100% at d=128, seq=256, 64 KV; a broken
scan-carry or broken delta rule produces <60%.

Usage:
    python bench/mqar.py                       # default config
    python bench/mqar.py --n-kv 32 --seq-len 128   # easier config

Pass: test accuracy ≥ 0.95 at default config.
"""

import argparse
import math
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from atlas_jax.model import Atlas, AtlasConfig


# =============================================================================
# Zoology MQAR data generator (ported from zoology/data/multiquery_ar.py).
# =============================================================================

def generate_mqar(num_examples, input_seq_len, vocab_size, num_kv_pairs,
                  power_a=0.01, random_non_queries=True, seed=0):
    """Returns (inputs, labels), each (num_examples, input_seq_len) int32.

    Layout: [k0, v0, k1, v1, ..., k_{N-1}, v_{N-1}, filler ... queries ...]
    Queries placed at positions `gaps*2` in the filler region (even offsets).
    Labels are -100 except at position-after-each-query, where label = matching value.
    """
    assert input_seq_len % 2 == 0
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 + num_kv_pairs * 2 <= input_seq_len

    rng = np.random.default_rng(seed)
    context_size = num_kv_pairs * 2
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)          # 0 reserved as filler
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys = np.stack([rng.choice(key_choices, size=num_kv_pairs, replace=False)
                     for _ in range(num_examples)])
    values = np.stack([rng.choice(value_choices, size=num_kv_pairs, replace=False)
                       for _ in range(num_examples)])

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # Query positions (biased toward small gaps — queries soon after KV prefix).
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()
    gaps = np.stack([rng.choice(space, size=num_kv_pairs, replace=False, p=p)
                     for _ in range(num_examples)])

    # Build the post-KV region: length = input_seq_len - context_size + 1 so that
    # labels[gaps*2 + context_size + 1] is addressable; we trim below.
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)   # (N, seq+1)
    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values=values, axis=1)

    inputs = examples[:, :-1]
    labels = labels[:, 1:]                              # shift for standard LM
    if random_non_queries:
        rand = rng.integers(vocab_size, size=inputs.shape)
        inputs = np.where(inputs == 0, rand, inputs)
    return inputs.astype(np.int32), labels.astype(np.int32)


# =============================================================================
# Loss / accuracy — mask positions with label == -100.
# =============================================================================

def masked_loss(model, inputs, labels, dropout_key=None):
    logits = model(inputs, dropout_key=dropout_key)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    mask = (labels != -100).astype(jnp.float32)
    safe = jnp.where(mask > 0, labels, 0)
    nll = -jnp.take_along_axis(log_probs, safe[..., None], axis=-1).squeeze(-1)
    return jnp.sum(nll * mask) / jnp.clip(jnp.sum(mask), 1.0)


@eqx.filter_jit(donate='all')
def train_step(model, opt_state, optimizer, inputs, labels, dropout_key):
    loss, grads = eqx.filter_value_and_grad(masked_loss)(model, inputs, labels, dropout_key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, loss


@eqx.filter_jit
def batch_accuracy(model, inputs, labels):
    logits = model(inputs)
    preds = jnp.argmax(logits, axis=-1)
    mask = (labels != -100)
    return jnp.sum((preds == labels) & mask).astype(jnp.float32), jnp.sum(mask).astype(jnp.float32)


def evaluate(model, inputs, labels, batch_size):
    correct = total = 0.0
    for i in range(0, len(inputs), batch_size):
        c, t = batch_accuracy(model, jnp.asarray(inputs[i:i + batch_size]),
                              jnp.asarray(labels[i:i + batch_size]))
        correct += float(c); total += float(t)
    return correct / max(total, 1.0)


# =============================================================================
# Entry point.
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab', type=int, default=8192)
    p.add_argument('--seq-len', type=int, default=256)
    p.add_argument('--n-kv', type=int, default=64)
    p.add_argument('--n-train', type=int, default=20000)
    p.add_argument('--n-test', type=int, default=1000)
    p.add_argument('--n-layer', type=int, default=2)
    p.add_argument('--n-head', type=int, default=2)
    p.add_argument('--n-embd', type=int, default=128)
    p.add_argument('--chunk-size', type=int, default=64)
    p.add_argument('--memory-expand', type=int, default=1,
                   help='Memory MLP expansion factor (paper default: 4).')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threshold', type=float, default=0.95)
    p.add_argument('--log-every', type=int, default=2)
    args = p.parse_args()

    jax.config.update('jax_default_matmul_precision', 'float32')
    print(f'JAX {jax.__version__} | devices: {jax.devices()}')

    # Data.
    train_in, train_lb = generate_mqar(args.n_train, args.seq_len, args.vocab,
                                        args.n_kv, seed=args.seed)
    test_in, test_lb = generate_mqar(args.n_test, args.seq_len, args.vocab,
                                      args.n_kv, seed=args.seed + 1)
    print(f'train {train_in.shape} | test {test_in.shape} | '
          f'vocab={args.vocab} n_kv={args.n_kv} seq={args.seq_len}')

    # Model.
    config = AtlasConfig(
        vocab_size=args.vocab,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        seq_len=args.seq_len, chunk_size=args.chunk_size,
        memory_expand=args.memory_expand, omega_window=4, conv_kernel=4,
    )
    key = jax.random.PRNGKey(args.seed)
    key, mkey = jax.random.split(key)
    model = Atlas(config, key=mkey)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f'Model: {n_params / 1e6:.2f}M params')

    # Optim.
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=100, decay_steps=args.epochs * (args.n_train // args.batch_size),
        end_value=args.lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    steps_per_epoch = args.n_train // args.batch_size
    print(f'Training: {args.epochs} epochs × {steps_per_epoch} steps = '
          f'{args.epochs * steps_per_epoch} total')
    print('-' * 72)

    rng = np.random.default_rng(args.seed + 10)
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        perm = rng.permutation(args.n_train)
        for s in range(steps_per_epoch):
            step += 1
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            bi = jnp.asarray(train_in[idx])
            bl = jnp.asarray(train_lb[idx])
            key, dk = jax.random.split(key)
            model, opt_state, loss = train_step(model, opt_state, optimizer, bi, bl, dk)

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            # Quick eval on first 256 test samples for signal.
            acc_probe = evaluate(model, test_in[:256], test_lb[:256], args.batch_size)
            print(f'epoch {epoch + 1:3d} | step {step:5d} | loss {float(loss):.4f} | '
                  f'test_acc(256) {acc_probe:.4f} | {time.time() - t0:.1f}s')

    # Final eval.
    print('-' * 72)
    test_acc = evaluate(model, test_in, test_lb, args.batch_size)
    status = 'PASS' if test_acc >= args.threshold else 'FAIL'
    print(f'[mqar] test_acc={test_acc:.4f} threshold={args.threshold} {status} '
          f'({time.time() - t0:.1f}s)')
    sys.exit(0 if test_acc >= args.threshold else 1)


if __name__ == '__main__':
    main()
