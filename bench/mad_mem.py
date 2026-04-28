"""MAD Memorization task (Poli et al. 2024) — correctness bench for atlas-jax.

Faithful reimplementation of the `memorization` task from
github.com/athms/mad-lab/blob/main/mad/data/instances.py. The task forces the
model to memorize a global key→value table in its parameters (not in-context
recall): a fixed KV map of 127 keys is sampled once with seed 12345, and every
sequence shows random keys interleaved with an `insert_token` at which the
model must output the matching value.

This is the single most discriminating sub-GPU-hour bench: Transformer ≈ 83.8%,
Atlas ≈ 91.4% (paper Table tab:MAD). A broken memory layer or bad gate init
sits at Transformer-level, not random.

Usage:
    python bench/mad_mem.py                  # default config
    python bench/mad_mem.py --epochs 100     # faster smoke
Pass: test accuracy ≥ 0.90 on value positions.
"""

import argparse
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from atlas_jax.model import Atlas, AtlasConfig


# =============================================================================
# Data — matches mad/data/instances.py::generate_memorization_instance.
# =============================================================================

def build_kv_map(vocab_size, seed=12345):
    """Global fixed KV map. insert_token = vocab-1, keys ∈ [0, vocab//2),
    values ∈ [vocab//2, vocab-1). kv[k] sampled i.i.d. from values."""
    insert_token = vocab_size - 1
    n_keys = vocab_size // 2                  # e.g. 128
    key_vocab = np.arange(n_keys)              # [0, 128)
    value_vocab = np.arange(vocab_size // 2, vocab_size - 1)   # [128, 255)
    rng = np.random.default_rng(seed)
    kv_map = rng.choice(value_vocab, size=n_keys, replace=True)
    return kv_map, key_vocab, insert_token


def generate_dataset(n_samples, seq_len, kv_map, key_vocab, insert_token, seed):
    """N sequences of `[k0, insert, k1, insert, ...]`. Targets are -100 at key
    positions, kv_map[k_i] at insert positions. No autoregressive shift —
    inputs[t] and targets[t] are paired 1:1."""
    assert seq_len % 2 == 0
    n_pairs = seq_len // 2
    rng = np.random.default_rng(seed)
    keys_chosen = rng.choice(key_vocab, size=(n_samples, n_pairs), replace=True)
    inputs = np.empty((n_samples, seq_len), dtype=np.int32)
    targets = np.full((n_samples, seq_len), -100, dtype=np.int32)
    inputs[:, 0::2] = keys_chosen
    inputs[:, 1::2] = insert_token
    targets[:, 1::2] = kv_map[keys_chosen]
    return inputs, targets


# =============================================================================
# Loss / accuracy — ignore target == -100.
# =============================================================================

def masked_loss(model, inputs, targets, dropout_key=None):
    logits = model(inputs, dropout_key=dropout_key)          # (B, T, V)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    mask = (targets != -100).astype(jnp.float32)
    safe_targets = jnp.where(mask > 0, targets, 0)
    nll = -jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1).squeeze(-1)
    return jnp.sum(nll * mask) / jnp.clip(jnp.sum(mask), 1.0)


@eqx.filter_jit(donate='all')
def train_step(model, opt_state, optimizer, inputs, targets, dropout_key):
    loss, grads = eqx.filter_value_and_grad(masked_loss)(model, inputs, targets, dropout_key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, loss


@eqx.filter_jit
def batch_accuracy(model, inputs, targets):
    logits = model(inputs)
    preds = jnp.argmax(logits, axis=-1)
    mask = (targets != -100)
    return jnp.sum((preds == targets) & mask).astype(jnp.float32), jnp.sum(mask).astype(jnp.float32)


def evaluate(model, inputs, targets, batch_size):
    correct = total = 0.0
    for i in range(0, len(inputs), batch_size):
        c, t = batch_accuracy(model, jnp.asarray(inputs[i:i + batch_size]),
                              jnp.asarray(targets[i:i + batch_size]))
        correct += float(c); total += float(t)
    return correct / max(total, 1.0)


# =============================================================================
# Entry point.
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab', type=int, default=256)
    p.add_argument('--seq-len', type=int, default=32)
    p.add_argument('--n-train', type=int, default=256)
    p.add_argument('--n-test', type=int, default=1280)
    p.add_argument('--n-layer', type=int, default=2)
    p.add_argument('--n-head', type=int, default=8)
    p.add_argument('--n-embd', type=int, default=128)
    p.add_argument('--chunk-size', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threshold', type=float, default=0.90)
    p.add_argument('--log-every', type=int, default=50)
    args = p.parse_args()

    jax.config.update('jax_default_matmul_precision', 'float32')
    print(f'JAX {jax.__version__} | devices: {jax.devices()}')

    # Data.
    kv_map, key_vocab, insert_token = build_kv_map(args.vocab, seed=12345)
    train_in, train_tg = generate_dataset(args.n_train, args.seq_len, kv_map, key_vocab,
                                          insert_token, seed=args.seed)
    test_in, test_tg = generate_dataset(args.n_test, args.seq_len, kv_map, key_vocab,
                                        insert_token, seed=args.seed + 1)
    print(f'train {train_in.shape} | test {test_in.shape} | '
          f'{len(key_vocab)} keys → values in [{args.vocab // 2}, {args.vocab - 1})')

    # Model.
    config = AtlasConfig(
        vocab_size=args.vocab,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        seq_len=args.seq_len, chunk_size=args.chunk_size,
        memory_expand=1, omega_window=4, conv_kernel=4,
    )
    key = jax.random.PRNGKey(args.seed)
    key, mkey = jax.random.split(key)
    model = Atlas(config, key=mkey)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f'Model: {n_params / 1e6:.2f}M params | config {config}')

    # Optim.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=args.lr, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Train.
    steps_per_epoch = args.n_train // args.batch_size
    rng = np.random.default_rng(args.seed + 10)
    print(f'Training: {args.epochs} epochs × {steps_per_epoch} step(s) = '
          f'{args.epochs * steps_per_epoch} total steps')
    print('-' * 72)

    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        perm = rng.permutation(args.n_train)
        for s in range(steps_per_epoch):
            step += 1
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            bi = jnp.asarray(train_in[idx])
            bt = jnp.asarray(train_tg[idx])
            key, dk = jax.random.split(key)
            model, opt_state, loss = train_step(model, opt_state, optimizer, bi, bt, dk)
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            train_acc = evaluate(model, train_in, train_tg, args.batch_size)
            print(f'epoch {epoch + 1:4d} | step {step:5d} | loss {float(loss):.4f} | '
                  f'train_acc {train_acc:.4f} | {time.time() - t0:.1f}s')

    # Final eval.
    print('-' * 72)
    train_acc = evaluate(model, train_in, train_tg, args.batch_size)
    test_acc = evaluate(model, test_in, test_tg, args.batch_size)
    status = 'PASS' if test_acc >= args.threshold else 'FAIL'
    print(f'[mad_mem] train_acc={train_acc:.4f} test_acc={test_acc:.4f} '
          f'threshold={args.threshold} target(paper)=0.914 {status} '
          f'({time.time() - t0:.1f}s)')
    sys.exit(0 if test_acc >= args.threshold else 1)


if __name__ == '__main__':
    main()
