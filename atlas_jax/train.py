"""Minimal single-GPU training loop for Atlas.

Supports enwik8 (byte-level) and climbmix (tiktoken parquet) via `--dataset`.
AdamW + linear warmup + cosine decay. Global grad-norm clip at 1.0.
"""

import argparse
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from atlas_jax.model import Atlas, AtlasConfig
from atlas_jax.data import enwik8_loader, climbmix_loader
from atlas_jax.tokenizer import get_tokenizer


# enwik8 is byte-level, so bits-per-byte = loss / ln(2).
# climbmix uses tiktoken (~3.3 chars/token), so BPB = loss / (ln(2) * 3.3).
BPB_DIVISOR = {'enwik8': math.log(2), 'climbmix': math.log(2) * 3.3}


def loss_fn(model, inputs, targets, dropout_key=None):
    logits = model(inputs, dropout_key=dropout_key)
    log_probs = jax.nn.log_softmax(logits.reshape(-1, logits.shape[-1]), axis=-1)
    return -jnp.mean(log_probs[jnp.arange(targets.size), targets.reshape(-1)])


@eqx.filter_jit(donate='all')
def train_step(model, opt_state, optimizer, inputs, targets, dropout_key):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, targets, dropout_key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, loss


@eqx.filter_jit
def eval_step(model, inputs, targets):
    return loss_fn(model, inputs, targets)


def save(model, opt_state, step, ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(ckpt_dir / 'model.eqx'), model)
    eqx.tree_serialise_leaves(str(ckpt_dir / 'opt_state.eqx'), opt_state)
    (ckpt_dir / 'meta.txt').write_text(str(step))


def load(model, opt_state, ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    if not (ckpt_dir / 'model.eqx').exists():
        return model, opt_state, 0
    model = eqx.tree_deserialise_leaves(str(ckpt_dir / 'model.eqx'), model)
    opt_state = eqx.tree_deserialise_leaves(str(ckpt_dir / 'opt_state.eqx'), opt_state)
    step = int((ckpt_dir / 'meta.txt').read_text().strip())
    print(f'[ckpt] resumed from step {step}')
    return model, opt_state, step


def main():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument('--n-layer', type=int, default=8)
    p.add_argument('--n-head', type=int, default=8)
    p.add_argument('--n-embd', type=int, default=448)
    p.add_argument('--seq-len', type=int, default=1024)
    p.add_argument('--chunk-size', type=int, default=64)
    p.add_argument('--memory-expand', type=int, default=1)
    p.add_argument('--omega-window', type=int, default=4)
    p.add_argument('--conv-kernel', type=int, default=4)

    # Optim
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--warmup-steps', type=int, default=200)
    p.add_argument('--total-steps', type=int, default=2000)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--eval-steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)

    # Data
    p.add_argument('--dataset', choices=['enwik8', 'climbmix'], required=True)
    p.add_argument('--data-path', type=str, required=True,
                   help='enwik8 file path, or climbmix parquet dir')
    p.add_argument('--tokenizer-dir', type=str, default=None,
                   help='Required for climbmix.')

    # Misc
    p.add_argument('--out-dir', type=str, default='out/atlas-jax')
    p.add_argument('--matmul-precision', choices=['float32', 'high', 'default'],
                   default='float32')
    args = p.parse_args()

    jax.config.update('jax_default_matmul_precision', args.matmul_precision)
    print(f'JAX {jax.__version__} | devices: {jax.devices()} | precision: {args.matmul_precision}')

    # Data + vocab.
    if args.dataset == 'enwik8':
        vocab_size = 256
        train_loader = enwik8_loader(args.data_path, args.batch_size, args.seq_len, 'train', seed=args.seed)
        val_loader = enwik8_loader(args.data_path, args.batch_size, args.seq_len, 'val', seed=args.seed + 1)
    else:
        assert args.tokenizer_dir, '--tokenizer-dir required for climbmix'
        tokenizer = get_tokenizer(args.tokenizer_dir)
        vocab_size = tokenizer.get_vocab_size()
        train_loader = climbmix_loader(args.data_path, tokenizer, args.batch_size, args.seq_len, 'train')
        val_loader = climbmix_loader(args.data_path, tokenizer, args.batch_size, args.seq_len, 'val')

    # Model.
    config = AtlasConfig(
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        seq_len=args.seq_len,
        chunk_size=args.chunk_size,
        memory_expand=args.memory_expand,
        omega_window=args.omega_window,
        conv_kernel=args.conv_kernel,
    )
    print(f'Config: {config}')

    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)
    model = Atlas(config, key=model_key)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f'Parameters: {n_params / 1e6:.1f}M')

    # Optimizer.
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=args.warmup_steps, decay_steps=args.total_steps,
        end_value=args.lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Resume if checkpoint exists.
    model, opt_state, start_step = load(model, opt_state, args.out_dir)

    tokens_per_step = args.batch_size * args.seq_len
    bpb_div = BPB_DIVISOR[args.dataset]
    print(f'Tokens/step: {tokens_per_step:,}')
    print('-' * 72)

    # Warmup compile.
    t0 = time.time()
    key, dk = jax.random.split(key)
    inputs, targets = next(train_loader)
    model, opt_state, loss = train_step(model, opt_state, optimizer, inputs, targets, dk)
    float(loss)
    print(f'Compiled in {time.time() - t0:.1f}s | initial loss {float(loss):.4f}')

    # Train.
    t_start = time.time()
    step_times = []
    step = start_step
    while step < args.total_steps:
        step += 1
        t = time.time()
        key, dk = jax.random.split(key)
        inputs, targets = next(train_loader)
        model, opt_state, loss = train_step(model, opt_state, optimizer, inputs, targets, dk)
        loss_val = float(loss)
        dt = time.time() - t
        step_times.append(dt)

        if step % 10 == 0 or step < 5:
            bpb = loss_val / bpb_div
            tps = tokens_per_step / dt
            elapsed = time.time() - t_start
            print(f'step {step:5d} | loss {loss_val:.4f} | bpb {bpb:.4f} | '
                  f'{dt * 1000:.0f}ms | {tps:.0f} tok/s | {elapsed:.0f}s')

        if args.eval_every > 0 and step % args.eval_every == 0:
            val_losses = []
            for _ in range(args.eval_steps):
                vi, vt = next(val_loader)
                val_losses.append(float(eval_step(model, vi, vt)))
            avg = sum(val_losses) / len(val_losses)
            print(f'  >>> eval | val_loss {avg:.4f} | val_bpb {avg / bpb_div:.4f}')
            save(model, opt_state, step, args.out_dir)

    # Final eval.
    val_losses = []
    for _ in range(args.eval_steps):
        vi, vt = next(val_loader)
        val_losses.append(float(eval_step(model, vi, vt)))
    avg = sum(val_losses) / len(val_losses)
    save(model, opt_state, step, args.out_dir)
    print('-' * 72)
    print(f'FINAL | val_loss {avg:.4f} | val_bpb {avg / bpb_div:.4f} | '
          f'{step} steps | {(step * tokens_per_step) / 1e6:.1f}M tokens')


if __name__ == '__main__':
    main()
