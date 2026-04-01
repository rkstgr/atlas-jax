"""Training loop for Atlas-JAX.

Single-GPU training with:
- eqx.filter_jit for compiled train/eval steps
- eqx.filter_value_and_grad for differentiation
- Soft logit capping (15.0 * tanh(logits / 15.0))
- MFU and BPB reporting
"""

import os
import sys
import time
import math
import argparse
from dataclasses import asdict

import jax
# Force f32 matmul precision — TF32 on H100 causes NaN in Newton-Schulz iterations
jax.config.update("jax_default_matmul_precision", "float32")
import jax.numpy as jnp
import equinox as eqx
import optax

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas
from atlas_jax.data import data_loader
from atlas_jax.tokenizer import get_tokenizer

# H100 SXM bf16 peak TFLOPS
GPU_PEAK_TFLOPS = {"H100": 989.4, "RTX8000": 32.6}


def estimate_flops_per_token(config, n_params, n_embed_params):
    """Estimate FLOPs per token (forward + backward = 6N + memory ops)."""
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


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------

@eqx.filter_jit
def train_step(model, opt_state, optimizer, inputs, targets):
    def loss_fn(model):
        logits, _ = model(inputs)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        loss = -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])
        return loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


@eqx.filter_jit
def eval_step(model, inputs, targets):
    logits, _ = model(inputs)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    targets_flat = targets.reshape(-1)
    log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
    loss = -jnp.mean(log_probs[jnp.arange(targets_flat.shape[0]), targets_flat])
    return loss


def main():
    parser = argparse.ArgumentParser(description='Atlas-JAX Training')
    parser.add_argument('--n-layer', type=int, default=8)
    parser.add_argument('--n-head', type=int, default=8)
    parser.add_argument('--n-embd', type=int, default=448)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--omega-window', type=int, default=16)
    parser.add_argument('--poly-degree', type=int, default=3)
    parser.add_argument('--deep-memory', action='store_true', default=True)
    parser.add_argument('--no-deep-memory', dest='deep_memory', action='store_false')
    parser.add_argument('--memory-expand', type=int, default=4)
    parser.add_argument('--pe-ste', action='store_true', default=False)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--ns-steps', type=int, default=5)

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--total-steps', type=int, default=2000)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--eval-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-peak-tflops', type=float, default=989.4,
                        help='GPU peak bf16 TFLOPS for MFU calc (H100=989.4)')

    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--tokenizer-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default='out/atlas-jax')

    args = parser.parse_args()

    print(f"JAX {jax.__version__} | devices: {jax.devices()}")
    key = jax.random.PRNGKey(args.seed)

    config = AtlasConfig(
        sequence_len=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        chunk_size=args.chunk_size,
        ns_steps=args.ns_steps,
        omega_window=args.omega_window,
        poly_degree=args.poly_degree,
        deep_memory=args.deep_memory,
        memory_expand=args.memory_expand,
        pe_ste=args.pe_ste,
    )
    print(f"Config: {asdict(config)}")

    key, model_key = jax.random.split(key)
    model = Atlas(config, key=model_key)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    n_embed_params = model.wte.weight.size
    flops_per_token = estimate_flops_per_token(config, n_params, n_embed_params)
    print(f"Parameters: {n_params:,} | FLOPs/token: {flops_per_token:,.0f}")

    # BPB conversion: BPB = loss * (1 / ln(2)) * (tokens / bytes)
    # For BPE tokenizers, ~3.3 chars/token on average for English text
    # More precisely: BPB = cross_entropy_nats / ln(2) / chars_per_token
    # We use the standard approximation: BPB ≈ loss / ln(2) / 3.3
    # This matches nanochat's convention
    CHARS_PER_TOKEN = 3.3
    BPB_FACTOR = 1.0 / (math.log(2) * CHARS_PER_TOKEN)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=args.warmup_steps,
        decay_steps=args.total_steps,
        end_value=args.lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Data
    tokenizer = get_tokenizer(args.tokenizer_dir)
    train_loader = data_loader(
        args.data_dir, tokenizer, args.batch_size, args.seq_len, split='train')
    val_loader = data_loader(
        args.data_dir, tokenizer, args.batch_size, args.seq_len, split='val')

    tokens_per_step = args.batch_size * args.seq_len
    gpu_peak_flops = args.gpu_peak_tflops * 1e12

    print(f"Tokens/step: {tokens_per_step:,} | GPU peak: {args.gpu_peak_tflops} TFLOPS")
    print(f"Starting training for {args.total_steps} steps...")
    print("-" * 80)

    # Warmup compilation with first batch
    print("Compiling train step (first step will be slow)...")
    t_compile = time.time()
    inputs, targets = next(train_loader)
    model, opt_state, loss = train_step(model, opt_state, optimizer, inputs, targets)
    float(loss)  # block until done
    compile_time = time.time() - t_compile
    print(f"Compilation done in {compile_time:.1f}s | initial loss: {float(loss):.4f}")
    print("-" * 80)

    for step in range(1, args.total_steps):
        t0 = time.time()
        inputs, targets = next(train_loader)
        model, opt_state, loss = train_step(model, opt_state, optimizer, inputs, targets)
        loss_val = float(loss)
        dt = time.time() - t0

        if step % 10 == 0 or step < 5:
            tps = tokens_per_step / dt
            mfu = (flops_per_token * tps) / gpu_peak_flops * 100
            bpb = loss_val * BPB_FACTOR
            print(f"step {step:5d} | loss {loss_val:.4f} | bpb {bpb:.4f} | "
                  f"{dt*1000:.0f}ms | {tps:.0f} tok/s | MFU {mfu:.2f}%")

        if args.eval_every > 0 and step % args.eval_every == 0:
            val_losses = []
            for _ in range(args.eval_steps):
                val_inputs, val_targets = next(val_loader)
                val_loss = eval_step(model, val_inputs, val_targets)
                val_losses.append(float(val_loss))
            avg_val_loss = sum(val_losses) / len(val_losses)
            val_bpb = avg_val_loss * BPB_FACTOR
            print(f"  >>> EVAL | val_loss {avg_val_loss:.4f} | val_bpb {val_bpb:.4f}")

    # Final eval
    print("-" * 80)
    print("Final evaluation...")
    val_losses = []
    for _ in range(50):
        val_inputs, val_targets = next(val_loader)
        val_loss = eval_step(model, val_inputs, val_targets)
        val_losses.append(float(val_loss))
    avg_val_loss = sum(val_losses) / len(val_losses)
    val_bpb = avg_val_loss * BPB_FACTOR
    print(f"FINAL | val_loss {avg_val_loss:.4f} | val_bpb {val_bpb:.4f}")
    print("Training complete.")


if __name__ == '__main__':
    main()
