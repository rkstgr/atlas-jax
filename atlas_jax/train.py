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
    parser.add_argument('--memory-expand', type=int, default=1)
    parser.add_argument('--pe-ste', action='store_true', default=True)
    parser.add_argument('--no-pe-ste', dest='pe_ste', action='store_false')
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--ns-steps', type=int, default=3)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--total-steps', type=int, default=2000)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--eval-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-peak-tflops', type=float, default=989.4,
                        help='GPU peak bf16 TFLOPS for MFU calc (H100=989.4)')

    parser.add_argument('--matmul-precision', type=str, default='float32',
                        choices=['float32', 'high', 'default'])
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--tokenizer-dir', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default='out/atlas-jax')
    parser.add_argument('--time-budget', type=int, default=0,
                        help='Training time budget in seconds (0 = use total-steps)')

    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    print(f"JAX {jax.__version__} | devices: {jax.devices()} | precision: {args.matmul_precision}")
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

    # Cast model to bf16 (PE upcasts to f32 internally; logits cast to f32 in model)
    def _to_bf16(x):
        return x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x
    model = jax.tree.map(_to_bf16, model, is_leaf=eqx.is_array)
    print(f"Model cast to bf16")

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
    use_time_budget = args.time_budget > 0
    if use_time_budget:
        print(f"Time budget: {args.time_budget}s")
    else:
        print(f"Training for {args.total_steps} steps")
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

    training_start = time.time()
    step = 0
    step_times = []

    while True:
        step += 1
        if use_time_budget:
            if time.time() - training_start >= args.time_budget:
                break
        else:
            if step >= args.total_steps:
                break

        t0 = time.time()
        inputs, targets = next(train_loader)
        model, opt_state, loss = train_step(model, opt_state, optimizer, inputs, targets)
        loss_val = float(loss)
        dt = time.time() - t0
        step_times.append(dt)

        if step % 10 == 0 or step < 5:
            tps = tokens_per_step / dt
            mfu = (flops_per_token * tps) / gpu_peak_flops * 100
            bpb = loss_val * BPB_FACTOR
            elapsed = time.time() - training_start
            print(f"step {step:5d} | loss {loss_val:.4f} | bpb {bpb:.4f} | "
                  f"{dt*1000:.0f}ms | {tps:.0f} tok/s | MFU {mfu:.2f}% | {elapsed:.0f}s")

        if args.eval_every > 0 and step % args.eval_every == 0:
            val_losses = []
            for _ in range(args.eval_steps):
                val_inputs, val_targets = next(val_loader)
                val_loss = eval_step(model, val_inputs, val_targets)
                val_losses.append(float(val_loss))
            avg_val_loss = sum(val_losses) / len(val_losses)
            val_bpb = avg_val_loss * BPB_FACTOR
            print(f"  >>> EVAL | val_loss {avg_val_loss:.4f} | val_bpb {val_bpb:.4f}")

    training_seconds = time.time() - training_start

    # Final eval
    print("-" * 80)
    print("Final evaluation...")
    val_losses = []
    for _ in range(args.eval_steps):
        val_inputs, val_targets = next(val_loader)
        val_loss = eval_step(model, val_inputs, val_targets)
        val_losses.append(float(val_loss))
    avg_val_loss = sum(val_losses) / len(val_losses)
    val_bpb = avg_val_loss * BPB_FACTOR
    total_seconds = time.time() - training_start
    total_tokens = step * tokens_per_step

    # Compute average MFU
    warmup_skip = min(5, len(step_times))
    if len(step_times) > warmup_skip:
        avg_dt = sum(step_times[warmup_skip:]) / len(step_times[warmup_skip:])
        avg_tps = tokens_per_step / avg_dt
        avg_mfu = (flops_per_token * avg_tps) / gpu_peak_flops * 100
    else:
        avg_tps = total_tokens / max(training_seconds, 1e-6)
        avg_mfu = (flops_per_token * avg_tps) / gpu_peak_flops * 100

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"val_loss:         {avg_val_loss:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"mfu_percent:      {avg_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {n_params / 1e6:.1f}")
    print(f"tokens_per_sec:   {avg_tps:.0f}")
    print(f"ms_per_step:      {avg_dt * 1000:.1f}" if len(step_times) > warmup_skip else "ms_per_step:      0.0")
    print(f"FINAL | val_loss {avg_val_loss:.4f} | val_bpb {val_bpb:.4f}")
    print("Training complete.")


if __name__ == '__main__':
    main()
