"""Enwik8 training script for atlas-jax — matches atlas_pytorch/train_atlas.py.

Single-GPU, f32, AdamW. For verifying JAX matches PyTorch on enwik8.
"""

import argparse
import math
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# Precision set after arg parse (--bf16 uses 'high', default uses 'float32')

from atlas_jax.config import AtlasConfig
from atlas_jax.model import Atlas
from atlas_jax.enwik8 import enwik8_data_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=320)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dim-head', type=int, default=64)
    parser.add_argument('--memory-expand', type=int, default=2)
    parser.add_argument('--poly-degree', type=int, default=2)
    parser.add_argument('--omega-window', type=int, default=2)
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--ns-steps', type=int, default=5)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--num-batches', type=int, default=5000)
    parser.add_argument('--validate-every', type=int, default=100)
    parser.add_argument('--eval-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-path', type=str, default='data/enwik8.gz')
    parser.add_argument('--stop-grad-chunks', action='store_true', default=False,
                        help='Use stop_gradient at chunk boundaries (paper mode)')
    parser.add_argument('--bf16', action='store_true', default=False,
                        help='Mixed precision: bf16 model weights, f32 optimizer')
    parser.add_argument('--fused-chunk', action='store_true', default=False,
                        help='Use fused Triton kernel')
    parser.add_argument('--model', type=str, default='lmm', choices=['lmm', 'mag'],
                        help='Architecture: lmm (memory only) or mag (memory + attention)')
    parser.add_argument('--window-size', type=int, default=64,
                        help='Sliding window size for MAG attention')
    parser.add_argument('--memory-layers', type=str, default=None,
                        help='Comma-separated layer indices for memory (MAG only, e.g. "1,3,5,7")')
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", "high" if args.bf16 else "float32")

    key = jax.random.PRNGKey(args.seed)

    config = AtlasConfig(
        vocab_size=256,
        sequence_len=args.seq_len,
        n_layer=args.depth,
        n_head=args.heads,
        n_embd=args.dim,
        dim_head=args.dim_head,
        chunk_size=args.chunk_size,
        ns_steps=args.ns_steps,
        omega_window=args.omega_window,
        poly_degree=args.poly_degree,
        deep_memory=True,
        memory_expand=args.memory_expand,
        pe_ste=True,
        use_checkpoint=True,
        fused_chunk=args.fused_chunk,
        dropout=0.0,
        gate_bias_init=0.0,
        max_lr=0.1,
        logit_softcap=0.0,
        stop_grad_chunks=args.stop_grad_chunks,
        geglu_ff=True,
        num_persist_mem_tokens=4 if args.model == 'lmm' else 0,
        window_size=args.window_size,
        neural_memory_layers=tuple(int(x) for x in args.memory_layers.split(',')) if args.memory_layers else None,
    )

    key, model_key = jax.random.split(key)
    if args.model == 'mag':
        from atlas_jax.mag_transformer import MAGTransformer
        model = MAGTransformer(config, key=model_key)
    else:
        model = Atlas(config, key=model_key, pad_vocab_size_to=1)

    if args.bf16:
        model = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x,
            model, is_leaf=eqx.is_array)

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Config: dim={args.dim}, depth={args.depth}, heads={args.heads}, "
          f"dim_head={args.dim_head}, expand={args.memory_expand}")

    # Optimizer: AdamW with gradient clipping and cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=100,
        decay_steps=args.num_batches,
        end_value=args.lr * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Data
    train_loader = enwik8_data_loader(
        args.data_path, args.batch_size, args.seq_len, split='train', seed=args.seed)
    val_loader = enwik8_data_loader(
        args.data_path, args.batch_size, args.seq_len, split='val', seed=args.seed + 1)

    @eqx.filter_jit
    def compute_loss(model, inputs, targets):
        logits, _ = model(inputs)
        logits_flat = logits.reshape(-1, logits.shape[-1]).astype(jnp.float32)
        targets_flat = targets.reshape(-1)
        return -jnp.mean(jax.nn.log_softmax(logits_flat, axis=-1)[
            jnp.arange(targets_flat.shape[0]), targets_flat])

    def _upcast_grads(grads):
        """Cast bf16 gradients to f32 for stable optimizer (clip_grad_norm overflows bf16)."""
        return jax.tree.map(
            lambda x: x.astype(jnp.float32) if eqx.is_array(x) and x.dtype == jnp.bfloat16 else x,
            grads, is_leaf=eqx.is_array)

    @eqx.filter_jit
    def train_step(model, opt_state, inputs, targets):
        loss, grads = eqx.filter_value_and_grad(
            lambda m: compute_loss(m, inputs, targets))(model)
        grads = _upcast_grads(grads)
        updates, new_opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), opt_state,
            eqx.filter(model, eqx.is_array))
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    # Gradient accumulation
    def train_step_with_accum(model, opt_state, loader, accum_steps):
        total_loss = 0.0
        for _ in range(accum_steps):
            inputs, targets = next(loader)
            model, opt_state, loss = train_step(model, opt_state, inputs, targets)
            total_loss += float(loss)
        return model, opt_state, total_loss / accum_steps

    # Warmup compilation
    print("Compiling...")
    t0 = time.time()
    inputs, targets = next(train_loader)
    model, opt_state, loss = train_step(model, opt_state, inputs, targets)
    float(loss)  # block
    print(f"Compilation done in {time.time()-t0:.1f}s | initial loss: {float(loss):.4f}")
    print("-" * 60)

    bpb_factor = 1.0 / math.log(2)  # nats to bits (byte-level, 1 char = 1 byte)

    for step in range(1, args.num_batches + 1):
        t0 = time.time()
        model, opt_state, avg_loss = train_step_with_accum(
            model, opt_state, train_loader, args.grad_accum)
        dt = time.time() - t0
        bpb = avg_loss * bpb_factor

        if step % 10 == 0 or step <= 5:
            tps = args.batch_size * args.seq_len * args.grad_accum / dt
            print(f"step {step:5d} | loss {avg_loss:.4f} | bpb {bpb:.4f} | "
                  f"{dt:.1f}s | {tps:.0f} tok/s")

        if step % args.validate_every == 0:
            # Validation
            val_losses = []
            for _ in range(args.eval_steps):
                v_inputs, v_targets = next(val_loader)
                v_loss = float(compute_loss(model, v_inputs, v_targets))
                val_losses.append(v_loss)
            val_loss = sum(val_losses) / len(val_losses)
            val_bpb = val_loss * bpb_factor
            print(f"  >>> EVAL | val_loss {val_loss:.4f} | val_bpb {val_bpb:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
