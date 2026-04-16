"""Needle-in-a-Haystack (S-NIAH) evaluation.

Tests a model's ability to retrieve a specific fact from a long context.
Inserts a "needle" (a random fact) at various depths in a "haystack" (filler text),
then queries the model to retrieve it.

Usage:
    python evals/eval_niah.py --checkpoint /path/to/ckpt --model mag --context-lengths 512,1024,2048,4096
"""

import argparse
import json
import math
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from atlas_jax.config import AtlasConfig


# Needle facts and queries
NEEDLES = [
    ("The secret password is 'elephant92'.", "What is the secret password?"),
    ("The capital of Zarkandia is Luminara.", "What is the capital of Zarkandia?"),
    ("The magic number is 7429.", "What is the magic number?"),
    ("Professor Smith's office is in room 314.", "What room is Professor Smith's office in?"),
    ("The launch code is ALPHA-BRAVO-42.", "What is the launch code?"),
]

# Filler text (repeated to fill context)
FILLER = """The weather patterns across the northern hemisphere showed significant variation during
the observed period. Temperature readings from multiple monitoring stations indicated a gradual
warming trend, consistent with seasonal expectations. Precipitation levels remained within
historical norms for most regions, though some coastal areas experienced above-average rainfall.
Wind patterns were largely stable, with occasional gusts reaching moderate speeds. Atmospheric
pressure readings showed typical fluctuations associated with passing weather systems. Cloud
cover varied from clear skies to overcast conditions depending on geographic location and time
of day. Humidity levels remained comfortable in most inland areas while coastal regions
experienced higher moisture content. Overall, the meteorological data collected during this
period aligned with established climate models and long-term historical trends for the region.
"""


def build_haystack_prompt(needle_text, query_text, context_length, depth_pct, tokenizer):
    """Build a prompt with needle inserted at given depth in filler text.

    Args:
        needle_text: The fact to hide
        query_text: The question to ask
        context_length: Target total context length in tokens
        depth_pct: Where to insert needle (0.0 = beginning, 1.0 = end)
        tokenizer: Callable that tokenizes text → list of token ids

    Returns:
        token_ids: Full prompt as token ids
        answer_start: Index where the answer should appear
    """
    # Tokenize components
    needle_tokens = tokenizer(needle_text)
    query_tokens = tokenizer(f"\n\nQuestion: {query_text}\nAnswer:")

    # How many filler tokens we need
    filler_budget = context_length - len(needle_tokens) - len(query_tokens) - 10  # margin
    if filler_budget < 0:
        filler_budget = 0

    # Generate filler by repeating and tokenizing
    filler_tokens = tokenizer(FILLER * 100)[:filler_budget]

    # Insert needle at depth
    insert_pos = int(len(filler_tokens) * depth_pct)
    context_tokens = filler_tokens[:insert_pos] + needle_tokens + filler_tokens[insert_pos:]

    # Full prompt: context + query
    full_tokens = context_tokens + query_tokens
    return full_tokens


def evaluate_niah(model, tokenizer_fn, context_lengths, depths, num_trials=5):
    """Run Needle-in-a-Haystack evaluation.

    Args:
        model: Callable model(token_ids) -> logits
        tokenizer_fn: Function text -> list[int]
        context_lengths: List of context lengths to test
        depths: List of depth percentages (0.0 to 1.0)
        num_trials: Number of needle/query pairs to test per cell

    Returns:
        results: Dict of {(ctx_len, depth): accuracy}
    """
    results = {}

    for ctx_len in context_lengths:
        for depth in depths:
            correct = 0
            total = 0

            for trial in range(min(num_trials, len(NEEDLES))):
                needle_text, query_text = NEEDLES[trial]

                # Build prompt
                prompt_tokens = build_haystack_prompt(
                    needle_text, query_text, ctx_len, depth, tokenizer_fn)

                # Truncate to model's capacity if needed
                prompt_tokens = prompt_tokens[-ctx_len:]

                # Forward pass
                input_ids = jnp.array([prompt_tokens], dtype=jnp.int32)
                logits, _ = model(input_ids)
                logits = logits[0]  # remove batch dim

                # Check if the model's next-token prediction contains the answer
                # We check if the top-1 predicted tokens after the query match
                # the needle's key content
                last_logit = logits[-1]  # logits for next token after prompt
                pred_token = int(jnp.argmax(last_logit))

                # Tokenize expected answer fragments
                # For "elephant92": check if predicted token is in the answer
                answer_key = needle_text.split("'")[1] if "'" in needle_text else needle_text.split("is ")[-1].rstrip(".")
                answer_tokens = tokenizer_fn(answer_key)

                if pred_token in answer_tokens[:3]:  # first 3 tokens of answer
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0.0
            results[(ctx_len, depth)] = accuracy
            print(f"  ctx={ctx_len:>6}, depth={depth:.0%}: {accuracy:.0%} ({correct}/{total})")

    return results


def print_heatmap(results, context_lengths, depths):
    """Print ASCII heatmap of results."""
    print(f"\n{'Ctx Len':>8}", end="")
    for d in depths:
        print(f" {d:.0%:>6}", end="")
    print()
    print("-" * (8 + 7 * len(depths)))

    for ctx_len in context_lengths:
        print(f"{ctx_len:>8}", end="")
        for d in depths:
            acc = results.get((ctx_len, d), 0.0)
            print(f" {acc:>5.0%}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack evaluation")
    parser.add_argument('--model', type=str, default='mag', choices=['lmm', 'mag'])
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--dim-head', type=int, default=64)
    parser.add_argument('--context-lengths', type=str, default='512,1024,2048,4096',
                        help='Comma-separated context lengths to test')
    parser.add_argument('--num-trials', type=int, default=5)
    args = parser.parse_args()

    jax.config.update("jax_default_matmul_precision", "float32")

    context_lengths = [int(x) for x in args.context_lengths.split(',')]
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Build model
    from atlas_jax.tokenizer import get_tokenizer
    tokenizer = get_tokenizer("/p/scratch/westai0047/nanochat/tokenizer")

    config = AtlasConfig(
        vocab_size=32768, n_layer=args.depth, n_head=args.heads, n_embd=args.dim,
        dim_head=args.dim_head, memory_expand=1, chunk_size=64, ns_steps=5,
        omega_window=2, poly_degree=2, deep_memory=True, pe_ste=True,
        fused_chunk=False, stop_grad_chunks=True, geglu_ff=True,
        window_size=64, neural_memory_layers=tuple(range(1, args.depth, 2)))

    key = jax.random.PRNGKey(42)
    if args.model == 'mag':
        from atlas_jax.mag_transformer import MAGTransformer
        model = MAGTransformer(config, key=key)
    else:
        from atlas_jax.model import Atlas
        model = Atlas(config, key=key)

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model: {args.model} {n_params/1e6:.1f}M params")
    print(f"Context lengths: {context_lengths}")
    print(f"Depths: {depths}")
    print()

    def tokenize(text):
        return tokenizer.encode(text)

    print("Running Needle-in-a-Haystack evaluation...")
    results = evaluate_niah(model, tokenize, context_lengths, depths, args.num_trials)
    print_heatmap(results, context_lengths, depths)


if __name__ == "__main__":
    main()
