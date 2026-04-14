"""JAX-to-PyTorch evaluation wrapper.

Makes atlas-jax models compatible with nanochat's evaluation framework
(CORE metric + BPB). Converts torch↔numpy↔jax at boundaries.

Usage:
    from atlas_jax.eval_wrapper import load_atlas_for_eval
    model_wrapper, tokenizer = load_atlas_for_eval(checkpoint_dir, model_type='mag')
    # model_wrapper is callable with nanochat's eval interface
"""

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class AtlasEvalWrapper:
    """Wraps a JAX Atlas/MAG model for use with nanochat's eval code.

    Interface:
        wrapper(input_ids) -> logits  (torch tensors)
        wrapper(x, y, loss_reduction='none') -> (B, T) per-token loss
        wrapper.max_seq_len -> int
        wrapper.get_device() -> torch.device
    """

    def __init__(self, jax_model, max_seq_len=2048, device='cuda'):
        assert _HAS_TORCH, "PyTorch required for eval wrapper"
        self.jax_model = jax_model
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)

    def __call__(self, x, y=None, loss_reduction='mean'):
        """Forward pass compatible with nanochat eval.

        Args:
            x: torch.Tensor (B, T) input token ids
            y: torch.Tensor (B, T) target token ids (optional)
            loss_reduction: 'mean', 'sum', or 'none'

        Returns:
            If y is None: logits (B, T, V) as torch tensor
            If y is given: loss scalar or (B, T) per-token loss
        """
        # Torch → numpy → JAX
        np_input = x.cpu().numpy()
        jax_input = jnp.array(np_input, dtype=jnp.int32)

        # Forward through JAX model
        jax_logits, _ = self.jax_model(jax_input)

        # JAX → numpy → Torch
        logits = torch.from_numpy(np.array(jax_logits)).to(self.device)

        if y is None:
            return logits

        # Compute loss
        if loss_reduction == 'none':
            # Per-token loss: (B, T)
            log_probs = F.log_softmax(logits, dim=-1)
            # Gather target log probs
            B, T, V = logits.shape
            loss = -log_probs.gather(2, y.unsqueeze(-1).clamp(min=0)).squeeze(-1)
            # Mask out -1 targets
            loss = loss * (y != -1).float()
            return loss
        else:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-1,
                reduction=loss_reduction)

    def get_device(self):
        return self.device


def load_atlas_for_eval(checkpoint_dir=None, model_type='mag', config_kwargs=None):
    """Load an Atlas model for evaluation.

    Args:
        checkpoint_dir: Path to checkpoint (None = random init for testing)
        model_type: 'mag' or 'lmm'
        config_kwargs: Override config parameters

    Returns:
        wrapper: AtlasEvalWrapper
    """
    from atlas_jax.config import AtlasConfig

    # Default config matching the 150M training run
    defaults = dict(
        vocab_size=32768, n_layer=12, n_head=12, n_embd=768, dim_head=64,
        memory_expand=1, chunk_size=64, ns_steps=5, omega_window=2,
        poly_degree=2, deep_memory=True, pe_ste=True, fused_chunk=False,
        stop_grad_chunks=True, geglu_ff=True, gate_bias_init=0.0, max_lr=0.1,
        window_size=64, neural_memory_layers=(1, 3, 5, 7, 9, 11))

    if config_kwargs:
        defaults.update(config_kwargs)

    config = AtlasConfig(**defaults)

    key = jax.random.PRNGKey(42)
    if model_type == 'mag':
        from atlas_jax.mag_transformer import MAGTransformer
        model = MAGTransformer(config, key=key)
    else:
        from atlas_jax.model import Atlas
        model = Atlas(config, key=key, pad_vocab_size_to=1)

    # Load checkpoint if provided
    if checkpoint_dir is not None:
        model = _load_checkpoint(model, checkpoint_dir)

    return AtlasEvalWrapper(model, max_seq_len=config.sequence_len)


def _load_checkpoint(model, checkpoint_dir):
    """Load model weights from equinox checkpoint."""
    import os
    import glob

    # Find latest checkpoint
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_*.eqx")))
    if not ckpt_files:
        # Try the format used by train_distributed.py
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "step_*")))

    if not ckpt_files:
        print(f"Warning: no checkpoint found in {checkpoint_dir}, using random init")
        return model

    latest = ckpt_files[-1]
    print(f"Loading checkpoint: {latest}")
    model = eqx.tree_deserialise_leaves(latest, model)
    return model
