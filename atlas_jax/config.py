"""AtlasConfig dataclass — all architecture hyperparameters."""

from dataclasses import dataclass


@dataclass
class AtlasConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1536
    dim_head: int | None = None  # per-head dim; if None, n_embd // n_head
    chunk_size: int = 64       # tokens per chunk for parallel memory computation
    conv_kernel: int = 4       # short causal convolution kernel size
    ns_steps: int = 5          # Polar Express orthogonalization iterations
    omega_window: int = 16     # Omega rule sliding window size (1 = online/Delta rule)
    poly_degree: int = 2       # polynomial feature mapping degree (0 = disabled)
    deep_memory: bool = True   # deep MLP memory vs linear matrix memory
    memory_expand: int = 2     # MLP expansion factor for deep memory (PyTorch default = 2)
    pe_ste: bool = False       # Polar Express straight-through estimator
    use_checkpoint: bool = True  # gradient checkpointing per chunk
    fused_chunk: bool = False   # FlashATLAS: fused Triton kernel for inner scan+PE+memory+output
    dropout: float = 0.0        # dropout rate (0.0 = disabled, PyTorch default)
    gate_bias_init: float = 0.0  # gate bias init (0.0 = sigmoid≈0.5, PyTorch default)
    max_lr: float = 0.1         # max learning rate for eta gate (PyTorch default_step_transform_max_lr)
    logit_softcap: float = 0.0  # logit soft-capping (0.0 = disabled, PyTorch has none)
    num_persist_mem_tokens: int = 0  # learnable prefix tokens (PyTorch default = 4)
    stop_grad_chunks: bool = False   # True = paper (frozen boundary), False = PyTorch (gradient flows)
    geglu_ff: bool = True       # SiLU-gated feedforward (True = PyTorch GEGLU, False = plain GELU)
    # MAG (Memory As Gate) settings
    window_size: int = 64        # sliding window size for attention
    neural_memory_layers: tuple | None = None  # which layers get memory (None = all)
