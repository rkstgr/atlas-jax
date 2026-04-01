"""AtlasConfig dataclass — all architecture hyperparameters."""

from dataclasses import dataclass


@dataclass
class AtlasConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1536
    chunk_size: int = 64       # tokens per chunk for parallel memory computation
    conv_kernel: int = 4       # short causal convolution kernel size
    ns_steps: int = 5          # Polar Express orthogonalization iterations
    omega_window: int = 16     # Omega rule sliding window size (1 = online/Delta rule)
    poly_degree: int = 3       # polynomial feature mapping degree (0 = disabled)
    deep_memory: bool = True   # deep MLP memory vs linear matrix memory
    memory_expand: int = 4     # MLP expansion factor for deep memory (paper default = 4)
    pe_ste: bool = False       # Polar Express straight-through estimator
    use_checkpoint: bool = True  # gradient checkpointing per chunk
