"""Enwik8 byte-level data loader.

Loads the enwik8 dataset (100M bytes of Wikipedia XML) for language modeling.
Byte-level encoding: each byte is a token, vocab_size=256, no tokenizer needed.

Split: first 90M bytes = train, next 5M = val, last 5M = test.
Matches atlas_pytorch/train_atlas.py data loading exactly.
"""

import gzip
import numpy as np
import jax.numpy as jnp


def load_enwik8(data_path):
    """Load enwik8 from gzip file, return numpy uint8 array."""
    with gzip.open(data_path) as f:
        data = np.frombuffer(f.read(int(95e6)), dtype=np.uint8).copy()
    return data


def enwik8_data_loader(data_path, batch_size, seq_len, split='train', seed=42):
    """Infinite generator yielding (inputs, targets) batches from enwik8.

    Args:
        data_path: path to enwik8.gz
        batch_size: number of sequences per batch
        seq_len: sequence length (inputs are seq_len, targets are next-token)
        split: 'train' (first 90M), 'val' (next 5M), or 'test' (last 5M)
        seed: random seed for reproducibility

    Yields:
        (inputs, targets): jnp.int32 arrays of shape (batch_size, seq_len)
    """
    data = load_enwik8(data_path)

    if split == 'train':
        data = data[:90_000_000]
    elif split == 'val':
        data = data[90_000_000:95_000_000]
    else:
        data = data[95_000_000:]

    rng = np.random.RandomState(seed)

    while True:
        starts = rng.randint(0, len(data) - seq_len - 1, size=batch_size)
        batch = np.stack([data[s:s + seq_len + 1] for s in starts])
        inputs = jnp.array(batch[:, :-1], dtype=jnp.int32)
        targets = jnp.array(batch[:, 1:], dtype=jnp.int32)
        yield inputs, targets
