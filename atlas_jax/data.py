"""Data loading. Two backends:

- enwik8: byte-level (vocab=256), first 100M - 5M = train, next 5M = val.
- climbmix: tiktoken-tokenized parquet shards with BOS-aligned best-fit packing.

Both yield `(inputs, targets)` as jnp.int32 arrays of shape `(B, T)`.
"""

import os

import numpy as np
import jax.numpy as jnp
import pyarrow.parquet as pq


# =============================================================================
# enwik8 (byte-level)
# =============================================================================

def load_enwik8_bytes(path: str):
    """Load enwik8 as a uint8 numpy array. Standard split: 90/5/5 (train/val/test)
    on the first 100M bytes."""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    n = 100_000_000
    data = data[:n]
    train = data[:90_000_000]
    val = data[90_000_000:95_000_000]
    test = data[95_000_000:]
    return train, val, test


def enwik8_loader(path: str, batch_size: int, seq_len: int, split: str = 'train',
                  seed: int = 0):
    """Infinite iterator of random contiguous (inputs, targets) from enwik8."""
    train, val, test = load_enwik8_bytes(path)
    data = {'train': train, 'val': val, 'test': test}[split]
    rng = np.random.default_rng(seed)
    N = len(data) - seq_len - 1
    while True:
        idx = rng.integers(0, N, size=batch_size)
        batch = np.stack([data[i:i + seq_len + 1] for i in idx]).astype(np.int32)
        yield jnp.asarray(batch[:, :-1]), jnp.asarray(batch[:, 1:])


# =============================================================================
# climbmix (parquet + tiktoken, BOS-aligned best-fit packing)
# =============================================================================

def _list_parquet_files(data_dir):
    files = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.parquet') and not f.endswith('.tmp'):
            files.append(os.path.join(data_dir, f))
    return files


def _document_batches(parquet_paths, tokenizer_batch_size=128, rank=0, world_size=1):
    """Infinite stream of document batches from parquet row-groups."""
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            rg_idx = rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size]
                rg_idx += world_size


def climbmix_loader(data_dir: str, tokenizer, batch_size: int, seq_len: int,
                    split: str = 'train', buffer_size: int = 1000,
                    rank: int = 0, world_size: int = 1):
    """BOS-aligned best-fit packing over tiktoken-tokenized parquet documents.

    Per row: pick the largest doc that fits, repeat, then crop a short doc to fill.
    Every row starts with BOS; 100% fill; ~35% token loss from end-of-row crops.
    """
    paths = _list_parquet_files(data_dir)
    assert paths, f"No parquet files in {data_dir}"
    paths = paths[:-1] if split == 'train' else paths[-1:]

    row_capacity = seq_len + 1
    bos = tokenizer.get_bos_token_id()
    doc_buffer = []
    batches = _document_batches(paths, rank=rank, world_size=world_size)

    def refill():
        docs = next(batches)
        for toks in tokenizer.encode(docs, prepend=bos):
            doc_buffer.append(toks)

    row_buffer = np.zeros((batch_size, row_capacity), dtype=np.int32)

    while True:
        for row_idx in range(batch_size):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill()
                remaining = row_capacity - pos

                # Largest doc that fits entirely.
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    n = len(doc)
                    if n <= remaining and n > best_len:
                        best_idx, best_len = i, n
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest.
                    s = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(s)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        yield (jnp.asarray(row_buffer[:, :-1], dtype=jnp.int32),
               jnp.asarray(row_buffer[:, 1:], dtype=jnp.int32))
