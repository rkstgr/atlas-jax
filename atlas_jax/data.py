"""Data loading: parquet reading + BOS-aligned best-fit packing in pure NumPy.

Reimplements nanochat's BOS-aligned dataloader without any PyTorch dependency.
Outputs jnp.array batches ready for training.

Algorithm for each row:
1. From buffered docs, pick the LARGEST doc that fits entirely
2. Repeat until no doc fits
3. When nothing fits, crop a doc to fill remaining space exactly

Key properties:
- Every row starts with BOS
- 100% utilization (no padding)
- ~35% of tokens discarded due to cropping
"""

import numpy as np
import jax.numpy as jnp
import pyarrow.parquet as pq

from atlas_jax.tokenizer import get_tokenizer


def list_parquet_files(data_dir):
    """List all .parquet files in data_dir, sorted by name."""
    import os
    files = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.parquet') and not f.endswith('.tmp'):
            files.append(os.path.join(data_dir, f))
    return files


def _document_batches(parquet_paths, tokenizer_batch_size=128, rank=0, world_size=1):
    """Infinite iterator over document batches from parquet files.

    Yields (text_batch, file_idx) tuples.
    Handles multi-worker sharding via rank/world_size.
    """
    epoch = 0
    while True:
        for pq_idx, filepath in enumerate(parquet_paths):
            pf = pq.ParquetFile(filepath)
            rg_idx = rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size], pq_idx
                rg_idx += world_size
        epoch += 1


def data_loader(
    data_dir: str,
    tokenizer,
    batch_size: int,
    seq_len: int,
    split: str = 'train',
    buffer_size: int = 1000,
    rank: int = 0,
    world_size: int = 1,
):
    """BOS-aligned best-fit packing data loader.

    Yields (inputs, targets) as jnp.int32 arrays of shape (B, T).
    """
    parquet_paths = list_parquet_files(data_dir)
    assert len(parquet_paths) > 0, f"No parquet files found in {data_dir}"

    if split == 'train':
        parquet_paths = parquet_paths[:-1]
    else:
        parquet_paths = parquet_paths[-1:]

    row_capacity = seq_len + 1
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    batches = _document_batches(parquet_paths, rank=rank, world_size=world_size)

    def refill_buffer():
        doc_batch, _ = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    row_buffer = np.zeros((batch_size, row_capacity), dtype=np.int32)

    while True:
        for row_idx in range(batch_size):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = doc
                    pos += doc_len
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        inputs = jnp.array(row_buffer[:, :-1], dtype=jnp.int32)
        targets = jnp.array(row_buffer[:, 1:], dtype=jnp.int32)
        yield inputs, targets
