"""Tokenizer: loads tiktoken encoding from nanochat's tokenizer.pkl directly.

Avoids importing nanochat (which pulls in torch, rustbpe, etc).
"""

import os
import pickle


class Tokenizer:
    """Thin wrapper around a tiktoken Encoding loaded from pickle."""

    def __init__(self, enc, bos_token_id):
        self.enc = enc
        self._bos_token_id = bos_token_id

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pkl_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        assert os.path.exists(pkl_path), f"tokenizer.pkl not found in {tokenizer_dir}"
        with open(pkl_path, "rb") as f:
            enc = pickle.load(f)
        bos_token_id = enc.encode_single_token("<|bos|>")
        return cls(enc, bos_token_id)

    def get_bos_token_id(self):
        return self._bos_token_id

    def get_vocab_size(self):
        return self.enc.n_vocab

    def encode(self, text, prepend=None, num_threads=8):
        """Encode text or list of texts. Returns list of token id lists."""
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            return ids
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
            return ids
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        return self.enc.decode(ids)


def get_tokenizer(tokenizer_dir):
    """Load tokenizer from directory containing tokenizer.pkl."""
    return Tokenizer.from_directory(tokenizer_dir)
