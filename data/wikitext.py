"""Wikitext data loading via HuggingFace `datasets` + tiktoken GPT-2 BPE.

The original `attention_moe` repo's wikitext URL is no longer reachable.
We swap to `wikitext-103-raw-v1` (or `wikitext-2-raw-v1`) via HF, then
tokenize with the GPT-2 BPE via `tiktoken`. Pack into fixed-length sequences
for the LM objective.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# Lazy tokenizer init to avoid hard dependency at import time.
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        import tiktoken

        _tokenizer = tiktoken.get_encoding("gpt2")
        # Reserve 0 for padding if ever needed.
        _tokenizer._eot = _tokenizer.eot_token  # noqa: SLF001
    return _tokenizer


@dataclass
class TokenizedSplit:
    """Memmap-backed tokenized split."""

    path: Path
    n_tokens: int

    def array(self) -> np.ndarray:
        return np.memmap(self.path, dtype=np.int32, mode="r", shape=(self.n_tokens,))


def tokenize_and_pack(
    texts: List[str],
    cache_path: Path,
    eot_token: int,
) -> TokenizedSplit:
    """Tokenize a list of strings, concatenate, and dump as int32 memmap."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        # Trust the file; user can delete to rebuild.
        n = int(np.fromfile(cache_path, dtype=np.int64, count=1)[0]) if False else None
        # Compute n_tokens from file size instead.
        n_tokens = cache_path.stat().st_size // 4  # int32
        return TokenizedSplit(path=cache_path, n_tokens=n_tokens)

    enc = _get_tokenizer()
    all_ids: List[int] = []
    for t in texts:
        ids = enc.encode_ordinary(t)
        ids.append(eot_token)
        all_ids.extend(ids)
    arr = np.asarray(all_ids, dtype=np.int32)
    arr.tofile(cache_path)
    return TokenizedSplit(path=cache_path, n_tokens=len(arr))


def load_wikitext(
    *,
    name: str = "wikitext-103-raw-v1",
    cache_dir: str = "data/cache",
) -> dict:
    """Load wikitext-{2,103}-raw-v1 via `datasets` and tokenize into memmaps.

    Returns a dict of `TokenizedSplit`s keyed by split name.
    """
    from datasets import load_dataset

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset("Salesforce/wikitext", name, cache_dir=str(cache_dir / "hf"))
    enc = _get_tokenizer()
    eot = enc.eot_token

    splits = {}
    for split_name in ["train", "validation", "test"]:
        texts = raw[split_name]["text"]
        out_path = cache_dir / f"{name.replace('/', '_')}__{split_name}.bin"
        splits[split_name] = tokenize_and_pack(texts, out_path, eot)
    return splits


class PackedLMDataset(Dataset):
    """Fixed-length chunks drawn from a contiguous token memmap."""

    def __init__(self, split: TokenizedSplit, seq_len: int):
        self.split = split
        self.seq_len = seq_len
        self.n_chunks = max(0, (split.n_tokens - 1) // seq_len)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict:
        arr = self.split.array()
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = arr[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return {"input_ids": x, "labels": y}


def make_dataloaders(
    splits: dict,
    *,
    seq_len: int,
    batch_size: int,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
) -> dict:
    eval_batch_size = eval_batch_size or batch_size
    train = PackedLMDataset(splits["train"], seq_len)
    valid = PackedLMDataset(splits["validation"], seq_len)
    test = PackedLMDataset(splits["test"], seq_len)
    return {
        "train": DataLoader(
            train, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=False,
        ),
        "validation": DataLoader(
            valid, batch_size=eval_batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=False,
        ),
        "test": DataLoader(
            test, batch_size=eval_batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=False,
        ),
    }


def vocab_size_gpt2() -> int:
    """Return GPT-2 vocab size (50257)."""
    return 50257
