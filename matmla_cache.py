"""Minimal MatMLA KV cache for sub-model benchmarking.

The cache stores, per layer:
    C_KV: [B, T_past, c_kv]       compressed KV latent (constant per sub-model)
    K_R:  [B, T_past, r_rope]     RoPE-K (constant per sub-model)

Both arrays are *identical* for every nested sub-model — that is the entire
point of the MLA cache layout. Decoding a sub-model with `active_q_heads`
decompresses on-demand using a prefix slice of `W_UK`/`W_UV`/`W_UQ` and a
prefix-col slice of `W_O`.

This module exposes a small helper class plus a function that prints a
cache-size table for a given model config — handy in `evaluate.py` to make
the MLA benefit visible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from model.matmla import MatMLAAttention


@dataclass
class MatMLACache:
    """Per-layer compressed-KV cache state."""

    c_kv: Optional[torch.Tensor] = None   # [B, T, c_kv]
    k_rope: Optional[torch.Tensor] = None  # [B, T, r_rope]

    def seq_len(self) -> int:
        if self.c_kv is None:
            return 0
        return int(self.c_kv.size(1))

    def update(self, new_c_kv: torch.Tensor, new_k_rope: torch.Tensor) -> "MatMLACache":
        """Append new tokens to the cache (decode step)."""
        if self.c_kv is None:
            self.c_kv = new_c_kv
            self.k_rope = new_k_rope
        else:
            self.c_kv = torch.cat([self.c_kv, new_c_kv], dim=1)
            self.k_rope = torch.cat([self.k_rope, new_k_rope], dim=1)
        return self

    def bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Bytes per token for the cache at this layer.

        Default dtype is fp16 (2 bytes). Layout is identical for all
        sub-models since only the *decompression* is sliced, not the cache.
        """
        if self.c_kv is None:
            return 0
        c_kv_dim = self.c_kv.size(-1)
        r_rope = self.k_rope.size(-1) if self.k_rope is not None else 0
        return (c_kv_dim + r_rope) * dtype_bytes


def cache_table_for_config(cfg: dict) -> List[Dict[str, float]]:
    """Return a list of dicts describing per-layer cache sizes for the given config.

    Useful for printing a comparison table between sub-models.
    """
    c_kv = int(cfg["model"]["c_kv"])
    r_rope = int(cfg["model"]["r_rope"])
    n_layers = int(cfg["model"]["n_layers"])

    # MatMLA cache per token per layer = (c_kv + r_rope) * dtype_bytes.
    bytes_per_token_per_layer_fp16 = (c_kv + r_rope) * 2
    return [
        {
            "n_layers": n_layers,
            "c_kv": c_kv,
            "r_rope": r_rope,
            "bytes_per_token_per_layer_fp16": bytes_per_token_per_layer_fp16,
            "total_kb_per_1k_tokens_fp16": (bytes_per_token_per_layer_fp16 * n_layers * 1024) / 1024,
        }
    ]


def cache_comparison_vs_naive(cfg: dict) -> List[Dict[str, float]]:
    """Compare MatMLA cache to a "naive" full MHA-style per-head K/V cache.

    The MatMLA paper uses MHA-on-decompressed-side: after decompression the
    model has `n_q_heads` K and `n_q_heads` V heads, all distinct. So the
    fair "naive MHA" baseline is `n_q_heads * head_dim` for K and V per token
    per layer (NOT GQA: there is no GQA on the decompressed side).
    """
    m = cfg["model"]
    c_kv = int(m["c_kv"])
    r_rope = int(m["r_rope"])
    n_q_heads = int(m["n_q_heads"])
    head_dim = int(m["head_dim"])
    n_layers = int(m["n_layers"])

    matmla_per_layer = (c_kv + r_rope) * 2
    naive_per_layer = (n_q_heads * head_dim * 2) * 2  # K and V, fp16

    return [
        {
            "scheme": "MatMLA (compressed latent)",
            "bytes_per_token_per_layer_fp16": matmla_per_layer,
            "kb_per_1k_tokens_fp16_total": matmla_per_layer * n_layers * 1024 / 1024,
        },
        {
            "scheme": "Naive MHA (full per-head K+V)",
            "bytes_per_token_per_layer_fp16": naive_per_layer,
            "kb_per_1k_tokens_fp16_total": naive_per_layer * n_layers * 1024 / 1024,
        },
    ]


def cache_comparison_vs_gqa(cfg: dict, n_kv_heads: int) -> List[Dict[str, float]]:
    """Compare MatMLA cache to a GQA baseline with `n_kv_heads` KV heads.

    Use this when you also train a GQA baseline model and want a fair
    cache-size comparison. The GQA cache stores `n_kv_heads * head_dim` per
    token per layer (for K and V combined), which can be larger or smaller
    than MatMLA's `c_kv + r_rope` depending on the configuration.
    """
    m = cfg["model"]
    c_kv = int(m["c_kv"])
    r_rope = int(m["r_rope"])
    head_dim = int(m["head_dim"])
    n_layers = int(m["n_layers"])

    matmla_per_layer = (c_kv + r_rope) * 2
    gqa_per_layer = (n_kv_heads * head_dim * 2) * 2  # K and V, fp16

    return [
        {
            "scheme": "MatMLA (compressed latent)",
            "bytes_per_token_per_layer_fp16": matmla_per_layer,
            "kb_per_1k_tokens_fp16_total": matmla_per_layer * n_layers * 1024 / 1024,
        },
        {
            "scheme": f"GQA-{n_kv_heads} (full per-head K+V)",
            "bytes_per_token_per_layer_fp16": gqa_per_layer,
            "kb_per_1k_tokens_fp16_total": gqa_per_layer * n_layers * 1024 / 1024,
        },
    ]


# Back-compat: a couple of older callers expected `cache_comparison_vs_naive`.
# Keep the function name working but route to the MHA comparison.
def cache_comparison_vs_naive_mha(cfg: dict) -> List[Dict[str, float]]:
    return cache_comparison_vs_naive(cfg)


def make_dummy_cache(layer, batch_size: int, dtype: torch.dtype = torch.float16) -> MatMLACache:
    """Create a zero-filled cache matching the layer's compression shape."""
    device = next(layer.parameters()).device
    return MatMLACache(
        c_kv=torch.zeros(batch_size, 0, layer.c_kv, device=device, dtype=dtype),
        k_rope=torch.zeros(batch_size, 0, layer.r_rope, device=device, dtype=dtype),
    )
