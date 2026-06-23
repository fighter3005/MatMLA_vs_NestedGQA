"""Shared transformer backbone (decoder block + LM model wrapper).

Used by all four model variants. The attention block is provided externally
so this stays neutral between MHA / GQA / MatMLA.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DynamicEmbedding, DynamicRMSNorm, RoPE, SwiGLU


class TransformerBlock(nn.Module):
    """Pre-LN decoder block: x + attn(rmsnorm(x)); x + ffn(rmsnorm(x))."""

    def __init__(
        self,
        d_model: int,
        attn: nn.Module,
        ffn: nn.Module,
    ):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.norm1 = DynamicRMSNorm(d_model)
        self.norm2 = DynamicRMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_kwargs: Optional[dict] = None,
        ffn_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        attn_kwargs = attn_kwargs or {}
        ffn_kwargs = ffn_kwargs or {}
        x = x + self.attn(self.norm1(x), **attn_kwargs)
        x = x + self.ffn(self.norm2(x), **ffn_kwargs)
        return x


class TransformerLM(nn.Module):
    """Standard transformer LM wrapper.

    The attention block is injected (MHA / GQA / NestedGQA / MatMLA). The
    residual stream `d_model` is always full. Sub-model selection is delegated
    to the attention (and optionally the FFN) via `block_kwargs`.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        block_factory: Callable[[int], TransformerBlock],
        tie_embeddings: bool = True,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings
        self.embed = DynamicEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList([block_factory(i) for i in range(n_layers)])
        self.norm_f = DynamicRMSNorm(d_model)
        if tie_embeddings:
            self.lm_head = None  # tied at forward time
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed_scale = math.sqrt(d_model)

    def forward(self, input_ids: torch.Tensor, **block_kwargs) -> torch.Tensor:
        x = self.embed(input_ids) * self.embed_scale
        for layer in self.layers:
            x = layer(x, **block_kwargs)
        x = self.norm_f(x)
        if self.lm_head is None:
            logits = F.linear(x, self.embed.weight)
        else:
            logits = self.lm_head(x)
        return logits
