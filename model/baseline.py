"""Baseline MHA and GQA models.

These are non-slicing baselines. They accept the same `active_q_heads` /
`active_kv_heads` kwargs for forward compatibility but only the full values
are well-defined; the model factory ignores nested granularities for them
(via the per-config `granularities` setting).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import TransformerBlock, TransformerLM
from .common import (
    DynamicLinear,
    DynamicRMSNorm,
    RoPE,
    SwiGLU,
    apply_rope,
    causal_mask,
    repeat_kv,
)


@dataclass
class AttentionConfig:
    d_model: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    rope_base: float = 10000.0
    qk_norm: bool = False
    attn_dropout: float = 0.0


class _BaseAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = DynamicLinear(cfg.d_model, cfg.n_q_heads * cfg.head_dim, bias=False)
        self.k_proj = DynamicLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = DynamicLinear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = DynamicLinear(cfg.n_q_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.rope = RoPE(cfg.head_dim, base=cfg.rope_base)
        self.qk_norm = cfg.qk_norm
        self.attn_dropout = cfg.attn_dropout
        if cfg.qk_norm:
            self.q_norm = DynamicRMSNorm(cfg.head_dim)
            self.k_norm = DynamicRMSNorm(cfg.head_dim)
        # For per-head repeat factor (1 if MHA, n_q/n_kv if GQA).
        self.n_rep = cfg.n_q_heads // cfg.n_kv_heads

    def _qkv(self, x: torch.Tensor):
        q = self.q_proj(x).view(*x.shape[:-1], self.cfg.n_q_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(*x.shape[:-1], self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(*x.shape[:-1], self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        active_q_heads: Optional[int] = None,
        active_kv_heads: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.cfg
        n_q = active_q_heads or cfg.n_q_heads
        n_kv = active_kv_heads or cfg.n_kv_heads
        assert n_kv == cfg.n_kv_heads, f"{type(self).__name__} does not support KV slicing"
        assert n_q == cfg.n_q_heads, f"{type(self).__name__} does not support Q slicing"

        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, cfg.n_q_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope.get(t, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        if self.n_rep > 1:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, cfg.n_q_heads * cfg.head_dim)
        return self.o_proj(out)


class MHAAttention(_BaseAttention):
    """Standard MHA: n_kv_heads == n_q_heads."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, rope_base: float = 10000.0, qk_norm: bool = False, attn_dropout: float = 0.0):
        cfg = AttentionConfig(d_model=d_model, n_q_heads=n_heads, n_kv_heads=n_heads, head_dim=head_dim, rope_base=rope_base, qk_norm=qk_norm, attn_dropout=attn_dropout)
        super().__init__(cfg)


class GQAAttention(_BaseAttention):
    """Standard GQA: n_kv_heads < n_q_heads, fixed."""

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int, head_dim: int, rope_base: float = 10000.0, qk_norm: bool = False, attn_dropout: float = 0.0):
        assert n_q_heads % n_kv_heads == 0
        cfg = AttentionConfig(
            d_model=d_model, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads, head_dim=head_dim, rope_base=rope_base, qk_norm=qk_norm, attn_dropout=attn_dropout,
        )
        super().__init__(cfg)


# ----------------------------------------------------------------------------
# LM factories
# ----------------------------------------------------------------------------


def build_baseline_mha(
    *,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    head_dim: int,
    n_layers: int,
    d_intermediate: int,
    rope_base: float = 10000.0,
    qk_norm: bool = False,
    tie_embeddings: bool = True,
    dropout: float = 0.0,
) -> TransformerLM:
    def make_block(_idx: int) -> TransformerBlock:
        attn = MHAAttention(d_model, n_heads, head_dim, rope_base=rope_base, qk_norm=qk_norm)
        ffn = SwiGLU(d_model, d_intermediate)
        block = TransformerBlock(d_model, attn, ffn, residual_dropout=dropout)

        # Wrap block.forward to swallow sub-model kwargs (baseline ignores them).
        original_forward = block.forward

        def shim(x: torch.Tensor, **kwargs):
            return original_forward(x)

        block.forward = shim  # type: ignore[assignment]
        return block
    return TransformerLM(vocab_size, d_model, n_layers, make_block, tie_embeddings=tie_embeddings)


def build_baseline_gqa(
    *,
    vocab_size: int,
    d_model: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    d_intermediate: int,
    rope_base: float = 10000.0,
    qk_norm: bool = False,
    tie_embeddings: bool = True,
    dropout: float = 0.0,
) -> TransformerLM:
    def make_block(_idx: int) -> TransformerBlock:
        attn = GQAAttention(d_model, n_q_heads, n_kv_heads, head_dim, rope_base=rope_base, qk_norm=qk_norm)
        ffn = SwiGLU(d_model, d_intermediate)
        block = TransformerBlock(d_model, attn, ffn, residual_dropout=dropout)

        original_forward = block.forward

        def shim(x: torch.Tensor, **kwargs):
            return original_forward(x)

        block.forward = shim  # type: ignore[assignment]
        return block
    return TransformerLM(vocab_size, d_model, n_layers, make_block, tie_embeddings=tie_embeddings)
