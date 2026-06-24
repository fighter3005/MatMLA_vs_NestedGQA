"""Nested GQA: GQA block with prefix-sliceable Q heads (uniform per KV group).

Slicing rules:
  - `active_q_heads` may be any divisor of `n_q_heads`.
  - `active_kv_heads` is fixed at the full `n_kv_heads`.
  - Active Q heads must split uniformly across KV groups, i.e. the per-group
    Q count `active_q_heads // n_kv_heads` is the same for every group.
    E.g. with 12 Q / 3 KV groups this admits {3, 6, 9, 12}.
  - FFN intermediate may be sliced via `active_intermediate` (any positive
    integer <= d_intermediate).
"""
from __future__ import annotations

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
    repeat_kv,
    resolve_submodel_dim,
)


class NestedGQAAttention(nn.Module):
    """GQA attention with prefix-sliceable Q heads (uniform per-KV-group).

    Internally the Q projection rows are laid out in **KV-group-major** order:
        [KV0 head0, KV1 head0, ..., KV_{n_kv-1} head0,
         KV0 head1, KV1 head1, ...,
         KV0 head_{q_per_kv-1}, ...]
    so that a prefix slice `out_rows=active_q*head_dim` corresponds to
    "take the first (active_q // n_kv_heads) heads from EACH KV group".
    The O projection columns are permuted in lockstep so the attention output
    is invariant to this reordering.
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads"
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_per_kv = n_q_heads // n_kv_heads

        self.q_proj = DynamicLinear(d_model, n_q_heads * head_dim, bias=False)
        self.k_proj = DynamicLinear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = DynamicLinear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = DynamicLinear(n_q_heads * head_dim, d_model, bias=False)
        self.rope = RoPE(head_dim, base=rope_base)
        self.attn_dropout = attn_dropout

        # KV-group-major reordering of Q rows + O cols. This is a fixed
        # permutation applied at init; it does not affect the model's
        # representational power, only the order of Q heads. After this
        # perm, `active_q_heads * head_dim` prefix slices are equivalent to
        # taking the first `active_q_heads // n_kv_heads` heads from each
        # KV group (uniform per-group slicing).
        self._apply_group_major_perm()

    def _apply_group_major_perm(self) -> None:
        q_per_kv = self.q_per_kv
        n_kv = self.n_kv_heads
        perm = []
        for i in range(q_per_kv):
            for g in range(n_kv):
                perm.append(g * q_per_kv + i)
        perm_t = torch.tensor(perm, dtype=torch.long)
        with torch.no_grad():
            w = self.q_proj.weight.data.view(self.n_q_heads, self.head_dim, -1)
            w = w.index_select(0, perm_t)
            self.q_proj.weight.data.copy_(w.reshape(self.n_q_heads * self.head_dim, -1))
            w = self.o_proj.weight.data.view(-1, self.n_q_heads, self.head_dim)
            w = w.index_select(1, perm_t)
            self.o_proj.weight.data.copy_(w.reshape(-1, self.n_q_heads * self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        active_q_heads: Optional[int] = None,
        active_kv_heads: Optional[int] = None,
    ) -> torch.Tensor:
        n_q = resolve_submodel_dim(self.n_q_heads, active_q_heads, name="active_q_heads")
        n_kv = self.n_kv_heads  # KV never sliced for NestedGQA
        if n_q % n_kv != 0:
            raise ValueError(
                f"active_q_heads={n_q} not divisible by n_kv_heads={n_kv}; "
                "uniform per-KV-group slicing requires this"
            )
        q_per_kv = n_q // n_kv

        b, t, _ = x.shape
        # Prefix-slice Q (group-major layout ensures uniform per-group selection).
        q = self.q_proj.forward_sliced(x, out_rows=n_q * self.head_dim)
        q = q.view(b, t, n_q, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, n_kv, self.head_dim).transpose(1, 2)

        cos, sin = self.rope.get(t, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        # Repeat each KV head q_per_kv times to match Q heads.
        k = repeat_kv(k, q_per_kv)
        v = repeat_kv(v, q_per_kv)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, n_q * self.head_dim)
        # o_proj slices its INPUT dim to match the active head budget.
        return self.o_proj.forward_sliced(out, in_cols=n_q * self.head_dim)


class _NestedFFN(nn.Module):
    """Wraps a SwiGLU block to support per-call `active_intermediate`."""

    def __init__(self, d_model: int, d_intermediate: int):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_intermediate)

    def forward(
        self,
        x: torch.Tensor,
        active_intermediate: Optional[int] = None,
    ) -> torch.Tensor:
        ai = resolve_submodel_dim(self.swiglu.d_intermediate, active_intermediate, name="active_intermediate")
        return self.swiglu(x, active_intermediate=ai)


def build_nested_gqa(
    *,
    vocab_size: int,
    d_model: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    d_intermediate: int,
    rope_base: float = 10000.0,
    tie_embeddings: bool = True,
    dropout: float = 0.0,
) -> TransformerLM:
    def make_block(_idx: int) -> TransformerBlock:
        attn = NestedGQAAttention(d_model, n_q_heads, n_kv_heads, head_dim, rope_base=rope_base)
        ffn = _NestedFFN(d_model, d_intermediate)

        block = TransformerBlock(d_model, attn, ffn, residual_dropout=dropout)

        # Wrap block.forward to translate top-level kwargs.
        original_forward = block.forward

        def shim(x: torch.Tensor, **kwargs):
            attn_kwargs = {"active_q_heads": kwargs.get("active_q_heads")}
            ffn_kwargs = {"active_intermediate": kwargs.get("active_intermediate")}
            return original_forward(x, attn_kwargs=attn_kwargs, ffn_kwargs=ffn_kwargs)

        block.forward = shim  # type: ignore[assignment]
        return block

    return TransformerLM(vocab_size, d_model, n_layers, make_block, tie_embeddings=tie_embeddings)
