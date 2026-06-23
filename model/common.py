"""Common building blocks: DynamicLinear, RMSNorm, RoPE, SwiGLU, CausalMask.

Slicing convention (matches hydralvlm's `modeling_common.DynamicLinear`):
  - `DynamicLinear.forward_sliced(x, out_rows=k, in_cols=j)` returns
        F.linear(x[..., :j], self.weight[:k, :j], bias[:k] if any)
  - `DynamicRMSNorm.forward(x, in_cols=j)` slices both input and weight to `j`.
  - `DynamicEmbedding.forward(input_ids, out_cols=j)` slices embedding output to `j`.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DynamicLinear (slicing wrapper around nn.Linear)
# ---------------------------------------------------------------------------


def linear_sliced(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_rows: Optional[int] = None,
    in_cols: Optional[int] = None,
) -> torch.Tensor:
    if in_cols is not None and input.size(-1) > in_cols:
        input = input[..., :in_cols]
    if out_rows is not None:
        weight = weight[:out_rows, :]
        if bias is not None:
            bias = bias[:out_rows]
    if in_cols is not None:
        weight = weight[:, :in_cols]
    return F.linear(input, weight, bias)


class DynamicLinear(nn.Linear):
    """An nn.Linear that supports prefix-sliced inference."""

    def forward_sliced(
        self,
        input: torch.Tensor,
        out_rows: Optional[int] = None,
        in_cols: Optional[int] = None,
    ) -> torch.Tensor:
        return linear_sliced(input, self.weight, self.bias, out_rows, in_cols)


# ---------------------------------------------------------------------------
# DynamicRMSNorm
# ---------------------------------------------------------------------------


class DynamicRMSNorm(nn.Module):
    """RMSNorm that can operate on sliced tensors.

    Mirrors Llama-style RMSNorm: output = weight * (x / rms(x)).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor, in_cols: Optional[int] = None) -> torch.Tensor:
        if in_cols is None:
            in_cols = x.size(-1)
        x_sliced = x[..., :in_cols] if x.size(-1) > in_cols else x
        w = self.weight[:in_cols]
        in_dtype = x_sliced.dtype
        out = self._norm(x_sliced.float()).to(in_dtype)
        return out * w


# ---------------------------------------------------------------------------
# DynamicEmbedding
# ---------------------------------------------------------------------------


class DynamicEmbedding(nn.Embedding):
    """Embedding whose output can be sliced to a prefix of hidden channels."""

    def forward(self, input_ids: torch.Tensor, out_cols: Optional[int] = None) -> torch.Tensor:
        emb = F.embedding(
            input_ids, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )
        if out_cols is not None and emb.size(-1) > out_cols:
            emb = emb[..., :out_cols].contiguous()
        return emb


# ---------------------------------------------------------------------------
# Rotary positional embeddings
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE. q/k: [B, H, T, D]. cos/sin: [B, T, D]."""
    cos = cos.unsqueeze(1)  # broadcast over heads
    sin = sin.unsqueeze(1)
    head_dim = q.size(-1)
    cos = cos[..., :head_dim]
    sin = sin[..., :head_dim]
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


class RoPE(nn.Module):
    """Rotary positional embedding generator."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.base = base
        self._cached: dict[Tuple[torch.device, torch.dtype, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype, seq_len)
        if key in self._cached:
            return self._cached[key]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)[None, :, :]  # [1, T, head_dim]
        sin = emb.sin().to(dtype)[None, :, :]
        self._cached[key] = (cos, sin)
        return cos, sin


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------


def causal_mask(T: int, device: torch.device, dtype: torch.dtype = torch.bool) -> torch.Tensor:
    """Return [T, T] bool mask: True where positions should be masked out."""
    return torch.triu(torch.ones(T, T, dtype=dtype, device=device), diagonal=1)


# ---------------------------------------------------------------------------
# SwiGLU MLP (Llama-style)
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU FFN block with optional prefix slicing of the intermediate dim.

    When `ffn_ratio=1.0` the block is the standard Llama MLP. When `ffn_ratio<1.0`
    the intermediate size is reduced (prefix-sliced), as in MatFormer's FFN nesting.
    """

    def __init__(
        self,
        d_model: int,
        d_intermediate: int,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.gate_proj = DynamicLinear(d_model, d_intermediate, bias=bias)
        self.up_proj = DynamicLinear(d_model, d_intermediate, bias=bias)
        self.down_proj = DynamicLinear(d_intermediate, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        active_intermediate: Optional[int] = None,
        active_embed: Optional[int] = None,
    ) -> torch.Tensor:
        ai = active_intermediate if active_intermediate is not None else self.d_intermediate
        ae = active_embed if active_embed is not None else self.d_model
        gate = self.gate_proj.forward_sliced(x, out_rows=ai, in_cols=ae)
        up = self.up_proj.forward_sliced(x, out_rows=ai, in_cols=ae)
        h = F.silu(gate) * up
        return self.down_proj.forward_sliced(h, out_rows=ae, in_cols=ai)


# ---------------------------------------------------------------------------
# repeat_kv (for GQA-style attention)
# ---------------------------------------------------------------------------


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, H_kv, T, D] -> [B, H_kv*n_rep, T, D] via repeat_interleave."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


# ---------------------------------------------------------------------------
# Submodule validation helper
# ---------------------------------------------------------------------------


def resolve_submodel_dim(
    full_dim: int,
    active_dim: Optional[int],
    *,
    divisors: Optional[list[int]] = None,
    name: str = "dim",
) -> int:
    """Resolve and validate an active sub-model dimension.

    - `None` → return full_dim.
    - Otherwise clamp to [1, full_dim].
    - If `divisors` is provided, also assert `active_dim in divisors`.
    """
    if active_dim is None:
        return full_dim
    a = max(1, min(int(active_dim), int(full_dim)))
    if divisors is not None and a not in divisors:
        raise ValueError(
            f"{name}={a} not in allowed divisors {divisors} of full {full_dim}"
        )
    return a
