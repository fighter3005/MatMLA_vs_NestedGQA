"""MatMLA: Multi-Head Latent Attention with nested decompression + nested FFN.

Architecture per layer (mirrors the MatMLA paper, Section 3.1):

  Inputs:
    h                 ∈ R^{d}                     (residual stream, never sliced)

  Compression (always full, regardless of sub-model):
    C_KV = h @ W_DKV    ∈ R^{c_kv}
    C_Q  = h @ W_DQ     ∈ R^{c_q}
    K_R  = RoPE(h @ W_KR) ∈ R^{r_rope}           (small, separate, full)

  Decompression (MHA-on-decompressed-side, prefix-sliced):
    K_full = C_KV @ W_UK     ∈ R^{n_q · d_h}
    V_full = C_KV @ W_UV     ∈ R^{n_q · d_h}
    Q_full = C_Q  @ W_UQ     ∈ R^{n_q · d_h}

    For a sub-model with active_q heads:
      K_i = K_full[:, :active_q·d_h]
      V_i = V_full[:, :active_q·d_h]
      Q_i = Q_full[:, :active_q·d_h]   (W_UQ also output-slice for symmetry)

  Attention:
    attn_out = SDPA(reshape(Q_i), reshape(K_i), reshape(V_i))   # plain MHA
    attn_out = concat(attn_out, K_R)                            # (Llama-style)
    attn_out = attn_out @ W_O[:, :active_q·d_h + r_rope]        # col prefix slice

  FFN (nested, same convention as MatFormer):
    h + FFN(rmsnorm(h), active_intermediate)

The cache stores `(C_KV, K_R)` per token. Layout is identical for every
sub-model, which is the entire point of MLA-style caching.

There is **no GQA at the decompressed side**: the paper has `n_kv_heads == n_q_heads`
after decompression (MHA-style), with the K and V dimension fully exposed via
the decompression. This module enforces that invariant internally.
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
    resolve_submodel_dim,
)


class MatMLAAttention(nn.Module):
    """MatMLA block with prefix-sliced decompression (MHA after decompression).

    All three decompression matrices (`W_UK`, `W_UV`, `W_UQ`) output
    `n_q_heads * head_dim` features. A sub-model simply uses the first
    `active_q_heads * head_dim` columns of each (output-side prefix slice).
    The output projection `W_O` has input dim `n_q_heads * head_dim + r_rope`
    and is col-sliced to `active_q_heads * head_dim + r_rope` for the
    sub-model.

    The cached state per token is `(C_KV, K_R)`. Both are computed at full
    width; their layout is identical for every sub-model.
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        head_dim: int,
        c_kv: int,
        c_q: int,
        r_rope: int,
        rope_base: float = 10000.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        assert r_rope % 2 == 0, "r_rope must be even"
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        # MHA-on-decompressed-side: KV head count equals Q head count.
        self.n_kv_heads = n_q_heads
        self.head_dim = head_dim
        self.c_kv = c_kv
        self.c_q = c_q
        self.r_rope = r_rope

        # Compression (always full; output dim is the cached state size).
        self.W_DKV = DynamicLinear(d_model, c_kv, bias=False)
        self.W_DQ = DynamicLinear(d_model, c_q, bias=False)
        self.W_KR = DynamicLinear(d_model, r_rope, bias=False)

        # Decompression matrices, all output n_q * head_dim.
        # K and V are at full output dim and sliced at reshape time.
        # Q is output-sliced via DynamicLinear to keep the slicing explicit
        # at the matmul level (matches the paper's `W_UQ[:, :d_i]` form).
        self._decomp_dim = n_q_heads * head_dim
        self.W_UK = DynamicLinear(c_kv, self._decomp_dim, bias=False)
        self.W_UV = DynamicLinear(c_kv, self._decomp_dim, bias=False)
        self.W_UQ = DynamicLinear(c_q, self._decomp_dim, bias=False)
        self.W_O = DynamicLinear(self._decomp_dim + r_rope, d_model, bias=False)

        # RoPE for the small K_R path (r_rope must be even).
        self.rope_kr = RoPE(r_rope, base=rope_base)
        self.attn_dropout = attn_dropout

    def forward(
        self,
        x: torch.Tensor,
        active_q_heads: Optional[int] = None,
        active_kv_heads: Optional[int] = None,
    ) -> torch.Tensor:
        # `active_kv_heads` accepted but ignored: with MHA-on-decompressed-side,
        # the KV head count is structurally equal to `n_q_heads`, and we slice
        # by reshaping K and V to `[B, n_q, T, head_dim]` then narrow to
        # `[:, :active_q]`.
        del active_kv_heads

        n_q = resolve_submodel_dim(self.n_q_heads, active_q_heads, name="active_q_heads")
        d_i = n_q * self.head_dim

        b, t, _ = x.shape
        # Compression (full; cached).
        c_kv = self.W_DKV(x)               # [B, T, c_kv]
        c_q = self.W_DQ(x)                 # [B, T, c_q]
        k_r_pre = self.W_KR(x)             # [B, T, r_rope]

        # RoPE on the small K_R path. Treat the r_rope vector as a single
        # "head" of dimension r_rope.
        cos_kr, sin_kr = self.rope_kr.get(t, x.device, x.dtype)
        cos_kr = cos_kr.expand(b, t, self.r_rope)
        sin_kr = sin_kr.expand(b, t, self.r_rope)
        k_r = apply_rope(
            k_r_pre.unsqueeze(1),          # [B, 1, T, r_rope]
            k_r_pre.unsqueeze(1),
            cos_kr,
            sin_kr,
        )[0].squeeze(1)                     # [B, T, r_rope]

        # Nested decompression: K, V are at full n_q*head_dim (we slice at
        # reshape time); Q is output-side sliced via DynamicLinear.
        q = self.W_UQ.forward_sliced(c_q, out_rows=d_i)   # [B, T, d_i]
        k = self.W_UK(c_kv)                                # [B, T, n_q*head_dim]
        v = self.W_UV(c_kv)                                # [B, T, n_q*head_dim]

        q = q.view(b, t, n_q, self.head_dim).transpose(1, 2)        # [B, n_q, T, head_dim]
        k = k.view(b, t, self.n_q_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_q_heads, self.head_dim).transpose(1, 2)
        # Slice K and V to the active head budget (MHA-style prefix).
        k = k[:, :n_q, :, :]
        v = v[:, :n_q, :, :]

        # Apply RoPE on Q,K (head_dim). Cache a tiny RoPE per (head_dim, base, T).
        cos_hd, sin_hd = _self_rope_cache(self.head_dim, self.rope_kr.base, t, x.device, x.dtype)
        q, k = apply_rope(q, k, cos_hd, sin_hd)

        # Plain MHA — no repeat_kv, since n_kv == n_q structurally.
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )                                  # [B, n_q, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(b, t, d_i)

        # Concat with k_r along the feature dim; W_O col-slices.
        out = torch.cat([out, k_r], dim=-1)               # [B, T, d_i + r_rope]
        out = self.W_O.forward_sliced(out, in_cols=d_i + self.r_rope)
        return out


# Tiny module-local cache for the head_dim RoPE.
_ROPE_HD_CACHE: dict = {}


def _self_rope_cache(head_dim: int, base: float, seq_len: int, device, dtype):
    key = (head_dim, base, seq_len, device, dtype)
    if key in _ROPE_HD_CACHE:
        return _ROPE_HD_CACHE[key]
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)[None, :, :]
    sin = emb.sin().to(dtype)[None, :, :]
    _ROPE_HD_CACHE[key] = (cos, sin)
    return cos, sin


class _NestedFFN(nn.Module):
    def __init__(self, d_model: int, d_intermediate: int):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_intermediate)

    def forward(self, x: torch.Tensor, active_intermediate: Optional[int] = None) -> torch.Tensor:
        ai = resolve_submodel_dim(self.swiglu.d_intermediate, active_intermediate, name="active_intermediate")
        return self.swiglu(x, active_intermediate=ai)


def build_matmla(
    *,
    vocab_size: int,
    d_model: int,
    n_q_heads: int,
    head_dim: int,
    n_layers: int,
    d_intermediate: int,
    c_kv: int,
    c_q: int,
    r_rope: int,
    rope_base: float = 10000.0,
    tie_embeddings: bool = True,
) -> TransformerLM:
    """Build a MatMLA LM.

    Note: `n_kv_heads` is intentionally absent. The paper has
    `n_kv_heads == n_q_heads` after decompression (MHA-style). To use a
    GQA variant of MLA (DeepSeek-V2 style), build a separate variant.
    """
    def make_block(_idx: int) -> TransformerBlock:
        attn = MatMLAAttention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            head_dim=head_dim,
            c_kv=c_kv,
            c_q=c_q,
            r_rope=r_rope,
            rope_base=rope_base,
        )
        ffn = _NestedFFN(d_model, d_intermediate)
        block = TransformerBlock(d_model, attn, ffn)

        original_forward = block.forward

        def shim(x: torch.Tensor, **kwargs):
            attn_kwargs = {"active_q_heads": kwargs.get("active_q_heads")}
            ffn_kwargs = {"active_intermediate": kwargs.get("active_intermediate")}
            return original_forward(x, attn_kwargs=attn_kwargs, ffn_kwargs=ffn_kwargs)

        block.forward = shim  # type: ignore[assignment]
        return block

    return TransformerLM(vocab_size, d_model, n_layers, make_block, tie_embeddings=tie_embeddings)
