"""Nested GQA + per-head MoE on the Q and O projections.

This is the hybrid discussed in the design notes: it keeps the GQA backbone
(``n_q`` query heads sharing ``n_kv`` KV groups) and NestedGQA's prefix
head-slicing, but replaces the single dense ``q_proj`` / ``o_proj`` with a
**per-head Mixture-of-Experts**:

  - Each query head owns ``n_experts`` Q expert matrices and a router. Per
    token, the router picks ``top-k`` experts and their gate-weighted average
    forms that head's effective Q (``W_eff = sum_e gate_e * W_e``).
  - The output projection O is likewise per-head MoE, routed from the same
    block input (decoupled router, SwitchHead-style).
  - K and V stay dense, one matrix per KV group, broadcast to the active
    heads via tiling. The KV cache is therefore identical for every sub-model
    (unchanged from NestedGQA).

Two orthogonal sub-model axes:

  - ``active_q_heads``  : prefix of heads (uniform per KV group), like NestedGQA.
  - ``active_q_topk``   : how many experts each active head routes to.

FFN ``active_intermediate`` slicing is inherited unchanged.

Head layout is **group-major**: head ``i`` belongs to KV group ``i % n_kv``.
A prefix ``[:active_q]`` with ``active_q = m * n_kv`` therefore takes the first
``m`` heads of *every* KV group, and the K/V tile factor is ``m``.

Memory note: all expert weights are resident regardless of the active sub-model.
Slicing/top-k save *compute* (and activations), not weight memory. Weight memory
only shrinks at export time, where experts can be pruned per head (see
``extract_submodel_state``).

Compute note: the expert combine here is implemented densely (all experts are
evaluated, then masked by a sparse top-k gate). This is numerically identical
to a sparse router and is portable to CPU/MPS/CUDA. True FLOP savings from
top-k require a grouped-GEMM (the original repo's ``cvmm`` CUDA kernel); the
hook for that would replace ``_moe_q`` / ``_moe_o`` below.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import TransformerBlock, TransformerLM
from .common import (
    DynamicLinear,
    RoPE,
    SwiGLU,
    apply_rope,
    resolve_submodel_dim,
)


class NestedGQAMoEAttention(nn.Module):
    """GQA attention with prefix-sliceable heads and per-head Q/O MoE."""

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        head_dim: int,
        n_experts: int,
        moe_k: int = 2,
        selection_mode: str = "softmax",
        rope_base: float = 10000.0,
        attn_dropout: float = 0.0,
        aux_loss_weight: float = 1e-2,
    ):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads"
        assert selection_mode in ("softmax", "sigmoid")
        assert 1 <= moe_k <= n_experts
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_experts = n_experts
        self.moe_k = moe_k
        self.selection_mode = selection_mode
        self.attn_dropout = attn_dropout
        self.aux_loss_weight = aux_loss_weight

        # K / V: dense, one matrix per KV group (never sliced; cache stays full).
        self.k_proj = DynamicLinear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = DynamicLinear(d_model, n_kv_heads * head_dim, bias=False)

        # Q experts:   (n_q, E, head_dim, d_model)   out_p = sum_d W[h,e,p,d] x[d]
        # O experts:   (n_q, E, d_model, head_dim)   out_D = sum_p W[h,e,D,p] a[p]
        self.q_experts = nn.Parameter(torch.empty(n_q_heads, n_experts, head_dim, d_model))
        self.o_experts = nn.Parameter(torch.empty(n_q_heads, n_experts, d_model, head_dim))
        # Routers: per head, E logits from the block input.
        self.q_router = nn.Parameter(torch.empty(n_q_heads, n_experts, d_model))
        self.o_router = nn.Parameter(torch.empty(n_q_heads, n_experts, d_model))

        self.rope = RoPE(head_dim, base=rope_base)

        # Aux (load-balancing) loss from the most recent forward, or None.
        self._last_aux: Optional[torch.Tensor] = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Expert matrices: fan-in scaled normal (init_weights does not reach
        # raw nn.Parameters, so initialise them here).
        q_std = 1.0 / math.sqrt(self.d_model)
        o_std = 1.0 / math.sqrt(self.head_dim)
        with torch.no_grad():
            nn.init.normal_(self.q_experts, std=q_std)
            nn.init.normal_(self.o_experts, std=o_std)
            nn.init.normal_(self.q_router, std=q_std)
            nn.init.normal_(self.o_router, std=o_std)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, x: torch.Tensor, router: torch.Tensor, k: int):
        """Return (gate, logits, top_idx).

        gate: (B, T, H, E) sparse — gate weights on the top-k experts, 0 else.
        logits: (B, T, H, E) raw router logits (for aux loss).
        top_idx: (B, T, H, k) selected expert indices (for aux loss).
        """
        logits = torch.einsum("btd,hed->bthe", x, router)  # (B,T,H,E)
        top_val, top_idx = logits.topk(k, dim=-1)           # (B,T,H,k)
        if self.selection_mode == "softmax":
            w = top_val.softmax(dim=-1)
        else:  # sigmoid
            w = top_val.sigmoid()
        gate = torch.zeros_like(logits).scatter(-1, top_idx, w)
        return gate, logits, top_idx

    def _aux_loss(self, logits: torch.Tensor, top_idx: torch.Tensor, k: int) -> torch.Tensor:
        """Switch-style load-balancing loss for one router (averaged over heads).

        loss = E * mean_h sum_e f_{h,e} * P_{h,e}
          f = fraction of (token) assignments to expert e
          P = mean softmax probability of expert e
        Minimised when load is uniform.
        """
        E = logits.shape[-1]
        probs = logits.softmax(dim=-1)                       # (B,T,H,E)
        P = probs.mean(dim=(0, 1))                            # (H,E)
        counts = F.one_hot(top_idx, E).sum(dim=-2).float()   # (B,T,H,E) in [0,k]
        f = counts.mean(dim=(0, 1)) / k                      # (H,E)
        return (P * f).sum(dim=-1).mean() * E

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        active_q_heads: Optional[int] = None,
        active_q_topk: Optional[int] = None,
    ) -> torch.Tensor:
        n_q = resolve_submodel_dim(self.n_q_heads, active_q_heads, name="active_q_heads")
        n_kv = self.n_kv_heads
        if n_q % n_kv != 0:
            raise ValueError(
                f"active_q_heads={n_q} not divisible by n_kv_heads={n_kv}; "
                "uniform per-KV-group slicing requires this"
            )
        m = n_q // n_kv  # heads per group == KV tile factor
        k = self.moe_k if active_q_topk is None else int(active_q_topk)
        k = max(1, min(k, self.n_experts))

        b, t, _ = x.shape
        p = self.head_dim

        # ---- K / V: dense per group ----
        k_ = self.k_proj(x).view(b, t, n_kv, p).transpose(1, 2)  # (B,n_kv,T,p)
        v_ = self.v_proj(x).view(b, t, n_kv, p).transpose(1, 2)

        # ---- Q: per-head MoE ----
        q_gate, q_logits, q_idx = self._route(x, self.q_router[:n_q], k)
        # all experts: (B,T,n_q,E,p); combine by sparse gate -> (B,T,n_q,p)
        all_q = torch.einsum("btd,hepd->bthep", x, self.q_experts[:n_q])
        q = torch.einsum("bthep,bthe->bthp", all_q, q_gate.to(all_q.dtype))
        q = q.permute(0, 2, 1, 3)  # (B,n_q,T,p)

        # ---- RoPE ----
        cos, sin = self.rope.get(t, x.device, x.dtype)
        q, k_ = apply_rope(q, k_, cos, sin)

        # ---- Expand K/V to active heads (group-major tiling) ----
        # head i attends to group (i % n_kv); repeat tiles groups m times.
        k_e = k_.repeat(1, m, 1, 1)  # (B,n_q,T,p)
        v_e = v_.repeat(1, m, 1, 1)

        attn = F.scaled_dot_product_attention(
            q, k_e, v_e,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )  # (B,n_q,T,p)

        # ---- O: per-head MoE (routed from the block input x) ----
        o_gate, o_logits, o_idx = self._route(x, self.o_router[:n_q], k)
        attn_h = attn.permute(0, 2, 1, 3)  # (B,T,n_q,p)
        # fold gate into the head output, then contract heads+experts+p -> D
        gated = attn_h.unsqueeze(3) * o_gate.unsqueeze(-1).to(attn_h.dtype)  # (B,T,n_q,E,p)
        out = torch.einsum("bthep,hedp->btd", gated, self.o_experts[:n_q])    # (B,T,D)

        # ---- Aux load-balancing loss (only when training & weighted) ----
        if self.training and self.aux_loss_weight > 0:
            aux = self._aux_loss(q_logits, q_idx, k) + self._aux_loss(o_logits, o_idx, k)
            self._last_aux = self.aux_loss_weight * 0.5 * aux
        else:
            self._last_aux = None

        return out


class _NestedFFN(nn.Module):
    """Wraps a SwiGLU block to support per-call ``active_intermediate``."""

    def __init__(self, d_model: int, d_intermediate: int):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_intermediate)

    def forward(self, x: torch.Tensor, active_intermediate: Optional[int] = None) -> torch.Tensor:
        ai = resolve_submodel_dim(
            self.swiglu.d_intermediate, active_intermediate, name="active_intermediate"
        )
        return self.swiglu(x, active_intermediate=ai)


def build_gqa_moe_nested(
    *,
    vocab_size: int,
    d_model: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    d_intermediate: int,
    n_experts: int,
    moe_k: int = 2,
    selection_mode: str = "softmax",
    aux_loss_weight: float = 1e-2,
    rope_base: float = 10000.0,
    tie_embeddings: bool = True,
    dropout: float = 0.0,
) -> TransformerLM:
    def make_block(_idx: int) -> TransformerBlock:
        attn = NestedGQAMoEAttention(
            d_model, n_q_heads, n_kv_heads, head_dim,
            n_experts=n_experts, moe_k=moe_k, selection_mode=selection_mode,
            rope_base=rope_base, aux_loss_weight=aux_loss_weight,
        )
        ffn = _NestedFFN(d_model, d_intermediate)
        block = TransformerBlock(d_model, attn, ffn, residual_dropout=dropout)

        original_forward = block.forward

        def shim(x: torch.Tensor, **kwargs):
            attn_kwargs = {
                "active_q_heads": kwargs.get("active_q_heads"),
                "active_q_topk": kwargs.get("active_q_topk"),
            }
            ffn_kwargs = {"active_intermediate": kwargs.get("active_intermediate")}
            return original_forward(x, attn_kwargs=attn_kwargs, ffn_kwargs=ffn_kwargs)

        block.forward = shim  # type: ignore[assignment]
        return block

    return TransformerLM(vocab_size, d_model, n_layers, make_block, tie_embeddings=tie_embeddings)


# ---------------------------------------------------------------------------
# Aux-loss collection (used by the training loop)
# ---------------------------------------------------------------------------


def collect_moe_aux_loss(model: nn.Module) -> Optional[torch.Tensor]:
    """Sum the per-layer MoE load-balancing losses from the last forward.

    Returns ``None`` if the model has no MoE attention layers (or none produced
    an aux term this step), so callers can cheaply skip the addition.
    """
    total: Optional[torch.Tensor] = None
    for mod in model.modules():
        if isinstance(mod, NestedGQAMoEAttention) and mod._last_aux is not None:
            total = mod._last_aux if total is None else total + mod._last_aux
    return total


# ---------------------------------------------------------------------------
# Export: prune to a concrete sub-model
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_submodel_state(
    model: nn.Module,
    *,
    active_q_heads: Optional[int] = None,
    experts_per_head: Optional[int] = None,
) -> dict:
    """Build a pruned state_dict for a concrete sub-model.

    - ``active_q_heads``: keep the first ``active_q_heads`` heads (group-major
      prefix). K/V stay full.
    - ``experts_per_head``: keep, for each kept head, the ``experts_per_head``
      experts with the largest router-gate L2 norm (a cheap importance proxy).
      The router rows are pruned in lockstep. ``None`` keeps all experts.

    This is the memory-saving export path: the returned tensors are strictly
    smaller than the trained model's. (Re-loading requires constructing a
    matching-shape module; this helper just produces the tensors + the head /
    expert index maps needed to do so.)
    """
    state: dict = {"layers": [], "meta": {}}
    for mod in model.modules():
        if not isinstance(mod, NestedGQAMoEAttention):
            continue
        n_q = active_q_heads or mod.n_q_heads
        if n_q % mod.n_kv_heads != 0:
            raise ValueError(
                f"active_q_heads={n_q} not divisible by n_kv_heads={mod.n_kv_heads}"
            )
        qe = mod.q_experts[:n_q]   # (n_q, E, p, D)
        oe = mod.o_experts[:n_q]
        qr = mod.q_router[:n_q]    # (n_q, E, D)
        orr = mod.o_router[:n_q]

        if experts_per_head is not None and experts_per_head < mod.n_experts:
            ke = int(experts_per_head)
            # importance proxy: router row norm per (head, expert)
            imp = qr.norm(dim=-1) + orr.norm(dim=-1)         # (n_q, E)
            idx = imp.topk(ke, dim=-1).indices               # (n_q, ke)
            gather_e = lambda w, e_extra: torch.gather(
                w, 1, idx.view(n_q, ke, *([1] * e_extra)).expand(n_q, ke, *w.shape[2:])
            )
            qe = gather_e(qe, 2)
            oe = gather_e(oe, 2)
            qr = gather_e(qr, 1)
            orr = gather_e(orr, 1)
            expert_idx = idx
        else:
            expert_idx = (
                torch.arange(mod.n_experts, device=qe.device)
                .unsqueeze(0)
                .expand(n_q, mod.n_experts)
            )

        state["layers"].append(
            {
                "q_experts": qe.clone(),
                "o_experts": oe.clone(),
                "q_router": qr.clone(),
                "o_router": orr.clone(),
                "k_proj": mod.k_proj.weight.detach().clone(),
                "v_proj": mod.v_proj.weight.detach().clone(),
                "kept_heads": n_q,
                "expert_idx": expert_idx.clone(),
            }
        )
    state["meta"] = {
        "active_q_heads": active_q_heads,
        "experts_per_head": experts_per_head,
    }
    return state
