"""Model factory.

Maps a YAML `model.variant` key to the corresponding builder. Also exposes
helpers to enumerate sub-model granularities and to sample one for the
current training step.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch.nn as nn

from .baseline import build_baseline_gqa, build_baseline_mha
from .matmla import build_matmla
from .nested_gqa import build_nested_gqa


@dataclass
class SubModelSpec:
    active_q_heads: Optional[int] = None       # None -> full
    active_kv_heads: Optional[int] = None      # None -> full
    active_intermediate: Optional[int] = None  # None -> full

    @property
    def tag(self) -> str:
        q = "full" if self.active_q_heads is None else f"q{self.active_q_heads}"
        kv = "" if self.active_kv_heads is None else f"_kv{self.active_kv_heads}"
        ffn = "" if self.active_intermediate is None else f"_ffn{self.active_intermediate}"
        return f"{q}{kv}{ffn}"

    @property
    def is_full(self) -> bool:
        return (
            self.active_q_heads is None
            and self.active_kv_heads is None
            and self.active_intermediate is None
        )


def build_model(cfg: dict) -> nn.Module:
    """Build the model specified by `cfg.model`."""
    m = cfg["model"]
    variant = m["variant"]
    common = dict(
        vocab_size=cfg["data"]["vocab_size"],
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        d_intermediate=m["d_intermediate"],
        rope_base=m.get("rope_base", 10000.0),
        tie_embeddings=m.get("tie_embeddings", True),
        dropout=m.get("dropout", 0.0),
    )

    if variant == "baseline_mha":
        return build_baseline_mha(
            **common,
            n_heads=m["n_heads"],
            head_dim=m["head_dim"],
        )
    if variant == "baseline_gqa":
        return build_baseline_gqa(
            **common,
            n_q_heads=m["n_q_heads"],
            n_kv_heads=m["n_kv_heads"],
            head_dim=m["head_dim"],
        )
    if variant == "nested_gqa":
        return build_nested_gqa(
            **common,
            n_q_heads=m["n_q_heads"],
            n_kv_heads=m["n_kv_heads"],
            head_dim=m["head_dim"],
        )
    if variant == "matmla":
        # MatMLA is MHA-on-decompressed-side; n_kv_heads is structurally equal
        # to n_q_heads and is not passed.
        return build_matmla(
            **common,
            n_q_heads=m["n_q_heads"],
            head_dim=m["head_dim"],
            c_kv=m["c_kv"],
            c_q=m["c_q"],
            r_rope=m["r_rope"],
        )
    raise ValueError(f"Unknown model.variant: {variant!r}")


def enumerate_submodels(cfg: dict) -> List[SubModelSpec]:
    """Enumerate all sub-model granularities defined in the config.

    Returns a list that always includes the full sub-model last. Note that
    the FULL spec is included for evaluation sweeps; per-step sampling
    (`sample_submodel`) explicitly chooses among the non-full specs.
    """
    m = cfg["model"]
    gran = m.get("granularities", {})
    q_list: Sequence[int] = list(gran.get("q_heads", []))
    ffn_list: Sequence[Optional[int]] = list(gran.get("ffn_ratio", []))
    # Convert FFN ratios to actual intermediate sizes (prefix slices).
    d_inter = int(m["d_intermediate"])
    ffn_dims: List[Optional[int]] = []
    for r in ffn_list:
        if r is None:
            ffn_dims.append(None)
        else:
            ffn_dims.append(max(1, int(round(d_inter * float(r)))))

    specs: List[SubModelSpec] = []
    n_q_full = int(m.get("n_q_heads", m.get("n_heads", 0)))
    for q in q_list:
        for f in ffn_dims:
            if q is None or q == n_q_full:
                q_full = None
            else:
                q_full = int(q)
            specs.append(SubModelSpec(active_q_heads=q_full, active_intermediate=f))

    # Always include the full sub-model last.
    specs.append(SubModelSpec())
    return specs


def sample_submodel(cfg: dict, step: int, *, rng: Optional[random.Random] = None) -> SubModelSpec:
    """Pick one `SubModelSpec` to train on for the current step.

    Algorithm: with probability `train.sample.full_prob` we return the full
    sub-model; otherwise we pick uniformly at random from the cartesian
    product of the granularity lists (q_heads x ffn_ratio).

    This is regime B from the design discussion: each step trains one
    sub-model and that sub-model is used for every microbatch of the step.
    """
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    sample_cfg = train_cfg.get("sample", {}) if isinstance(train_cfg, dict) else {}
    full_prob = float(sample_cfg.get("full_prob", 0.25))
    full_prob = max(0.0, min(1.0, full_prob))

    rng = rng or random

    if rng.random() < full_prob:
        return SubModelSpec()

    non_full = [s for s in enumerate_submodels(cfg) if not s.is_full]
    if not non_full:
        # No non-full granularities configured (e.g., baseline_mha). Fall back
        # to the full sub-model so we always have something to train on.
        return SubModelSpec()
    return rng.choice(non_full)


def submodel_kv_size_bytes(cfg: dict, spec: SubModelSpec, dtype_bytes: int = 2) -> int:
    """Compute per-layer KV-cache byte size for a given sub-model.

    - For MatMLA: layout is `(c_kv + r_rope) * dtype_bytes` regardless of sub-model.
    - For GQA / MHA: layout is `n_kv_heads * head_dim * 2 (K+V) * dtype_bytes`
      per layer, with active_q_heads not affecting the cache.
    """
    m = cfg["model"]
    n_layers = int(m["n_layers"])
    variant = m["variant"]
    if variant == "matmla":
        per_layer = (int(m["c_kv"]) + int(m["r_rope"])) * dtype_bytes
    elif variant in ("nested_gqa", "baseline_gqa", "baseline_mha"):
        n_kv = int(m.get("n_kv_heads", m.get("n_heads", 0)))
        head_dim = int(m["head_dim"])
        per_layer = (n_kv * head_dim * 2) * dtype_bytes  # K + V
    else:
        raise ValueError(f"unknown variant {variant}")
    return per_layer * n_layers
