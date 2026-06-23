"""Trunc-Normal init scaled per Pre-LN convention."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def init_weights(module: nn.Module, *, n_layers: int, std: float = 0.02) -> None:
    """Initialize all parameters in the module.

    - Linears: trunc_normal_(weight), zero bias, scaled by sqrt(2 / n_layers).
    - Embeddings: trunc_normal_(weight).
    - RMSNorm: ones for weight, zeros for bias.
    """
    scale = math.sqrt(2.0 / n_layers)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
            m.weight.data.mul_(scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
        elif isinstance(m, nn.Parameter) and m.dim() > 1:
            nn.init.trunc_normal_(m, std=std, a=-2 * std, b=2 * std)
    for m in module.modules():
        if isinstance(m, _RMSNormBase):
            nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)


class _RMSNormBase(nn.Module):
    pass


# Patch: our RMSNorm doesn't subclass nn.LayerNorm, so we use isinstance check
# by name during iteration. Easier to just match by attribute.
def init_rmsnorm(module: nn.Module) -> None:
    for name, m in module.named_modules():
        if hasattr(m, "weight") and isinstance(m.weight, nn.Parameter) and m.__class__.__name__.endswith("RMSNorm"):
            nn.init.ones_(m.weight)
