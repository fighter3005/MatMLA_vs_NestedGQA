"""Cross-entropy loss for next-token prediction with optional masking."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token cross-entropy.

    logits: [B, T, V], labels: [B, T].
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
    )
