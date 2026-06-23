"""Cosine LR schedule with linear warmup."""
from __future__ import annotations

import math
from torch.optim import Optimizer


class CosineWarmupScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.max_steps = max(1, int(max_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, step: int) -> None:
        if step < self.warmup_steps:
            scale = (step + 1) / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            progress = min(max(progress, 0.0), 1.0)
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * scale
