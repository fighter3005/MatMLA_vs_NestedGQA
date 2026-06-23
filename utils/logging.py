"""Unified logger supporting wandb, tensorboard, both, or none.

Backend is selected by a single string: "wandb", "tb", "both", "none".
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


class UnifiedLogger:
    """A small wrapper that fans out metric dicts to one or more backends."""

    def __init__(
        self,
        backend: str,
        *,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        save_dir: str = "save",
    ):
        self.backend = backend
        self.run_name = run_name or "run"
        self.save_dir = Path(save_dir) / self.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._tb = None
        self._wb = None
        self._step = 0

        if backend in ("tb", "both"):
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(self.save_dir / "tb"))
        if backend in ("wandb", "both"):
            import wandb

            self._wb = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "matmla-vs-nestedgqa"),
                name=self.run_name,
                config=config or {},
                dir=str(self.save_dir),
                resume="allow",
            )

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if step is not None:
            self._step = step
        else:
            self._step += 1
        step = self._step
        # Sanitize: drop non-scalars.
        cleaned = {}
        for k, v in metrics.items():
            try:
                cleaned[k] = float(v)
            except (TypeError, ValueError):
                continue
        if self._tb is not None:
            for k, v in cleaned.items():
                self._tb.add_scalar(k, v, step)
        if self._wb is not None:
            self._wb.log(cleaned, step=step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
        if self._wb is not None:
            import wandb

            wandb.finish()
