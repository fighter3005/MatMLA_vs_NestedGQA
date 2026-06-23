"""Tiny YAML loader with dot-attribute access and CLI override merging."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge `override` into `base`. Lists are replaced, not appended."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class DotDict(dict):
    """Dict with attribute access for ergonomics."""

    def __getattr__(self, k):
        v = self[k]
        if isinstance(v, dict) and not isinstance(v, DotDict):
            v = DotDict(v)
            self[k] = v
        return v

    def __setattr__(self, k, v):
        self[k] = v


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> DotDict:
    """Load YAML config and apply `overrides` (dotted keys)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            _set_dotted(cfg, k, v)
    return DotDict(cfg)


def _set_dotted(d: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
