"""Evaluate a trained checkpoint across all sub-model granularities.

Prints a per-sub-model table of val loss + perplexity, plus (for MatMLA) a
cache-size comparison table.

Run:
    python evaluate.py -config configs/matmla.yaml -ckpt save/matmla/final.pt
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from data.wikitext import (
    load_wikitext,
    make_dataloaders,
    vocab_size_gpt2,
)
from loss import lm_loss
from matmla_cache import cache_comparison_vs_naive
from model.factory import SubModelSpec, build_model, enumerate_submodels, submodel_kv_size_bytes
from utils.config import load_config
from train import evaluate, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-config", type=str, required=True)
    p.add_argument("-ckpt", type=str, required=True)
    p.add_argument("-split", type=str, default="validation", choices=["validation", "test"])
    p.add_argument("-max_batches", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["data"]["vocab_size"] = vocab_size_gpt2(cfg.data.get("max_vocab_size"))

    splits = load_wikitext(
        name=cfg.data.get("dataset", "wikitext-103-raw-v1"),
        cache_dir=cfg.data.get("cache_dir", "data/cache"),
        max_vocab_size=cfg.data.get("max_vocab_size"),
    )
    loaders = make_dataloaders(
        splits,
        seq_len=cfg.data.seq_len,
        batch_size=cfg.train.batch_size,
        eval_batch_size=cfg.train.get("eval_batch_size"),
    )
    device = pick_device()
    model = build_model(dict(cfg)).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    submodels = enumerate_submodels(dict(cfg))
    print(f"=== Evaluating {args.ckpt} on {args.split} ===")
    print(f"{'sub-model':>22s}  {'loss':>9s}  {'ppl':>9s}  {'kv_kb/1k_tokens':>17s}")
    for spec in submodels:
        m = evaluate(
            model, loaders[args.split], device,
            submodel=spec, max_batches=args.max_batches,
        )
        kv_bytes = submodel_kv_size_bytes(dict(cfg), spec)
        kv_kb_per_1k = kv_bytes * 1024 / 1024
        print(
            f"{spec.tag:>22s}  {m['val/loss']:9.4f}  "
            f"{m['val/perplexity']:9.2f}  {kv_kb_per_1k:17.2f}"
        )

    if cfg.model.variant == "matmla":
        print()
        print("=== MatMLA cache comparison (fp16, per-token, all layers) ===")
        for row in cache_comparison_vs_naive(dict(cfg)):
            print(
                f"  {row['scheme']:<28s} "
                f"per_layer={row['bytes_per_token_per_layer_fp16']:>5d} B  "
                f"total/1k tokens = {row['kb_per_1k_tokens_fp16_total']:>8.1f} KB"
            )


if __name__ == "__main__":
    main()
