"""Train a small LM from scratch with optional nested sub-model evaluation.

Run:
    python train.py -config configs/matmla.yaml -log both -name matmla_test
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from data.wikitext import (
    load_wikitext,
    make_dataloaders,
    vocab_size_gpt2,
)
from loss import lm_loss
from model.factory import (
    SubModelSpec,
    build_model,
    enumerate_submodels,
    sample_submodel,
)
from utils.config import load_config
from utils.init import init_weights
from utils.logging import UnifiedLogger
from utils.schedule import CosineWarmupScheduler


# ---------------------------------------------------------------------------
# Device + seeding
# ---------------------------------------------------------------------------


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-config", type=str, required=True)
    p.add_argument("-log", type=str, default="both", choices=["wandb", "tb", "both", "none"])
    p.add_argument("-name", type=str, default=None)
    p.add_argument("-max_steps", type=int, default=None)
    p.add_argument("-seed", type=int, default=None)
    p.add_argument("-save_dir", type=str, default="save")
    p.add_argument("-eval_only", action="store_true")
    p.add_argument("-ckpt", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model,
    loader: Iterable[dict],
    device: torch.device,
    *,
    submodel: SubModelSpec,
    max_batches: int | None = None,
    eval_loss_chunk: int = 4,
) -> dict:
    """Evaluate `model` with the given `submodel` spec over `loader`.

    Cross-entropy is computed in chunks along the batch dimension to keep
    peak memory bounded by `eval_loss_chunk * seq_len * vocab_size`. With
    seq_len=512 and vocab=50257, fp32 logits are ~1 MB per (1, T), so a
    chunk of 4 keeps the log-softmax buffer under 1 GB.

    Returns {"val/loss": float, "val/perplexity": float, "val/tokens": int}.
    """
    model.eval()
    total_loss_sum = 0.0    # sum of per-token losses (so loss = sum / N)
    total_tokens = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        logits = model(
            x,
            active_q_heads=submodel.active_q_heads,
            active_intermediate=submodel.active_intermediate,
        )
        # Chunk over batch dim to avoid one giant log_softmax allocation.
        B = y.shape[0]
        chunk = max(1, min(int(eval_loss_chunk), B))
        loss_sum = 0.0
        n_tokens = 0
        for j in range(0, B, chunk):
            yc = y[j : j + chunk]
            lc = logits[j : j + chunk]
            loss_chunk = torch.nn.functional.cross_entropy(
                lc.reshape(-1, lc.size(-1)),
                yc.reshape(-1),
                reduction="sum",
            )
            loss_sum += float(loss_chunk.item())
            n_tokens += yc.numel()
        total_loss_sum += loss_sum
        total_tokens += n_tokens
    model.train()
    avg = total_loss_sum / max(1, total_tokens)
    return {"val/loss": avg, "val/perplexity": math.exp(avg), "val/tokens": total_tokens}


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.name:
        cfg["train"]["name"] = args.name
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = int(args.max_steps)
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
    cfg["data"]["vocab_size"] = vocab_size_gpt2()

    run_name = cfg.train.get("name") or Path(args.config).stem
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.train.get("seed"))
    device = pick_device()
    print(f"[setup] device={device}, run={run_name}, config={args.config}")

    # ---- Data ----
    print("[data] loading wikitext via HF datasets...")
    splits = load_wikitext(
        name=cfg.data.get("dataset", "wikitext-103-raw-v1"),
        cache_dir=cfg.data.get("cache_dir", "data/cache"),
    )
    print(
        f"[data] tokens: train={splits['train'].n_tokens:,}, "
        f"val={splits['validation'].n_tokens:,}, test={splits['test'].n_tokens:,}"
    )
    loaders = make_dataloaders(
        splits,
        seq_len=cfg.data.seq_len,
        batch_size=cfg.train.batch_size,
        eval_batch_size=cfg.train.get("eval_batch_size"),
        num_workers=cfg.train.get("num_workers", 0),
    )

    # ---- Model ----
    model = build_model(dict(cfg)).to(device)
    init_weights(model, n_layers=cfg.model.n_layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] variant={cfg.model.variant} params={n_params/1e6:.2f}M")

    # ---- Optimizer + schedule ----
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.train.get("weight_decay", 0.1),
    )
    sched = CosineWarmupScheduler(
        opt,
        warmup_steps=cfg.train.get("warmup_steps", 200),
        max_steps=cfg.train.max_steps,
        min_lr_ratio=cfg.train.get("min_lr_ratio", 0.1),
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(cfg.train.get("amp", False))
    )

    # ---- Logger ----
    logger = UnifiedLogger(
        args.log,
        run_name=run_name,
        config=dict(cfg),
        save_dir=args.save_dir,
    )

    # ---- Resume from checkpoint if provided, or auto-resume from final.pt in save_dir ----
    start_step = 0
    ckpt_path = Path(args.ckpt) if args.ckpt else (save_dir / "final.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt and not args.eval_only:
            opt.load_state_dict(ckpt["optimizer"])
        if "step" in ckpt:
            start_step = int(ckpt["step"])
        print(f"[ckpt] resumed from {ckpt_path} at step {start_step}")
    else:
        if args.ckpt:
            print(f"[ckpt] {args.ckpt} not found, starting from scratch")
        else:
            print(f"[ckpt] no final.pt in {save_dir}, starting from scratch")

    if args.eval_only:
        submodels = enumerate_submodels(dict(cfg))
        print(f"[eval] {len(submodels)} granularities")
        for spec in submodels:
            metrics = evaluate(
                model, loaders["validation"], device,
                submodel=spec, max_batches=cfg.train.get("eval_max_batches"),
            )
            tag = f"val/{spec.tag}"
            print(f"  {tag}: loss={metrics['val/loss']:.4f} ppl={metrics['val/perplexity']:.2f}")
            logger.log({f"{tag}/loss": metrics["val/loss"], f"{tag}/ppl": metrics["val/perplexity"]}, step=start_step)
        logger.close()
        return

    # ---- Train loop ----
    submodels = enumerate_submodels(dict(cfg))
    train_iter = iter(loaders["train"])
    model.train()
    log_interval = cfg.train.get("log_interval", 20)
    eval_interval = cfg.train.get("eval_interval", 500)
    grad_accum = max(1, int(cfg.train.get("grad_accum", 1)))
    grad_clip = cfg.train.get("grad_clip", 1.0)

    # Sampling RNG is seeded independently of torch/numpy so it doesn't
    # disturb the data-loader RNG state.
    sample_rng = random.Random(cfg.train.get("seed"))

    t0 = time.time()
    for step in range(start_step, cfg.train.max_steps):
        sched.step(step)

        # Sample ONE sub-model for this step. All microbatches of the step
        # use the same spec so the gradient direction is consistent.
        train_spec = sample_submodel(dict(cfg), step, rng=sample_rng)

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        n_tokens_total = 0
        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loaders["train"])
                batch = next(train_iter)
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=bool(cfg.train.get("amp", False))):
                logits = model(
                    x,
                    active_q_heads=train_spec.active_q_heads,
                    active_intermediate=train_spec.active_intermediate,
                )
                # Chunked cross-entropy to keep peak log_softmax buffer small.
                # Per microbatch: log_softmax buffer ~ chunk * T * V * 4 bytes.
                B = y.shape[0]
                chunk = max(1, int(cfg.train.get("loss_chunk", 4)))
                chunk = min(chunk, B)
                loss_sum = 0.0
                n_tokens = 0
                for j in range(0, B, chunk):
                    yc = y[j : j + chunk]
                    lc = logits[j : j + chunk]
                    loss_chunk = torch.nn.functional.cross_entropy(
                        lc.reshape(-1, lc.size(-1)),
                        yc.reshape(-1),
                        reduction="sum",
                    )
                    loss_sum = loss_sum + loss_chunk
                    n_tokens += yc.numel()
                loss = loss_sum / max(1, n_tokens) / grad_accum
            scaler.scale(loss).backward()
            loss_accum += float((loss_sum / max(1, n_tokens)).item())
            n_tokens_total += n_tokens
        if grad_clip and grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()

        if (step + 1) % log_interval == 0 or step == 0:
            elapsed = time.time() - t0
            metrics = {
                "train/loss": loss_accum,
                "train/lr": opt.param_groups[0]["lr"],
                "train/tokens_seen": (step + 1) * cfg.train.batch_size * grad_accum * cfg.data.seq_len,
                "train/sec_per_step": elapsed / max(1, step + 1 - start_step),
                "train/sampled_tag": train_spec.tag,
                "train/active_q_heads": float(train_spec.active_q_heads) if train_spec.active_q_heads is not None else -1.0,
                "train/active_intermediate": float(train_spec.active_intermediate) if train_spec.active_intermediate is not None else -1.0,
            }
            logger.log(metrics, step=step + 1)
            print(
                f"[step {step+1:6d}] loss={loss_accum:.4f} "
                f"lr={opt.param_groups[0]['lr']:.2e} "
                f"spec={train_spec.tag} "
                f"({elapsed:.0f}s, {metrics['train/sec_per_step']:.2f}s/step)"
            )

        if (step + 1) % eval_interval == 0:
            print(f"[eval @ step {step+1}]")
            eval_metrics = {}
            for spec in submodels:
                m = evaluate(
                    model, loaders["validation"], device,
                    submodel=spec, max_batches=cfg.train.get("eval_max_batches"),
                )
                tag = f"val/{spec.tag}"
                eval_metrics[f"{tag}/loss"] = m["val/loss"]
                eval_metrics[f"{tag}/ppl"] = m["val/perplexity"]
                print(
                    f"  {tag:>18s}: loss={m['val/loss']:.4f} ppl={m['val/perplexity']:.2f}"
                )
            logger.log(eval_metrics, step=step + 1)

    # ---- Save ----
    ckpt_path = save_dir / "final.pt"
    torch.save(
        {"model": model.state_dict(), "step": cfg.train.max_steps, "config": dict(cfg)},
        ckpt_path,
    )
    print(f"[done] saved checkpoint to {ckpt_path}")
    logger.close()


if __name__ == "__main__":
    main()
