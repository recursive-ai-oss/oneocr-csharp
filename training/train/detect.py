"""Trains the DBNet-style detector + layout model (models/dbnet.py). Resumable, chunked, appends metrics, writes a run manifest.

Usage: uv run python -m train.detect configs/dbnet-lite-v0.yaml

Losses: DB (Liao et al. 2020): BCE with 3:1 hard-negative mining on the probability map, dice on the approximate binary map
sigmoid(50 (P - T)), L1 on the threshold map (weight 10). Layout: BCE + dice per class on the shrunk-region maps, only for the classes a
source labels. Orientation: cross entropy on the page rotation.

A run stops after `max_minutes` of training (a resumable chunk) and continues from ckpt-latest.pt the next time it is started.
Paths in the config may use environment variables (${ONEOCR_HOME}).
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from eval.detect import evaluate_records, to_input  # noqa: F401  (to_input kept for callers)
from models.dbnet import DBNetLite, count_macs, count_params
from train.detect_data import DetDataset, load_records

BASE = Path(__file__).resolve().parents[1]


def lr_at(step: int, tc: dict) -> float:
    if step < tc["warmup"]:
        return tc["lr"] * (step + 1) / tc["warmup"]
    p = (step - tc["warmup"]) / max(1, tc["steps"] - tc["warmup"])
    return tc["lr"] * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))


class StepSampler:
    """One batch of (record index, nonce) per step from a generator seeded by (seed, step): a resumed run sees the same batches."""

    def __init__(self, bounds: list[tuple[int, int]], weights: list[float], batch: int, seed: int, start: int, stop: int):
        self.bounds, self.weights, self.batch, self.seed, self.start, self.stop = bounds, weights, batch, seed, start, stop

    def __len__(self):
        return self.stop - self.start

    def __iter__(self):
        for step in range(self.start, self.stop):
            r = random.Random(self.seed * 1_000_003 + step)
            out = []
            for j in range(self.batch):
                lo, hi = r.choices(self.bounds, self.weights)[0]
                out.append((r.randrange(lo, hi), step * self.batch + j))
            yield out


def dice(p: torch.Tensor, g: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    p, g = p * m, g * m
    return 1 - (2 * (p * g).sum() + 1) / (p.sum() + g.sum() + 1)


def det_loss(out: dict, b: dict, lc: dict) -> dict:
    valid = b["text_valid"].view(-1, 1, 1, 1) * b["text_ok"]
    gt, thr_gt, thr_m = b["prob"], b["thr"], b["thr_mask"]
    logit = out["prob"].float()
    p = logit.sigmoid()
    # probability map: BCE with hard-negative mining over the samples that have line boxes
    bce = F.binary_cross_entropy_with_logits(logit, gt, reduction="none")
    pos = (gt > 0.5).float() * valid
    neg = (gt <= 0.5).float() * valid
    npos = int(pos.sum().item())
    nneg = min(int(neg.sum().item()), max(3 * npos, 64 if npos == 0 else 0))
    if nneg > 0:
        nl = (bce * neg).flatten()
        top = nl.topk(nneg).values.sum()
    else:
        top = bce.sum() * 0
    l_s = ((bce * pos * b["pos_w"]).sum() + top) / max(npos + nneg, 1)
    t = out["thresh"].float().sigmoid()
    l_b = dice(torch.sigmoid(50 * (p - t)), gt, valid.expand_as(gt))
    mm = thr_m * valid
    l_t = (torch.abs(t - thr_gt) * mm).sum() / mm.sum().clamp(min=1)
    # layout
    ll = out["layout"].float()
    cm = b["cmask"].view(*b["cmask"].shape, 1, 1)
    lbce = (
        F.binary_cross_entropy_with_logits(ll, b["layout"], pos_weight=torch.tensor(3.0, device=ll.device), reduction="none") * cm
    ).sum() / (cm.sum() * ll.shape[2] * ll.shape[3]).clamp(min=1)
    lp = ll.sigmoid()
    ldice = 0.0
    nc = 0
    for c in range(ll.shape[1]):
        m = cm[:, c : c + 1]
        if m.sum() > 0:
            ldice = ldice + dice(lp[:, c : c + 1], b["layout"][:, c : c + 1], m.expand_as(lp[:, c : c + 1]))
            nc += 1
    ldice = ldice / max(nc, 1)
    l_o = F.cross_entropy(out["orient"].float(), b["orient"])
    total = (
        lc["prob"] * l_s
        + lc["binary"] * l_b
        + lc["thresh"] * l_t
        + lc["layout_bce"] * lbce
        + lc["layout_dice"] * ldice
        + lc["orient"] * l_o
    )
    return {"total": total, "prob": l_s, "binary": l_b, "thresh": l_t, "layout_bce": lbce, "layout_dice": ldice, "orient": l_o}


def main() -> None:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cfg = json.loads(os.path.expandvars(json.dumps(cfg)))
    tc, lc = cfg["train"], cfg["loss"]
    seed = cfg["seed"]
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = os.environ.get("ONEOCR_DEVICE", "cuda")  # "cpu" only for smoke runs
    torch.manual_seed(seed)
    model = DBNetLite(cfg.get("model")).to(dev).to(memory_format=torch.channels_last)

    root = Path(cfg["datasets_root"])
    train_recs, bounds, weights, prov = [], [], [], {}
    for src in cfg["sources"]:
        recs = load_records(src["name"], root, "train", src)
        bounds.append((len(train_recs), len(train_recs) + len(recs)))
        weights.append(src["weight"])
        train_recs += recs
        prov[src["name"]] = {
            "images": len(recs),
            "with_line_boxes": sum(r["lines"] is not None for r in recs),
            "label_source": src["label_source"],
        }
    val_sets = {}
    for src in cfg["sources"]:
        v = load_records(src["name"], root, "val", src)
        val_sets[src["name"]] = v[:: max(1, len(v) // tc["val_pages"])][: tc["val_pages"]]
    ds = DetDataset(
        train_recs, cfg["size"], True, seed, tuple(cfg["scale"]), cfg["rot_p"], cfg.get("min_core", 0.0), cfg.get("small_weight", 1.0)
    )

    macs = count_macs(model, 1024, 736)["total"]
    manifest = {
        "config": cfg, "seed": seed, "git_commit": os.environ.get("ONEOCR_GIT_COMMIT", "unknown"),
        "data": prov, "parameters_inference": count_params(model), "macs_a4_1024x736": macs,
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }  # fmt: skip
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    step, best = 0, -1.0
    ckpt_path = run_dir / "ckpt-latest.pt"
    if not ckpt_path.exists() and cfg.get("init_from"):  # continue from our own earlier run's weights, fresh optimizer and schedule
        model.load_state_dict(torch.load(cfg["init_from"], map_location=dev, weights_only=False)["model"])
        print(f"initialised from {cfg['init_from']}", flush=True)
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck.get("best", -1.0)
        print(f"resumed from step {step}", flush=True)
    log = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")

    def emit(rec: dict) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    def save(path: Path) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "config": model.cfg}, path)

    sampler = StepSampler(bounds, weights, tc["batch"], seed, step, tc["steps"])
    loader = DataLoader(
        ds, batch_sampler=sampler, num_workers=tc["workers"], pin_memory=dev == "cuda", prefetch_factor=4, persistent_workers=False
    )
    t0 = time.time()
    acc: dict[str, float] = {}
    n_acc = 0
    for batch in loader:
        if step >= tc["steps"] or (time.time() - t0) / 60 > tc["max_minutes"]:
            break
        b = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        x = to_input_batch(b["image"])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        model.train()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            out = model(x.contiguous(memory_format=torch.channels_last))
        ls = det_loss(out, b, lc)
        opt.zero_grad(set_to_none=True)
        ls["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        for k, v in ls.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n_acc += 1
        if step % tc["log_every"] == 0:
            emit(
                {
                    "step": step,
                    "lr": lr_at(step, tc),
                    "elapsed_s": round(time.time() - t0, 1),
                    **{k: round(v / n_acc, 4) for k, v in acc.items()},
                }
            )
            acc, n_acc = {}, 0
        if step % tc["eval_every"] == 0 or step == tc["steps"]:
            model.eval()
            v = {n: evaluate_records(model, recs, dev) for n, recs in val_sets.items()}
            # CORD labels only part of each receipt, so its precision means nothing: it contributes text recall
            score = sum(
                r["layout"]["mAP"] + (r["text"]["recall" if n == "cord" else "f1"] if "text" in r else 0.0) for n, r in v.items()
            ) / len(v)
            emit({"step": step, "val": v, "score": score})
            if score > best:  # picked on validation pages only
                best = score
                save(run_dir / "best.pt")
        if step % tc["ckpt_every"] == 0:
            save(ckpt_path)
    save(ckpt_path)
    if step >= tc["steps"]:
        save(run_dir / "final.pt")
    print(json.dumps({"stopped_at_step": step, "done": step >= tc["steps"]}), flush=True)


def to_input_batch(img: torch.Tensor) -> torch.Tensor:
    return img.float().div(127.5).sub(1.0)


if __name__ == "__main__":
    main()
