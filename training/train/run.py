"""Trains crnn-v0 on rendered line crops with CTC. Resumable; appends metrics to a log; writes a run manifest.

Usage: uv run python -m train.run configs/crnn-smoke.yaml

Paths in the config may use environment variables (e.g. ${ONEOCR_HOME}) so worker locations stay out of the repo.
CRNN: Shi et al. 2015. CTC: Graves et al. 2006.
"""

from __future__ import annotations

import hashlib
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

from data import charset as cs
from eval.lines import evaluate
from models.crnn import CRNN, DEFAULT_CONFIG, output_length
from train.data import LineSet

BASE = Path(__file__).resolve().parents[1]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_manifest(cfg: dict, data_dirs: dict[str, Path], charset_path: Path) -> dict:
    return {
        "config": cfg,
        "seed": cfg["seed"],
        "git_commit": os.environ.get("ONEOCR_GIT_COMMIT", "unknown"),
        "charset_sha256": sha256_file(charset_path),
        "data": {
            k: {"labels_sha256": sha256_file(d / "labels.jsonl"), "label_provenance": "exact-truth synthetic"} for k, d in data_dirs.items()
        },
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def lr_at(step: int, tc: dict) -> float:
    if step < tc["warmup"]:
        return tc["lr"] * (step + 1) / tc["warmup"]
    p = (step - tc["warmup"]) / max(1, tc["steps"] - tc["warmup"])
    return tc["lr"] * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p)))


def main() -> None:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cfg = json.loads(os.path.expandvars(json.dumps(cfg)))
    tc = cfg["train"]
    seed = cfg["seed"]
    charset_path = BASE / cfg["charset"]
    charset = cs.load(charset_path)
    data_root = Path(cfg["data_dir"])
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    random.seed(seed)
    torch.manual_seed(seed)
    model_cfg = cfg.get("model") or {**DEFAULT_CONFIG, "classes": len(charset)}
    assert model_cfg["classes"] == len(charset)
    model = CRNN(model_cfg).to(dev)
    params = sum(p.numel() for p in model.parameters())

    sets = {k: LineSet(data_root / k, charset) for k in ("train", "val", "eval")}
    manifest = run_manifest(cfg, {k: data_root / k for k in sets}, charset_path)
    manifest["parameters"] = params
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    step = 0
    ckpt_path = run_dir / "ckpt-latest.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed from step {step}", flush=True)

    train = sets["train"]
    steps_per_epoch = math.ceil(len(train) / tc["batch"])
    log = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")

    def emit(rec: dict) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    def save(path: Path) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": model_cfg,
                    "charset": charset, "manifest": manifest}, path)  # fmt: skip

    t0 = time.time()
    run_loss, run_n = 0.0, 0
    cached_epoch, batches = -1, []
    while step < tc["steps"]:
        epoch, i = divmod(step, steps_per_epoch)
        if epoch != cached_epoch:
            cached_epoch, batches = epoch, train.epoch_batches(tc["batch"], random.Random(seed * 100003 + epoch))
        idx = batches[i]
        x, widths = train.batch(idx)
        x = x.to(dev, non_blocking=True)
        tgt = torch.tensor([t for k in idx for t in train.targets[k]])
        tlen = torch.tensor([len(train.targets[k]) for k in idx])
        ilen = torch.tensor([output_length(int(w), model.config) for w in widths])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, widths)
        lp = logits.float().log_softmax(-1).transpose(0, 1)
        loss = F.ctc_loss(lp, tgt, ilen, tlen, blank=0, reduction="mean", zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        run_loss += float(loss)
        run_n += 1
        if step % tc["log_every"] == 0:
            emit({"step": step, "loss": run_loss / run_n, "lr": lr_at(step, tc), "elapsed_s": round(time.time() - t0, 1)})
            run_loss, run_n = 0.0, 0
        if step % tc["eval_every"] == 0 or step == tc["steps"]:
            emit({"step": step, "val": evaluate(model, sets["val"], charset, dev, limit=500),
                  "eval_heldout": evaluate(model, sets["eval"], charset, dev, limit=500)})  # fmt: skip
        if step % tc["ckpt_every"] == 0 or step == tc["steps"]:
            save(ckpt_path)
    save(run_dir / "final.pt")
    final = {"step": step, "val": evaluate(model, sets["val"], charset, dev),
             "eval_heldout": evaluate(model, sets["eval"], charset, dev), "train_seconds": round(time.time() - t0, 1)}  # fmt: skip
    emit({"final": final})
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
