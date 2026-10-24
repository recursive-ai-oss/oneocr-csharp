"""Trains crnn-v0 on rendered line crops with CTC. Resumable; appends metrics to a log; writes a run manifest.

Usage: uv run python -m train.run CONFIG.yaml [--until STEP]

--until stops the process at that step after writing the resumable checkpoint, so a long schedule runs as chunks of a
few hours: start the same command again and it continues from `ckpt-latest.pt` (weights, optimizer, step, best validation
CER). Nothing else is state: the learning rate is a function of the step, the batch order of the seed and epoch, and the
degradation draws of the seed and step, so a resumed run sees the same batches as an uninterrupted one.

Training lines are the clean packs of the line set (DATA_DIR/clean/train); every step degrades a fresh batch on the GPU
(degrade/gpu.py, training ranges), seeded from the run seed and the step so a resumed run draws the same batches.
Validation and held-out sets are the materialised crops (degrade/materialize.py). Any architecture in `models`.

Paths in the config may use environment variables (e.g. ${ONEOCR_HOME}) so worker locations stay out of the repo.
CRNN: Shi et al. 2015. SVTR: Du et al. 2022. CTC: Graves et al. 2006.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import queue
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data import charset as cs
from degrade.gpu import degrade
from eval.lines import evaluate
from models import build, default_config, macs_per_column
from train.data import LineSet, PackSet, to_input

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
    until = int(sys.argv[sys.argv.index("--until") + 1]) if "--until" in sys.argv else None
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
    if tc.get("deterministic"):  # bit-reproducible kernels, slower; used to check that resuming is exact
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model_cfg = cfg.get("model") or default_config(cfg["arch"], len(charset))
    assert model_cfg["classes"] == len(charset)
    model = build(model_cfg).to(dev)
    params = sum(p.numel() for p in model.parameters())

    sets = {"train": PackSet(data_root / "clean" / "train", charset), "val": LineSet(data_root / "val", charset),
            "eval": LineSet(data_root / "eval", charset)}  # fmt: skip
    side = yaml.safe_load((BASE / cfg["line_set"]).read_text(encoding="utf-8"))["gpu_degrade"]["train"]
    gen = torch.Generator(device=dev)
    dirs = {"train": data_root / "clean" / "train", "val": data_root / "val", "eval": data_root / "eval"}
    manifest = run_manifest(cfg, dirs, charset_path)
    manifest["parameters"] = params
    manifest["macs_per_column"] = macs_per_column(model.config)
    if not (run_dir / "run_manifest.json").exists():
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    step = 0
    best = float("inf")
    ckpt_path = run_dir / "ckpt-latest.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        best = ck.get("best", float("inf"))
        print(f"resumed from step {step}", flush=True)

    if step == 0 and cfg.get("init_from"):  # a second stage starts from an earlier checkpoint's weights (optimizer and schedule are fresh)
        model.load_state_dict(torch.load(cfg["init_from"], map_location=dev, weights_only=False)["model"])
        print(f"initialised from {cfg['init_from']}", flush=True)

    train = sets["train"]
    steps_per_epoch = math.ceil(len(train) / tc["batch"])
    log = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")

    def emit(rec: dict) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    def save(path: Path) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "config": model_cfg, "charset": charset, "manifest": manifest}, path)  # fmt: skip

    def producer(first: int, q: queue.Queue) -> None:
        """Builds batches on a background thread so the GPU is not waiting for numpy."""
        cached_epoch, batches = -1, []
        for st in range(first, tc["steps"] if until is None else min(until, tc["steps"])):
            epoch, i = divmod(st, steps_per_epoch)
            if epoch != cached_epoch:
                cached_epoch, batches = epoch, train.epoch_batches(tc["batch"], random.Random(seed * 100003 + epoch))
            idx = batches[i]
            x, widths = train.batch(idx)
            tgt = torch.tensor([t for k in idx for t in train.targets[k]])
            tlen = torch.tensor([len(train.targets[k]) for k in idx])
            q.put((x.pin_memory(), widths, tgt, tlen))

    q: queue.Queue = queue.Queue(maxsize=24)
    threading.Thread(target=producer, args=(step, q), daemon=True).start()

    t0 = time.time()
    stop = tc["steps"] if until is None else min(until, tc["steps"])
    emit({"chunk_start": step, "until": stop, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": manifest["git_commit"]})
    run_loss, run_n = 0.0, 0
    while step < stop:
        x, widths, tgt, tlen = q.get()
        gen.manual_seed(seed * 1_000_003 + step)
        x, widths, _ = degrade(x.to(dev, non_blocking=True), widths.to(dev), side, gen)
        ilen = torch.tensor([model.out_len(int(w)) for w in widths])
        x = to_input(x, dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, widths)
        lp = logits.float().log_softmax(-1).transpose(0, 1)
        if tc.get("deterministic"):  # the CUDA CTC backward uses atomic adds, whose order varies from run to run; the CPU one is exact
            loss = F.ctc_loss(lp.cpu(), tgt, ilen, tlen, blank=0, reduction="mean", zero_infinity=True).to(dev)
        else:
            loss = F.ctc_loss(lp, tgt, ilen, tlen, blank=0, reduction="mean", zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        run_loss += loss.item()
        run_n += 1
        if step % tc["log_every"] == 0:
            emit({"step": step, "loss": run_loss / run_n, "lr": lr_at(step, tc), "elapsed_s": round(time.time() - t0, 1)})
            run_loss, run_n = 0.0, 0
        if step % tc["eval_every"] == 0 or step == tc["steps"]:
            v = evaluate(model, sets["val"], charset, dev, limit=1500)
            emit({"step": step, "val": v, "eval_heldout": evaluate(model, sets["eval"], charset, dev, limit=1500)})
            if v["cer"] < best:  # the best checkpoint is picked on validation (training fonts), never on the held-out set
                best = v["cer"]
                save(run_dir / "best.pt")
        if step % tc["ckpt_every"] == 0 or step == stop:
            save(ckpt_path)
    if step < tc["steps"]:
        emit({"chunk_end": step, "elapsed_s": round(time.time() - t0, 1)})
        return
    save(run_dir / "final.pt")
    final = {"step": step, "val": evaluate(model, sets["val"], charset, dev),
             "eval_heldout": evaluate(model, sets["eval"], charset, dev), "train_seconds": round(time.time() - t0, 1)}  # fmt: skip
    emit({"final": final})
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
