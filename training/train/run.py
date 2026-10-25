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
from train.adaptive import Adaptive
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


def augment_real(x: torch.Tensor, widths: torch.Tensor, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Light photometric and width jitter for real crops (uint8 [B,1,32,W], paper-padded): contrast and brightness, gamma, a little
    blur and noise, width scaled by 0.85-1.15. Real crops are the scarcest data, so they are seen many times with small changes."""
    dev = x.device
    b = x.shape[0]
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(b, generator=g, device=dev)  # noqa: E731
    f = x.float() / 255.0
    f = f ** u(0.75, 1.35).view(b, 1, 1, 1)
    f = (f - 0.5) * u(0.7, 1.1).view(b, 1, 1, 1) + 0.5 + u(-0.08, 0.08).view(b, 1, 1, 1)
    sig = u(0.3, 0.9) * (torch.rand(b, generator=g, device=dev) < 0.3)
    k = torch.arange(-2, 3, device=dev).float().view(1, -1)
    w = torch.exp(-(k**2) / (2 * sig.clamp(min=1e-3).view(-1, 1) ** 2))
    w = torch.where(sig.view(-1, 1) > 0, w / w.sum(1, keepdim=True), (k == 0).float())
    f = torch.nn.functional.conv2d(torch.nn.functional.pad(f, (2, 2, 0, 0), mode="replicate").transpose(0, 1), w.view(b, 1, 1, 5), groups=b)
    f = f.transpose(0, 1)
    f = f + torch.randn(f.shape, generator=g, device=dev) * u(0.0, 0.03).view(b, 1, 1, 1)
    f = f.clamp(0, 1) * 255.0
    outs, ws = [], []
    for i in range(b):
        wi = int(widths[i])
        nw = max(4, int(round(wi * float(u(0.85, 1.15)[i]))))
        outs.append(torch.nn.functional.interpolate(f[i : i + 1, :, :, :wi], size=(f.shape[2], nw), mode="bilinear", align_corners=False))
        ws.append(nw)
    return pad_group(outs, ws)


def pad_group(imgs: list[torch.Tensor], widths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stacks [1,1,H,w] images to a common width, repeating each image's last column (paper) as padding."""
    wmax = max(widths)
    pad = torch.nn.functional.pad
    out = torch.stack([pad(im[0], (0, wmax - w), mode="replicate") if w < wmax else im[0] for im, w in zip(imgs, widths, strict=True)])
    return out, torch.tensor(widths)


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
    for k, v in cfg.get(
        "main_override", {}
    ).items():  # e.g. the reader's resampling on the production mix: {p: {resample: 0.3}, ranges: {...}}
        side = {**side, k: {**side[k], **v}}
    # extra training sources mixed into every batch (fine-tune stages): a clean pack degraded on the GPU with its own side of a line-set
    # config (kind pack), or real crops already at model height (kind crops); extra validation sets are scored next to `val`
    sources = []
    for sc in cfg.get("sources", []):
        if sc["kind"] == "pack":
            ds = PackSet(Path(sc["dir"]), charset)
            sd = yaml.safe_load((BASE / sc["line_set"]).read_text(encoding="utf-8"))["gpu_degrade"][sc["side"]]
        else:
            ds, sd = LineSet(Path(sc["dir"]), charset), None
        sources.append((sc, ds, sd))
    adaptive = None
    for sc, ds, sd in sources:
        if sc.get("adaptive"):  # the targeted-error stream: weighted draws from this pack, degradation bins from the measured errors
            stream = PackSet(Path(sc["adaptive"]["stream_dir"]), charset)
            adaptive = Adaptive(ds, stream, charset, sc["adaptive"], sd, seed, run_dir)
            adaptive_src = sc["name"]
    extra_val = {k: LineSet(Path(v), charset) for k, v in cfg.get("extra_val", {}).items()}
    n_extra = sum(sc["per_batch"] for sc, _, _ in sources)
    main_batch = tc["batch"] - n_extra
    gen = torch.Generator(device=dev)
    dirs = {"train": data_root / "clean" / "train", "val": data_root / "val", "eval": data_root / "eval"}
    manifest = run_manifest(cfg, dirs, charset_path)
    manifest["parameters"] = params
    manifest["sources"] = [{"name": sc["name"], "kind": sc["kind"], "per_batch": sc["per_batch"], "label_provenance": sc["provenance"],
                            "labels_sha256": sha256_file(Path(sc["dir"]) / "labels.jsonl")} for sc, _, _ in sources]  # fmt: skip
    manifest["extra_val"] = {k: sha256_file(Path(v) / "labels.jsonl") for k, v in cfg.get("extra_val", {}).items()}
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
    steps_per_epoch = math.ceil(len(train) / main_batch)
    log = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")

    def emit(rec: dict) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    def save(path: Path) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "config": model_cfg, "charset": charset, "manifest": manifest}, path)  # fmt: skip

    if extra_val:  # starting model's CER on every validation set: the unit the selection score is measured in
        if step == 0:
            tc["val_baseline"] = {"val": evaluate(model, sets["val"], charset, dev, limit=1500)["cer"]} | {
                k: evaluate(model, d, charset, dev, limit=3000)["cer"] for k, d in extra_val.items()}  # fmt: skip
            emit({"baseline": tc["val_baseline"]})
        else:
            lines = (run_dir / "metrics.jsonl").read_text().splitlines()
            tc["val_baseline"] = next(json.loads(ln)["baseline"] for ln in lines if '"baseline"' in ln)

    def producer(first: int, q: queue.Queue) -> None:
        """Builds batches on a background thread so the GPU is not waiting for numpy."""
        cached_epoch, batches = -1, []
        for st in range(first, tc["steps"] if until is None else min(until, tc["steps"])):
            epoch, i = divmod(st, steps_per_epoch)
            if epoch != cached_epoch:
                cached_epoch, batches = epoch, train.epoch_batches(main_batch, random.Random(seed * 100003 + epoch))
            idx = batches[i]
            x, widths = train.batch(idx)
            groups = [(0, x.pin_memory(), widths, [train.targets[k] for k in idx])]
            srng = random.Random(seed * 7919 + st)
            for si, (sc, ds, _) in enumerate(sources, 1):
                if adaptive is not None and sc["name"] == adaptive_src:
                    sidx = adaptive.sample(st, sc["per_batch"])
                else:
                    sidx = [srng.randrange(len(ds)) for _ in range(sc["per_batch"])]
                sx, sw = ds.batch(sidx)
                groups.append((si, sx.pin_memory(), sw, [ds.targets[k] for k in sidx]))
            q.put(groups)

    if adaptive is not None:  # weights versions are files next to the checkpoint; the producer needs the current one before it starts
        for v in (step // adaptive.every - 1, step // adaptive.every):
            if v >= 0:
                adaptive.load(v)
        if step % adaptive.every and (step // adaptive.every) not in adaptive.versions:
            raise SystemExit(f"adapt-v{step // adaptive.every:03d}.json is missing next to the checkpoint")

    q: queue.Queue = queue.Queue(maxsize=24)
    threading.Thread(target=producer, args=(step, q), daemon=True).start()

    t0 = time.time()
    stop = tc["steps"] if until is None else min(until, tc["steps"])
    emit({"chunk_start": step, "until": stop, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_commit": manifest["git_commit"]})
    run_loss, run_n = 0.0, 0
    while step < stop:
        if adaptive is not None and step % adaptive.every == 0 and (step // adaptive.every) not in adaptive.versions:
            v = step // adaptive.every
            if not adaptive.load(v):
                prev = adaptive.versions[v - 1] if v - 1 in adaptive.versions else adaptive.initial()
                state, report = adaptive.measure(model, v, dev, prev)
                adaptive.publish(v, state)
                emit({"adapt": report})
        groups = q.get()
        gen.manual_seed(seed * 1_000_003 + step)
        imgs, ws, tl = [], [], []
        for si, gx, gw, gt in groups:
            gx, gw = gx.to(dev, non_blocking=True), gw.to(dev)
            if si == 0:
                gx, gw, _ = degrade(gx, gw, side, gen)
            elif sources[si - 1][2] is not None:
                sd = sources[si - 1][2]
                if adaptive is not None and sources[si - 1][0]["name"] == adaptive_src:
                    sd = {**sd, "_bins": adaptive.bins_for(step // adaptive.every)}
                gx, gw, _ = degrade(gx, gw, sd, gen)
            else:
                gx, gw = augment_real(gx, gw, gen)
            for k in range(gx.shape[0]):
                imgs.append(gx[k : k + 1, :, :, : int(gw[k])].float())
                ws.append(int(gw[k]))
            tl += gt
        x, widths = pad_group(imgs, ws)
        x = x.round().clamp(0, 255)
        tgt = torch.tensor([t for ts in tl for t in ts])
        tlen = torch.tensor([len(ts) for ts in tl])
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
            ev = {k: evaluate(model, d, charset, dev, limit=3000) for k, d in extra_val.items()}
            emit({"step": step, "val": v, "extra_val": ev, "eval_heldout": evaluate(model, sets["eval"], charset, dev, limit=1500)})
            score = v["cer"]
            # fine-tune: mean CER relative to the starting model on each validation set, plus a penalty for the line set getting worse
            if ev:
                base = tc["val_baseline"]
                score = (v["cer"] / base["val"] + sum(r["cer"] / base[k] for k, r in ev.items())) / (1 + len(ev))
                score += tc.get("val_penalty", 3.0) * max(0.0, v["cer"] / base["val"] - 1.0)
            if score < best:  # the best checkpoint is picked on validation sets, never on the held-out or page evaluation sets
                best = score
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
