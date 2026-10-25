"""Trains the document extraction model (models/docextract.py). Resumable, chunked, appends metrics, writes a run manifest.

Usage: uv run python -m train.docextract configs/docextract-v0.yaml

Data: pickles from train.docextract_data (generated pages and CORD train, each as truth words and as OCR words). Every step draws a
batch from the sources by weight. Truth-word examples get light augmentation (box jitter, look-alike character swaps) so that the
model does not rely on perfectly clean words; OCR-word examples are used as they are. Labels come only from the generator's truth
and CORD's annotations (see train/docextract_data.py).

Loss: cross entropy on the field tags, the roles, the document type, and on the two pointer relations (a softmax over the page's
words, the target being the anchor word; the word itself means "none"), summed.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from eval.docextract import evaluate
from models.docextract import DocExtract, collate, count_params, featurize

# single-character look-alikes that a reader confuses
LOOKALIKE = {
    "0": "O",
    "O": "0",
    "1": "l",
    "l": "1",
    "I": "l",
    "5": "S",
    "S": "5",
    "8": "B",
    "B": "8",
    ",": ".",
    ".": ",",
    "e": "c",
    "c": "e",
    "m": "n",
    "n": "m",
    "u": "v",
    "v": "u",
    "2": "Z",
    "Z": "2",
}


def lr_at(step: int, tc: dict) -> float:
    if step < tc["warmup"]:
        return tc["lr"] * (step + 1) / tc["warmup"]
    p = (step - tc["warmup"]) / max(1, tc["steps"] - tc["warmup"])
    return tc["lr"] * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def augment(ex: dict, rng: random.Random, max_words: int) -> dict:
    """Window to max_words and, for truth-word examples, jitter boxes and swap look-alike characters."""
    n = len(ex["texts"])
    start = 0
    if n > max_words:
        start = rng.randint(0, n - max_words)
    end = min(n, start + max_words)
    texts = ex["texts"][start:end]
    boxes = ex["boxes"][start:end].astype(np.float64).copy()
    if ex["meta"]["input"] == "truth":
        h = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
        boxes[:, [0, 2]] += np.random.default_rng(rng.getrandbits(32)).normal(0, 0.15, (len(boxes), 2)) * h[:, None]
        boxes[:, [1, 3]] += np.random.default_rng(rng.getrandbits(32)).normal(0, 0.05, (len(boxes), 2)) * h[:, None]
        out = []
        for t in texts:
            if rng.random() < 0.15:
                cs = list(t)
                for i, c in enumerate(cs):
                    if rng.random() < 0.08 and c in LOOKALIKE:
                        cs[i] = LOOKALIKE[c]
                t = "".join(cs)
            out.append(t)
        texts = out
    f = featurize(texts, boxes.tolist(), ex["width"], ex["height"])
    item = dict(f)
    for key in ("tags", "roles"):
        item[key] = ex[key][start:end]
    for key in ("group_ptr", "kv_ptr"):
        p = ex[key][start:end].copy()
        p = np.where(p >= 0, p - start, p)
        p[(p < 0) | (p >= end - start)] = -100
        item[key] = p
    item["doc"] = ex["doc"]
    return item


def loss_fn(out: dict, b: dict, w: dict) -> dict:
    tags = F.cross_entropy(out["tags"].float().transpose(1, 2), b["tags"], ignore_index=-100)
    roles = F.cross_entropy(out["roles"].float().transpose(1, 2), b["roles"], ignore_index=-100)
    doc = F.cross_entropy(out["doc"].float(), b["doc"])
    rel = out["rel"].float()
    rg = F.cross_entropy(rel[:, 0].transpose(1, 2), b["group_ptr"], ignore_index=-100)
    rk = F.cross_entropy(rel[:, 1].transpose(1, 2), b["kv_ptr"], ignore_index=-100)
    rg = torch.nan_to_num(rg)
    rk = torch.nan_to_num(rk)
    total = w["tags"] * tags + w["roles"] * roles + w["doc"] * doc + w["group"] * rg + w["kv"] * rk
    return {"total": total, "tags": tags.detach(), "roles": roles.detach(), "doc": doc.detach(), "group": rg.detach(), "kv": rk.detach()}


def select_score(m: dict) -> float:
    return (m["all"]["header_micro"]["f1"] + m["all"]["item_field"]["f1"] + m["all"]["kv"]["f1"]) / 3


def main() -> None:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cfg = json.loads(os.path.expandvars(json.dumps(cfg)))
    tc, lw = cfg["train"], cfg["loss"]
    seed = cfg["seed"]
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = os.environ.get("ONEOCR_DEVICE", "cuda")  # "cpu" only for smoke runs
    torch.manual_seed(seed)
    model = DocExtract(cfg.get("model")).to(dev)

    pools, weights, prov = [], [], {}
    for src in cfg["sources"]:
        d = pickle.load(open(src["pickle"], "rb"))[src["set"]]
        ex = d[src["kind"]]
        pools.append(ex)
        weights.append(src["weight"])
        prov[f"{src['set']}/{src['kind']}"] = {
            "pages": len(ex),
            "weight": src["weight"],
            "label_source": src["label_source"],
            "pickle": Path(src["pickle"]).name,
        }
    val = {}
    for v in cfg["val"]:
        d = pickle.load(open(v["pickle"], "rb"))[v["set"]]
        val[f"{v['set']}/{v['kind']}"] = d[v["kind"]][: v.get("limit")]
    macs_note = {"parameters": count_params(model)}
    manifest = {
        "config": cfg,
        "seed": seed,
        "git_commit": os.environ.get("ONEOCR_GIT_COMMIT", "unknown"),
        "data": prov,
        **macs_note,
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"], betas=(0.9, 0.98))
    step, best = 0, -1.0
    ckpt_path = run_dir / "ckpt-latest.pt"
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

    t0 = time.time()
    acc: dict[str, float] = {}
    n_acc = 0
    while step < tc["steps"] and (time.time() - t0) / 60 <= tc["max_minutes"]:
        rng = random.Random(seed * 1_000_003 + step)  # a resumed run draws the same batch as an uninterrupted one
        items = []
        for _ in range(tc["batch"]):
            pool = rng.choices(pools, weights)[0]
            items.append(augment(pool[rng.randrange(len(pool))], rng, tc["max_words"]))
        b = {k: v.to(dev) for k, v in collate(items).items()}
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc)
        model.train()
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            out = model(b["chars"], b["feats"], b["bins"], b["geo"], b["mask"])
        ls = loss_fn(out, b, lw)
        opt.zero_grad(set_to_none=True)
        ls["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        for k, v in ls.items():
            acc[k] = acc.get(k, 0.0) + float(v.detach())
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
            res = {k: evaluate(model, ex, dev, by_kind=False) for k, ex in val.items()}
            score = float(np.mean([select_score(m) for m in res.values()]))
            emit(
                {
                    "step": step,
                    "val": {
                        k: {
                            "header_f1": m["all"]["header_micro"]["f1"],
                            "item_field_f1": m["all"]["item_field"]["f1"],
                            "kv_f1": m["all"]["kv"]["f1"],
                            "doc": m["all"]["doc_type_accuracy"],
                        }
                        for k, m in res.items()
                    },
                    "score": score,
                }
            )
            if score > best:  # picked on validation pages and CORD dev only
                best = score
                save(run_dir / "best.pt")
        if step % tc["ckpt_every"] == 0:
            save(ckpt_path)
    save(ckpt_path)
    if step >= tc["steps"]:
        save(run_dir / "final.pt")
    print(json.dumps({"stopped_at_step": step, "done": step >= tc["steps"]}), flush=True)


if __name__ == "__main__":
    main()
