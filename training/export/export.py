"""Exports a crnn-v0 checkpoint (or a seeded random-init model) to model container v0.

Usage:
  uv run python -m export.export --checkpoint final.pt --out model.onm [--version 0.1.0] [--eval-data LABEL=DIR ...]
  uv run python -m export.export --random-seed 7 --out model.onm

BN is folded into the conv weights in fp64 then rounded once to fp16. With --eval-data (repeatable) the fp16-rounded model is
evaluated next to the fp32 checkpoint, and single-thread CPU latency is reported (a proxy for the C# kernels).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from data import charset as cs
from export.container import read_container, write_container
from models import build, fold, load_folded, macs_per_column
from models.crnn import CRNN, DEFAULT_CONFIG
from models.crnn import FoldedReference as CrnnReference
from models.svtr import FoldedReference as SvtrReference

ARCHITECTURE_IDS = {"crnn-v0": "crnn-v0", "svtr-lite": "svtr-lite-v1"}  # svtr-lite-v1 is a proposal: see export/svtr-lite-architecture.md

BASE = Path(__file__).resolve().parents[1]
INPUT = {"height": 32, "channels": 1, "mean": 0.5, "std": 0.5, "invert": False}


def random_model(seed: int, config: dict) -> CRNN:
    """Random init with non-trivial BN statistics so that BN folding is actually exercised."""
    torch.manual_seed(seed)
    m = CRNN(config)
    for b in m.conv:
        bn = b[1]
        bn.weight.data.uniform_(0.6, 1.4)
        bn.bias.data.normal_(0, 0.1)
        bn.running_mean.data.normal_(0, 0.2)
        bn.running_var.data.uniform_(0.5, 2.0)
    return m.eval()


def export(model, charset: list[str], path: Path, version: str, provenance: dict) -> dict:
    tensors = fold(model.eval())
    manifest = {
        "architecture": ARCHITECTURE_IDS[model.config.get("arch", "crnn-v0")],
        "name": "line-recognizer",
        "version": version,
        "license": "MIT OR Apache-2.0",
        "input": INPUT,
        "charset": charset,
        "config": model.config,
        "provenance": provenance,
    }
    write_container(path, manifest, {k: v.numpy() for k, v in tensors.items()})
    return manifest


def single_thread_latency(model: CRNN, width: int = 320, reps: int = 20) -> float:
    """Milliseconds per line (32 x width) on one CPU thread."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    m = model.cpu().eval()
    x = torch.randn(1, 1, 32, width)
    with torch.no_grad():
        for _ in range(3):
            m(x)
        t = time.perf_counter()
        for _ in range(reps):
            m(x)
        dt = (time.perf_counter() - t) / reps
    torch.set_num_threads(prev)
    return dt * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--random-seed", type=int)
    ap.add_argument("--charset", default="data/charset-v1.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--eval-data", action="append", default=[], help="LABEL=DIR of a line set (labels.jsonl + images/); repeatable")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    charset = cs.load(BASE / a.charset)
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model = build(ck["config"])
        model.load_state_dict(ck["model"])
        charset = ck["charset"]
        provenance = {"run": ck["manifest"], "step": ck["step"]}
    else:
        model = random_model(a.random_seed, {**DEFAULT_CONFIG, "classes": len(charset)})
        provenance = {"random_init_seed": a.random_seed, "note": "untrained, for parity tests only"}
    export(model, charset, out, a.version, provenance)

    # round trip: the reader must return exactly the fp16 values that were written
    m2, tensors = read_container(out)
    assert m2["charset"] == charset and m2["config"] == model.config
    ref = {k: v.half().float().numpy() for k, v in fold(model).items()}
    assert all((tensors[k] == ref[k]).all() for k in ref), "round trip mismatch"

    params = sum(p.numel() for p in model.parameters())
    folded_params = sum(v.size for v in tensors.values())
    report = {
        "file": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "bytes": out.stat().st_size,
        "train_parameters": params,
        "container_parameters": folded_params,
        "macs_per_column": macs_per_column(model.config),
        "fp16_tensor_mb": round(folded_params * 2 / 1e6, 2),
        "cpu_ms_per_line_w320_1thread": round(single_thread_latency(model), 2),
    }
    if a.eval_data:
        from eval.lines import evaluate
        from train.data import LineSet

        t = {k: torch.from_numpy(v) for k, v in tensors.items()}
        folded = load_folded(model.config, t).cuda()
        fp32 = model.cuda().eval()
        for item in a.eval_data:
            label, _, d = item.partition("=")
            ds = LineSet(Path(d), charset)
            report[f"{label}_fp32"] = evaluate(fp32, ds, charset)
            report[f"{label}_fp16_folded"] = evaluate(folded, ds, charset)
    # reference forward on random input, as a sanity check that fixtures will be producible
    reference = SvtrReference if model.config.get("arch") == "svtr-lite" else CrnnReference
    reference(model.config, {k: torch.from_numpy(v) for k, v in tensors.items()}).forward(torch.randn(1, 1, 32, 100))
    Path(str(out) + ".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
