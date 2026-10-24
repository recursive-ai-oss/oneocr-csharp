"""Text-line detection on the bench-v1 eval sets, scored against line boxes built from the word-box sidecars.

Usage: uv run python -m eval.detect_bench --checkpoint best.pt --bench DIR [--out report.json] [--limit N]

The sidecars hold word boxes (layout-box truth: the box of a word spans the line height and the justified gap, so an ink-tight
detector cannot reach F1 1.0 on them). The detector is line-level, so words of one block that sit on one baseline are joined into
line boxes with the same rule used for DocLayNet cells (train.detect_data.merge_runs); F1 is at IoU 0.5. Sets are reported separately.
These pages are never trained on; run this only for final reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.detect import infer, match_counts, prf
from models.dbnet import DBNetLite
from train.detect_data import merge_runs

SETS = ["clean300-v1", "degraded-v1", "layout-v1", "phone-v1", "screenshot-v1", "receipts-v1"]


def truth_lines(words: list[dict]) -> np.ndarray:
    by_block: dict[int, list[list[float]]] = {}
    for w in words:
        b = w["box"]
        by_block.setdefault(w.get("block", 0), []).append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    lines = [ln for runs in by_block.values() for ln in merge_runs(runs, gap=1.5, overlap=0.5)]
    return np.array(lines, np.float32).reshape(-1, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expand", type=float, default=1.0)
    ap.add_argument("--text-thr", type=float, default=0.3)
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = DBNetLite(ck["config"])
    model.load_state_dict(ck["model"])
    model = model.to(a.device).eval()
    report = {"checkpoint_step": ck["step"], "expand": a.expand, "text_threshold": a.text_thr, "sets": {}}
    for s in SETS:
        files = sorted((Path(a.bench) / s / "truth").glob("*.json"))[: a.limit]
        tp = np.zeros(3, np.int64)
        orient_ok = 0
        for f in files:
            t = json.loads(f.read_text())
            res = infer(
                model,
                Image.open(next((Path(a.bench) / s / "pages").glob(f.stem + ".*"))),
                a.device,
                expand_text=a.expand,
                text_thr=a.text_thr,
            )
            tp += match_counts(res["text"], truth_lines(t["words"]))
            orient_ok += int(res["orient"].argmax() == 0)
        report["sets"][s] = {**prf(*[int(v) for v in tp]), "pages": len(files), "orient_upright_acc": orient_ok / max(len(files), 1)}
        print(s, json.dumps(report["sets"][s]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
