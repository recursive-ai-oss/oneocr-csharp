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

from eval.detect import infer, iou_matrix, match_counts, prf
from models.dbnet import DBNetLite
from train.detect_data import merge_runs

SETS = ["clean300-v1", "degraded-v1", "layout-v1", "phone-v1", "screenshot-v1", "receipts-v1"]


def truth_lines(words: list[dict], key: str = "box", gap: float = 1.5) -> np.ndarray:
    """Line boxes from word boxes; key "ink" uses the bench-v2 ink-tight boxes (falls back to the layout box)."""
    by_block: dict[int, list[list[float]]] = {}
    for w in words:
        b = w.get(key) or w["box"]
        by_block.setdefault(w.get("block", 0), []).append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
    lines = [ln for runs in by_block.values() for ln in merge_runs(runs, gap=gap, overlap=0.5)]
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
    ap.add_argument("--boxes", default="box", choices=["box", "ink"], help="truth box kind (bench-v2 has ink)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--gap", type=float, default=1.5, help="truth words on one baseline join into a line below this gap (line heights)")
    ap.add_argument("--kinds", help="only pages whose truth `kind` is one of these (comma list; layout-v1 has table, form, mixed, ...)")
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = DBNetLite(ck["config"])
    model.load_state_dict(ck["model"])
    model = model.to(a.device).eval()
    report = {"checkpoint_step": ck["step"], "expand": a.expand, "text_threshold": a.text_thr, "sets": {}}
    for s in a.sets.split(","):
        files = sorted((Path(a.bench) / s / "truth").glob("*.json"))[: a.limit]
        tp = np.zeros(3, np.int64)
        orient_ok = 0
        n_pages = 0
        hr, wr = [], []
        for f in files:
            t = json.loads(f.read_text())
            if a.kinds and t.get("kind") not in a.kinds.split(","):
                continue
            res = infer(
                model,
                Image.open(next((Path(a.bench) / s / "pages").glob(f.stem + ".*"))),
                a.device,
                expand_text=a.expand,
                text_thr=a.text_thr,
            )
            n_pages += 1
            gt = truth_lines(t["words"], a.boxes, a.gap)
            tp += match_counts(res["text"], gt, a.iou)
            if len(gt) and len(res["text"]):  # size of each prediction relative to its best-overlapping truth line
                m = iou_matrix(res["text"], gt)
                for i in range(len(res["text"])):
                    j = int(m[i].argmax())
                    if m[i, j] >= 0.3:
                        hr.append((res["text"][i, 3] - res["text"][i, 1]) / (gt[j, 3] - gt[j, 1]))
                        wr.append((res["text"][i, 2] - res["text"][i, 0]) / (gt[j, 2] - gt[j, 0]))
            orient_ok += int(res["orient"].argmax() == 0)
        report["sets"][s] = {
            **prf(*[int(v) for v in tp]),
            "pages": n_pages,
            "orient_upright_acc": orient_ok / max(n_pages, 1),
            "height_ratio_median": float(np.median(hr)) if hr else None,
            "width_ratio_median": float(np.median(wr)) if wr else None,
        }
        print(s, json.dumps(report["sets"][s]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
