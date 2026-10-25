"""Word recall of the text detector on CORD receipts that no detector run trains on or selects with.

A truth word counts as found when one predicted line box covers at least half of the word's area. CORD labels only the item and total
rows, so precision is not measured here; the number of predicted lines per page is printed to watch for runaway extras, and ink F1 on
the benchmark pages remains the check for those. Recall is split by word shape (one character, digits, other) and by height.

    python -m datasets.cord_crops --out DIR                      # first: writes DIR/{train,calib,val}/{pages,truth}
    uv run python -m eval.receipt_recall --checkpoint best.pt --crops DIR --splits val,calib [--text-thr 0.3 0.4] [--expand 0.4]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.detect import decode, raw_maps
from models.dbnet import DBNetLite

HEIGHT_BINS = [0, 16, 24, 40, 1e9]


def covered(words: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Per truth word: the largest fraction of its area inside one predicted box."""
    if len(boxes) == 0 or len(words) == 0:
        return np.zeros(len(words), np.float32)
    ix = np.clip(np.minimum(words[:, None, 2], boxes[None, :, 2]) - np.maximum(words[:, None, 0], boxes[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(words[:, None, 3], boxes[None, :, 3]) - np.maximum(words[:, None, 1], boxes[None, :, 1]), 0, None)
    area = np.maximum((words[:, 2] - words[:, 0]) * (words[:, 3] - words[:, 1]), 1e-9)
    return (ix * iy / area[:, None]).max(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--splits", default="val,calib")
    ap.add_argument("--text-thr", type=float, nargs="+", default=[0.3])
    ap.add_argument("--expand", type=float, nargs="+", default=[0.4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = DBNetLite(ck["config"])
    model.load_state_dict(ck["model"])
    model = model.to(a.device).eval()
    result: dict = {"checkpoint_step": ck["step"]}
    for split in a.splits.split(","):
        truth = sorted((Path(a.crops) / split / "truth").glob("*.json"))[: a.limit]
        pages = []
        for tf in truth:
            t = json.loads(tf.read_text())
            img = Image.open(next((Path(a.crops) / split / "pages").glob(tf.stem + ".*")))
            pages.append((raw_maps(model, img, a.device), t["words"]))
        for thr, exp in itertools.product(a.text_thr, a.expand):
            found, single, digit, hgt, lines = [], [], [], [], 0
            for m, words in pages:
                boxes = decode(m, text_thr=thr, expand_text=exp)["text"]
                lines += len(boxes)
                wb = np.array([w["box"] for w in words], np.float32).reshape(-1, 4)
                f = covered(wb, boxes) >= 0.5
                found += f.tolist()
                single += [len(w["text"].strip()) == 1 for w in words]
                digit += [any(c.isdigit() for c in w["text"]) for w in words]
                hgt += (wb[:, 3] - wb[:, 1]).tolist()
            found, single, digit, hgt = (np.array(v) for v in (found, single, digit, hgt))
            rec = lambda mask: round(float(found[mask].mean()), 4) if mask.any() else None  # noqa: E731, B023
            r = {
                "pages": len(pages), "words": int(len(found)), "missed": int((~found).sum()),
                "recall": rec(np.ones_like(found)),
                "single_char": {"n": int(single.sum()), "recall": rec(single), "missed": int((~found & single).sum())},
                "has_digit_multi": rec(digit & ~single), "other": rec(~digit & ~single),
                "by_height": {f"{lo}-{hi}": {"n": int(((hgt >= lo) & (hgt < hi)).sum()), "recall": rec((hgt >= lo) & (hgt < hi))}
                              for lo, hi in itertools.pairwise(HEIGHT_BINS)},
                "lines_per_page": round(lines / max(len(pages), 1), 2),
            }  # fmt: skip
            result[f"{split} thr={thr} expand={exp}"] = r
            print(split, f"thr={thr} expand={exp}", json.dumps(r), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
