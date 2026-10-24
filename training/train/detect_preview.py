"""Draws a few records with their line boxes, layout boxes and training targets, to check labels by eye.

Usage: uv run python -m train.detect_preview OUT_DIR [doclaynet|publaynet] [N]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from train.detect_data import CLASSES, DetDataset, load_records

COLORS = ["#d00", "#08c", "#0a0", "#c80", "#a0a", "#555", "#0cc", "#c0c"]


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    name = sys.argv[2] if len(sys.argv) > 2 else "doclaynet"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    root = Path(os.environ.get("ONEOCR_DATASETS", Path.home() / "oneocr" / "datasets"))
    recs = load_records(name, root, "val")[:: max(1, 5000 // 40)][:n]
    ds = DetDataset(recs, 1024, train=False)
    for i, r in enumerate(recs):
        im = Image.open(r["path"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for c, x0, y0, x1, y1 in r["layout"]:
            d.rectangle([x0, y0, x1, y1], outline=COLORS[int(c)], width=2)
        for b in r["lines"] or []:
            d.rectangle(b, outline="#f0f", width=1)
        im.save(out / f"{name}-{i}-boxes.png")
        s = ds[i]
        canvas = Image.fromarray(s["image"][0].numpy()).convert("RGB")
        arr = np.asarray(canvas).astype(np.float32)
        prob = np.kron(s["prob"][0].numpy(), np.ones((2, 2)))
        arr[..., 1] = np.clip(arr[..., 1] + 160 * prob, 0, 255)
        lay = s["layout"].numpy()
        for c in range(lay.shape[0]):
            m = np.kron(lay[c], np.ones((8, 8)))
            col = np.array([int(COLORS[c][1:][k] * 2, 16) if len(COLORS[c]) == 4 else 0 for k in range(3)], np.float32)
            arr = np.where(m[..., None] > 0, 0.6 * arr + 0.4 * col, arr)
        Image.fromarray(arr.astype(np.uint8)).save(out / f"{name}-{i}-targets.png")
    print("classes", CLASSES)


if __name__ == "__main__":
    main()
