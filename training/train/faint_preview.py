"""Looks at low-contrast training windows before a run: writes a few faded windows with their line boxes drawn, and reports the
1st-99th percentile spread of the grey values inside each small line box (the legibility check: the text must still be there).

    uv run python -m train.faint_preview WEB_WINDOWS OUT_DIR [--n 12]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from train.detect_data import DetDataset, load_web_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("windows")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--marks", action="store_true", help="draw the faint non-text marks too and save the windows")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = [r for r in load_web_records(Path(a.windows), "train") if sum(1 for d, e in zip(r["dark"], r["em_px"], strict=True) if d and e < 16) >= 2]
    rng = random.Random(3)
    spreads = []
    for i in range(a.n):
        r = rng.choice(recs)
        img = np.asarray(Image.open(r["path"]).convert("L"))
        faded = DetDataset.faint_text(img, rng)
        if a.marks:
            q = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for x0, y0, x1, y1 in r["lines"]], np.float32)
            faded = DetDataset.faint_marks(faded, q, np.ones(img.shape, np.uint8), rng)
        for x0, y0, x1, y1 in r["lines"]:
            if y1 - y0 > 20:
                continue
            for name, im in (("orig", img), ("faint", faded)):
                p = im[int(y0) : int(y1) + 1, int(x0) : int(x1) + 1]
                if p.size:
                    lo, hi = np.percentile(p, [1, 99])
                    if name == "faint":
                        spreads.append((hi - lo) / 255)
        if i < 6:
            h = min(img.shape[0], 400)
            canvas = Image.fromarray(np.concatenate([img[:h], faded[:h]], 1)).convert("RGB")
            d = ImageDraw.Draw(canvas)
            for x0, y0, x1, y1 in r["lines"]:
                if y0 < h:
                    d.rectangle([x0 + img.shape[1], y0, x1 + img.shape[1], y1], outline=(255, 0, 0))
            canvas.save(out / f"faint{i}.png")
    s = np.array(spreads)
    print(f"faded small lines: {len(s)}  spread p10/p50/p90 = {np.percentile(s, [10, 50, 90]).round(3)}  share < 0.05: {(s < 0.05).mean():.3f}")


if __name__ == "__main__":
    main()
