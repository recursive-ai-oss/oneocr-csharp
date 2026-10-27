"""Second, independent pixel check of a written web line set: the saved 32 px crops against their DOM boxes.

    uv run python -m gen.web_lines_check --data OUT/train [--limit N]

gen/web_lines.py checks each line on the full-resolution screenshot before it cuts the crop. This reads only what was written
(the 32 px PNG, the crop box, the scale, the DOM box) and asks again whether the ink of the crop sits where the DOM says the text
is, which also catches mistakes in the crop and rescale arithmetic. Exit status 1 when more than 1% of the crops fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def check_row(root: Path, r: dict) -> str | None:
    g = np.asarray(Image.open(root / "images" / f"{r['id']}.png").convert("L"), dtype=np.float32)
    sc = r["scale"]
    cx0, cy0 = r["crop"][0], r["crop"][1]
    x0, y0, x1, y1 = r["dom_box"]
    bx0, by0, bx1, by1 = (x0 - cx0) * sc, (y0 - cy0) * sc, (x1 - cx0) * sc, (y1 - cy0) * sc
    bg = float(np.bincount(g.astype(np.uint8).ravel(), minlength=256).argmax())
    d = np.abs(g - bg)
    top = float(np.percentile(d, 99.5))
    if top < 12:
        return "no contrast in the saved crop"
    ys, xs = np.nonzero(d > 0.3 * top)
    if len(xs) < 6:
        return "no ink in the saved crop"
    em = r["em_px"] * sc
    tol = 0.35 * em + 2
    if abs(xs.min() - bx0) > 0.8 * em + 2 or abs(xs.max() + 1 - bx1) > 0.8 * em + 2:
        return f"ink x {xs.min()}-{xs.max() + 1} vs DOM {bx0:.0f}-{bx1:.0f}"
    if ys.min() < by0 - 0.2 * (by1 - by0) - 2 and (by0 > tol):  # ink above the DOM box: a neighbouring line (allowed once it is slight)
        if ys.min() < by0 - 0.5 * (by1 - by0):
            return "ink far above the DOM box"
    if np.nonzero(d[:, int(max(0, bx0)) : int(min(g.shape[1], bx1))].max(axis=0) > 0.3 * top)[0].size < 0.3 * max(1.0, bx1 - bx0):
        return "DOM box mostly empty"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    root = Path(a.data)
    rows = [json.loads(x) for x in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines() if x]
    rows = rows[: a.limit] if a.limit else rows
    bad = [(r["id"], why) for r in rows if (why := check_row(root, r))]
    share = round(len(bad) / max(1, len(rows)), 4)
    print(json.dumps({"checked": len(rows), "failed": len(bad), "failed_share": share, "examples": bad[:10]}, indent=1))
    sys.exit(1 if len(bad) > 0.01 * len(rows) else 0)


if __name__ == "__main__":
    main()
