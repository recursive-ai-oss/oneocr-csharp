# ruff: noqa: E501
"""Line crops cut from generated receipt photos, with exact labels.

Why: receipts photographed on a desk are tilted by up to 30 degrees, printed in faded thermal ink at 8-16 px type. The recognizer's
synthetic receipt lines are straight and rendered clean before degradation, so the faint and tilted case is covered only by the
engine crops of the (few) generated photos. The page generator (`pages/`, configs/pages-receiptphoto.yaml) renders a receipt, puts
it through the camera model and moves every word box with the image, so the line quad of each printed line is exact. This script cuts
each line (or each side of a wide gap inside a line, which the page reader also reads separately) out of the photo by that quad, straightens
it and scales it to the recognizer's 32 px height. The label is the text the generator printed; nothing is read from an image.

The crop box is the line quad grown by a random margin (sideways 0.15-0.6 line heights, above and below -0.05 to +0.3), because the reader's
own boxes vary by that much. A crop whose ink barely differs from its paper (the thermal fade took the whole line) is dropped: a label with
no ink behind it teaches the network to invent text.

    uv run python -m gen.receipt_lines --pages PAGES_DIR/train --out OUT/train --n 60000 --seed 129 [--workers 4]

PAGES_DIR/<split> holds images/ and truth/<id>.json (oneocr.page, written by `python -m pages.generate`). Output: OUT/images/<id>.png
(8-bit gray, 32 px high), labels.jsonl (id, text, page, and the crop parameters), run.json.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from data import charset as cs

BASE = Path(__file__).resolve().parents[1]
OUT_H = 32
GAP_SPLIT = 1.5  # a gap wider than this many line heights ends a segment (the reader cuts wide gaps)


def segments(words: list[dict]) -> list[list[dict]]:
    """Splits a line's words at wide gaps. Quads are (TL, TR, BR, BL) as x0 y0 x1 y1 ..., in the line's own reading order."""
    out: list[list[dict]] = [[words[0]]]
    for a, b in zip(words, words[1:], strict=False):
        q, r = a["quad"], b["quad"]
        h = math.hypot(q[6] - q[0], q[7] - q[1])
        gap = math.hypot(r[0] - q[2], r[1] - q[3])
        if h > 0 and gap > GAP_SPLIT * h:
            out.append([b])
        else:
            out[-1].append(b)
    return out


def cut(img: Image.Image, seg: list[dict], rng: random.Random) -> tuple[Image.Image, dict] | None:
    """Straightens the segment's quad, grown by random margins, to a 32 px high gray crop."""
    f, e = seg[0]["quad"], seg[-1]["quad"]
    tl, bl = np.array(f[0:2]), np.array(f[6:8])
    tr, br = np.array(e[2:4]), np.array(e[4:6])
    ax = tr - tl
    length = float(np.hypot(*ax))
    hgt = float(np.hypot(*(bl - tl)))
    if length < 4 or hgt < 4:
        return None
    ux, uy = ax / length, (bl - tl) / hgt
    mx = rng.uniform(0.15, 0.6) * hgt, rng.uniform(0.15, 0.6) * hgt
    my = rng.uniform(-0.05, 0.3) * hgt, rng.uniform(-0.05, 0.3) * hgt
    ntl, ntr = tl - ux * mx[0] - uy * my[0], tr + ux * mx[1] - uy * my[0]
    nbl, nbr = bl - ux * mx[0] + uy * my[1], br + ux * mx[1] + uy * my[1]
    box_h = hgt + my[0] + my[1]
    if box_h > 64:  # the apparent type is big: scaling it down would alias, and such lines are not the case this data is for
        return None
    w = int(round(OUT_H * (length + mx[0] + mx[1]) / box_h))
    if not 16 <= w <= 1600:
        return None
    pts = np.stack([ntl, nbl, nbr, ntr])
    iw, ih = img.size
    if pts[:, 0].min() < -0.15 * hgt or pts[:, 1].min() < -0.15 * hgt or pts[:, 0].max() > iw + 0.15 * hgt or pts[:, 1].max() > ih + 0.15 * hgt:
        return None
    crop = img.transform((w, OUT_H), Image.Transform.QUAD, tuple(float(v) for v in pts.flatten()), resample=Image.Resampling.BICUBIC)
    a = np.asarray(crop)
    if np.percentile(a, 95) - np.percentile(a, 5) < 14:
        return None
    return crop, {"w": w, "line_h_px": round(hgt, 2), "margin_x": [round(m / hgt, 3) for m in mx], "margin_y": [round(m / hgt, 3) for m in my]}


def work(job: tuple[str, str, int, list[str]]) -> list[tuple[dict, bytes]]:
    pages_dir, pid, seed, charset = job
    root = Path(pages_dir)
    truth = json.loads((root / "truth" / f"{pid}.json").read_text(encoding="utf-8"))
    image_path = next(root.joinpath("images").glob(pid + ".*"))
    img = Image.open(image_path).convert("L")
    allowed = set(charset)
    rng = random.Random(f"{seed}/{pid}")
    rows = []
    n = 0
    for block in truth["pages"][0]["blocks"]:
        for line in block["lines"]:
            words = [w for w in line["words"] if w.get("quad") and w["text"]]
            if not words:
                continue
            for seg in segments(words):
                text = " ".join(w["text"] for w in seg)
                if not 1 <= len(text) <= 120 or any(c not in allowed for c in text):
                    continue
                r = cut(img, seg, rng)
                if r is None:
                    continue
                crop, params = r
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                rows.append(({"id": f"{pid}-{n:03d}", "text": text, "page": pid} | params, buf.getvalue()))
                n += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True, help="directory with images/ and truth/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True, help="crops to keep (random subset of every valid one)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    charset = cs.load(BASE / "data/charset-v1.txt")
    ids = sorted(p.stem for p in (Path(a.pages) / "truth").glob("*.json"))
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        per_page = pool.map(work, [(a.pages, i, a.seed, charset) for i in ids], chunksize=16)
    rows = [r for page in per_page for r in page]
    random.Random(a.seed).shuffle(rows)
    rows = sorted(rows[: a.n], key=lambda r: r[0]["id"])
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for lab, png in rows:
            (out / "images" / f"{lab['id']}.png").write_bytes(png)
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    commit = os.environ.get("ONEOCR_GIT_COMMIT", "unknown")
    meta = {"generator": "gen.receipt_lines", "seed": a.seed, "pages": len(ids), "valid_crops": sum(len(p) for p in per_page), "kept": len(rows),
            "labels_sha256": hashlib.sha256((out / "labels.jsonl").read_bytes()).hexdigest(), "git_commit": commit,
            "label_provenance": "exact-truth synthetic: text printed by pages.generate, quads from the renderer's boxes through the camera homography",
            "seconds": round(time.time() - t0, 1)}
    (out / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
