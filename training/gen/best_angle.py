"""Rotate photos by trial angles, and pick for each photo the angle whose engine reading looks best, without using any label.

    uv run python -m gen.best_angle rotate --images DIR --out DIR --angles -30,-20,-10,10,20,30 [--ids FILE]
    uv run python -m gen.best_angle pick   --engine 0=DIR --engine -10=DIR ... --images DIR --rotated DIR --out-engine DIR --out-images DIR

`rotate` writes OUT/a<angle>/<id>.jpg (bicubic, expanded canvas filled with the photo's median colour). `pick` reads the word
sidecars (`OneOcr.Bench run --words 1 --pagejson 1`) of every angle and takes, per photo, the angle with the largest sum over words with
confidence >= 0.9 of confidence x length. The rule is fixed in advance and never sees a transcription. It links the chosen angle's page
model, sidecar and image under OUT-ENGINE / OUT-IMAGES, so `gen.receipt_photos align` can read them as if they were the photo's own.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def cmd_rotate(a: argparse.Namespace) -> None:
    ids = set(Path(a.ids).read_text().split()) if a.ids else None
    for ang in (int(v) for v in a.angles.split(",")):
        d = Path(a.out) / f"a{ang}"
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(a.images).glob("*.jpg")):
            if ids is not None and f.stem not in ids:
                continue
            im = Image.open(f).convert("RGB")
            fill = tuple(int(v) for v in np.median(np.asarray(im).reshape(-1, 3), axis=0))
            im.rotate(ang, expand=True, resample=Image.BICUBIC, fillcolor=fill).save(d / f.name, quality=95)


def score(sidecar: Path) -> float:
    if not sidecar.exists():
        return 0.0
    words = json.loads(sidecar.read_text())["words"]
    return sum(w.get("conf", 0) * len(w["text"]) for w in words if w.get("conf", 0) >= 0.9)


def cmd_pick(a: argparse.Namespace) -> None:
    engines = {int(k): Path(v) for k, v in (e.split("=", 1) for e in a.engine)}
    oe, oi = Path(a.out_engine), Path(a.out_images)
    oe.mkdir(parents=True, exist_ok=True)
    oi.mkdir(parents=True, exist_ok=True)
    hist: dict[int, int] = {}
    for f in sorted(Path(a.images).glob("*.jpg")):
        cand = {ang: score(d / f"{f.stem}.json") for ang, d in engines.items() if (d / f"{f.stem}.json").exists()}
        if not cand:
            continue
        best = max(cand, key=lambda k: cand[k])
        hist[best] = hist.get(best, 0) + 1
        for suffix in (".json", ".page.json", ".txt"):
            src = engines[best] / f"{f.stem}{suffix}"
            if src.exists():
                (oe / src.name).unlink(missing_ok=True)
                os.symlink(src, oe / src.name)
        img = f if best == 0 else Path(a.rotated) / f"a{best}" / f.name
        (oi / f.name).unlink(missing_ok=True)
        os.symlink(img, oi / f.name)
    print("chosen angle histogram", dict(sorted(hist.items())))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rotate")
    r.add_argument("--images", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--angles", default="-30,-20,-10,10,20,30")
    r.add_argument("--ids", help="file of photo ids (whitespace separated); default all")
    p = sub.add_parser("pick")
    p.add_argument("--engine", action="append", required=True, help="ANGLE=DIR of engine output on that rotation (0 = the photo as is)")
    p.add_argument("--images", required=True)
    p.add_argument("--rotated", required=True)
    p.add_argument("--out-engine", required=True)
    p.add_argument("--out-images", required=True)
    a = ap.parse_args()
    {"rotate": cmd_rotate, "pick": cmd_pick}[a.cmd](a)


if __name__ == "__main__":
    main()
