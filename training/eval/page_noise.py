# ruff: noqa: E501
"""Page-noise measures that a page path can compute cheaply, per page, for several page sets (to see which separates degraded pages).

Two families, both on the gray image at detector scale (long side 1024, the detector's own resize, unpadded area only):

  sigma    Immerkaer's noise estimate with a robust median: the 3x3 Laplacian-difference N = [[1,-2,1],[-2,4,-2],[1,-2,1]] is applied
           to every interior pixel, |I * N| is an integer 0..2040 and for white Gaussian noise of deviation s its median is 0.6745 * 6 s,
           so sigma = median / 4.047. Text edges raise only the tail, so a page that is mostly paper reads the paper. Exact on integers.
  junk     from the detector's text map (probability, stride 2): the number of 4-connected components of P > 0.3 with at most 12 cells
           and a mean probability under 0.5, per megapixel of detector input. These are the blobs that become one-glyph "lines".

    uv run python -m eval.page_noise --checkpoint CKPT --set NAME=PAGES_DIR [--set ...] [--out noise.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from eval.detect import prepare, raw_maps
from models.dbnet import DBNetLite

N = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.int32)


def sigma_median(gray: np.ndarray) -> float:
    g = gray.astype(np.int32)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    r = np.zeros((h - 2, w - 2), np.int32)
    for dy in range(3):
        for dx in range(3):
            r += N[dy, dx] * g[dy : dy + h - 2, dx : dx + w - 2]
    hist = np.bincount(np.abs(r).ravel(), minlength=2041)
    half = (hist.sum() + 1) // 2  # lower median of the integer values
    med = int(np.searchsorted(np.cumsum(hist), half))
    return med / 4.047


def junk_density(prob: np.ndarray, thr: float = 0.3, max_cells: int = 12, max_score: float = 0.5) -> float:
    lab, n = ndimage.label(prob > thr)
    if n == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    cnt = ndimage.sum(np.ones_like(prob), lab, idx)
    score = ndimage.mean(prob, lab, idx)
    junk = int(((cnt <= max_cells) & (cnt >= 2) & (score < max_score)).sum())
    mpx = prob.size * 4 / 1e6  # the map is stride 2
    return junk / mpx


def map_stats(prob: np.ndarray, thr: float = 0.3) -> dict:
    """Cheap statistics of the detector's text map (stride 2): what the text on the page looks like, not how noisy the paper is.

    comps/coverage/height/prob are over components of P > thr with at least 2 cells (the detector's lines): `comps` per megapixel of
    detector input, `coverage` the share of cells on, `height` the median short side in cells of the components with more than 12 cells
    (0 when none), `prob` the median of component mean probabilities.
    """
    lab, n = ndimage.label(prob > thr)
    mpx = prob.size * 4 / 1e6
    out = {"comps": 0.0, "coverage": float((prob > thr).mean()), "height": 0.0, "prob": 0.0, "nlines": 0.0}
    if n == 0:
        return out
    idx = np.arange(1, n + 1)
    cnt = np.asarray(ndimage.sum(np.ones_like(prob), lab, idx))
    score = np.asarray(ndimage.mean(prob, lab, idx))
    keep = cnt >= 2
    out["comps"] = float(keep.sum() / mpx)
    out["nlines"] = float((cnt > 12).sum())
    if keep.any():
        out["prob"] = float(np.median(score[keep]))
    big = [k + 1 for k in np.nonzero(cnt > 12)[0]]
    if big:
        objs = ndimage.find_objects(lab)
        out["height"] = float(np.median([min(objs[k - 1][0].stop - objs[k - 1][0].start, objs[k - 1][1].stop - objs[k - 1][1].start) for k in big]))
    return out


def gray_stats(g: np.ndarray) -> dict:
    """Paper level (median gray), ink contrast (median minus 2nd percentile) and the page's height / width."""
    return {"paper": float(np.median(g)), "contrast": float(np.median(g) - np.percentile(g, 2)), "aspect": g.shape[0] / g.shape[1]}


def unpadded_gray(img: Image.Image) -> np.ndarray:
    canvas, s = prepare(img)
    w, h = img.size
    return canvas[: max(1, round(h * s)), : max(1, round(w * s))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--set", action="append", required=True, help="NAME=PAGES_DIR")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = DBNetLite(ck["config"])
    model.load_state_dict(ck["model"])
    model = model.to(a.device).eval()
    res: dict[str, dict] = {}
    for spec in a.set:
        name, d = spec.split("=", 1)
        rows = {}
        for f in sorted(Path(d).iterdir()):
            if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img = Image.open(f)
            m = raw_maps(model, img, a.device)
            g = unpadded_gray(img.convert("L"))
            rows[f.stem] = {"sigma": sigma_median(g), "junk": junk_density(m["prob"]), **map_stats(m["prob"]), **gray_stats(g)}
        res[name] = rows
        for k in next(iter(rows.values())):
            v = np.array([r[k] for r in rows.values()])
            print(f"{name:16s} {k:6s} n={len(v)} min/p10/p50/p90/max " + " / ".join(f"{np.percentile(v, p):.2f}" for p in (0, 10, 50, 90, 100)), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
