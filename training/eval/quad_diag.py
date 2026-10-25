# ruff: noqa: E501, B905
"""How well does a text-line angle read off the detector's probability map match the true line angle?

For every connected component of the text map (one detected line), the angle is the principal axis of its cells; the true angle is a
least-squares line through the centres of the truth words under it. Compared with the single page skew the engine removes today.
This sizes a rotated-box / quad output for straight line crops without building it.

Usage: uv run python -m eval.quad_diag --checkpoint best.pt --bench DIR --engine-out DIR [--sets phone-v1,degraded-v1]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from eval.detect import raw_maps
from models.dbnet import DBNetLite
from train.detect_data import TEXT_STRIDE


def pct(a, ps=(50, 90, 99)):
    return {p: round(float(np.percentile(a, p)), 2) for p in ps} if len(a) else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--engine-out", required=True)
    ap.add_argument("--sets", default="phone-v1,degraded-v1")
    ap.add_argument("--thr", type=float, default=0.3)
    a = ap.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = DBNetLite(ck["config"])
    model.load_state_dict(ck["model"])
    model.eval()
    for s in a.sets.split(","):
        e_loc, e_glob, e_best = [], [], []
        n_lines = n_multi = 0
        for f in sorted((Path(a.bench) / s / "truth").glob("*.json")):
            t = json.loads(f.read_text())
            img = Image.open(next((Path(a.bench) / s / "pages").glob(f.stem + ".*")))
            m = raw_maps(model, img, "cpu")
            sc = m["scale"]
            skew = json.loads((Path(a.engine_out) / s / f"{f.stem}.page.json").read_text())["pages"][0].get("skew", 0.0)
            words = [w for w in t["words"] if w.get("ink_quad")]
            cx = np.array([np.mean(w["ink_quad"][0::2]) for w in words])
            cy = np.array([np.mean(w["ink_quad"][1::2]) for w in words])
            wang = np.array([math.degrees(math.atan2(w["ink_quad"][3] - w["ink_quad"][1], w["ink_quad"][2] - w["ink_quad"][0])) for w in words])
            med = float(np.median(wang))
            sgn = 1.0 if np.median(abs(wang - skew)) <= np.median(abs(wang + skew)) else -1.0
            lab, n = ndimage.label(m["prob"] > a.thr)
            for k in range(1, n + 1):
                ys, xs = np.nonzero(lab == k)
                if len(xs) < 40 or np.ptp(xs) < 8 * max(np.ptp(ys), 1):  # long thin components only (a line, not a speck or a block)
                    continue
                X, Y = xs * TEXT_STRIDE / sc, ys * TEXT_STRIDE / sc
                c = np.cov(np.vstack([X, Y]))
                w_, v_ = np.linalg.eigh(c)
                vx, vy = v_[:, 1]
                th = math.degrees(math.atan2(vy, vx))
                th = (th + 90) % 180 - 90
                # truth words whose centre is inside the component (dilated by one cell)
                ix = np.clip((cx * sc / TEXT_STRIDE).astype(int), 0, lab.shape[1] - 1)
                iy = np.clip((cy * sc / TEXT_STRIDE).astype(int), 0, lab.shape[0] - 1)
                sel = lab[iy, ix] == k
                if sel.sum() < 4 or np.ptp(cx[sel]) < 150:
                    continue
                n_lines += 1
                if np.ptp(cy[sel]) > 1.2 * np.median([w["ink_quad"][5] - w["ink_quad"][3] for w, s_ in zip(words, sel) if s_]) + 0.2 * np.ptp(cx[sel]):
                    n_multi += 1  # several truth lines under one component
                    continue
                slope = np.polyfit(cx[sel], cy[sel], 1)[0]
                tt = math.degrees(math.atan(slope))
                e_loc.append(abs(th - tt))
                e_glob.append(abs(sgn * skew - tt))
                e_best.append(abs(med - tt))
        print(f"== {s}: {len(e_loc)} single lines compared ({n_multi} multi-line components skipped, {n_lines} candidates)")
        for name, e in (("detector map angle", e_loc), ("page skew removed today", e_glob), ("best single page angle (oracle median)", e_best)):
            e = np.array(e)
            print(f"  {name:40s} |err| deg {pct(e)}  <0.3: {100 * (e < 0.3).mean():.0f}%  <0.5: {100 * (e < 0.5).mean():.0f}%  <1: {100 * (e < 1).mean():.0f}%")


if __name__ == "__main__":
    main()
