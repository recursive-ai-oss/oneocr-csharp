# ruff: noqa: E501
"""Pixel gap at the cut of words we split with a space: is there any real gap in the ink, or did the recognizer invent the space?

For every truth word of class split-space (page_causes.py JSON with `all_words`), takes the truth ink box, thresholds the page crop
(Otsu), and measures the widest empty column run within it in units of the word's height. The same is measured for the word gaps of the
page's real spaces (truth neighbours on one line) so the two can be compared. A split whose widest internal gap is as small as the
letter gaps is a space the network made up; one as wide as real word gaps is a detector/segmentation cut.

    python eval/split_gap.py CAUSES.json BENCH_ROOT [--sets ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def otsu(g: np.ndarray) -> float:
    h = np.bincount(g.ravel(), minlength=256).astype(float)
    p = h / h.sum()
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    s = (m[-1] * w - m) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return float(np.argmax(s))


def widest_gap(img: np.ndarray, b: list[float]) -> tuple[int, int]:
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    crop = img[max(y0, 0): y1, max(x0, 0): x1]
    if crop.size == 0 or crop.shape[1] < 3:
        return 0, 0
    t = otsu(crop)
    ink = (crop < t) if np.median(crop) > t else (crop > t)
    cols = ink.any(axis=0)
    idx = np.flatnonzero(cols)
    if len(idx) < 2:
        return 0, crop.shape[0]
    inner = cols[idx[0]: idx[-1] + 1]
    best = run = 0
    for v in inner:
        run = 0 if v else run + 1
        best = max(best, run)
    return best, crop.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("causes")
    ap.add_argument("bench")
    ap.add_argument("--sets", nargs="*")
    a = ap.parse_args()
    res = json.loads(Path(a.causes).read_text())
    for name in a.sets or res:
        gaps, word_gaps = [], []
        by_page: dict[str, list[int]] = {}
        for p, i, *_rest, c, _e, _hh in [(w[0], w[1], w[2], w[3], w[4], w[5], w[6]) for w in res[name]["all_words"]]:
            if c == "split-space":
                by_page.setdefault(p, []).append(i)
        for p, ids in by_page.items():
            t = json.loads((Path(a.bench) / name / "truth" / f"{p}.json").read_text())
            tw = [w for w in t["words"] if " ".join(w["text"].split())]
            img = None
            for ext in ("png", "jpg"):
                f = Path(a.bench) / name / "pages" / f"{p}.{ext}"
                if f.exists():
                    img = np.asarray(Image.open(f).convert("L"))
                    break
            if img is None:
                continue
            for i in ids:
                b = tw[i].get("ink") or tw[i]["box"]
                g, h = widest_gap(img, b)
                gaps.append(g / max(h, 1))
            # real word gaps on this page: neighbouring truth words on one line
            bs = [w.get("ink") or w["box"] for w in tw]
            for k in range(len(bs) - 1):
                a_, b_ = bs[k], bs[k + 1]
                if len(a_) == 4 and len(b_) == 4 and abs((a_[1] + a_[3]) - (b_[1] + b_[3])) < 0.5 * (a_[3] - a_[1]) and b_[0] > a_[2]:
                    word_gaps.append((b_[0] - a_[2]) / max(a_[3] - a_[1], 1))
                    if len(word_gaps) > 4000:
                        break
        if gaps:
            q = np.percentile(gaps, [10, 50, 90])
            wq = np.percentile(word_gaps, [10, 50, 90]) if word_gaps else [float("nan")] * 3
            print(f"{name}: {len(gaps)} split words; widest inner ink gap / height p10 {q[0]:.2f} p50 {q[1]:.2f} p90 {q[2]:.2f} | real word gaps p10 {wq[0]:.2f} p50 {wq[1]:.2f} p90 {wq[2]:.2f}")


if __name__ == "__main__":
    main()
