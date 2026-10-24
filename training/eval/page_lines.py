"""Line records for evaluation crops cut from page images by their truth word boxes.

Usage: uv run python -m eval.page_lines PAGES_DIR TRUTH_DIR OUT.jsonl

PAGES_DIR/<id>.<png|jpg> and TRUTH_DIR/<id>.json (words with text and [x0, y0, x1, y1] boxes).
Words are grouped into lines by geometry: words whose boxes are centred at nearly the same height (within 40% of the box
height) and follow one another left to right without a wide gap form a line. The line text is its words joined by single spaces; the
line box is the union of the word boxes. A word wrapped at a hyphen has one box over two lines; those two lines are left out.
Fourth argument: `words` makes every word its own axis-aligned line crop; `deskew` turns each page back by its skew (median top-edge
angle of the word quads) before grouping lines, as the library will before it cuts lines; `rectwords` rectifies every word quad
into an upright patch (photographs: perspective and tilt), and the crop convention then runs inside the patch.
`LineGen --page-lines` turns the records into crops. Evaluation only: nothing here feeds training.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def group_lines(words: list[dict]) -> list[list[dict]]:
    """Words sorted by vertical centre are cut into lines where the centre jumps by more than 40% of the word height from the
    line's running mean (a page skewed by half a degree drifts by a few pixels along a line, far less than that), then each line
    is ordered left to right and split where two words are more than 2.5 heights apart (column gutters, price columns)."""
    clusters: list[list[dict]] = []
    means: list[float] = []
    for w in sorted(words, key=lambda w: w["box"][1] + w["box"][3]):
        cy, h = (w["box"][1] + w["box"][3]) / 2, w["box"][3] - w["box"][1]
        if clusters and abs(cy - means[-1]) <= 0.4 * h:
            clusters[-1].append(w)
            means[-1] += (cy - means[-1]) / len(clusters[-1])
        else:
            clusters.append([w])
            means.append(cy)
    lines: list[list[dict]] = []
    for c in clusters:
        c.sort(key=lambda w: w["box"][0])
        cur = [c[0]]
        for w in c[1:]:
            if w["box"][0] - cur[-1]["box"][2] > 2.5 * (w["box"][3] - w["box"][1]):
                lines.append(cur)
                cur = []
            cur.append(w)
        lines.append(cur)
    return lines


def drop_wrapped(words: list[dict], lines: list[list[dict]]) -> tuple[list[list[dict]], int]:
    """A word broken across two lines at a hyphen has one box covering both lines, so the line holding it and the line that
    continues it do not match their text. Every line whose box meets such a word's box is left out; returns the remaining lines
    and how many were dropped."""
    hs = sorted(w["box"][3] - w["box"][1] for w in words)
    median = hs[len(hs) // 2]
    tall = [w["box"] for w in words if w["box"][3] - w["box"][1] > 1.6 * median]

    def hit(ln: list[dict]) -> bool:
        x0, y0 = min(w["box"][0] for w in ln), min(w["box"][1] for w in ln)
        x1, y1 = max(w["box"][2] for w in ln), max(w["box"][3] for w in ln)
        return any(x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1 for b in tall)

    keep = [ln for ln in lines if not hit(ln)]
    return keep, len(lines) - len(keep)


def page_skew(words: list[dict]) -> float:
    """Median angle (radians) of the top edges of the word quads: the page's skew."""
    return float(np.median([math.atan2(w["quad"][3] - w["quad"][1], w["quad"][2] - w["quad"][0]) for w in words]))


def deskewed_words(words: list[dict], theta: float) -> list[dict]:
    """Words with axis-aligned boxes in the frame where the page is turned back by theta (about the page origin)."""
    c, s = math.cos(theta), math.sin(theta)
    out = []
    for w in words:
        q = w["quad"]
        pts = [(q[i] * c + q[i + 1] * s, -q[i] * s + q[i + 1] * c) for i in (0, 2, 4, 6)]
        us, vs = [p[0] for p in pts], [p[1] for p in pts]
        out.append({"text": w["text"], "box": [min(us), min(vs), max(us), max(vs)]})
    return out


# Intermediate images are written uncompressed (compress_level=0): the crop tool's PNG reader rejects some deflate streams the usual
# encoder writes ("Too many length or distance codes").
def turn_page(page: Path, theta: float, dst: Path) -> None:
    """Writes the page turned back by theta: output (u, v) samples the page at R(theta)(u, v), bicubic, paper-coloured fill."""
    with Image.open(page) as im:
        g = im.convert("L")
    fill = int(np.median(np.asarray(g)))
    c, s = math.cos(theta), math.sin(theta)
    g.transform(g.size, Image.AFFINE, (c, -s, 0, s, c, 0), Image.BICUBIC, fillcolor=fill).save(dst, compress_level=0)


def word_patch(g: Image.Image, quad: list[float], dst: Path) -> list[float]:
    """Rectifies one word quad (TL, TR, BR, BL) into an upright patch with a margin of 0.7 word heights all around; returns the
    word's box inside the patch."""
    tl, tr, br, bl = [np.array(quad[i : i + 2], dtype=float) for i in (0, 2, 4, 6)]
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    ex, ey = (tr - tl) / max(np.linalg.norm(tr - tl), 1e-6), (bl - tl) / max(np.linalg.norm(bl - tl), 1e-6)
    m = 0.7 * h
    ow, oh = int(math.ceil(w + 2 * m)), int(math.ceil(h + 2 * m))
    o = tl - m * ex - m * ey
    corners = [o, o + oh * ey, o + ow * ex + oh * ey, o + ow * ex]  # source corners for output (0,0), (0,H), (W,H), (W,0)
    g.transform((ow, oh), Image.QUAD, tuple(float(v) for p in corners for v in p), Image.BICUBIC).save(dst, compress_level=0)
    return [m, m, m + w, m + h]


def main() -> None:
    pages, truth, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else "lines"
    words_only = mode == "words"
    rows, dropped = [], 0
    work = out.parent / "rect-pages"
    if mode in ("deskew", "rectwords"):
        work.mkdir(parents=True, exist_ok=True)
    for tp in sorted(truth.glob("*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        page = next(pages.glob(t["id"] + ".*"))
        if mode == "rectwords":
            hs = sorted(w["box"][3] - w["box"][1] for w in t["words"])
            keep = [w for w in t["words"] if w["box"][3] - w["box"][1] <= 1.6 * hs[len(hs) // 2]]
            with Image.open(page) as im:
                g = im.convert("L")
            for i, w in enumerate(keep):
                patch = work / f"{t['id']}-l{i:03d}.png"
                box = word_patch(g, w["quad"], patch)
                rows.append({"id": f"{t['id']}-l{i:03d}", "text": w["text"], "page": str(patch), "box": [round(v, 2) for v in box]})
            continue
        if mode == "deskew":
            theta = page_skew(t["words"])
            dst = work / f"{t['id']}.png"
            turn_page(page, theta, dst)
            page, words = dst, deskewed_words(t["words"], theta)
            lines = group_lines(words)
            lines, n = drop_wrapped(words, lines)
            dropped += n
        elif words_only:
            # photographed pages are tilted by more than a line's height over its length, so an axis-aligned line crop would
            # cut through the neighbouring lines; each word is cropped on its own instead
            hs = sorted(w["box"][3] - w["box"][1] for w in t["words"])
            lines = [[w] for w in t["words"] if w["box"][3] - w["box"][1] <= 1.6 * hs[len(hs) // 2]]  # not a word wrapped at a hyphen
        else:
            lines = group_lines(t["words"])
            lines, n = drop_wrapped(t["words"], lines)
            dropped += n
        for i, ln in enumerate(lines):
            box = [min(w["box"][0] for w in ln), min(w["box"][1] for w in ln), max(w["box"][2] for w in ln), max(w["box"][3] for w in ln)]
            rows.append({"id": f"{t['id']}-l{i:03d}", "text": " ".join(w["text"] for w in ln), "page": str(page),
                         "box": [round(v, 2) for v in box]})  # fmt: skip
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} lines from {len(list(truth.glob('*.json')))} pages -> {out} ({dropped} lines around hyphen-wrapped words left out)")


if __name__ == "__main__":
    main()
