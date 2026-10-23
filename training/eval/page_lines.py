"""Line records for evaluation crops cut from page images by their truth word boxes.

Usage: uv run python -m eval.page_lines PAGES_DIR TRUTH_DIR OUT.jsonl

PAGES_DIR/<id>.<png|jpg> and TRUTH_DIR/<id>.json (words with text and [x0, y0, x1, y1] boxes).
Words are grouped into lines by geometry: words whose boxes are centred at nearly the same height (within 40% of the box
height) and follow one another left to right without a wide gap form a line. The line text is its words joined by single spaces; the
line box is the union of the word boxes. A word wrapped at a hyphen has one box over two lines; those two lines are left out.
With the word `words` as fourth argument every word is its own line (for tilted photographs).
`LineGen --page-lines` turns the records into crops. Evaluation only: nothing here feeds training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


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


def main() -> None:
    pages, truth, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    words_only = sys.argv[4:] == ["words"]
    rows, dropped = [], 0
    for tp in sorted(truth.glob("*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        page = next(pages.glob(t["id"] + ".*"))
        if words_only:
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
