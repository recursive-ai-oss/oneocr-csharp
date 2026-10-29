"""Where CORD's one-character words sit in the detector's training lines (`train.detect_data.cord_lines` grouping).

A one-character word that is the first or last word of a longer line lies partly outside that line's shrunk core, and only a word of
its own gets a core of its own. Prints counts of single-character words by position in their line and, for words of their own, the
gap to the neighbouring word in line heights, on the official train split.

    uv run python -m eval.cord_single_stats --datasets ROOT [--split train]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--gap", type=float, default=1.2)
    a = ap.parse_args()
    ds = Path(a.datasets) / "cord" / "v2"
    ann = [json.loads(x) for x in (ds / "annotations" / f"{a.split}.jsonl").read_text().splitlines()]
    pos, own_gaps, sizes = collections.Counter(), [], collections.Counter()
    for r in ann:
        rows: dict[int, list] = {}
        for line in json.loads(r["ground_truth"])["valid_line"]:
            for w in line["words"]:
                if w["text"].strip():
                    q = w["quad"]
                    p = np.array([[q["x1"], q["y1"]], [q["x2"], q["y2"]], [q["x3"], q["y3"]], [q["x4"], q["y4"]]], np.float32)
                    rows.setdefault(w["row_id"], []).append((p, w["text"].strip(), line.get("category", "")))
        for words in rows.values():
            words.sort(key=lambda t: float(t[0][:, 0].mean()))
            groups, cur = [], [words[0]]
            for t in words[1:]:
                h = float(np.linalg.norm(cur[-1][0][3] - cur[-1][0][0]))
                gap = float(t[0][:, 0].min() - cur[-1][0][:, 0].max()) / max(h, 1.0)
                if gap > a.gap:
                    groups.append(cur)
                    cur = []
                cur.append(t)
            groups.append(cur)
            for g in groups:
                for i, (p, text, cat) in enumerate(g):
                    if len(text) != 1:
                        continue
                    where = "own line" if len(g) == 1 else "first of line" if i == 0 else "last of line" if i == len(g) - 1 else "middle"
                    pos[where] += 1
                    sizes[(where, cat.split(".")[-1])] += 1
                    if len(g) == 1:
                        own_gaps.append(float(np.linalg.norm(p[1] - p[0]) / max(np.linalg.norm(p[3] - p[0]), 1.0)))
    print("single-character words by position:", dict(pos))
    print("by position and CORD field:", {f"{k[0]}/{k[1]}": v for k, v in sizes.most_common(12)})
    if own_gaps:
        print("own-line width / height: median", round(float(np.median(own_gaps)), 2), "p10", round(float(np.percentile(own_gaps, 10)), 2))


if __name__ == "__main__":
    main()
