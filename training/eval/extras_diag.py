# ruff: noqa: E501
"""Recognized words that sit on no truth word ("extras"), for one or more engine outputs on the same set, by size and confidence.

Each output directory holds the engine's `<id>.json` sidecars (word boxes, text, calibrated confidence) for the pages of one set; the
truth directory holds the set's `<id>.json` with ink boxes. A word is an extra when under 30 % of every truth word's area is covered
by it. The table says how many extras there are, how many characters they add to the error, how small and how unsure they are, and
how much of the extra count a size or confidence rule would remove without touching a word that has truth under it.

    python extras_diag.py --truth TRUTH_DIR --out NAME=DIR [--out NAME=DIR ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def box(w: dict) -> list[float]:
    return w.get("ink") or w.get("box") or [min(w["quad"][0::2]), min(w["quad"][1::2]), max(w["quad"][0::2]), max(w["quad"][1::2])]


def covered(tb: np.ndarray, hb: list[float]) -> bool:
    iw = np.clip(np.minimum(tb[:, 2], hb[2]) - np.maximum(tb[:, 0], hb[0]), 0, None)
    ih = np.clip(np.minimum(tb[:, 3], hb[3]) - np.maximum(tb[:, 1], hb[1]), 0, None)
    area = np.maximum((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1]), 1e-9)
    return bool(((iw * ih) / area > 0.3).any())


def collect(truth: Path, out: Path) -> tuple[list[dict], list[dict], int]:
    legit, extra, truth_chars = [], [], 0
    for tf in sorted(truth.glob("*.json")):
        t = json.loads(tf.read_text())
        tb = np.array([box(w) for w in t["words"]], np.float32).reshape(-1, 4)
        truth_chars += sum(len(w["text"]) for w in t["words"])
        of = out / tf.name
        if not of.exists():
            continue
        for w in json.loads(of.read_text())["words"]:
            b = box(w)
            r = {"h": b[3] - b[1], "w": b[2] - b[0], "conf": float(w.get("conf", 0.0)), "chars": len(w.get("text", "")), "page": tf.stem}
            (legit if covered(tb, b) else extra).append(r)
    return legit, extra, truth_chars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", action="append", required=True, help="NAME=DIR")
    a = ap.parse_args()
    for spec in a.out:
        name, d = spec.split("=", 1)
        legit, extra, tc = collect(Path(a.truth), Path(d))
        ec = sum(r["chars"] for r in extra)
        print(f"== {name}: words with truth {len(legit)}, extras {len(extra)} ({ec} chars = {100 * ec / tc:.2f}% of truth chars)")
        if not extra:
            continue
        for label, key in (("height px", "h"), ("width px", "w"), ("confidence", "conf")):
            for grp, rows in (("extras", extra), ("legit", legit)):
                v = np.array([r[key] for r in rows])
                print(f"  {label:10s} {grp:6s} p05/25/50/75/95 " + " / ".join(f"{np.percentile(v, p):.2f}" for p in (5, 25, 50, 75, 95)))
        print("  extras by height (px):", {f"<{t}": sum(r["h"] < t for r in extra) for t in (4, 8, 12, 16, 20)})
        print("  extras by confidence:", {f"<{t}": sum(r["conf"] < t for r in extra) for t in (0.1, 0.3, 0.5, 0.8, 0.95)})
        by_page: dict[str, int] = {}
        for r in extra:
            by_page[r["page"]] = by_page.get(r["page"], 0) + 1
        top = sorted(by_page.items(), key=lambda kv: -kv[1])[:6]
        print("  pages with most extras:", ", ".join(f"{k} {v}" for k, v in top), f"({len(by_page)} pages have any)")
        # rules a post-filter could apply: what they remove of the extras, and what legitimate words they would cost
        for h, c in ((8, 0.5), (12, 0.5), (12, 0.8), (16, 0.5), (16, 0.8)):
            rm = sum(r["h"] < h and r["conf"] < c for r in extra)
            cost = sum(r["h"] < h and r["conf"] < c for r in legit)
            print(f"  rule height<{h} and conf<{c}: removes {rm}/{len(extra)} extras, {cost}/{len(legit)} legit words")


if __name__ == "__main__":
    main()
