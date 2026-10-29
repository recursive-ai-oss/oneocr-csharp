"""Per-page reading-order errors of one or two output directories against one truth directory (scorer v3 truth text).

    uv run python scripts/order_pages.py --truth SET/truth A_DIR [B_DIR] [--only l016,l019]

Prints the page's order errors (word edit distance), words and content word errors for each directory, and the totals, so a change
to the page stage can be read page by page.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import score_page, truth_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--scorer", default="v3")
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    truth = Path(a.truth)
    only = set(filter(None, a.only.split(",")))
    totals = [[0, 0, 0] for _ in a.dirs]
    rows = []
    for t in sorted(truth.glob("*.json")):
        if only and t.stem not in only:
            continue
        txt = truth_text(truth, t.stem, a.scorer)
        per = []
        for i, d in enumerate(a.dirs):
            p = Path(d) / f"{t.stem}.txt"
            s = score_page(txt, p.read_text(encoding="utf-8") if p.exists() else "")
            per.append(s)
            totals[i][0] += s["order_errors"]
            totals[i][1] += s["ref_words"]
            totals[i][2] += s["word_errors"]
        rows.append((t.stem, per))
    for stem, per in rows:
        if len({s["order_errors"] for s in per}) > 1 or only:
            print(stem, "  ".join(f"order {s['order_errors']:4d}/{s['ref_words']:4d}" for s in per))
    for d, (o, w, e) in zip(a.dirs, totals):
        print(f"{d}: order {o}/{w} = {100.0 * o / max(1, w):.3f}%  word errors {e}")


if __name__ == "__main__":
    main()
