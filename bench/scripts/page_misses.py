"""Missed characters (recall-CER numerator) per page for several outputs of one set, to see which pages a variant moves.

    uv run python scripts/page_misses.py --truth SET/truth [--scorer v3] [--top 20] NAME=DIR [NAME=DIR ...]

The first output is the base. Prints the totals per output, then the pages where the base misses most with every output's misses.
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
    ap.add_argument("--scorer", default="v3")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    truth = Path(a.truth)
    runs = [(r.split("=", 1)[0], Path(r.split("=", 1)[1])) for r in a.runs]
    rows: dict[str, list[int]] = {}
    chars = 0
    words = [0] * len(runs)
    for t in sorted(truth.glob("*.txt")):
        tt = truth_text(truth, t.stem, a.scorer)
        rows[t.stem] = []
        for k, (_, d) in enumerate(runs):
            h = d / f"{t.stem}.txt"
            s = score_page(tt, h.read_text(encoding="utf-8", errors="replace") if h.exists() else "", a.scorer == "v3.1")
            rows[t.stem].append(s["char_misses"])
            words[k] += s["word_misses"]
        chars += s["ref_chars"]
    print("run".ljust(14), "recall-CER   missed words")
    for k, (n, _) in enumerate(runs):
        print(n.ljust(14), f"{100 * sum(r[k] for r in rows.values()) / chars:8.3f}   {words[k]}")
    print("\npage".ljust(10) + "".join(n.rjust(12) for n, _ in runs))
    for p, r in sorted(rows.items(), key=lambda kv: -kv[1][0])[: a.top]:
        print(p.ljust(10) + "".join(str(v).rjust(12) for v in r))
    better = sum(1 for r in rows.values() if min(r[1:]) < r[0]) if len(runs) > 1 else 0
    print(f"\npages where some other run misses fewer characters than {runs[0][0]}: {better}")


if __name__ == "__main__":
    main()
