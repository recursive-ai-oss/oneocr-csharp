"""Counts what scorer v3.1's quote fold changes: per set, the pages and the words (truth and hypothesis) whose normalised form differs.

    uv run python scripts/quote_fold_diff.py TRUTH_ROOT HYP_ROOT SET...

TRUTH_ROOT holds <set>/truth, HYP_ROOT holds <set>/<id>.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oneocr_bench.score import normalize, truth_text


def main(truth_root: str, hyp_root: str, sets: list[str]) -> None:
    for s in sets:
        td, hd = Path(truth_root) / s / "truth", Path(hyp_root) / s
        pages = pw_t = pw_h = tot_t = tot_h = 0
        for t in sorted(td.glob("*.txt")):
            truth = truth_text(td, t.stem, "v3")
            h = hd / t.name
            hyp = h.read_text(encoding="utf-8", errors="replace") if h.exists() else ""
            a, b = normalize(truth), normalize(truth, True)
            c, d = normalize(hyp), normalize(hyp, True)
            dt = sum(1 for w in a if "''" in w)
            dh = sum(1 for w in c if "''" in w)
            tot_t += len(a)
            tot_h += len(c)
            pw_t += dt
            pw_h += dh
            pages += (a != b) or (c != d)
        print(f"{s}: pages changed {pages}; truth words changed {pw_t} of {tot_t}; hypothesis words changed {pw_h} of {tot_h}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
