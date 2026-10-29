"""Check that a newer scorer reproduces an older one's CER, WER and reading-order numbers exactly.

    python scripts/compare_scorers.py OLD_SCORE_PY OLD_TRUTH_ROOT NEW_TRUTH_ROOT OUTPUT_ROOT [set ...]

OLD_SCORE_PY is the older `score.py` saved to a file (for example `git show bench-v1:bench/oneocr_bench/score.py`). The old scorer
runs on OLD_TRUTH_ROOT/<set>/truth, the scorer in this checkout on NEW_TRUTH_ROOT/<set>/truth, both on OUTPUT_ROOT/<set>
(`<id>.txt` and, where present, sidecars). Every content number, in total and per page, must be equal to the last bit; the text
truth files must be byte-identical. Exit status 1 on any difference.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oneocr_bench import score as new

SETS = ["clean300-v1", "degraded-v1", "screenshot-v1", "layout-v1", "phone-v1", "receipts-v1"]
KEYS = ["pages", "missing", "cer", "wer", "order_err", "cer_recall", "wer_recall", "totals", "per_page"]


def main() -> int:
    spec = importlib.util.spec_from_file_location("old_score", sys.argv[1])
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old_root, new_root, out_root = (Path(p) for p in sys.argv[2:5])
    names = sys.argv[5:] or SETS
    bad = 0
    print(f"{'set':14} {'pages':>5} {'CER %':>10} {'WER %':>10} {'order %':>10}  identical")
    for n in names:
        a = old.score_dir(old_root / n / "truth", out_root / n)
        b = new.score_dir(new_root / n / "truth", out_root / n)
        same = all(a[k] == b[k] for k in KEYS)
        same_txt = all(
            (old_root / n / "truth" / f.name).read_bytes() == f.read_bytes() for f in sorted((new_root / n / "truth").glob("*.txt"))
        ) and len(list((old_root / n / "truth").glob("*.txt"))) == len(list((new_root / n / "truth").glob("*.txt")))
        ok = same and same_txt
        bad += not ok
        verdict = "yes" if ok else f"NO (scores equal: {same}, truth text equal: {same_txt})"
        print(f"{n:14} {b['pages']:>5} {b['cer'] * 100:>10.6f} {b['wer'] * 100:>10.6f} {b['order_err'] * 100:>10.6f}  {verdict}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
