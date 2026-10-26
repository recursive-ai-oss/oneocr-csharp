"""CER / WER / order of one or more engine output roots over eval sets, side by side (the bench scorer's definitions).

    python eval/score_roots.py --eval EVAL_ROOT --sets a b c --roots ROOT1 ROOT2 ...

EVAL_ROOT/<set>/truth/<id>.json and ROOT/<set>/<id>.txt; sets starting with `receipts` are scored on recall-CER / recall-WER.
Prints `set | root1 cer/wer/order | root2 ...` in percent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from oneocr_bench.score import score_page  # noqa: E402


def totals(eval_root: Path, root: Path, name: str) -> tuple[float, float, float]:
    a = dict.fromkeys(("char_errors", "word_errors", "order_errors", "ref_chars", "ref_words", "char_misses", "word_misses"), 0)
    for tf in sorted((eval_root / name / "truth").glob("*.json")):
        hyp = root / name / f"{tf.stem}.txt"
        r = score_page(json.loads(tf.read_text())["text"], hyp.read_text() if hyp.exists() else "")
        for k in a:
            a[k] += r[k]
    rec = name.startswith("receipts")
    return (100 * a["char_misses" if rec else "char_errors"] / max(a["ref_chars"], 1), 100 * a["word_misses" if rec else "word_errors"] / max(a["ref_words"], 1), 100 * a["order_errors"] / max(a["ref_words"], 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--roots", nargs="+", required=True)
    a = ap.parse_args()
    print("set | " + " | ".join(Path(r).name for r in a.roots))
    for s in a.sets:
        print(s + " | " + " | ".join("%.3f/%.3f/%.2f" % totals(Path(a.eval), Path(r), s) for r in a.roots))


if __name__ == "__main__":
    main()
