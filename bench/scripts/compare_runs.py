"""Before/after quality table for two sets of engine outputs over the same pages, under one scorer.

    uv run python scripts/compare_runs.py --scorer v3.1 --before NAME=ROOT --after NAME=ROOT \
        [--truth-root ROOT:SET,SET ...]  SET=TRUTH_DIR[:BEFORE_DIR:AFTER_DIR] ...

Each SET argument is `label=TRUTH_DIR,BEFORE_DIR,AFTER_DIR` (directories holding <id>.txt and, for ECE, <id>.json). Prints CER / WER / order
(percent) and ECE for each run, the change, the number of pages whose text differs, and for the recall sets the recall-CER and recall-WER.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import score_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="v3.1")
    ap.add_argument("--recall", default="", help="comma-separated labels whose truth is partial: also print recall-CER / recall-WER")
    ap.add_argument("sets", nargs="+", help="label=TRUTH_DIR,BEFORE_DIR,AFTER_DIR")
    a = ap.parse_args()
    recall = set(a.recall.split(","))
    print(f"{'set':<18}{'CER before':>11}{'after':>9}{'WER before':>11}{'after':>9}{'order before':>13}{'after':>9}{'ECE b':>8}{'ECE a':>8}{'pages changed':>15}")
    for spec in a.sets:
        label, dirs = spec.split("=", 1)
        t, b, n = (Path(d) for d in dirs.split(","))
        rb, rn = score_dir(t, b, scorer=a.scorer), score_dir(t, n, scorer=a.scorer)
        changed = sum(1 for p in t.glob("*.txt") if (b / p.name).read_text(errors="replace") != (n / p.name).read_text(errors="replace"))

        def pct(x: float) -> str:
            return f"{100 * x:.3f}"

        def ece(r: dict) -> str:
            e = r.get("ece")
            return "-" if not e else f"{e.get('ece_matched' if label in recall else 'ece', e.get('ece', 0)):.3f}"

        print(f"{label:<18}{pct(rb['cer']):>11}{pct(rn['cer']):>9}{pct(rb['wer']):>11}{pct(rn['wer']):>9}{pct(rb['order_err']):>13}{pct(rn['order_err']):>9}"
              f"{ece(rb):>8}{ece(rn):>8}{changed:>8} of {rb['pages']}")
        if label in recall:
            print(f"{'  recall':<18}{pct(rb['cer_recall']):>11}{pct(rn['cer_recall']):>9}{pct(rb['wer_recall']):>11}{pct(rn['wer_recall']):>9}")


if __name__ == "__main__":
    main()
