"""Truth words with no recognised word box covering enough of them, from engine outputs and truth in the benchmark layout.

The page cause analysis calls such a word missed at 30% coverage. This does the same count for sets whose truth has only word boxes
(CORD's partial truth), split by single-character words, for comparing post-processing variants of one detector.

    python -m eval.missed_words TRUTH_DIR HYP_DIR [--min-cover 0.3]       # <id>.json truth and <id>.json hypothesis words with boxes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.receipt_recall import covered


def box(w: dict) -> list[float]:
    if "box" in w:
        return [float(v) for v in w["box"]]
    q = w["quad"]
    return [min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("hyp")
    ap.add_argument("--min-cover", type=float, default=0.3)
    a = ap.parse_args()
    n = miss = n1 = miss1 = 0
    for tf in sorted(Path(a.truth).glob("*.json")):
        tw = json.loads(tf.read_text())["words"]
        hf = Path(a.hyp) / tf.name
        hw = json.loads(hf.read_text())["words"] if hf.exists() else []
        tb = np.array([box(w) for w in tw], np.float32).reshape(-1, 4)
        hb = np.array([box(w) for w in hw], np.float32).reshape(-1, 4)
        lost = covered(tb, hb) < a.min_cover
        one = np.array([len(w["text"].strip()) == 1 for w in tw], bool)
        n, miss, n1, miss1 = n + len(tw), miss + int(lost.sum()), n1 + int(one.sum()), miss1 + int((lost & one).sum())
    print(f"missed {miss}/{n} words, single-character {miss1}/{n1} (coverage < {a.min_cover})")


if __name__ == "__main__":
    main()
