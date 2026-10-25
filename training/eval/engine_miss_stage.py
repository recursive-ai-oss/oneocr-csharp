"""Why an engine output misses a truth word: nothing near it, a box that covers part of it, or a box that read something else.

For every truth word with no recognised box covering `--min-cover` of it, looks at the recognised boxes that overlap it at all and
reports counts by case plus a few examples (truth text, what overlapping boxes read, their confidence). Truth and hypothesis are
in the benchmark layout: `<id>.json` with `words` (box or quad, text, optional conf).

    python -m eval.engine_miss_stage TRUTH_DIR HYP_DIR [--min-cover 0.3] [--examples 12]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from eval.missed_words import box
from eval.receipt_recall import covered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("hyp")
    ap.add_argument("--min-cover", type=float, default=0.3)
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()
    cases: collections.Counter = collections.Counter()
    cases1: collections.Counter = collections.Counter()
    ex = []
    for tf in sorted(Path(a.truth).glob("*.json")):
        tw = json.loads(tf.read_text())["words"]
        hf = Path(a.hyp) / tf.name
        hw = json.loads(hf.read_text())["words"] if hf.exists() else []
        tb = np.array([box(w) for w in tw], np.float32).reshape(-1, 4)
        hb = np.array([box(w) for w in hw], np.float32).reshape(-1, 4)
        cov = covered(tb, hb)
        for i, w in enumerate(tw):
            if cov[i] >= a.min_cover:
                continue
            one = len(w["text"].strip()) == 1
            # any recognised box touching the word, by the share of that box lying inside the word
            near = []
            for j, h in enumerate(hw):
                ix = max(0.0, min(tb[i, 2], hb[j, 2]) - max(tb[i, 0], hb[j, 0]))
                iy = max(0.0, min(tb[i, 3], hb[j, 3]) - max(tb[i, 1], hb[j, 1]))
                if ix * iy > 0:
                    near.append((h.get("text", ""), round(float(h.get("conf", 1.0)), 2)))
            c = "no recognised box touches it" if not near else "a box touches it but covers less than the share"
            cases[c] += 1
            cases1[c] += one
            if len(ex) < a.examples and one and near:
                ex.append((tf.stem, w["text"], near[:3], round(float(cov[i]), 2)))
    print("missed words by case:", dict(cases))
    print("  of which single characters:", dict(cases1))
    for e in ex:
        print("  example", e)


if __name__ == "__main__":
    main()
