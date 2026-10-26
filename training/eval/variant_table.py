# ruff: noqa: E501
"""Recall-CER and recall-WER of several engine output directories on one truth set, split by kind of truth word.

    cd bench && uv run python ../training/eval/variant_table.py TRUTH_DIR NAME=DIR [NAME=DIR ...]

Per variant: the scorer's CER, WER, recall-CER and recall-WER (the .txt files), and the recall-CER misses (characters counted by
the scorer's pairing, `recall_attribution.attribute_page` on the word sidecars) per kind of truth word: with a digit (prices,
codes, dates), letters only, other. Shows whether a decoding or word-joining switch changes item names or codes.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oneocr_bench.score import normalize, score_dir  # noqa: E402
from recall_attribution import attribute_page  # noqa: E402


def kind(t: str) -> str:
    if any(c.isdigit() for c in t):
        return "digit"
    return "alpha" if any(c.isalpha() for c in t) else "other"


def main() -> None:
    truth = Path(sys.argv[1])
    print("| variant | CER | WER | recall-CER | recall-WER | digit words (miss chars of) | alpha | other |\n|---|---:|---:|---:|---:|---|---|---|")
    for spec in sys.argv[2:]:
        name, d = spec.split("=", 1)
        r = score_dir(truth, Path(d))
        miss: collections.Counter = collections.Counter()
        tot: collections.Counter = collections.Counter()
        for tf in sorted(truth.glob("*.json")):
            tw = json.loads(tf.read_text())["words"]
            hf = Path(d) / tf.name
            hw = json.loads(hf.read_text())["words"] if hf.exists() else []
            for w in tw:
                for t in normalize(w["text"]):
                    tot[kind(t)] += len(t)
            for t, _w, m, _c, _tx, _p in attribute_page(tw, hw):
                miss[kind(t)] += m
        cell = lambda k: f"{miss[k]} / {tot[k]} = {100 * miss[k] / max(tot[k], 1):.2f}%"  # noqa: E731, B023
        print(f"| {name} | {100 * r['cer']:.2f} | {100 * r['wer']:.2f} | {100 * r['cer_recall']:.2f} | {100 * r['wer_recall']:.2f} | {cell('digit')} | {cell('alpha')} | {cell('other')} |")


if __name__ == "__main__":
    main()
