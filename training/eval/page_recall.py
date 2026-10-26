"""Recall-CER / recall-WER (errors on truth words only), plus plain CER / WER and order, of page readings against truth.

    uv run python -m eval.page_recall --truth DIR --hyp DIR [--hyp DIR ...]

TRUTH/<id>.json carries "text" (or TRUTH/<id>.txt is the text); HYP/<id>.txt is the reading. Uses the benchmark scorer's page function, so the numbers are the ones its
`score` command reports (receipts have partial truth, so recall is the judged pair there).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from oneocr_bench.score import score_page  # noqa: E402

KEYS = ["char_errors", "word_errors", "order_errors", "ref_chars", "ref_words", "char_misses", "word_misses"]


def totals(truth: Path, hyp: Path) -> dict:
    a = [0] * 7
    n = 0
    for tf in sorted(truth.glob("*.json")) or sorted(truth.glob("*.txt")):
        text = json.loads(tf.read_text(encoding="utf-8"))["text"] if tf.suffix == ".json" else tf.read_text(encoding="utf-8")
        h = hyp / (tf.stem + ".txt")
        r = score_page(text, h.read_text(encoding="utf-8") if h.exists() else "")
        for k, key in enumerate(KEYS):
            a[k] += r[key]
        n += 1
    return {"pages": n, "cer": 100 * a[0] / a[3], "wer": 100 * a[1] / a[4], "order": 100 * a[2] / a[4],
            "recall_cer": 100 * a[5] / a[3], "recall_wer": 100 * a[6] / a[4]}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--hyp", action="append", required=True)
    a = ap.parse_args()
    for h in a.hyp:
        print(h, json.dumps({k: round(v, 3) for k, v in totals(Path(a.truth), Path(h)).items()}))


if __name__ == "__main__":
    main()
