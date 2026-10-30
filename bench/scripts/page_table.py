"""Per-page share of each engine's errors on one set, under the scorer the gate reads.

    uv run python scripts/page_table.py --truth SET/truth [--scorer v3.1] NAME=DIR [NAME=DIR ...]

The first engine is the subject. For every page the table gives each engine's character and word errors as percentage points of the
whole set's truth (so the rows add up to the set's CER and WER), then the subject's excess over the best of the others on the same
page, largest first: these are the pages that carry a loss. DIR holds <id>.txt per page.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import score_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--scorer", default="v3.1")
    ap.add_argument("engines", nargs="+")
    a = ap.parse_args()
    truth = Path(a.truth)
    names, results = [], []
    for spec in a.engines:
        name, d = spec.split("=", 1)
        names.append(name)
        results.append(score_dir(truth, Path(d), scorer=a.scorer))
    tot = results[0]["totals"]
    chars, words = tot["ref_chars"], tot["ref_words"]
    pages = sorted(results[0]["per_page"])
    rows = []
    for p in pages:
        cer = [r["per_page"][p]["char_errors"] * 100 / chars for r in results]
        wer = [r["per_page"][p]["word_errors"] * 100 / words for r in results]
        best = min(cer[1:]) if len(cer) > 1 else 0.0
        rows.append((cer[0] - best, p, cer, wer, results[0]["per_page"][p]["ref_words"]))
    rows.sort(reverse=True)
    head = "page  words " + " ".join(f"{n + ' CER/WER':>18}" for n in names) + "   excess CER vs best other"
    print(head)
    for excess, p, cer, wer, n in rows:
        print(f"{p:<6}{n:>5} " + " ".join(f"{c:>8.3f}/{w:<8.3f}" for c, w in zip(cer, wer)) + f"   {excess:+.3f}")
    print("total       " + " ".join(f"{r['cer'] * 100:>8.3f}/{r['wer'] * 100:<8.3f}" for r in results))


if __name__ == "__main__":
    main()
