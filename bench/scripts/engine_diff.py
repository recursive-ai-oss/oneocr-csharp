"""Page-by-page and word-class comparison of our output with other engines' output on the same eval set (black-box text only).

    uv run python scripts/engine_diff.py --truth SET/truth --ours OUT/SET [--ref NAME=DIR ...] [--scorer v3] [--top 12] [--classes]

Per page and engine it prints CER, WER and order error under the chosen scorer (the numbers the gate reads, micro-averaged), the pages where
our order error is furthest above the best reference, and, with --classes, how many truth words (bag-of-words leftovers, as the scorer
counts them) each engine gets wrong by word class, split into words only we miss, only the reference misses, and both.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import normalize, score_page, truth_text  # noqa: E402


def word_class(w: str) -> str:
    if re.search(r"\d", w):
        return "has digit"
    if re.search(r"https?:|www\.|\.(gov|com|org)\b|/", w):
        return "url or slash"
    if re.search(r"[\"'\u2018\u2019\u201c\u201d]", w):
        return "quote mark"
    if re.fullmatch(r"\W+", w):
        return "punctuation only"
    if re.search(r"[-\u2010-\u2014_]", w):
        return "hyphen or underscore"
    core = re.sub(r"^\W+|\W+$", "", w)
    if len(core) <= 1:
        return "single letter"
    if core.isupper():
        return "ALL CAPS"
    if core[0].isupper():
        return "Capitalized"
    if re.search(r"^\W|\W$", w):
        return "lower + edge punctuation"
    return "plain lowercase"


def run(truth_dir: Path, hyp_dir: Path, scorer: str) -> dict:
    tot = Counter()
    pages = {}
    for t in sorted(truth_dir.glob("*.txt")):
        h = hyp_dir / t.name
        txt = truth_text(truth_dir, t.stem, scorer)
        hyp = h.read_text(encoding="utf-8") if h.exists() else ""
        s = score_page(txt, hyp)
        pages[t.stem] = (s, normalize(txt), normalize(hyp))
        for k, v in s.items():
            tot[k] += v
    return {"tot": tot, "pages": pages}


def rates(tot: Counter) -> str:
    return f"CER {100 * tot['char_errors'] / tot['ref_chars']:.2f}  WER {100 * tot['word_errors'] / tot['ref_words']:.2f}  order {100 * tot['order_errors'] / tot['ref_words']:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", action="append", default=[], help="NAME=DIR with <id>.txt")
    ap.add_argument("--scorer", default="v3")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--classes", action="store_true")
    a = ap.parse_args()
    truth = Path(a.truth)
    engines = {"ours": run(truth, Path(a.ours), a.scorer)}
    for r in a.ref:
        n, _, d = r.partition("=")
        engines[n] = run(truth, Path(d), a.scorer)
    print(f"scorer {a.scorer}, {len(engines['ours']['pages'])} pages")
    for n, e in engines.items():
        print(f"  {n:24} {rates(e['tot'])}")

    ours = engines["ours"]["pages"]
    refs = [n for n in engines if n != "ours"]
    if refs:
        print(f"\npages where our order error is furthest above the best reference (errors = word edit distance; share of the set's order errors):")
        total_o = engines["ours"]["tot"]["order_errors"]
        rows = []
        for pid, (s, _, _) in ours.items():
            best = min(engines[n]["pages"][pid][0]["order_errors"] for n in refs)
            rows.append((s["order_errors"] - best, pid, s["order_errors"], best, s["ref_words"], s["word_errors"]))
        rows.sort(reverse=True)
        gap = sum(max(r[0], 0) for r in rows)
        print(f"  sum of per-page excess over the best reference: {gap} errors = {100 * gap / engines['ours']['tot']['ref_words']:.2f} pp of order; our total {100 * total_o / engines['ours']['tot']['ref_words']:.2f}")
        for ex, pid, o, b, n, we in rows[: a.top]:
            print(f"  {pid}: ours {o} ({100 * o / n:.1f}%), best ref {b} ({100 * b / n:.1f}%), words {n}, our content word errors {we}")
        cum = 0
        print("  cumulative excess by page rank: " + ", ".join(f"top{k}={100 * sum(max(r[0], 0) for r in rows[:k]) / engines['ours']['tot']['ref_words']:.2f}pp" for k in (1, 3, 5, 10)))

    if a.classes and refs:
        for n in refs:
            miss = {"ours only": Counter(), "ref only": Counter(), "both": Counter(), "words": Counter()}
            for pid, (s, rw, hw) in ours.items():
                rc = Counter(rw)
                lo = rc - Counter(hw)
                lr = rc - Counter(engines[n]["pages"][pid][2])
                for w, c in rc.items():
                    miss["words"][word_class(w)] += c
                for w in set(lo) | set(lr):
                    cl = word_class(w)
                    a_, b_ = lo[w], lr[w]
                    both = min(a_, b_)
                    miss["both"][cl] += both
                    miss["ours only"][cl] += a_ - both
                    miss["ref only"][cl] += b_ - both
            print(f"\ntruth-word leftovers by class, ours vs {n} (counts; WER pp = count / {engines['ours']['tot']['ref_words']} truth words)")
            print(f"  {'class':26}{'truth':>8}{'ours only':>11}{'ref only':>10}{'both':>7}{'net pp (ours-ref)':>20}")
            nw = engines["ours"]["tot"]["ref_words"]
            for cl, _ in sorted(miss["words"].items(), key=lambda kv: -(miss["ours only"][kv[0]] - miss["ref only"][kv[0]])):
                o, r, b = miss["ours only"][cl], miss["ref only"][cl], miss["both"][cl]
                print(f"  {cl:26}{miss['words'][cl]:>8}{o:>11}{r:>10}{b:>7}{100 * (o - r) / nw:>20.3f}")


if __name__ == "__main__":
    main()
