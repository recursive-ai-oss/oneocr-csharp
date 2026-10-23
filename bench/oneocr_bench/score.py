"""Score OCR hypotheses against truth text: order-free CER/WER and reading-order error.

Normalisation (applied to truth and hypothesis alike):
  1. Unicode NFC;
  2. typographic quotes/dashes folded to ASCII (' " -) and the ligatures fi/fl split,
     so an engine is not charged for a font's choice of glyph;
  3. all whitespace runs collapse to one space; text is split into words on whitespace.
Case and punctuation are kept.

Order-free content metrics (cer, wer): words are treated as a bag. Words present in
both sides cancel exactly; each leftover truth word is paired with the closest leftover
hypothesis word (greedy, smallest edit distance first); unpaired leftovers are
deletions / insertions.
  wer = max(#leftover truth words, #leftover hyp words) / #truth words
  cer = (sum of edit distances of pairs + chars of unpaired words) / #truth chars (no spaces)
A word split or merged by the engine therefore costs a little more than in an ordered alignment.

Reading-order error (order_err): normalised word-level edit distance of the two word
sequences in order (substitutions + insertions + deletions) / #truth words. It also
contains recognition errors; on a page with few of them it is dominated by order.

    python -m oneocr_bench score --truth DIR --hyp DIR [--json OUT]
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-",
    "\u00a0": " ",
})


def normalize(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text).translate(FOLD).replace("\ufb01", "fi").replace("\ufb02", "fl")
    return t.split()


def content_errors(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """(char errors, word errors) of the bag-of-words comparison."""
    rc, hc = Counter(ref), Counter(hyp)
    r_left = list((rc - hc).elements())
    h_left = list((hc - rc).elements())
    word_err = max(len(r_left), len(h_left))
    char_err = 0
    if r_left and h_left:
        dist_fn = Levenshtein.distance
        pairs = sorted((dist_fn(r_left[i], h_left[j]), i, j) for i in range(len(r_left)) for j in range(len(h_left)))
        used_r: set[int] = set()
        used_h: set[int] = set()
        for dist, i, j in pairs:
            if i in used_r or j in used_h:
                continue
            used_r.add(i)
            used_h.add(j)
            char_err += min(dist, max(len(r_left[i]), len(h_left[j])))
            if len(used_r) == len(r_left) or len(used_h) == len(h_left):
                break
        char_err += sum(len(w) for k, w in enumerate(r_left) if k not in used_r)
        char_err += sum(len(w) for k, w in enumerate(h_left) if k not in used_h)
    else:
        char_err = sum(len(w) for w in r_left) + sum(len(w) for w in h_left)
    return char_err, word_err


def score_page(truth: str, hyp: str) -> dict:
    ref, hy = normalize(truth), normalize(hyp)
    ce, we = content_errors(ref, hy)
    return {
        "ref_words": len(ref), "ref_chars": sum(map(len, ref)),
        "char_errors": ce, "word_errors": we,
        "order_errors": Levenshtein.distance(ref, hy),
    }


def score_dir(truth_dir: Path, hyp_dir: Path) -> dict:
    """Micro-averaged over pages; a missing hypothesis file counts as empty output."""
    pages = {}
    for t in sorted(truth_dir.glob("*.txt")):
        h = hyp_dir / t.name
        text = h.read_text(encoding="utf-8", errors="replace") if h.exists() else ""
        pages[t.stem] = score_page(t.read_text(encoding="utf-8"), text)
    tot = {k: sum(p[k] for p in pages.values()) for k in next(iter(pages.values()))}
    return {
        "pages": len(pages),
        "missing": sum(not (hyp_dir / f"{p}.txt").exists() for p in pages),
        "cer": tot["char_errors"] / tot["ref_chars"],
        "wer": tot["word_errors"] / tot["ref_words"],
        "order_err": tot["order_errors"] / tot["ref_words"],
        "totals": tot,
        "per_page": pages,
    }


def main(truth: str, hyp: str, out: str | None) -> None:
    r = score_dir(Path(truth), Path(hyp))
    print(f"pages {r['pages']} (missing {r['missing']})  CER {r['cer']:.4%}  WER {r['wer']:.4%}  order {r['order_err']:.4%}")
    if out:
        Path(out).write_text(json.dumps(r, indent=1) + "\n")
