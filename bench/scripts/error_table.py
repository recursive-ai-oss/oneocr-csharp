"""Every content error of one or more engines on a set, under the scorer the gate reads, sorted into causes.

    uv run python scripts/error_table.py --truth SET/truth [--scorer v3.1] [--list] NAME=DIR [NAME=DIR ...]

DIR holds <id>.txt per page (black-box output for competitors). The comparison is the scorer's own: identical words cancel, the leftover
truth words and leftover hypothesis words are paired closest first (`content_errors`). Each leftover is then put in one cause:

  missed          a truth word with no hypothesis leftover
  extra           a hypothesis leftover with no truth word
  merged          a hypothesis leftover that is two adjacent truth leftovers written together (a space not read)
  split           two adjacent hypothesis leftovers that together are one truth leftover (a space inserted)
  misread:<kind>  a paired word, by what the edit is: edge deletion (first or last letter lost), inner deletion, insertion,
                  case, punctuation/symbol, look-alike letter substitution, other substitution

Numbers are characters of the scorer's own CER numerator (and word errors for WER), in percentage points of the set's truth
characters/words. With --list every single error is printed (page, truth, read), for engines you want to read line by line.
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import normalize, truth_text

LOOKALIKE = {frozenset(p) for p in ["l1", "lI", "1I", "O0", "oO", "o0", "cC", "sS", "vV", "wW", "xX", "zZ", "uU", "pP", "il", "ce", "no", "hb", "bo"]}


def pair_leftovers(ref: list[str], hyp: list[str]):
    rc, hc = Counter(ref), Counter(hyp)
    r_left = list((rc - hc).elements())
    h_left = list((hc - rc).elements())
    pairs = []
    used_r: set[int] = set()
    used_h: set[int] = set()
    if r_left and h_left:
        cand = sorted((Levenshtein.distance(r_left[i], h_left[j]), i, j) for i in range(len(r_left)) for j in range(len(h_left)))
        for d, i, j in cand:
            if i in used_r or j in used_h:
                continue
            used_r.add(i)
            used_h.add(j)
            pairs.append((r_left[i], h_left[j]))
            if len(used_r) == len(r_left) or len(used_h) == len(h_left):
                break
    missed = [w for k, w in enumerate(r_left) if k not in used_r]
    extra = [w for k, w in enumerate(h_left) if k not in used_h]
    return pairs, missed, extra


def kind_of(t: str, h: str) -> str:
    if t.casefold() == h.casefold():
        return "case"
    ops = [o for o in Levenshtein.opcodes(t, h) if o.tag != "equal"]
    if len(ops) == 1:
        o = ops[0]
        if o.tag == "delete":
            gone = t[o.src_start:o.src_end]
            if o.src_start == 0 or o.src_end == len(t):
                return "deletion at word edge (punct)" if all(not c.isalnum() for c in gone) else "deletion at word edge (letter)"
            return "deletion inside (punct)" if all(not c.isalnum() for c in gone) else "deletion inside (letter)"
        if o.tag == "insert":
            return "insertion (punct)" if all(not c.isalnum() for c in h[o.dest_start:o.dest_end]) else "insertion (letter)"
        a, b = t[o.src_start:o.src_end], h[o.dest_start:o.dest_end]
        if all(not c.isalnum() for c in a + b):
            return "substitution punct/symbol"
        if len(a) == 1 and len(b) == 1 and frozenset((a, b)) in LOOKALIKE:
            return "substitution look-alike"
        return "substitution letter"
    nonalnum_t = "".join(c for c in t if c.isalnum())
    nonalnum_h = "".join(c for c in h if c.isalnum())
    if nonalnum_t == nonalnum_h:
        return "punctuation only (several)"
    return "several edits"


def analyse(truth_dir: Path, hyp_dir: Path, scorer: str):
    rows = []  # (page, cause, truth, hyp, chars, words)
    n_chars = n_words = 0
    for t in sorted(truth_dir.glob("*.txt")):
        stem = t.stem
        ref = normalize(truth_text(truth_dir, stem, scorer), scorer == "v3.1")
        h = hyp_dir / f"{stem}.txt"
        hyp = normalize(h.read_text(encoding="utf-8", errors="replace") if h.exists() else "", scorer == "v3.1")
        n_chars += sum(map(len, ref))
        n_words += len(ref)
        pairs, missed, extra = pair_leftovers(ref, hyp)
        # merged / split among leftovers: a paired hypothesis word that is two truth leftovers together (or the reverse)
        # merged / split among leftovers: a paired hypothesis word that is two truth leftovers together (or the reverse)
        miss_set = Counter(missed)
        pend_pairs = list(pairs)
        used = set()
        for k, (tw, hw) in enumerate(pend_pairs):
            if k in used:
                continue
            for k2, (tw2, hw2) in enumerate(pend_pairs):
                if k2 == k or k2 in used:
                    continue
                # two truth words, one hyp word = merged (one of the pairs holds the fused word, the other is a miss)
            # fall through to the simple rules below
        # merged: a paired/extra hyp word equals the join of two truth leftovers; split: truth leftover equals the join of two hyp leftovers
        r_left = [w for p in pairs for w in (p[0],)] + missed
        h_left = [p[1] for p in pairs] + extra
        done_t: set[int] = set()
        done_h: set[int] = set()
        for j, hw in enumerate(h_left):
            for a in range(len(r_left)):
                if a in done_t:
                    continue
                for b in range(len(r_left)):
                    if b == a or b in done_t:
                        continue
                    if r_left[a] + r_left[b] == hw:
                        done_t.update({a, b})
                        done_h.add(j)
                        rows.append((stem, "merged", r_left[a] + " " + r_left[b], hw, Levenshtein.distance(r_left[a] + " " + r_left[b], hw), 2))
                        break
                if j in done_h:
                    break
        for a, tw in enumerate(r_left):
            if a in done_t:
                continue
            for j1, h1 in enumerate(h_left):
                if j1 in done_h:
                    continue
                for j2, h2 in enumerate(h_left):
                    if j2 == j1 or j2 in done_h:
                        continue
                    if h1 + h2 == tw:
                        done_t.add(a)
                        done_h.update({j1, j2})
                        rows.append((stem, "split", tw, h1 + " " + h2, 1, 1))
                        break
                if a in done_t:
                    break
        for i, (tw, hw) in enumerate(pairs):
            if i in done_t or i in done_h:
                continue
            d = min(Levenshtein.distance(tw, hw), max(len(tw), len(hw)))
            rows.append((stem, "misread:" + kind_of(tw, hw), tw, hw, d, 1))
        for i, tw in enumerate(missed):
            if len(pairs) + i in done_t:
                continue
            rows.append((stem, "missed", tw, "", len(tw), 1))
        for i, hw in enumerate(extra):
            if len(pairs) + i in done_h:
                continue
            rows.append((stem, "extra", "", hw, len(hw), 1))
    return rows, n_chars, n_words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--scorer", default="v3.1")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    truth = Path(a.truth)
    runs = [(r.split("=", 1)[0], Path(r.split("=", 1)[1])) for r in a.runs]
    results = {n: analyse(truth, d, a.scorer) for n, d in runs}
    causes = sorted({c for rows, _, _ in results.values() for _, c, *_ in rows})
    nc = next(iter(results.values()))[1]
    nw = next(iter(results.values()))[2]
    print(f"truth: {nw} words, {nc} chars, scorer {a.scorer}.  cells: CER-pp / WER-pp (count of items)")
    print("cause".ljust(44) + "".join(n.rjust(26) for n in results))
    tot = {n: [0, 0] for n in results}
    for c in causes:
        line = c.ljust(44)
        for n, (rows, _, _) in results.items():
            ch = sum(r[4] for r in rows if r[1] == c)
            wd = sum(r[5] for r in rows if r[1] == c)
            cnt = sum(1 for r in rows if r[1] == c)
            tot[n][0] += ch
            tot[n][1] += wd
            line += f"{100 * ch / nc:8.3f}/{100 * wd / nw:6.3f} ({cnt:3d})".rjust(26)
        print(line)
    print("total".ljust(44) + "".join(f"{100 * tot[n][0] / nc:8.3f}/{100 * tot[n][1] / nw:6.3f}".rjust(26) for n in results))
    if a.list:
        for n, (rows, _, _) in results.items():
            print(f"\n--- {n}")
            for stem, c, tw, hw, ch, wd in rows:
                print(f"{stem} {c:34s} {tw!r} -> {hw!r}")


if __name__ == "__main__":
    main()
