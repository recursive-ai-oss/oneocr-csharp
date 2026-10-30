"""Where the recall-CER / recall-WER of a receipt set comes from, word by word, for a partial truth (text lines, no boxes).

    uv run python scripts/receipt_causes.py --truth SET/truth --ours OUT/SET \
        [--ref NAME=DIR] [--alt LABEL=DIR ...] [--scorer v3.1] [--examples 6] [--pages 12]

The scorer's pairing is repeated exactly (identical words cancel; the leftover truth words are paired with the leftover
hypothesis words, closest first; a truth word counts min(edit distance, its length) characters, or all of them when unpaired),
and each leftover truth word's counted characters are put in one cause:

  <alt label>   the word is read exactly in an alternative run of our engine (LABEL=DIR, in the order given): a stage of the
                page path (noise filter, angle search, ...) turned a right word into a wrong one in the base run
  merged        the paired hypothesis word contains the word glued to something else
  split         two or three consecutive hypothesis words join to (nearly) the word
  undetected    nothing in our output resembles the word and the reference engine reads it in a box that none of our boxes touch
  garbled       nothing resembles the word, but one of our boxes sits where the reference engine read it (a box with a wrong reading)
  absent        nothing resembles the word and the reference engine does not read it either (or there is no reference)
  wrong         a hypothesis word resembles the word but differs (misreading, a clipped box)

Misses are also split by the truth line's type (totals, prices on item lines, header, masks, other) and by word class, and by the
height of the reference engine's box when it reads the word (small print). The reference engine's output is only ever compared,
never used as a label.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import Levenshtein, normalize, truth_text

MONEY = re.compile(r"^[-(]?[$€£]?\d[\d,]*[.,]\d{2}\)?[A-Za-z]?$")
TOTALS = re.compile(r"\b(sub\s*-?total|total|tax|balance|change|amount|due|cash|visa|mastercard|master|debit|credit|tip|discount|savings|tender|paid|payment|approved|auth)\b", re.I)
PHONE = re.compile(r"\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}")
STREET = re.compile(r"\b(st|street|ave|avenue|rd|road|blvd|dr|drive|hwy|suite|ste|ln|way)\b\.?", re.I)


def line_type(line: str, index: int, count: int) -> str:
    if "*" in line or re.search(r"[xX#]{3,}", line):
        return "mask"
    if TOTALS.search(line):
        return "totals/payment"
    if index < 4 or PHONE.search(line) or STREET.search(line):
        return "header/store"
    if any(MONEY.match(w) for w in line.split()) or re.match(r"^\d+\s*[xX@]?\s", line):
        return "item"
    if index >= count - 4:
        return "footer"
    return "other text"


def word_class(w: str) -> str:
    if MONEY.match(w):
        return "money"
    if re.search(r"\d", w):
        return "digits"
    if len(re.sub(r"\W", "", w)) <= 1:
        return "single char"
    return "letters"


def load_words(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("words", [])


def touches(a: list[float], b: list[float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def leftovers(ref: list[str], hyp: list[str]):
    """The scorer's pairing: (truth word, paired hypothesis word or None, characters counted) for each leftover truth word."""
    rc, hc = Counter(ref), Counter(hyp)
    r_left = list((rc - hc).elements())
    h_left = list((hc - rc).elements())
    pair: dict[int, int] = {}
    if r_left and h_left:
        pairs = sorted((Levenshtein.distance(r_left[i], h_left[j]), i, j) for i in range(len(r_left)) for j in range(len(h_left)))
        used: set[int] = set()
        for _, i, j in pairs:
            if i in pair or j in used:
                continue
            pair[i] = j
            used.add(j)
            if len(pair) == len(r_left) or len(used) == len(h_left):
                break
    for i, t in enumerate(r_left):
        p = h_left[pair[i]] if i in pair else None
        yield t, p, (min(Levenshtein.distance(t, p), len(t)) if p is not None else len(t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", default="", help="NAME=DIR of a reference engine's output (text and word boxes)")
    ap.add_argument("--alt", action="append", default=[], help="LABEL=DIR of an alternative run of our engine")
    ap.add_argument("--scorer", default="v3.1")
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--pages", type=int, default=12)
    a = ap.parse_args()
    fold = a.scorer == "v3.1"
    truth_dir, ours_dir = Path(a.truth), Path(a.ours)
    ref_name, ref_dir = (a.ref.split("=", 1) if a.ref else ("", ""))
    alts = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.alt]

    tot_chars = 0
    tot_words = 0
    cause_c: Counter = Counter()
    cause_w: Counter = Counter()
    cross: Counter = Counter()          # (cause, line type) -> chars
    cross_w: Counter = Counter()
    cls_c: Counter = Counter()          # (word class) -> chars
    cls_w: Counter = Counter()
    size_c: Counter = Counter()
    ref_right_c: Counter = Counter()    # cause -> chars the reference reads exactly
    ref_right_w: Counter = Counter()
    ref_known = 0
    page_c: dict[str, int] = {}
    page_chars: dict[str, int] = {}
    examples: dict[str, list] = defaultdict(list)
    ref_miss_c = 0

    for tf in sorted(truth_dir.glob("*.txt")):
        stem = tf.stem
        lines = truth_text(truth_dir, stem, a.scorer).split("\n")
        ref_words: list[str] = []
        line_of: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for li, ln in enumerate(lines):
            for w in normalize(ln, fold):
                ref_words.append(w)
                line_of[w].append(li)
        ours_txt = (ours_dir / f"{stem}.txt").read_text(encoding="utf-8", errors="replace") if (ours_dir / f"{stem}.txt").exists() else ""
        hyp = normalize(ours_txt, fold)
        ours_boxes = [(normalize(w["text"], fold), [float(v) for v in w["box"]]) for w in load_words(ours_dir / f"{stem}.json")]
        alt_counts = []
        for label, d in alts:
            tx = (d / f"{stem}.txt").read_text(encoding="utf-8", errors="replace") if (d / f"{stem}.txt").exists() else ""
            alt_counts.append((label, Counter(normalize(tx, fold))))
        refw: list[tuple[str, list[float]]] = []
        ref_counts: Counter = Counter()
        if ref_dir:
            rw = load_words(Path(ref_dir) / f"{stem}.json")
            for w in rw:
                for t in normalize(w["text"], fold):
                    refw.append((t, [float(v) for v in w["box"]]))
            ref_counts = Counter(t for t, _ in refw)
        heights = sorted(b[3] - b[1] for _, b in refw)
        med_h = heights[len(heights) // 2] if heights else 0.0
        chars = sum(map(len, ref_words))
        tot_chars += chars
        tot_words += len(ref_words)
        page_chars[stem] = chars
        page_c[stem] = 0
        hc = Counter(hyp)
        used_alt = Counter()
        for t, paired, miss in leftovers(ref_words, hyp):
            li = line_of[t].pop() if line_of[t] else 0
            ltype = line_type(lines[li], li, len(lines))
            cause = None
            for label, cnt in alt_counts:
                if cnt[t] > hc[t] + used_alt[(label, t)] - 0:
                    # the alternative run reads one more copy of the word than the base run does
                    used_alt[(label, t)] += 1
                    cause = label
                    break
            near = paired is not None and Levenshtein.distance(t, paired) < max(1, 0.6 * len(t))
            if cause is None:
                if paired is not None and len(paired) > len(t) and t in paired:
                    cause = "merged"
                else:
                    cause = None
                    for n in (2, 3):
                        for i in range(len(hyp) - n + 1):
                            j = "".join(hyp[i:i + n])
                            if Levenshtein.distance(t, j) <= (1 if len(t) > 4 else 0):
                                cause = "split"
                                break
                        if cause:
                            break
                if cause is None:
                    if near:
                        cause = "wrong"
                    else:
                        # nothing resembles the word: where does the reference read it?
                        rb = next((b for tt, b in refw if tt == t), None)
                        if rb is None:
                            cause = "absent"
                        elif any(touches(rb, b) for _, b in ours_boxes):
                            cause = "garbled"
                        else:
                            cause = "undetected"
            rb = next((b for tt, b in refw if tt == t), None)
            small = "ref size n/a"
            if rb is not None and med_h:
                small = "small (<0.8 median)" if (rb[3] - rb[1]) < 0.8 * med_h else "normal"
            cause_c[cause] += miss
            cause_w[cause] += 1
            cross[(cause, ltype)] += miss
            cross_w[(cause, ltype)] += 1
            cls_c[(cause, word_class(t))] += miss
            size_c[(cause, small)] += miss
            page_c[stem] += miss
            if ref_dir:
                if ref_counts[t] > 0:
                    ref_right_c[cause] += miss
                    ref_right_w[cause] += 1
            if len(examples[cause]) < a.examples and miss:
                examples[cause].append((stem, t, paired, ltype))
        if ref_dir:
            rr = Counter(normalize("\n".join(lines), fold)) - Counter(t for t, _ in refw)
            ref_miss_c += sum(len(w) * n for w, n in rr.items())  # upper bound: unpaired view of the reference's misses

    pp = lambda c: 100.0 * c / max(tot_chars, 1)
    pw = lambda w: 100.0 * w / max(tot_words, 1)
    print(f"truth: {tot_words} words, {tot_chars} chars; recall-CER {pp(sum(cause_c.values())):.2f}  recall-WER {pw(sum(cause_w.values())):.2f}")
    print("\ncause              words   chars    pp-CER   pp-WER" + (f"   ref-reads: chars  pp" if ref_dir else ""))
    for c, v in sorted(cause_c.items(), key=lambda kv: -kv[1]):
        extra = f"   {ref_right_c[c]:6d} {pp(ref_right_c[c]):5.2f}" if ref_dir else ""
        print(f"{c:16s} {cause_w[c]:7d} {v:7d} {pp(v):9.2f} {pw(cause_w[c]):8.2f}{extra}")
    if ref_dir:
        print(f"  ({ref_name}: a truth word counts as read when the engine outputs it at least once)")
    print("\ncause x line type (pp-CER)")
    types = sorted({k[1] for k in cross})
    print(f"{'':16s}" + "".join(f"{t:>16s}" for t in types))
    for c in sorted(cause_c, key=lambda c: -cause_c[c]):
        print(f"{c:16s}" + "".join(f"{pp(cross[(c, t)]):16.2f}" for t in types))
    print("\nline type totals: pp-CER / pp-WER")
    for t in types:
        print(f"  {t:16s} {pp(sum(v for (c, tt), v in cross.items() if tt == t)):6.2f} {pw(sum(v for (c, tt), v in cross_w.items() if tt == t)):6.2f}")
    print("\ncause x word class (pp-CER)")
    classes = sorted({k[1] for k in cls_c})
    print(f"{'':16s}" + "".join(f"{t:>14s}" for t in classes))
    for c in sorted(cause_c, key=lambda c: -cause_c[c]):
        print(f"{c:16s}" + "".join(f"{pp(cls_c[(c, t)]):14.2f}" for t in classes))
    print("\ncause x size by reference box height (pp-CER)")
    sizes = sorted({k[1] for k in size_c})
    print(f"{'':16s}" + "".join(f"{t:>22s}" for t in sizes))
    for c in sorted(cause_c, key=lambda c: -cause_c[c]):
        print(f"{c:16s}" + "".join(f"{pp(size_c[(c, t)]):22.2f}" for t in sizes))
    worst = sorted(page_c.items(), key=lambda kv: -kv[1])
    n = len(worst)
    for k in (5, 10, 20):
        print(f"\nworst {k} pages carry {100 * sum(v for _, v in worst[:k]) / max(sum(page_c.values()), 1):.0f}% of the missed characters ({n} pages)", end="")
    print("\n\nworst pages (page, missed chars / truth chars):")
    print("  " + ", ".join(f"{p} {v}/{page_chars[p]}" for p, v in worst[: a.pages]))
    for c, e in examples.items():
        print(f"\n{c} examples (page, truth, paired hypothesis, line type):")
        for x in e:
            print("  ", x)


if __name__ == "__main__":
    main()
