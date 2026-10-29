"""List where scorer v2 and scorer v3 treat a hypothesis word differently, and sample those cases for a manual check.

    python scripts/scorer_diff.py --eval EVAL_ROOT --engine NAME=OUTPUT_ROOT [--engine ...] [--sets S ...]
                                  [--samples 20 --seed 112 --out spotcheck.md]

EVAL_ROOT holds <set>/truth; every OUTPUT_ROOT holds <set>/<id>.txt (an engine's page outputs). For each hypothesis word the outcome in the
bag pairing of `oneocr_bench.score` (exact, paired with a truth word, extra) is computed against the v2 truth sequence and the v3 one. A word
whose outcome or partner differs is a difference; the counts are by direction. Each page's two truth sequences are also aligned, and
every region where they differ must be a pure re-tokenisation (the same characters cut differently), which the script checks.
"""

from __future__ import annotations

import argparse
import difflib
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

from oneocr_bench.score import normalize, truth_text, word_outcomes


def pairing(ref: list[str], hyp: list[str]) -> tuple[list[str], list[str | None]]:
    """`word_outcomes` plus the truth word each hypothesis word was matched or paired with (None for extra)."""
    status = word_outcomes(ref, hyp)
    rc = Counter(ref)
    partner: list[str | None] = [None] * len(hyp)
    taken: Counter = Counter()
    for j, w in enumerate(hyp):
        if status[j] == "exact":
            partner[j] = w
            taken[w] += 1
    left_h = [j for j, st in enumerate(status) if st != "exact"]
    left_r = list((rc - Counter(hyp)).elements())
    if left_r and left_h and any(status[j] == "paired" for j in left_h):
        d = cdist(left_r, [hyp[j] for j in left_h], scorer=Levenshtein.distance)
        used_r: set[int] = set()
        used_h: set[int] = set()
        m = len(left_h)
        for flat in np.argsort(d, axis=None, kind="stable"):
            i, k = divmod(int(flat), m)
            if i in used_r or k in used_h:
                continue
            used_r.add(i)
            used_h.add(k)
            partner[left_h[k]] = left_r[i]
            if len(used_r) == len(left_r) or len(used_h) == m:
                break
    return status, partner


def near(seq: list[str], word: str | None, hyp: list[str], j: int) -> int | None:
    """Index in `seq` of the truth word `word` that the hypothesis word j sits next to: the occurrence whose neighbours agree most with the
    hypothesis neighbours, then the one nearest to the same relative position."""
    if word is None:
        return None
    idx = [i for i, x in enumerate(seq) if x == word]
    if not idx:
        return None
    t = j / max(len(hyp), 1) * len(seq)

    def key(i: int) -> tuple[int, float]:
        agree = sum(1 for d in range(-4, 5) if d and 0 <= i + d < len(seq) and 0 <= j + d < len(hyp) and seq[i + d] == hyp[j + d])
        return (-agree, abs(i - t))
    return min(idx, key=key)


def ctx(seq: list[str], i: int | None, k: int = 3) -> str:
    if i is None:
        return "(none)"
    return " ".join(seq[max(0, i - k):i]) + f" **[{seq[i]}]** " + " ".join(seq[i + 1:i + 1 + k])


def regions(a: list[str], b: list[str]) -> list[tuple[list[str], list[str]]]:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(a[i1:i2], b[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def is_mark(t: str) -> bool:
    return bool(t) and all(unicodedata.category(c)[0] in "PS" for c in t)


def kind(a: list[str], b: list[str]) -> str:
    if any(is_mark(t) for t in a) and len(b) < len(a):
        return "touching mark joined"
    if len(b) > len(a):
        return "wrapped fragment split"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--engine", action="append", required=True, metavar="NAME=ROOT")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=112)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ev = Path(a.eval)
    sets = a.sets or sorted(d.name for d in ev.iterdir() if (d / "truth").is_dir())
    diffs: list[dict] = []
    direction: Counter = Counter()
    per_engine_set: dict = defaultdict(Counter)
    region_kinds: Counter = Counter()
    bad_regions: list[str] = []
    total_hyp = Counter()
    for spec in a.engine:
        name, _, root = spec.partition("=")
        for s in sets:
            hyp_dir = Path(root) / s
            if not hyp_dir.is_dir():
                continue
            truth_dir = ev / s / "truth"
            for t in sorted(truth_dir.glob("*.txt")):
                h = hyp_dir / t.name
                hyp = normalize(h.read_text(encoding="utf-8", errors="replace")) if h.exists() else []
                r2 = normalize(truth_text(truth_dir, t.stem, "v2"))
                r3 = normalize(truth_text(truth_dir, t.stem, "v3"))
                total_hyp[(name, s)] += len(hyp)
                if r2 == r3:
                    continue
                if name == a.engine[0].partition("=")[0]:
                    for ra, rb in regions(r2, r3):
                        if "".join(ra) != "".join(rb):
                            bad_regions.append(f"{s}/{t.stem}: {ra} vs {rb}")
                        region_kinds[kind(ra, rb)] += 1
                st2, p2 = pairing(r2, hyp)
                st3, p3 = pairing(r3, hyp)
                changed_regions = regions(r2, r3)
                for j, w in enumerate(hyp):
                    if st2[j] == st3[j] and p2[j] == p3[j]:
                        continue
                    label = f"{st2[j]}->{st3[j]}" if st2[j] != st3[j] else f"{st2[j]} (other partner)"
                    direction[label] += 1
                    per_engine_set[(name, s)][label] += 1
                    pos = j / max(len(hyp), 1)
                    diffs.append({"engine": name, "set": s, "page": t.stem, "j": j, "hyp": hyp, "st2": st2[j], "st3": st3[j], "p2": p2[j], "p3": p3[j],
                                  "r2": r2, "r3": r3, "pos": pos, "label": label,
                                  "kind": next((kind(ra, rb) for ra, rb in changed_regions
                                                if (p3[j] in rb and p3[j] not in ra) or (p2[j] in ra and p2[j] not in rb)), "outside a changed region")})
    kinds = Counter(d["kind"] for d in diffs)
    outside = Counter(d["label"] for d in diffs if d["kind"] == "outside a changed region")
    lines = ["# Scorer v2 against v3: word-level differences", "",
             f"Engines: {', '.join(e.partition('=')[0] for e in a.engine)}; sets: {', '.join(sets)}.", "",
             f"Truth regions where the two sequences differ (first engine's pages): {dict(region_kinds)}; "
             f"regions that are not a re-tokenisation of the same characters: {len(bad_regions)}", ""]
    lines += [f"- NOT PURE: {b}" for b in bad_regions[:20]]
    lines += ["", f"Hypothesis words whose outcome or partner changed: {len(diffs)} of {sum(total_hyp.values())}", "",
              "| direction | words |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(direction.items(), key=lambda kv: -kv[1])]
    lines += ["", "By cause (the truth region the word's partner sits in): " + ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items())),
              "Outside every changed region (a leftover pairing shifted by the changed words; none of these may be a match gained): " + str(dict(outside)), "",
              "| engine | set | " + " | ".join(sorted(direction)) + " |", "|---|---|" + "---|" * len(direction)]
    for (e, s), c in sorted(per_engine_set.items()):
        lines.append(f"| {e} | {s} | " + " | ".join(str(c.get(k, 0)) for k in sorted(direction)) + " |")
    pick = random.Random(a.seed).sample(diffs, min(a.samples, len(diffs)))
    lines += ["", f"## {len(pick)} sampled differences (seed {a.seed})", ""]
    for n, d in enumerate(pick, 1):
        h = d["hyp"]
        lines += [f"### {n}. {d['engine']} / {d['set']} / {d['page']}  ({d['label']}; {d['kind']})", "",
                  f"- hypothesis: {ctx(h, d['j'], 4)}",
                  f"- truth v2 (text): {ctx(d['r2'], near(d['r2'], d['p2'], h, d['j']))}",
                  f"- truth v3 (boxes): {ctx(d['r3'], near(d['r3'], d['p3'], h, d['j']))}",
                  f"- v2 outcome: {d['st2']}" + (f" with `{d['p2']}`" if d['p2'] else "") + f"; v3 outcome: {d['st3']}" + (f" with `{d['p3']}`" if d['p3'] else ""), ""]
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print("\n".join(lines[:40]))
    return 1 if bad_regions else 0


if __name__ == "__main__":
    raise SystemExit(main())
