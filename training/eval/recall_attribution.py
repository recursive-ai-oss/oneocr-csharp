"""Where the order-free recall-CER of a page output comes from, attributed word by word.

The benchmark scorer cancels identical words, pairs the leftover truth words with the leftover hypothesis words, closest first, and
counts min(edit distance, truth length) characters per pair and the whole word when it is left unpaired. This repeats that pairing
(same function, same order) and puts each leftover truth word's counted characters in one class from geometry:

  undetected  no recognised box touches the word's truth box
  merged      a touching box reads text that contains the word as a part (glued to a neighbour)
  split       the word's text is not read, but two or more touching boxes read pieces that join to it
  wrong       a touching box reads something else (recognition error, or a box clipped to a part of the word)

    cd bench && uv run python ../training/eval/recall_attribution.py TRUTH_DIR HYP_DIR
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from oneocr_bench.score import Levenshtein, normalize  # noqa: E402


def box(w: dict) -> list[float]:
    if "box" in w:
        return [float(v) for v in w["box"]]
    q = w["quad"]
    return [min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])]


def touches(a: list[float], b: list[float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("hyp")
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()
    cost = collections.Counter()
    words = collections.Counter()
    ref_chars = 0
    ex: dict[str, list] = collections.defaultdict(list)
    for tf in sorted(Path(a.truth).glob("*.json")):
        tw = json.loads(tf.read_text())["words"]
        hf = Path(a.hyp) / tf.name
        hw = json.loads(hf.read_text())["words"] if hf.exists() else []
        ref = [(t, w) for w in tw for t in normalize(w["text"])]
        hyp = [(t, w) for w in hw for t in normalize(w["text"])]
        ref_chars += sum(len(t) for t, _ in ref)
        rc, hc = collections.Counter(t for t, _ in ref), collections.Counter(t for t, _ in hyp)
        # leftover truth words: for equal words the earlier ones cancel, as `normalize`d lists are compared as multisets
        seen: collections.Counter = collections.Counter()
        left_r = []
        for t, w in ref:
            seen[t] += 1
            if seen[t] > hc[t]:
                left_r.append((t, w))
        seen = collections.Counter()
        left_h = []
        for t, w in hyp:
            seen[t] += 1
            if seen[t] > rc[t]:
                left_h.append((t, w))
        pair: dict[int, int] = {}
        if left_r and left_h:
            pairs = sorted((Levenshtein.distance(left_r[i][0], left_h[j][0]), i, j) for i in range(len(left_r)) for j in range(len(left_h)))
            used_h: set[int] = set()
            for _, i, j in pairs:
                if i in pair or j in used_h:
                    continue
                pair[i] = j
                used_h.add(j)
                if len(pair) == len(left_r) or len(used_h) == len(left_h):
                    break
        for i, (t, w) in enumerate(left_r):
            if i in pair:
                miss = min(Levenshtein.distance(t, left_h[pair[i]][0]), len(t))
            else:
                miss = len(t)
            tb = box(w)
            near = [h for h in hw if touches(tb, box(h))]
            texts = [x for h in near for x in normalize(h["text"])]
            if not near:
                c = "undetected"
            elif any(t != x and t in x for x in texts):
                c = "merged"
            elif len(near) > 1 and t in "".join(texts):
                c = "split"
            else:
                c = "wrong"
            words[c] += 1
            cost[c] += miss
            if len(ex[c]) < a.examples and miss:
                ex[c].append((tf.stem, t, texts[:3], miss))
    total = sum(cost.values())
    print(
        f"recall-CER {100 * total / max(ref_chars, 1):.2f}% of {ref_chars} truth chars; misses by class (words, chars, pp of recall-CER):"
    )
    for c in ("undetected", "merged", "split", "wrong"):
        print(f"  {c:11s} {words[c]:5d} words {cost[c]:5d} chars {100 * cost[c] / max(ref_chars, 1):6.2f} pp")
    for c, e in ex.items():
        print(f"  {c} examples (page, truth, touching boxes read, chars counted): {e}")


if __name__ == "__main__":
    main()
