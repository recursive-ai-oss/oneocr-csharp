"""Where the OCR-word ceiling of docextract goes: breaks the loss of every truth field instance on OCR words into causes.

Usage: uv run python -m eval.docextract_loss PICKLE...    (pickles from train.docextract_data build, with OCR words)

For each truth field of an OCR example, the OCR words that carry its class label (labels come from box overlap with the truth words, see
train/docextract_data.py) are joined in reading order and compared with the printed text (white space removed, NFC):
  exact          equal: reachable by any model
  no_words       no OCR word overlaps the field's truth words: detection/finder miss, or the alignment threshold (cover < 0.5)
  order          same characters, different sequence: reading order of the OCR words inside the field
  recognition    similarity of the two strings >= 0.75 (edit ratio): misread characters
  merge_or_split lower similarity: words fused with neighbours, cut apart, or partly missing (detection/line finding)
Counts are over truth fields, per class family (header / item), so the shares add up to 1 within a family.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, defaultdict

from rapidfuzz.distance import Levenshtein

from eval.docextract import norm
from models.docextract import CLASSES, GROUP_CLASSES


def classify(ex: dict) -> list[tuple[str, str]]:
    out = []
    by_cls: dict[str, list[str]] = defaultdict(list)
    n = len(ex["texts"])
    tags = ex["tags"]
    spans: list[tuple[int, list[str]]] = []
    i = 0
    while i < n:
        t = int(tags[i])
        if t > 0:
            c = (t - 1) // 2
            j = i + 1
            while j < n and int(tags[j]) == 2 + 2 * c:
                j += 1
            spans.append((c, ex["texts"][i:j]))
            i = j
        else:
            i += 1
    for c, ws in spans:
        by_cls[CLASSES[c]].append(norm(" ".join(ws)))
    left = {k: list(v) for k, v in by_cls.items()}
    for f in ex["truth"]["fields"]:
        want = norm(f["text"])
        fam = "item" if f["cls"] in GROUP_CLASSES else "header"
        cands = left.get(f["cls"], [])
        if want in cands:
            cands.remove(want)
            out.append((fam, "exact"))
            continue
        if not cands:
            out.append((fam, "no_words"))
            continue
        best = max(cands, key=lambda g: 1 - Levenshtein.normalized_distance(g, want))
        sim = 1 - Levenshtein.normalized_distance(best, want)
        if sorted(best) == sorted(want):
            out.append((fam, "order"))
        elif sim >= 0.75:
            out.append((fam, "recognition"))
        else:
            out.append((fam, "merge_or_split"))
    return out


def main() -> None:
    for path in sys.argv[1:]:
        d = pickle.load(open(path, "rb"))
        for src in ("generated", "cord"):
            exs = d.get(src, {}).get("ocr") or []
            if not exs:
                continue
            cnt: dict[str, Counter] = defaultdict(Counter)
            for ex in exs:
                if ex["meta"].get("partial"):
                    # CORD: only fields whose class is mapped have labels; the same count applies
                    pass
                for fam, why in classify(ex):
                    cnt[fam][why] += 1
            res = {fam: {k: round(v / sum(c.values()), 4) for k, v in c.most_common()} | {"n": sum(c.values())} for fam, c in cnt.items()}
            print(path.split("/")[-1], src, json.dumps(res))


if __name__ == "__main__":
    main()
