#!/usr/bin/env python3
"""Reading-order error of `oneocr ocr` text against the truth words of tools/skew_pages.py.

    python tools/skew_order.py MANIFEST_DIR TEXT_DIR

The error of a page is the word-level edit distance between the truth and the page text divided by the truth word count, in
percent (it includes recognition errors, so read it as a difference between two runs on the same pages). Prints the mean per
angle and per layout, then every page above 5 percent.
"""
import collections
import json
import os
import sys


def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main(root, texts):
    entries = json.load(open(os.path.join(root, "manifest.json")))
    by_angle, by_layout, bad = collections.defaultdict(list), collections.defaultdict(list), []
    for e in entries:
        path = os.path.join(texts, e["file"].replace(".png", ".txt"))
        got = open(path, encoding="utf-8").read().split() if os.path.exists(path) else []
        err = 100.0 * distance(e["words"], got) / len(e["words"])
        by_angle[e["angle"]].append(err)
        by_layout[e["layout"]].append(err)
        if err > 5:
            bad.append((e["file"], e["angle"], e["layout"], round(err, 1)))
    mean = lambda v: sum(v) / len(v)
    print("all pages: %.2f" % mean([x for v in by_angle.values() for x in v]))
    print("by angle:", {k: round(mean(v), 2) for k, v in sorted(by_angle.items())})
    print("by layout:", {k: round(mean(v), 2) for k, v in sorted(by_layout.items())})
    for b in bad:
        print("  >5%:", *b)


main(sys.argv[1], sys.argv[2])
