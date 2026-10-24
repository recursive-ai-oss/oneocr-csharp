"""Per page kind (table, form, article, ...) CER, WER and reading-order error from `score --json` files.

    python3 table_compare.py ROOT [ROOT2]      ROOT holds {nodet,det}-{layout-v1,screenshot-v1}.json (see table_rows.sh)

With two roots, lists the pages where the second is worse on any count.
"""
import collections
import glob
import json
import sys
from pathlib import Path

EVAL = Path.home() / "oneocr/eval/bench-v1"


def kinds(set_name):
    out = {}
    for f in glob.glob(f"{EVAL}/{set_name}/truth/*.json"):
        d = json.load(open(f))
        out[d["id"]] = d.get("kind") or d.get("template") or d.get("screen") or "?"
    return out


def agg(root, cfg, set_name):
    k = kinds(set_name)
    pp = json.load(open(f"{root}/{cfg}-{set_name}.json"))["per_page"]
    a = collections.defaultdict(lambda: collections.Counter())
    for pid, v in pp.items():
        for key in ("ref_words", "ref_chars", "char_errors", "word_errors", "order_errors"):
            a[k[pid]][key] += v[key]
            a["ALL"][key] += v[key]
    return a, pp, k


def row(c):
    return f"CER {100 * c['char_errors'] / max(1, c['ref_chars']):6.2f}  WER {100 * c['word_errors'] / max(1, c['ref_words']):6.2f}  order {100 * c['order_errors'] / max(1, c['ref_words']):6.2f}"


roots = sys.argv[1:]
for s in ("layout-v1", "screenshot-v1"):
    for cfg in ("nodet", "det"):
        print(f"== {s} {cfg}")
        results = [agg(r, cfg, s) for r in roots]
        for kind in sorted(results[0][0]):
            print(f"  {kind:9}", " | ".join(row(res[0][kind]) for res in results))
        if len(results) == 2:
            (_, a, k), (_, b, _) = results
            for pid in sorted(a):
                x, y = a[pid], b[pid]
                diffs = [(key, x[key], y[key]) for key in ("char_errors", "word_errors", "order_errors") if y[key] > x[key]]
                if diffs:
                    print("   ", pid, k[pid], "ref_words", x["ref_words"], "worse:", diffs)
