#!/usr/bin/env python3
"""Converts CORD v2 test annotations (CC BY 4.0) into the small JSON the grounding evaluation reads.

Usage: cord_to_json.py <annotations/test.jsonl> <out.json>

For each receipt: its annotated words in reading order (top to bottom, left to right) with boxes, and the annotated
fields (menu names, counts, prices, sub totals, tax, totals, cash) as lists of word indices. The output is derived from
CORD's own annotations; nothing is added by hand. It is not committed: CORD is fetched separately.
"""
import json
import sys

CATEGORIES = {
    "menu.nm", "menu.cnt", "menu.price", "menu.unitprice",
    "sub_total.subtotal_price", "sub_total.tax_price", "sub_total.discount_price", "sub_total.service_price",
    "total.total_price", "total.cashprice", "total.changeprice", "total.creditcardprice",
}


def box(q):
    xs = [q["x1"], q["x2"], q["x3"], q["x4"]]
    ys = [q["y1"], q["y2"], q["y3"], q["y4"]]
    return [min(xs), min(ys), max(xs), max(ys)]


def main(src, dst):
    out = []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            gt = json.loads(rec["ground_truth"])
            size = gt["meta"]["image_size"]
            words = []  # (text, box, line index, word index within line)
            lines = []
            for li, vl in enumerate(gt["valid_line"]):
                ids = []
                for w in vl["words"]:
                    if not w["text"].strip():
                        continue
                    words.append({"text": w["text"], "box": box(w["quad"]), "line": li})
                    ids.append(len(words) - 1)
                lines.append((vl["category"], ids))
            # Reading order: cluster words into lines by vertical overlap, lines top to bottom, words left to right.
            by_top = sorted(range(len(words)), key=lambda i: words[i]["box"][1])
            rows = []
            for i in by_top:
                b = words[i]["box"]
                cy = (b[1] + b[3]) / 2
                for row in rows:
                    if abs(row["cy"] - cy) <= 0.5 * (b[3] - b[1]):
                        row["ids"].append(i)
                        row["cy"] = sum((words[j]["box"][1] + words[j]["box"][3]) / 2 for j in row["ids"]) / len(row["ids"])
                        break
                else:
                    rows.append({"cy": cy, "ids": [i]})
            rows.sort(key=lambda r: r["cy"])
            order = [i for r in rows for i in sorted(r["ids"], key=lambda j: words[j]["box"][0])]
            rank = {old: new for new, old in enumerate(order)}
            fields = []
            for cat, ids in lines:
                if cat in CATEGORIES and ids:
                    fields.append({"category": cat, "words": sorted(rank[i] for i in ids)})
            out.append({
                "id": rec["id"], "width": size["width"], "height": size["height"],
                "words": [{"text": words[i]["text"], "box": words[i]["box"]} for i in order],
                "fields": fields,
            })
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(len(out), "receipts")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
