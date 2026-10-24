"""Oracle check of labels, assembly and metrics: feeds the labels themselves through the assembler as if they were perfect model output.

Usage: uv run python -m eval.docextract_oracle PICKLE...

On truth words the scores must be 1.0 (the labels survive tagging, grouping and pointer decoding). On OCR words they show the ceiling that
OCR errors and word splits/merges leave to any model: a field whose words were misread cannot match the printed text.
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np

from eval.docextract import new_acc, score_page, summarize
from models.docextract import NREL, NTAGS


def oracle_output(ex: dict) -> dict:
    n = len(ex["texts"])
    big = 20.0
    tags = np.zeros((n, NTAGS), dtype=np.float32)
    roles = np.zeros((n, 5), dtype=np.float32)
    rel = np.zeros((NREL, n, n), dtype=np.float32)
    for i in range(n):
        tags[i, max(int(ex["tags"][i]), 0)] = big
        roles[i, max(int(ex["roles"][i]), 0)] = big
        for r, key in enumerate(("group_ptr", "kv_ptr")):
            p = int(ex[key][i])
            rel[r, i, p if p >= 0 else i] = big
    doc = np.zeros(3, dtype=np.float32)
    doc[ex["doc"]] = big
    return {"tags": tags, "roles": roles, "doc": doc, "rel": rel}


def main() -> None:
    for path in sys.argv[1:]:
        d = pickle.load(open(path, "rb"))
        for src in ("generated", "cord"):
            for kind in ("truth", "ocr"):
                exs = d.get(src, {}).get(kind) or []
                if not exs:
                    continue
                acc = new_acc()
                for ex in exs:
                    score_page(ex, oracle_output(ex), acc)
                s = summarize(acc)
                out = {
                    "pages": s["pages"],
                    "header_f1": round(s["header_micro"]["f1"], 4),
                    "item_field_f1": round(s["item_field"]["f1"], 4),
                    "item_exact_f1": round(s["item_exact"]["f1"], 4) if s["item_exact"] else None,
                    "tax_row_f1": round(s["tax_row"]["f1"], 4) if s["tax_row"] else None,
                    "kv_f1": round(s["kv"]["f1"], 4),
                    "doc": s["doc_type_accuracy"],
                }
                print(path.split("/")[-1], src, kind, json.dumps(out))


if __name__ == "__main__":
    main()
