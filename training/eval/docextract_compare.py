"""Scores the C# extractor's assembled pages (OneOcr.Bench docextract-eval) with the metrics of eval/docextract.py.

The C# run reads the same OCR page models the examples were built from, so both sides see the same words. With --model the
PyTorch model is scored on the same examples as well, and the per-field difference is reported.

  uv run python -m eval.docextract_compare --data val.pkl --src generated --jsonl csharp-val.jsonl [--model model.onm] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from eval.docextract import evaluate, load_model, new_acc, score_assembled, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--src", choices=["generated", "cord"], required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = pickle.load(open(a.data, "rb"))
    examples = d[a.src]["ocr"]
    cs = {}
    for line in Path(a.jsonl).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        cs[r["id"]] = r
    acc = new_acc()
    kinds: dict = {}
    missing = mismatched = 0
    for ex in examples:
        key = ex["meta"]["id"].split("/")[-1]
        r = cs.get(key)
        if r is None:
            missing += 1
            continue
        if r["words"] != len(ex["texts"]):
            mismatched += 1
            continue
        score_assembled(ex, r, acc)
        score_assembled(ex, r, kinds.setdefault(ex["meta"]["kind"], new_acc()))
    report = {"pages": len(examples), "missing": missing, "word_count_mismatch": mismatched, "csharp": {"all": summarize(acc), "by_kind": {k: summarize(v) for k, v in sorted(kinds.items())}}}
    if a.model:
        report["pytorch"] = evaluate(load_model(a.model), examples)
    text = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    c = report["csharp"]["all"]
    print(f"pages {len(examples)} missing {missing} mismatched {mismatched}")
    print(f"C#      header F1 {c['header_micro']['f1']:.4f} item-field F1 {c['item_field']['f1']:.4f} doc acc {c['doc_type_accuracy']}")
    if a.model:
        p = report["pytorch"]["all"]
        print(f"PyTorch header F1 {p['header_micro']['f1']:.4f} item-field F1 {p['item_field']['f1']:.4f} doc acc {p['doc_type_accuracy']}")


if __name__ == "__main__":
    main()
