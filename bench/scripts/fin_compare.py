"""Scores runs of the financial sets side by side: CER, WER, order, amount exact match (box and text-only), quantity cells, reconciliation.

    uv run python scripts/fin_compare.py --eval EVAL_ROOT --set fin-scan [--set ...] LABEL=RUN_ROOT [LABEL=RUN_ROOT ...]

Each RUN_ROOT holds one directory per set with the sidecars of `OneOcr.Bench run --words 1`. The number of words each run reports is
printed too (a rise on a set whose truth did not change is extra words).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.finreport import pct, pooled, score_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    root = Path(a.eval)
    results = {}
    for spec in a.runs:
        label, path = spec.split("=", 1)
        dirs = {s: Path(path) / s for s in a.set if (Path(path) / s).is_dir()}
        results[label] = (score_config(root, label, dirs, False), dirs)
    print("| set | run | CER | WER | order | amount box | amount text | qty | recon | words |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in [*a.set, "ALL"]:
        for label, (res, dirs) in results.items():
            names = list(a.set) if s == "ALL" else [s]
            r = pooled(res["sets"], names)
            if not r:
                continue
            qty = [res["sets"][n]["amounts"] for n in names if res["sets"][n].get("amounts")]
            qv = sum(x["counts"]["values"].get("qty", 0) for x in qty)
            qr = sum(x["counts"]["read"].get("qty", 0) for x in qty)
            words = sum(len(json.loads(f.read_text())["words"]) for n in names for f in dirs[n].glob("*.json") if f.with_suffix(".txt").exists())
            print(f"| {s} | {label} | {pct(r['cer'])} | {pct(r['wer'])} | {pct(r['order'])} | {pct(r['amount'])} | {pct(r['text_only'])} | "
                  f"{pct(qr / qv if qv else None, 1)} ({qr}/{qv}) | {pct(r['reconciliation'], 1)} | {words} |")


if __name__ == "__main__":
    main()
