"""Regression table: two or more runs of the same sets, scorer v3.1 (or --scorer), with the changed pages and the extra words listed.

    uv run python scripts/regress_table.py --scorer v3.1 BASE=ROOT NEW=ROOT --pair v4:clean300-v4=EVAL/clean300-v4 ...

Each ROOT holds <group>/<set>/ (group v4 or v5, one directory per group). The first run is the base. For every set: CER/WER/order of each run (in %),
the number of pages whose text changed, the pages themselves, and the number of words each run reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import score_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="v3.1")
    ap.add_argument("--set", action="append", required=True, help="GROUP:NAME=TRUTH_PARENT (truth in TRUTH_PARENT/NAME/truth)")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    runs = [r.split("=", 1) for r in a.runs]
    print("| set | run | CER | WER | order | words | pages changed |")
    print("|---|---|---|---|---|---|---|")
    for spec in a.set:
        key, parent = spec.split("=", 1)
        group, name = key.split(":")
        truth = Path(parent) / name / "truth"
        base = None
        for label, root in runs:
            hyp = Path(root) / group / name
            if not hyp.is_dir():
                continue
            r = score_dir(truth, hyp, "ink", a.scorer)
            words = sum(len(json.loads(f.read_text())["words"]) for f in hyp.glob("*.json") if f.with_suffix(".txt").exists())
            changed = []
            if base is None:
                base = hyp
            else:
                changed = [t.stem for t in sorted(hyp.glob("*.txt")) if (base / t.name).read_text() != t.read_text()]
            print(f"| {group}/{name} | {label} | {100 * r['cer']:.3f} | {100 * r['wer']:.3f} | {100 * r['order_err']:.3f} | {words} | {len(changed)}: {' '.join(changed[:12])} |")


if __name__ == "__main__":
    main()
