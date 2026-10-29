# ruff: noqa: E501
"""Deletion / insertion / substitution split of character errors per set, for several models side by side.

    uv run python -m eval.err_split --sets a,b,c NAME=HYPS_DIR [NAME=HYPS_DIR ...]

Each HYPS_DIR holds <set>.hyps.jsonl from eval.model_hyps. Cells are `CER% (del/ins/sub)` in percent of reference characters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.capacity_diag import align


def split(path: Path) -> tuple[float, float, float, int]:
    d = i = s = n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        r = json.loads(line)
        n += len(r["ref"])
        if r["ref"] == r["hyp"]:
            continue
        for kind, _a, _b in align(r["ref"], r["hyp"]):
            d += kind == "del"
            i += kind == "ins"
            s += kind == "sub"
    return 100 * d / n, 100 * i / n, 100 * s / n, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("models", nargs="+", help="NAME=HYPS_DIR")
    a = ap.parse_args()
    models = [m.split("=", 1) for m in a.models]
    print("| set | " + " | ".join(n for n, _ in models) + " |\n|---|" + "---|" * len(models))
    for s in a.sets.split(","):
        cells = []
        for _, d in models:
            p = Path(d) / f"{s}.hyps.jsonl"
            if not p.exists():
                cells.append("-")
                continue
            de, ins, su, _n = split(p)
            cells.append(f"{de + ins + su:.2f} ({de:.2f}/{ins:.2f}/{su:.2f})")
        print(f"| {s} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
