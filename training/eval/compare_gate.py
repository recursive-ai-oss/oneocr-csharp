"""Compares two line-gate tables (eval.gate --json) row by row and lists every row the older model won that the newer one does not.

    uv run python -m eval.compare_gate OLD.json NEW.json [--old-name 0.5.0 --new-name 0.6.0]

A row counts as won on a metric when ours is below both Tesseract models' values. The newer model may replace the older one only
when no metric of any gate row is lost and no gate row's CER or WER rises by more than `--tolerance` (relative, default 3%) without
being named in the output. Exit status 1 when a won metric is lost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def won(r: dict, m: str) -> bool:
    return r["ours"][m] < min(r["tess_best"][m], r["tess_fast"][m])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--old-name", default="old")
    ap.add_argument("--new-name", default="new")
    ap.add_argument("--tolerance", type=float, default=0.03)
    a = ap.parse_args()
    old = {r["row"]: r for r in json.loads(Path(a.old).read_text())}
    new = {r["row"]: r for r in json.loads(Path(a.new).read_text())}
    lost, worse = [], []
    print(f"| row | {a.old_name} CER / WER % | {a.new_name} CER / WER % | change CER / WER | verdict |\n|---|---:|---:|---:|---|")
    for k, o in old.items():
        n = new[k]
        gate = "(info)" not in n["verdict"]
        d = [100 * (n["ours"][m] / o["ours"][m] - 1) for m in ("cer", "wer")]
        lose = [m for m in ("cer", "wer") if won(o, m) and not won(n, m)]
        if gate and lose:
            lost.append((k, lose))
        if gate and any(x > 100 * a.tolerance for x in d):
            worse.append(k)
        f = lambda r: f"{100 * r['ours']['cer']:.2f} / {100 * r['ours']['wer']:.2f}"  # noqa: E731
        mark = "LOST " + ",".join(lose) if gate and lose else n["verdict"]
        print(f"| {k} | {f(o)} | {f(n)} | {d[0]:+.0f}% / {d[1]:+.0f}% | {mark} |")
    print(f"\nlost won metrics: {lost or 'none'}\nrows more than {100 * a.tolerance:.0f}% worse on CER or WER: {worse or 'none'}")
    sys.exit(1 if lost else 0)


if __name__ == "__main__":
    main()
