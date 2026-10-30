"""Scores every set an engine run directory holds (scorer v3.1) and prints one table: the base run's numbers, then each candidate's
change against it, with extra-word counts.

    cd bench && uv run python ../training/eval/det_compare.py --out OUT_ROOT --base NAME --runs NAME [NAME ...]

OUT_ROOT/<run>/<group>/<set>/ is what eval/det_table.sh writes. Set roots come from the same environment variables (B4_ROOT, V5_ROOT,
RG_ROOT, T1_ROOT, T2_ROOT, T3_ROOT). Columns are micro-averaged CER, WER and reading-order error in percent, the recall-CER and
recall-WER (partial-truth sets: receipts), and the number of hypothesis words with no truth word left ("extra", see extra_words.py).
Changes are candidate minus base; negative is better except in the extra column, where fewer is better too.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extra_words import count  # noqa: E402
from oneocr_bench.score import score_dir  # noqa: E402

GROUPS = {"b4": "B4_ROOT", "v5": "V5_ROOT", "rg": "RG_ROOT", "t1": "T1_ROOT", "t2": "T2_ROOT", "t3": "T3_ROOT"}


def measure(root: Path, run_dir: Path) -> dict | None:
    if not run_dir.is_dir() or not list(run_dir.glob("*.txt")):
        return None
    r = score_dir(root / "truth", run_dir, scorer="v3.1")
    e = count(root / "truth", run_dir)
    return {"cer": 100 * r["cer"], "wer": 100 * r["wer"], "order": 100 * r["order_err"], "rcer": 100 * r["cer_recall"],
            "rwer": 100 * r["wer_recall"], "extra": e["extra"], "hyp": e["hyp_words"], "pages": r["pages"],
            "ece": (r["ece"] or {}).get("ece")}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    a = ap.parse_args()
    out = Path(a.out)
    print(f"{'set':27s}{'pages':>5s} | {'base: CER':>9s}{'WER':>7s}{'order':>7s}{'rCER':>7s}{'rWER':>7s}{'extra':>6s}")
    for g in a.groups:
        base_root = Path(os.path.expanduser(os.environ[GROUPS[g]]))
        for s in sorted(p.name for p in (out / a.base / g).iterdir() if p.is_dir()) if (out / a.base / g).is_dir() else []:
            b = measure(base_root / s, out / a.base / g / s)
            if b is None:
                continue
            print(
                f"{g + ' ' + s:27s}{b['pages']:5d} | {b['cer']:9.3f}{b['wer']:7.3f}{b['order']:7.2f}"
                f"{b['rcer']:7.2f}{b['rwer']:7.2f}{b['extra']:6d}"
            )
            for run in a.runs:
                c = measure(base_root / s, out / run / g / s)
                if c is None:
                    print(f"{'':32s}{run:>9s}: missing")
                    continue
                print(
                    f"{'':32s}{run:>9s}: {c['cer'] - b['cer']:+8.3f}{c['wer'] - b['wer']:+7.3f}{c['order'] - b['order']:+7.2f}"
                    f"{c['rcer'] - b['rcer']:+7.2f}{c['rwer'] - b['rwer']:+7.2f}{c['extra'] - b['extra']:+6d}"
                )


if __name__ == "__main__":
    main()
