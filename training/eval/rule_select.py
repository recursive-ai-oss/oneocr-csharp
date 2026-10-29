# ruff: noqa: E501
"""Choose a page-quality rule for the detector's text threshold on held-out pages, by emulating it page by page.

A rule "if the page's noise measure exceeds T, use post-processing variant HI, else variant LO" acts on one page at a time and
changes nothing else, so its output on a page is exactly the page output of the chosen variant. This tool takes the engine outputs of
several manifest-only variants (the same detector with different `config.postprocess`) over held-out sets, the per-page measures
from `eval.page_noise`, and scores every (measure, T, HI) by assembling the per-page outputs and running the benchmark scorer.

    cd bench && uv run python ../training/eval/rule_select.py --root RUN --noise noise.json --lo t0.3-e0.4 \
        --hi t0.4-e0.4,t0.5-e0.4 --truth NAME=TRUTH_DIR ... [--target degraded-tune] [--tol 0.02]

RUN holds `<variant>/<set>/` engine outputs (`<id>.txt`, `<id>.json`). The rule is picked to minimise the target set's CER subject
to every other set not getting worse than LO by more than `tol` percentage points in CER, WER or order.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from oneocr_bench.score import score_dir


def mixed(run: Path, pages: list[str], choose: dict[str, str], name: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for pid in pages:
        src = run / choose[pid] / name
        for ext in (".txt", ".json"):
            f = src / f"{pid}{ext}"
            if f.exists():
                shutil.copy(f, dest / f.name)
    return dest


def score(truth: Path, out: Path, recall: bool) -> tuple[float, float, float]:
    r = score_dir(truth, out, "ink")
    return (r["cer_recall"], r["wer_recall"], r["order_err"]) if recall else (r["cer"], r["wer"], r["order_err"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--noise", required=True)
    ap.add_argument("--lo", required=True)
    ap.add_argument("--hi", required=True, help="comma list of variants")
    ap.add_argument("--truth", action="append", required=True, help="NAME=TRUTH_DIR; NAME also names the set under each variant")
    ap.add_argument("--target", default="degraded-tune")
    ap.add_argument("--tol", type=float, default=0.02, help="allowed loss on the other sets, percentage points")
    ap.add_argument("--max-candidates", type=int, default=30, help="thresholds tried per measure (evenly spaced in rank)")
    ap.add_argument("--recall", default="sel-cord,receipts-v1", help="sets scored on recall (partial truth)")
    a = ap.parse_args()
    root = Path(a.root)
    noise = json.loads(Path(a.noise).read_text())
    truth = dict(s.split("=", 1) for s in a.truth)
    recall = set(a.recall.split(","))
    tmp = Path(tempfile.mkdtemp())
    pages = {n: sorted(p.stem for p in Path(t).glob("*.txt")) for n, t in truth.items()}
    base: dict[str, tuple[float, float, float]] = {}
    for n, t in truth.items():
        base[n] = score(Path(t), root / a.lo / n, n in recall)
        print(f"LO {a.lo:12s} {n:16s} CER {100 * base[n][0]:.3f} WER {100 * base[n][1]:.3f} order {100 * base[n][2]:.3f}")
        for v in a.hi.split(","):
            s = score(Path(t), root / v / n, n in recall)
            print(f"   {v:12s} {n:16s} CER {100 * s[0]:.3f} WER {100 * s[1]:.3f} order {100 * s[2]:.3f}")
    rows = []
    for measure in ("sigma", "junk"):
        values = sorted({noise[n][p][measure] for n in truth for p in pages[n] if p in noise.get(n, {})})
        mids = [(x + y) / 2 for x, y in zip(values, values[1:], strict=False)]
        cands = mids if len(mids) <= a.max_candidates else [mids[round(i * (len(mids) - 1) / (a.max_candidates - 1))] for i in range(a.max_candidates)]
        for hi in a.hi.split(","):
            for T in cands:
                res = {}
                for n, t in truth.items():
                    choose = {p: (hi if noise[n][p][measure] > T else a.lo) for p in pages[n]}
                    d = mixed(root, pages[n], choose, n, tmp / f"{measure}-{hi}-{T:.4f}" / n)
                    res[n] = score(Path(t), d, n in recall)
                    shutil.rmtree(d, ignore_errors=True)
                harm = max((res[n][k] - base[n][k]) * 100 for n in truth if n != a.target for k in range(3))
                rows.append((res[a.target][0], harm, measure, hi, T, res))
    ok = sorted((r for r in rows if r[1] <= a.tol), key=lambda r: (r[0], r[4]))
    print("\nbest rules within tolerance (target CER, worst loss on other sets in pp):")
    for cer, harm, m, hi, T, res in ok[:8]:
        print(f"  {m} > {T:.4f} -> {hi}: {a.target} CER {100 * cer:.3f} WER {100 * res[a.target][1]:.3f} order {100 * res[a.target][2]:.3f}; worst other loss {harm:+.3f} pp")
    if not ok:
        print("  none")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
