"""Choose the weight of page-level receipt-photo samples in the reporting curve by split-half, then fit the curve on both halves.

The two halves are the receipt-photo calibration split and the evaluation split of the generated photos (250 photos). For each
weight K: fit the curve on the synthetic and CORD calibration sets, the line crops of half A and the page samples of half A counted
K times, and measure ECE on the page samples of half B; then the same with the halves swapped. The weight with the lowest mean
held-out ECE wins, and the final curve is fitted on everything with that weight. Nothing else (no sealed or test data) is read.

    uv run python -m eval.split_half_fit --lines HYPS_DIR --page-calib A.hyps.jsonl --page-dev B.hyps.jsonl --out DIR [--weights 0 10 25 50 100]

HYPS_DIR holds calib, calib_receipt, calib6, calib_cord, gen_calib and gen_dev `.hyps.jsonl` (line crops, `eval.lines`),
the page files come from `eval.page_samples`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.calibrate import MAX_KNOTS, SCRIPTS, calibrated, ece, fit_map, word_samples  # noqa: F401
from eval.scoring import read_hyps

LINE_SETS = [("calib", 1), ("calib_receipt", 1), ("calib6", 1), ("calib_cord", 8)]


def fit(samples: list) -> dict:
    cal = {"raw": "word-mean-max-run", "scripts": {}}
    for s in SCRIPTS.values():
        g = [x for x in samples if x[0] == s]
        cal["scripts"][s] = fit_map(np.array([x[1] for x in g]), np.array([x[2] for x in g]))
    return cal


def held_out_ece(cal: dict, samples: list) -> float:
    ok = np.array([x[2] for x in samples], dtype=float)
    return ece(calibrated(samples, cal), ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--page-calib", required=True)
    ap.add_argument("--page-dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", type=int, nargs="+", default=[0, 10, 25, 50, 100])
    ap.add_argument("--guard", nargs="*", default=["eval", "val", "clean300-v1", "degraded-v1", "screenshot-v1", "layout-v1", "phone-v1", "receipts-v1", "cord_dev"],
                    help="line sets (in --lines) whose ECE must not suffer from a heavier page weight")
    ap.add_argument("--guard-max", type=float, default=0.03, help="a weight is allowed while every guard set's ECE stays at or below this (the calibration gate is 0.05)")
    a = ap.parse_args()
    d = Path(a.lines)
    base: list = []
    for name, k in LINE_SETS:
        base += word_samples(read_hyps(d / f"{name}.hyps.jsonl")) * k
    lines = {h: word_samples(read_hyps(d / f"{n}.hyps.jsonl")) for h, n in (("A", "gen_calib"), ("B", "gen_dev"))}
    pages = {"A": word_samples(read_hyps(a.page_calib)), "B": word_samples(read_hyps(a.page_dev))}
    guard_samples = {g: word_samples(read_hyps(d / f"{g}.hyps.jsonl")) for g in a.guard}
    rows = []
    for k in a.weights:
        ab = held_out_ece(fit(base + lines["A"] + pages["A"] * k), pages["B"])  # fit on calib, judged on evaluation photos
        ba = held_out_ece(fit(base + lines["B"] + pages["B"] * k), pages["A"])  # fit on evaluation photos, judged on calib
        everything = fit(base + lines["A"] + lines["B"] + (pages["A"] + pages["B"]) * k)
        guard = {g: held_out_ece(everything, guard_samples[g]) for g in a.guard}
        rows.append({"weight": k, "ece_fit_calib_test_dev": ab, "ece_fit_dev_test_calib": ba, "mean": (ab + ba) / 2, "guard": guard})
        print(rows[-1], flush=True)
    # Page samples of one kind of document pull the curve away from every other kind of text, so the weight is the one with the lowest
    # held-out ECE among those that keep every guard set at or below the cap.
    allowed = [r for r in rows if all(v <= a.guard_max for v in r["guard"].values())]
    best = min(allowed, key=lambda r: (r["mean"], r["weight"]))
    k = best["weight"]
    final = fit(base + lines["A"] + lines["B"] + (pages["A"] + pages["B"]) * k)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(final, indent=1), encoding="utf-8")
    allp = pages["A"] + pages["B"]
    report = {"allowed_weights": [r["weight"] for r in allowed], "rows": rows, "chosen_weight": k, "pages": {"calib": len(pages["A"]), "dev": len(pages["B"])},
              "in_sample_ece_all_pages": held_out_ece(final, allp), "ece_raw_all_pages": ece(np.array([x[1] for x in allp]), np.array([x[2] for x in allp], dtype=float))}
    (out / "split-half.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("chosen", k, json.dumps(report))


if __name__ == "__main__":
    main()
