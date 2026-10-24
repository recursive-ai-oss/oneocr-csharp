"""Hypotheses (text and word confidences) of a trained recognizer on every line set, as it ships: folded and fp16-rounded.

    uv run python -m eval.model_hyps --checkpoint best.pt --sets-dir LINES_DIR --bench-dir BENCH_LINES --out OUT

Writes OUT/<set>.hyps.jsonl for the synthetic sets (val, eval, diag_fonts, diag_degrade, calib) and for each page set's `finder` crops
(named like the page set). Raw confidences only; the calibration map is applied by eval.calibrate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eval.bakeoff import PAGE_SETS, shipped_model
from eval.confidence import hypotheses, write_hyps
from train.data import LineSet

SYNTH = ["val", "eval", "diag_fonts", "diag_degrade", "calib"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sets-dir", required=True)
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model, _ = shipped_model(ck)
    model = model.cuda()
    todo = {s: Path(a.sets_dir) / s for s in SYNTH} | {s: Path(a.bench_dir) / s / "finder" for s in PAGE_SETS}
    for name, d in todo.items():
        rows = hypotheses(model, LineSet(d, ck["charset"]), ck["charset"])
        write_hyps(rows, out / f"{name}.hyps.jsonl")
        print(name, len(rows), flush=True)


if __name__ == "__main__":
    main()
