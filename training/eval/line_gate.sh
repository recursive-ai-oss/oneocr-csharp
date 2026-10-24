#!/usr/bin/env bash
# Line-level gate for a checkpoint: hypotheses on every set, the calibration map (fitted on the calibration splits only), ECE before
# and after, and the table against the black-box baseline's stored outputs on the identical crops.
#   eval/line_gate.sh CHECKPOINT OUT_DIR [model_hyps options, e.g. --extra NAME=DIR for more sets to read]
# Environment: LINES_DIR (rendered line sets with calib/ and the held-out sets), BENCH_LINES (page-set crops, eval/make_bench_lines.sh),
# TESS_DIR (eval/tess_lines.py outputs per set), and optionally RECEIPT_CALIB and CORD_CALIB (extra calibration crops, rendered sets
# with images/ + labels.jsonl) with CORD_WEIGHT (how many times the small CORD calibration set counts in the fit, default 8).
# ALSO lists extra hyps sets for the ECE table; DEVICE=cpu runs the hypotheses without the GPU.
set -euo pipefail
ck=$1
out=$2
shift 2
cd "$(dirname "$0")/.."
mkdir -p "$out"
extra=()
fit=("$out/hyps/calib.hyps.jsonl")
if [ -n "${RECEIPT_CALIB:-}" ]; then
  extra+=(--extra "calib_receipt=$RECEIPT_CALIB")
  fit+=("$out/hyps/calib_receipt.hyps.jsonl")
fi
if [ -n "${CORD_CALIB:-}" ]; then
  extra+=(--extra "calib_cord=$CORD_CALIB")
  fit+=("$out/hyps/calib_cord.hyps.jsonl:${CORD_WEIGHT:-8}")
fi
uv run python -m eval.model_hyps --checkpoint "$ck" --sets-dir "$LINES_DIR" --bench-dir "$BENCH_LINES" --out "$out/hyps" ${DEVICE:+--device $DEVICE} "${extra[@]}" "$@"
uv run python -m eval.calibrate fit --hyps "${fit[@]}" --out "$out/calibration.json"
uv run python -m eval.calibrate table --calibration "$out/calibration.json" --hyps-dir "$out/hyps" --out "$out/ece.md" --also "${ALSO:-}"
uv run python -m eval.gate --ours "$out/hyps" --tess "$TESS_DIR" --out "$out/gate.md" --json "$out/gate.json"
