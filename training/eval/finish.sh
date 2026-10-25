#!/usr/bin/env bash
# Everything after a fine-tune run, on the CPU: hypotheses on every line set with the shipped (folded, fp16-rounded) weights, the
# calibration map, ECE, the line gate, both containers and the parity and calibration fixtures. No GPU, no timing.
#   eval/finish.sh RUN LABEL          RUN = run directory name (e.g. crnn-v8)
# Environment: ONEOCR_HOME (state of this stage: runs/, data/, exports/, eval-<label>/), ONEOCR_PREV (the production run's data/lines-v4),
# ONEOCR_FT (earlier fine-tune stage data: lines-v5 sets, CORD crops), BENCH_LINES (page-set crops), TESS_DIR (stored Tesseract outputs),
# VERSION (container version string), CORD_WEIGHT (default 8).
set -uo pipefail
export CUDA_VISIBLE_DEVICES=
cd "$(dirname "$0")/.."
run=$1
lab=$2
h=$ONEOCR_HOME
o=$h/eval-$lab
x=$h/exports/$run
ck=$h/runs/$run/best.pt
mkdir -p "$o" "$x"
extra=()
r1also=
for s in lines-v5/calib:calib_receipt cord-lines/calib/finder:calib_cord lines-v5/gaps_val:gaps_val cord-lines/val/finder:cord_dev lines-v5/val:receipt_synth_val \
  lines-v5/ui_val:ui_val lines-v5/tight_val:tight_val; do extra+=(--extra "${s#*:}=$ONEOCR_FT/data/${s%%:*}"); done
v6=${ONEOCR_V6:-$h}
for s in figures_val small_val reader_val shot_val calib; do extra+=(--extra "${s}6=$v6/data/lines-v6/$s"); done
# receipt-text sets of the r1 stage, when present: held-out fonts and books, exact truth
for s in recv_val faint_val dots_val ends_val; do [ -d "$h/data/lines-r1/$s" ] && extra+=(--extra "${s}=$h/data/lines-r1/$s") && r1also+=",${s}"; done
cordbox=$ONEOCR_FT/data/cord-lines/val/box
[ -d "$cordbox" ] && extra+=(--extra "cord_dev_box=$cordbox") && r1also+=",cord_dev_box"
# lines-v7 (separators, mixed-case interface words), when this stage rendered them: held-out validation sets and their calibration crops
v7=()
for s in sep_val uiword_val numneg_val calib7; do [ -d "$h/data/lines-v7/$s" ] && { extra+=(--extra "${s}=$h/data/lines-v7/$s"); v7+=("$s"); }; done
cal7=(); [ -d "$h/data/lines-v7/calib7" ] && cal7=("$o/hyps/calib7.hyps.jsonl")
also7=""; [ -d "$h/data/lines-v7/sep_val" ] && also7=",sep_val,uiword_val,numneg_val"
uv run python -m eval.model_hyps --device cpu --checkpoint "$ck" --sets-dir "$ONEOCR_PREV/data/lines-v4" --bench-dir "$BENCH_LINES" --out "$o/hyps" "${extra[@]}" | tail -2
uv run python -m eval.calibrate fit --hyps "$o/hyps/calib.hyps.jsonl" "$o/hyps/calib_receipt.hyps.jsonl" "$o/hyps/calib6.hyps.jsonl" "${cal7[@]}" \
  "$o/hyps/calib_cord.hyps.jsonl:${CORD_WEIGHT:-8}" --out "$o/calibration.json"
uv run python -m eval.calibrate table --calibration "$o/calibration.json" --hyps-dir "$o/hyps" --out "$o/ece.md" \
  --also "gaps_val,cord_dev,receipt_synth_val,ui_val,tight_val,figures_val6,small_val6,reader_val6,shot_val6${also7}${r1also:-}" > /dev/null
uv run python -m eval.gate --ours "$o/hyps" --tess "$TESS_DIR" --out "$o/gate.md" --json "$o/gate.json" > /dev/null
uv run python -m export.export --checkpoint "$ck" --out "$x/model-v0.onm" --version "${VERSION:?}" | grep -E '"sha256"|fp16_tensor|cpu_ms'
uv run python -m export.export --checkpoint "$ck" --out "$x/model-v1.onm" --version "$VERSION" --format-version 1 --calibration "$o/calibration.json" | grep '"sha256"'
f=$x/fixtures
rm -rf "$f"
uv run python -m parity.fixtures --model "$x/model-v0.onm" --model-v1 "$x/model-v1.onm" --data "$ONEOCR_PREV/data/lines-v4/val" --out "$f/parity" | tail -1 | cut -c1-120
uv run python -m parity.calibration_fixture --model "$x/model-v1.onm" --data "$ONEOCR_PREV/data/lines-v4/eval" --out "$f/parity/calibration-fixture" | tail -1
(cd "$f/parity" && find . -type f | sort | xargs sha256sum > "$f/SHA256SUMS")
sha256sum "$x/model-v0.onm" "$x/model-v1.onm"
