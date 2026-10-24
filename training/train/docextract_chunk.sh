#!/usr/bin/env bash
# One resumable training chunk of the document extraction model, then (when the run is finished) export and evaluation.
#   train/docextract_chunk.sh CONFIG RUN_DIR DATA_DIR EXPORT_DIR
# Run it under the worker's GPU lock. CONFIG may use ${ONEOCR_HOME}; DATA_DIR holds the pickles from docextract_make_data.sh.
# Evaluation reports go to EXPORT_DIR: the fp32 checkpoint and the fp16-rounded container on held-out generated pages (truth words and
# OCR words) and on CORD test; the same sets scored with the labels themselves as model output give the ceiling that OCR leaves.
set -euo pipefail
config=$1 run=$2 data=$3 exports=$4
cd "$(dirname "$0")/.."
mkdir -p "$exports"
uv run python -m train.docextract "$config"
[ -e "$run/final.pt" ] || { echo "chunk ended before the last step; run again to resume"; exit 0; }
ver=${ONEOCR_MODEL_VERSION:-0.1.0}
uv run python -m export.docextract --checkpoint "$run/best.pt" --run-manifest "$run/run_manifest.json" --version "$ver" --out "$exports/model.onm" \
  | tee "$exports/export.json"
sets=("$data/val.pkl" "$data/cord-test.pkl" "$data/cord-dev.pkl")
uv run python -m eval.docextract --model "$run/best.pt" --data "${sets[@]}" --device cuda --out "$exports/eval-fp32.json" >/dev/null
uv run python -m eval.docextract --model "$exports/model.onm" --data "${sets[@]}" --device cuda --latency --out "$exports/eval-fp16.json" >/dev/null
uv run python -m eval.docextract_oracle "${sets[@]}" | tee "$exports/oracle-ceiling.txt"
sha256sum "$exports/model.onm" | tee "$exports/SHA256SUMS"
