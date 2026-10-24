#!/usr/bin/env bash
# Cuts word crops from the fetched CORD v2 train and dev images, the way the receipts evaluation set is cut (each word's quad is
# rectified to an upright patch, then LineGen cuts it like the library's line finder), and writes them with CORD's own labels.
#   gen/make_cord_lines.sh OUT_DIR
# Reads $ONEOCR_DATASETS/cord/v2 (datasets/fetch_cord.py). OUT_DIR/{train,calib,val}/finder/{images,labels.jsonl}, OUT_DIR/split.json.
# Test images never enter: datasets/cord_crops.py leaves the official test split out. Needs dotnet and uv on PATH.
set -euo pipefail
out=$(realpath -m "$1")
cd "$(dirname "$0")/.."
dotnet build gen/LineGen -c Release 2>&1 | grep -E 'error|Warn' || true
uv run python datasets/cord_crops.py --out "$out"
for s in train calib val; do
  uv run python -m eval.page_lines "$out/$s/pages" "$out/$s/truth" "$out/$s/lines.jsonl" rectwords
  dotnet out/dotnet/bin/LineGen/release/LineGen.dll --page-lines "$out/$s/lines.jsonl" --out "$out/$s"
done
