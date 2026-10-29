#!/usr/bin/env bash
# Builds everything the document extraction model trains and is evaluated on:
#   train/docextract_make_data.sh OUT SEED N_TRAIN N_VAL OCR_MODEL [WORKERS]
# 1. generated pages (configs/pages-docs.yaml: invoices, receipts, forms and a few prose and table pages), train and val splits
# 2. our OCR over the page images, and over CORD v2 train/dev/test images (train/docextract_ocr.sh)
# 3. pickles for train/docextract.py and eval/docextract.py: OUT/{train,val}.pkl and OUT/cord-{train,dev,test}.pkl
# Needs ONEOCR_FONT_ROOT, ONEOCR_TEXT_ROOT, PLAYWRIGHT_BROWSERS_PATH (see pages/README.md), .NET 10 on PATH, and CORD fetched
# (datasets/fetch_cord.py) under ONEOCR_DATASETS (default ~/oneocr/datasets). Steps already done (a manifest or output exists) are kept.
set -euo pipefail
out=$1 seed=$2 ntrain=$3 nval=$4 model=$5 workers=${6:-16}
cd "$(dirname "$0")/.."
cord=${ONEOCR_DATASETS:-$HOME/oneocr/datasets}/cord/v2
mkdir -p "$out"
for split in train val; do
  n=$ntrain; [ $split = val ] && n=$nval
  [ -e "$out/pages/$split/manifest.jsonl" ] || uv run python -m pages --n "$n" --seed "$seed" --split $split --out "$out/pages" --workers "$workers" \
    --config configs/pages-docs.yaml
  [ -e "$out/ocr/$split.done" ] || { bash train/docextract_ocr.sh "$out/pages/$split/images" "$out/ocr/$split" "$model" "$workers"; touch "$out/ocr/$split.done"; }
done
for split in train dev test; do
  [ -e "$out/ocr-cord/$split.done" ] || { bash train/docextract_ocr.sh "$cord/images/$split" "$out/ocr-cord/$split" "$model" "$workers"; touch "$out/ocr-cord/$split.done"; }
  uv run python -m train.docextract_data build --cord "$cord" --cord-split $split --cord-ocr "$out/ocr-cord" --out "$out/cord-$split.pkl"
done
for split in train val; do
  uv run python -m train.docextract_data build --pages "$out/pages" --split $split --ocr "$out/ocr" --out "$out/$split.pkl"
done
uv run python -m pages.check "$out/pages" --split val --n "$nval" --workers "$workers" --report "$out/check-val.json" >/dev/null || true
echo "done: $(ls "$out"/*.pkl | tr '\n' ' ')"
