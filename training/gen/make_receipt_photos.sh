#!/usr/bin/env bash
# Receipt-photo pages for the detector (configs/pages-receiptphoto.yaml), built in resumable chunks of 500 pages:
#   gen/make_receipt_photos.sh OUT SEED N_TRAIN N_VAL [WORKERS] [MAX_CHUNKS]
# A finished chunk leaves OUT/<split>/chunk-<start>.done, so running the script again continues where it stopped. MAX_CHUNKS bounds the
# chunks built in one call (to keep a job under its time limit). Needs ONEOCR_FONT_ROOT, ONEOCR_TEXT_ROOT and a Chromium for Playwright.
set -euo pipefail
out=$1 seed=$2 ntrain=$3 nval=$4 workers=${5:-6} maxc=${6:-1000}
: "${ONEOCR_FONT_ROOT:?}" "${ONEOCR_TEXT_ROOT:?}"
cd "$(dirname "$0")/.."
cfg=configs/pages-receiptphoto.yaml
chunk=500
built=0
for split in train val; do
  n=$ntrain
  [ "$split" = val ] && n=$nval
  mkdir -p "$out/$split"
  for ((s = 0; s < n; s += chunk)); do
    [ -e "$out/$split/chunk-$s.done" ] && continue
    [ "$built" -ge "$maxc" ] && exit 0
    c=$(( n - s < chunk ? n - s : chunk ))
    uv run python -m pages --n "$c" --start "$s" --seed "$seed" --split "$split" --out "$out" --workers "$workers" --config "$cfg"
    touch "$out/$split/chunk-$s.done"
    built=$((built + 1))
  done
done
uv run python -m pages.check "$out" --split train --n 500 --workers "$workers" --config "$cfg" --report "$out/check-train.json" > /dev/null || true
uv run python -m pages.check "$out" --split val --n 250 --workers "$workers" --config "$cfg" --report "$out/check-val.json" > /dev/null || true
