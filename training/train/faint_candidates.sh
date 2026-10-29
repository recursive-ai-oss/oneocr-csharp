#!/usr/bin/env bash
# Blends a fine-tuned detector checkpoint with its parent, exports it (container v1, same postprocess config and trained-class list as
# the parent's export, plus the page-noise rule) and runs the page cause report over the tune sets and the receipt sets.
#   train/faint_candidates.sh PARENT.pt TUNED.pt RUN_MANIFEST OUT_DIR MODEL.onm CHARLM WORDFREQ TUNE_ROOT RECEIPT_ROOT ALPHA [ALPHA ...]
# ALPHA is the tuned checkpoint's share; "ref=FILE.onm" scores an existing export (the parent's) under the same sets instead.
set -euo pipefail
cd "$(dirname "$0")/.."
parent=$1 tuned=$2 runman=$3 out=$4 model=$5 charlm=$6 wordfreq=$7 troot=$8 rroot=$9
shift 9
mkdir -p "$out"
score() { # NAME DETECTOR
  nice -n 10 bash ../bench/scripts/page_causes.sh "$troot" "$out/causes-$1" "$model" "$2" "$charlm" "$wordfreq" \
    screenshot-real-tune screenshot-tune screenshot-tune2 >"$out/causes-$1.log" 2>&1 || true
  nice -n 10 bash ../bench/scripts/page_causes.sh "$rroot" "$out/causes-$1-cord" "$model" "$2" "$charlm" "$wordfreq" \
    cord-val cord-calib >"$out/causes-$1-cord.log" 2>&1 || true
}
for a in "$@"; do
  if [[ $a == ref=* ]]; then score "ref" "${a#ref=}"; continue; fi
  ck=$out/ft-a$a.pt
  uv run python -m export.blend_checkpoints "$parent" "$tuned" "$ck" --alpha "$a"
  uv run python -m export.dbnet --checkpoint "$ck" --out "$out/raw-a$a.onm" --version 0.3.1 --run-manifest "$runman" \
    --exclude-classes form_field,checkbox \
    --postprocess '{"text_threshold":0.3,"text_expand":0.4,"layout_threshold":0.6,"layout_expand":0.85}' >/dev/null
  uv run python -m export.add_postprocess_rule "$out/raw-a$a.onm" "$out/det-a$a.onm" --junk-max 1000000 --version 0.3.1 >/dev/null
  sha256sum "$out/det-a$a.onm" | tee -a "$out/SHA256SUMS"
  score "a$a" "$out/det-a$a.onm"
done
