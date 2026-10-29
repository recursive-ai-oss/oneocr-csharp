#!/usr/bin/env bash
# Blends a fine-tuned detector checkpoint with its parent, exports it (container v1, 0.3.0's postprocess config plus the page-noise
# rule) and runs the page cause report over the tune sets.
#   train/faint_candidates.sh PARENT.pt TUNED.pt OUT_DIR MODEL.onm CHARLM WORDFREQ EVAL_ROOT ALPHA [ALPHA ...]
set -euo pipefail
cd "$(dirname "$0")/.."
parent=$1 tuned=$2 out=$3 model=$4 charlm=$5 wordfreq=$6 eval_root=$7
shift 7
mkdir -p "$out"
for a in "$@"; do
  ck=$out/ft-a$a.pt
  uv run python -m export.blend_checkpoints "$parent" "$tuned" "$ck" --alpha "$a"
  uv run python -m export.dbnet --checkpoint "$ck" --out "$out/raw-a$a.onm" --version 0.3.1 \
    --postprocess '{"text_threshold":0.3,"text_expand":0.4,"layout_threshold":0.6,"layout_expand":0.85}'
  uv run python -m export.add_postprocess_rule "$out/raw-a$a.onm" "$out/det-a$a.onm" --junk-max 1000000 --version 0.3.1
  sha256sum "$out/det-a$a.onm" | tee -a "$out/SHA256SUMS"
  nice -n 10 bash ../bench/scripts/page_causes.sh "$eval_root" "$out/causes-a$a" "$model" "$out/det-a$a.onm" "$charlm" "$wordfreq" \
    screenshot-real-tune screenshot-tune screenshot-tune2 >"$out/causes-a$a.log" 2>&1 || true
done
