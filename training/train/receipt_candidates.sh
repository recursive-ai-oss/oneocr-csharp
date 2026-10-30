#!/usr/bin/env bash
# Blends a fine-tuned detector checkpoint with its parent and exports each blend as a container: same postprocess config and
# trained-class list as the parent's export, the page-noise rule, and the faint-text recovery members of the bundled detector.
#   train/receipt_candidates.sh PARENT.pt TUNED.pt RUN_MANIFEST OUT_DIR VERSION ALPHA [ALPHA ...]
# ALPHA is the tuned checkpoint's share (0 reproduces the parent). Writes OUT_DIR/det-a<ALPHA>.onm and SHA256SUMS.
set -euo pipefail
cd "$(dirname "$0")/.."
parent=$1 tuned=$2 runman=$3 out=$4 version=$5
shift 5
mkdir -p "$out"
for a in "$@"; do
  ck=$out/ft-a$a.pt
  uv run python -m export.blend_checkpoints "$parent" "$tuned" "$ck" --alpha "$a"
  uv run python -m export.dbnet --checkpoint "$ck" --out "$out/raw-a$a.onm" --version "$version" --run-manifest "$runman" \
    --exclude-classes form_field,checkbox \
    --postprocess '{"text_threshold":0.3,"text_expand":0.4,"layout_threshold":0.6,"layout_expand":0.85}' >/dev/null
  uv run python -m export.add_postprocess_rule "$out/raw-a$a.onm" "$out/rule-a$a.onm" --junk-max 1000000 --version "$version" >/dev/null
  uv run python -m export.add_faint_rule "$out/rule-a$a.onm" "$out/det-a$a.onm" --threshold 0.02 --bg-max 110 --min-cells 6 \
    --min-aspect 1.5 --min-confidence 0.9 --version "$version" >/dev/null
  rm -f "$out/raw-a$a.onm" "$out/rule-a$a.onm"
  sha256sum "$out/det-a$a.onm" | tee -a "$out/SHA256SUMS"
done
