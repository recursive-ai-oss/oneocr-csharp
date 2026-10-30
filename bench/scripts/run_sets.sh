#!/usr/bin/env bash
# Reads the pages of the named sets with a built OneOcr.Bench and the bundled-style model files.
#
#   bench/scripts/run_sets.sh BIN_DIR EVAL_ROOT OUT_ROOT MODEL DETECTOR CHARLM WORDFREQ DEGREE SET [SET ...]
#
# BIN_DIR is a directory holding OneOcr.Bench.dll (a copy of a build, so a later rebuild does not change a baseline).
# EXTRA (environment) adds bench options, for example EXTRA="--split 7:3.5".
# OUT_ROOT/<set>/ receives <id>.txt and <id>.json (words with boxes and confidences). Sets that already have run.log are skipped.
set -euo pipefail
bin=$1 eval_root=$2 out_root=$3 model=$4 detector=$5 charlm=$6 wordfreq=$7 degree=$8
shift 8
for s in "$@"; do
  [[ -f "$out_root/$s/run.log" ]] && continue
  mkdir -p "$out_root/$s"
  dotnet "$bin/OneOcr.Bench.dll" run --model "$model" --detector "$detector" --wordlm "$charlm" --wordfreq "$wordfreq" \
    --pages "$eval_root/$s/pages" --out "$out_root/$s" --degree "$degree" --words 1 ${EXTRA:-} >"$out_root/$s/run.log.tmp" 2>&1
  mv "$out_root/$s/run.log.tmp" "$out_root/$s/run.log"
done
