#!/usr/bin/env bash
# Runs the engine over layout-v1 and screenshot-v1 with and without the detector and scores each run.
#   bench/scripts/table_rows.sh LABEL        (cwd: repository root)
# Required environment:
#   ONEOCR_TABLE_OUT    directory for outputs; results go to $ONEOCR_TABLE_OUT/LABEL/
#   ONEOCR_EVAL         directory holding the bench-v1 sets (layout-v1/, screenshot-v1/)
#   ONEOCR_RECOGNIZER   recognizer model file
#   ONEOCR_DETECTOR     detector model file
#   ONEOCR_ARTIFACTS    build output directory
# Output: $ONEOCR_TABLE_OUT/LABEL/{nodet,det}/<set>/ and score tables in $ONEOCR_TABLE_OUT/LABEL/<config>-<set>.txt
set -euo pipefail
label=${1:?label}
: "${ONEOCR_TABLE_OUT:?set ONEOCR_TABLE_OUT}" "${ONEOCR_EVAL:?set ONEOCR_EVAL}" \
  "${ONEOCR_RECOGNIZER:?set ONEOCR_RECOGNIZER}" "${ONEOCR_DETECTOR:?set ONEOCR_DETECTOR}" \
  "${ONEOCR_ARTIFACTS:?set ONEOCR_ARTIFACTS}"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
out=$ONEOCR_TABLE_OUT/$label
eval=$ONEOCR_EVAL
rec=$ONEOCR_RECOGNIZER
det=$ONEOCR_DETECTOR
mkdir -p "$out"
nice -n 10 dotnet build -c Release bench/OneOcr.Bench -v quiet -clp:NoSummary 2>&1 | tail -3
bin=$(find "$ONEOCR_ARTIFACTS" -name OneOcr.Bench.dll -path '*/bin/*' | head -1)
for cfg in nodet det; do
  for set in layout-v1 screenshot-v1; do
    mkdir -p "$out/$cfg/$set"
    extra=()
    [ "$cfg" = det ] && extra=(--detector "$det")
    nice -n 10 dotnet "$bin" run --model "$rec" "${extra[@]}" --pages "$eval/$set/pages" --out "$out/$cfg/$set" --degree 4
    (cd bench && nice -n 10 uv run python -m oneocr_bench score "$eval/$set" "$out/$cfg/$set" --json "$out/$cfg-$set.json" > "$out/$cfg-$set.txt" 2>&1) || true
    tail -n 12 "$out/$cfg-$set.txt"
  done
done
