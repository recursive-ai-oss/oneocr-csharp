#!/usr/bin/env bash
# Runs the engine over layout-v1 and screenshot-v1 with and without the detector and scores each run.
#   bench/scripts/table_rows.sh LABEL        (cwd: repository root, on the worker)
# Output: ~/oneocr/u35/LABEL/{nodet,det}/<set>/ and score tables in ~/oneocr/u35/LABEL/<config>-<set>.txt
set -euo pipefail
label=${1:?label}
export PATH=$HOME/dotnet10:$PATH DOTNET_ROOT=$HOME/dotnet10 DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
export ONEOCR_ARTIFACTS=$HOME/oneocr/u35/art
out=$HOME/oneocr/u35/$label
eval=$HOME/oneocr/eval/bench-v1
rec=$HOME/oneocr/u14/exports/crnn-v4/model-v1.onm
det=$HOME/oneocr/u23/exports/dbnet-lite-v0/0.1.0/model.onm
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
