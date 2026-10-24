#!/usr/bin/env bash
# Peak RSS and time of each preparation stage alone, per degree. usage: stage_rows.sh <label> [set]
set -eu
label=$1; set=${2:-clean300-v1}
export PATH=~/dotnet10:$PATH DOTNET_ROOT=~/dotnet10 DOTNET_CLI_TELEMETRY_OPTOUT=1
out=~/oneocr/u27/mem/$label
mkdir -p "$out"
dotnet build bench/OneOcr.Bench -c Release -v quiet -clp:NoSummary -o "$out/bin" >"$out/build.log" 2>&1 || { tail -20 "$out/build.log"; exit 1; }
for d in 1 4; do for m in none gray grid find detect prep; do
  echo -n "degree $d "; dotnet "$out/bin/OneOcr.Bench.dll" stage --model ~/oneocr/u9/exports/crnn-v2/model.onm --pages ~/oneocr/eval/bench-v1/$set/pages --mode $m --degree $d
done; done
