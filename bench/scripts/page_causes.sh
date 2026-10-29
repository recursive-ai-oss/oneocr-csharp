#!/usr/bin/env bash
# Where the page error comes from, in one command: reads every page of the named sets with the bundled-style model files,
# then runs training/eval/page_causes.py over the word sidecars.
#
#   bench/scripts/page_causes.sh EVAL_ROOT OUT_ROOT MODEL DETECTOR CHARLM WORDFREQ SET [SET ...]
#
# EVAL_ROOT/<set>/{pages,truth} is a bench set root (for example a dev or tune root). OUT_ROOT receives <set>/<id>.txt,
# <id>.json (the engine's output) and causes.txt / causes.json (the report). Needs dotnet (net10.0) and, for the report, the
# bench Python environment (Pillow, rapidfuzz). Light: one process at degree 4; run it under nice on a shared machine.
# PAGE_CAUSES_ARGS is passed to the report (for example "--aggregate --ref DIR").
set -euo pipefail
eval_root=$1 out_root=$2 model=$3 detector=$4 charlm=$5 wordfreq=$6
shift 6
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
dotnet build -c Release "$repo/bench/OneOcr.Bench" -v quiet -clp:NoSummary >/dev/null
bin=$(find "${ONEOCR_ARTIFACTS:-$repo/../$(basename "$repo").artifacts}/bin/OneOcr.Bench" -name OneOcr.Bench.dll -path '*release*' | head -1)
for s in "$@"; do
  mkdir -p "$out_root/$s"
  dotnet "$bin" run --model "$model" --detector "$detector" --wordlm "$charlm" --wordfreq "$wordfreq" \
    --pages "$eval_root/$s/pages" --out "$out_root/$s" --degree 4 --words 1 >"$out_root/$s/run.log"
done
(cd "$repo/bench" && uv run python ../training/eval/page_causes.py --eval "$eval_root" --out "$out_root" --sets "$@" ${PAGE_CAUSES_ARGS:-} \
  --json "$out_root/causes.json") | tee "$out_root/causes.txt"
