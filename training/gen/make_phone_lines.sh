#!/usr/bin/env bash
# Word crops from page sets with word-box truth, cut the way the phone-photo evaluation crops are: each word quad rectified to an
# upright patch, then LineGen cuts it like the library's line finder. For validation sets (selection guards), never training.
#   gen/make_phone_lines.sh NAME=PAGES_DIR:TRUTH_DIR [...] -- OUT_DIR      -> OUT_DIR/NAME/{finder,box,margin}
set -euo pipefail
cd "$(dirname "$0")/.."
args=("$@")
out=$(realpath -m "${args[-1]}")
dotnet build gen/LineGen -c Release 2>&1 | grep -E ' error ' || true
for spec in "${args[@]:0:${#args[@]}-2}"; do
  name=${spec%%=*}; rest=${spec#*=}; pages=${rest%%:*}; truth=${rest#*:}
  mkdir -p "$out/$name"
  uv run python -m eval.page_lines "$pages" "$truth" "$out/$name/lines.jsonl" rectwords
  dotnet out/dotnet/bin/LineGen/release/LineGen.dll --page-lines "$out/$name/lines.jsonl" --out "$out/$name"
done
