#!/usr/bin/env bash
# Cuts line crops from every evaluation page set: lines are grouped from the truth word boxes (eval/page_lines.py) and cropped by
# LineGen three ways (finder / box / margin, see gen/LineGen/Program.cs). Evaluation data only; nothing here feeds training.
#   [RECT=1] eval/make_bench_lines.sh BENCH_ROOT OUT_DIR [set ...]
# RECT=1 cuts the skewed or photographed sets (degraded, phone, receipts) from deskewed pages / rectified word quads, the geometry the
# library's page stage hands to line recognition; without it every crop is axis-aligned in the page as imaged.
# BENCH_ROOT/<set>/{pages,truth}. Needs dotnet and uv on PATH (LineGen built as in gen/make_lines.sh).
set -euo pipefail
root=$1
out=$2
shift 2
sets=${*:-clean300-v1 degraded-v1 screenshot-v1 layout-v1 phone-v1 receipts-v1}
cd "$(dirname "$0")/.."
dotnet build gen/LineGen -c Release 2>&1 | grep -E 'error|Warn' || true
for s in $sets; do
  mkdir -p "$out/$s"
  uv run python -m eval.page_lines "$root/$s/pages" "$root/$s/truth" "$out/$s/lines.jsonl" $(case "$s" in
    degraded-v1) [ -n "${RECT:-}" ] && echo deskew;;
    phone-v1|receipts-v1) [ -n "${RECT:-}" ] && echo rectwords || echo words;;
  esac)
  dotnet out/dotnet/bin/LineGen/release/LineGen.dll --page-lines "$out/$s/lines.jsonl" --out "$out/$s"
done
