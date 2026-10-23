#!/usr/bin/env bash
# Builds a rendered line dataset from a line-set config: font coverage, plan, crops for every set, split checks.
#   gen/make_lines.sh configs/lines-v1.yaml OUT_DIR [PAGE_TRUTH_DIR]
# Needs: ONEOCR_FONT_ROOT and ONEOCR_TEXT_ROOT (see data/fonts.py, data/texts.py fetch), dotnet and uv on PATH.
set -euo pipefail
cfg=$1
out=$2
truth=${3:-}
cd "$(dirname "$0")/.."
: "${ONEOCR_FONT_ROOT:?}" "${ONEOCR_TEXT_ROOT:?}"
dotnet build gen/LineGen -c Release 2>&1 | grep -E 'error|Warn' || true
gen="dotnet out/dotnet/bin/LineGen/release/LineGen.dll"
mkdir -p "$out"
uv run python -c "import yaml;print('\n'.join(f['file'] for f in yaml.safe_load(open('data/fonts.yaml'))['fonts']))" > "$out/fonts.txt"
$gen --coverage "$out/fonts.txt" --charset data/charset-v1.txt --out "$out/coverage.json"
export ONEOCR_COVERAGE=$out/coverage.json
uv run python -m gen.plan "$cfg" "$out"
for s in $(uv run python -c "import yaml,sys;print(' '.join(yaml.safe_load(open('$cfg'))['sets']))"); do
  $gen --plan "$out/$s.plan.jsonl" --out "$out/$s"
done
uv run python -m gen.check "$out" $truth
