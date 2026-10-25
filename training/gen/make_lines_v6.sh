#!/usr/bin/env bash
# Renders the lines-v6 sets: plan, clean packs for every set (CPU), then the validation and calibration sets degraded once on the GPU.
# The training pack and the stream stay clean (the training loop degrades them with fresh draws).
#   gen/make_lines_v6.sh OUT_DIR [render|materialize]     (no second argument: both)
# Needs ONEOCR_FONT_ROOT, ONEOCR_TEXT_ROOT, dotnet and uv on PATH. `materialize` uses the GPU.
set -euo pipefail
out=$(realpath -m "$1")
stage=${2:-both}
cd "$(dirname "$0")/.."
: "${ONEOCR_FONT_ROOT:?}" "${ONEOCR_TEXT_ROOT:?}"
cfg=configs/lines-v6.yaml
if [ "$stage" != materialize ]; then
  mkdir -p "$out"
  dotnet build gen/LineGen -c Release 2>&1 | grep -E ' error |rror\(s\)' || true
  gen="dotnet out/dotnet/bin/LineGen/release/LineGen.dll"
  uv run python -c "import yaml;print('\n'.join(f['file'] for f in yaml.safe_load(open('data/fonts.yaml'))['fonts']))" > "$out/fonts.txt"
  $gen --coverage "$out/fonts.txt" --charset data/charset-v1.txt --out "$out/coverage.json"
  export ONEOCR_COVERAGE=$out/coverage.json
  uv run python -m gen.plan "$cfg" "$out"
  for s in $(uv run python -c "import yaml;print(' '.join(yaml.safe_load(open('$cfg'))['sets']))"); do
    $gen --plan "$out/$s.plan.jsonl" --out "$out/clean/$s" --clean --height 48
  done
fi
if [ "$stage" != render ]; then
  uv run python -m degrade.materialize "$cfg" "$out" figures_val small_val reader_val calib
fi
