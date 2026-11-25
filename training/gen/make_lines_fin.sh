#!/usr/bin/env bash
# Renders the lines-fin sets (financial strings and the replay of the production mix): plan, clean packs for every set (CPU), then the
# validation, evaluation and calibration sets degraded once on the GPU. Training packs stay clean (the training loop degrades them).
#   gen/make_lines_fin.sh OUT_DIR [render|materialize]
# Needs ONEOCR_FONT_ROOT, ONEOCR_TEXT_ROOT, dotnet and uv on PATH. `materialize` uses the GPU.
set -euo pipefail
out=$(realpath -m "$1")
stage=${2:-both}
cd "$(dirname "$0")/.."
export CFG=configs/lines-fin.yaml
if [ "$stage" != materialize ]; then
  gen/make_lines_r1.sh "$out" render
fi
if [ "$stage" != render ]; then
  SETS=$(uv run python -c "import yaml;print(' '.join(k for k in yaml.safe_load(open('$CFG'))['sets'] if not k.endswith('train')))") \
    gen/make_lines_r1.sh "$out" materialize
fi
