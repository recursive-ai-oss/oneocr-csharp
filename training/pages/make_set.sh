#!/usr/bin/env bash
# Builds a page set and checks it:  pages/make_set.sh OUT SEED N_TRAIN N_VAL [WORKERS]
# Needs ONEOCR_FONT_ROOT, ONEOCR_TEXT_ROOT and PLAYWRIGHT_BROWSERS_PATH (see pages/README.md).
set -euo pipefail
out=$1 seed=$2 ntrain=$3 nval=$4 workers=${5:-8}
cd "$(dirname "$0")/.."
uv run python -m pages --n "$ntrain" --seed "$seed" --split train --out "$out" --workers "$workers"
uv run python -m pages --n "$nval" --seed "$seed" --split val --out "$out" --workers "$workers"
uv run python -m pages.check "$out" --split train --n "$(( ntrain < 1000 ? ntrain : 1000 ))" --workers "$workers" --report "$out/check-train.json" > /dev/null || true
uv run python -m pages.check "$out" --split val --n "$(( nval < 250 ? nval : 250 ))" --workers "$workers" --report "$out/check-val.json" > /dev/null || true
