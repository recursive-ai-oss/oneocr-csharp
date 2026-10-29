#!/usr/bin/env bash
# Data for the tiny-text word-space fine-tune (configs/crnn-tinygap.yaml): Chromium-rendered tiny interface and running text with
# exact DOM truth, plus small-type views of the public-domain web lines that are already rendered.
#   gen/make_tinygap.sh OUT_DIR WEB_DIR          WEB_DIR = gen/web_lines.py output (train/, heldout/)
# Needs ONEOCR_FONT_ROOT, ONEOCR_TEXT_ROOT and a Chromium for Playwright. CPU only, 6 workers, about 15 minutes.
set -euo pipefail
out=$(realpath -m "$1")
web=$(realpath "$2")
cd "$(dirname "$0")/.."
: "${ONEOCR_FONT_ROOT:?}" "${ONEOCR_TEXT_ROOT:?}"
uv run python -m gen.ui_lines --config configs/ui-lines-tiny.yaml --split train --n "${N_TRAIN:-100000}" --seed 113 --out "$out/ui-tiny" --workers 6
uv run python -m gen.ui_lines --config configs/ui-lines-tiny.yaml --split val --n 4000 --seed 113 --out "$out/ui-tiny" --workers 6 --sheet "$out/ui-tiny/val-sheet.png"
# ink height of 14 device px or less: the type size where a word gap is two to four pixels
uv run python -m train.subset_crops --src "$web/train" --out "$out/web-small/train" --max-ink-h 14 --needs-space
uv run python -m train.subset_crops --src "$web/heldout" --out "$out/web-small/heldout" --max-ink-h 14 --needs-space
