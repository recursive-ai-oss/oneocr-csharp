#!/usr/bin/env bash
# Robustness fine-tune (configs/crnn-robust.yaml). Resumable: UNTIL=step stops there, and the same command continues from the latest checkpoint.
#   train/robust_chunk.sh
# Environment: as configs/crnn-robust.yaml (ONEOCR_HOME with start/best.pt, ONEOCR_PREV, ONEOCR_FT, ONEOCR_V6, ONEOCR_GEN, ONEOCR_MASKS, ONEOCR_WEB,
# ONEOCR_PHONE, ONEOCR_RECF, ONEOCR_TINY, ONEOCR_JIT), ONEOCR_GIT_COMMIT.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --frozen -q
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
uptime
uv run python -m train.run configs/crnn-robust.yaml ${UNTIL:+--until $UNTIL}
