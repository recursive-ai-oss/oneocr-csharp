#!/usr/bin/env bash
# One resumable chunk of the financial fine-tune: makes the starting checkpoint from the released container on the first call,
# then trains up to --until STEP (same command again continues from ckpt-latest.pt).
#   ONEOCR_HOME=... ONEOCR_FIN=... ONEOCR_CONTAINER=crnn.onm train/fin_chunk.sh [CONFIG] UNTIL
set -euo pipefail
cd "$(dirname "$0")/.."
cfg=${1:-configs/crnn-fin.yaml}
until=${2:?until step}
: "${ONEOCR_HOME:?}" "${ONEOCR_FIN:?}" "${ONEOCR_CONTAINER:?}"
export ONEOCR_START=$ONEOCR_HOME/start.pt
[ -e "$ONEOCR_START" ] || uv run python -m export.container_to_checkpoint "$ONEOCR_CONTAINER" "$ONEOCR_START"
uv run python -m train.run "$cfg" --until "$until"
