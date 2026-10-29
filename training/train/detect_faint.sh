#!/usr/bin/env bash
# One chunk of a low-contrast detector fine-tune under the worker lock; run again to resume from ckpt-latest.pt. Exits 75 when the
# lock is held.
#   train/detect_faint.sh MINUTES [CONFIG] [LOCK_LABEL]
# Environment: ONEOCR_HOME (run state), ONEOCR_BASE (checkpoint to start from), ONEOCR_DATASETS, ONEOCR_PAGES, ONEOCR_PAGES_DARK,
# ONEOCR_WEB_WINDOWS (see configs/dbnet-web.yaml), ONEOCR_GIT_COMMIT.
set -euo pipefail
cd "$(dirname "$0")/.."
mins=${1:-80}
cfg=${2:-configs/dbnet-faint.yaml}
label=${3:-detector-faint}
set -o noclobber
echo "oneocr $label $(date -Is) end $(date -Is -d "+$((mins + 5)) min")" > ~/WORKER-BENCH.lock || { cat ~/WORKER-BENCH.lock; exit 75; }
trap 'rm -f ~/WORKER-BENCH.lock' EXIT
set +o noclobber
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
uptime
timeout "$((mins * 60))" uv run python -m train.detect "$cfg"
