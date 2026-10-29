#!/usr/bin/env bash
# One locked GPU chunk of a production run (resumable): train/prod_chunk.sh CONFIG_NAME UNTIL_STEP
# Environment: ONEOCR_HOME and the data roots named in the config; the lock is taken here and released however the job ends.
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1; until=$2
ls ~/WORKER-BENCH.lock 2>/dev/null && { cat ~/WORKER-BENCH.lock; exit 75; }
set -o noclobber
echo "oneocr $name until $until $(date -Is) end $(date -Is -d '+115 min')" > ~/WORKER-BENCH.lock || exit 75
set +o noclobber
trap 'rm -f ~/WORKER-BENCH.lock' EXIT
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; uptime
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 6900 uv run python -m train.run "configs/$name.yaml" --until "$until" >> "$ONEOCR_HOME/$name.log" 2>&1
tail -n 2 "$ONEOCR_HOME/$name.log" | cut -c1-250
