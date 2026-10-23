#!/usr/bin/env bash
# Tesseract baselines for every set, one after another, on an otherwise idle machine.
#
#   ONEOCR_BENCH_DIR=... bash scripts/tess_all.sh
#
# If ONEOCR_LOCK names a file, the script waits until it does not exist, creates it atomically, and removes it on
# exit, so two timing runs on a shared machine never overlap. Load average is recorded in each result file.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${ONEOCR_LOCK:-}" ]]; then
  until ( set -o noclobber; echo "oneocr bench tess $(date -Is)" > "$ONEOCR_LOCK" ) 2>/dev/null; do
    sleep 30
  done
  trap 'rm -f "$ONEOCR_LOCK"' EXIT
fi

# let other work on the machine settle before timing (gives up after 10 minutes and records the load)
for _ in $(seq 20); do awk '{exit !($1 < 3)}' /proc/loadavg && break; sleep 30; done
uptime
for s in clean300-v1 degraded-v1 screenshot-v1 layout-v1 phone-v1 receipts-v1; do
  uv run python -m oneocr_bench tess --set "$s" --psms 3 4 6 11
done
uv run python -m oneocr_bench tess-report
uptime
