#!/usr/bin/env bash
# PaddleOCR and docTR rows for every evaluation set, one process per (configuration, thread setting, set).
#
#   bash competitors/run_all.sh [ROOT]          # default ROOT: ~/oneocr/bench/competitors
#
# Inputs:  ONEOCR_BENCH_DIR (default ~/oneocr/bench) with <set>/pages. Outputs: ROOT/out/<config>-<threads>/<set>/.
#   <threads> = "default": every page, engine's own threading (accuracy rows and default-threading timing)
#               "1t":      the first SUB pages (default 8) of each set with one CPU thread (timing; the text does
#                          not depend on the thread count)
# Claims ~/WORKER-BENCH.lock atomically first (exits 3 if it is busy) and removes it on exit. Load average, CPU and
# core count are recorded in every timing.json. CONFIGS and SETS can be overridden to run a part.
set -euo pipefail
root="${1:-$HOME/oneocr/bench/competitors}"
bench="${ONEOCR_BENCH_DIR:-$HOME/oneocr/bench}"
sub="${SUB:-8}"
lock="$HOME/WORKER-BENCH.lock"
here="$(cd "$(dirname "$0")" && pwd)"

( set -o noclobber; echo "oneocr competitor baselines (PaddleOCR, docTR) $(date -Is) expected_end=${EXPECTED_END:-about 5h later}" > "$lock" ) 2>/dev/null || { echo "worker lock busy: $(cat "$lock")"; exit 3; }
trap 'rm -f "$lock"' EXIT

export PADDLE_PDX_CACHE_HOME="$root/cache/paddlex" HF_HOME="$root/cache/hf" TORCH_HOME="$root/cache/torch" \
       DOCTR_CACHE_DIR="$root/cache/doctr" PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
configs=(${CONFIGS:-paddle:v5-mobile paddle:v5-server doctr:default doctr:mobile})
sets=(${SETS:-clean300-v1 degraded-v1 screenshot-v1 layout-v1 phone-v1 receipts-v1})
mkdir -p "$root/out"
{ date -Is; uptime; lscpu | grep -E 'Model name|^CPU\(s\)'; cat "$root/versions.txt"; } > "$root/out/environment-start.txt"

for c in "${configs[@]}"; do
  engine="${c%%:*}"; variant="${c##*:}"
  py="$root/venv-$engine/bin/python"
  extra=(); [[ "$engine" == paddle ]] && extra=(--det-side 1920)
  for set in "${sets[@]}"; do
    "$py" "$here/run.py" --engine "$engine" --variant "$variant" --pages "$bench/$set/pages" \
        --out "$root/out/$engine-$variant-default/$set" --threads 0 "${extra[@]}" 2>&1 | tail -1 | cut -c1-300
    "$py" "$here/run.py" --engine "$engine" --variant "$variant" --pages "$bench/$set/pages" \
        --out "$root/out/$engine-$variant-1t/$set" --threads 1 --limit "$sub" "${extra[@]}" 2>&1 | tail -1 | cut -c1-300
  done
done
{ date -Is; uptime; } > "$root/out/environment-end.txt"
