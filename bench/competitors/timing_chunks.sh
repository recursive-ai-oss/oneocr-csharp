#!/usr/bin/env bash
# Timing and peak-RSS rows for PaddleOCR and docTR in short chunks: one configuration x one set per chunk, each chunk
# claiming ~/WORKER-BENCH.lock atomically and releasing it at the end, so other jobs get the machine between chunks.
#
#   bash competitors/timing_chunks.sh [ROOT]
#
# A chunk is skipped (and retried after a back-off of BACKOFF seconds, default 600) when the lock is taken or when the
# 1-minute load average is above MAXLOAD (default 3) once the lock is ours. Each chunk times the first N_DEFAULT pages
# (default 5) with the engine's own threading and the first N_ONE pages (default 3) with one thread; accuracy comes from
# the full runs of run_all.sh. Output: ROOT/out-timing/<config>-{default,1t}/<set>/timing.json (+ the texts).
set -uo pipefail
root="${1:-$HOME/oneocr/bench/competitors}"
bench="${ONEOCR_BENCH_DIR:-$HOME/oneocr/bench}"
here="$(cd "$(dirname "$0")" && pwd)"
lock="$HOME/WORKER-BENCH.lock"
n_default="${N_DEFAULT:-5}"; n_one="${N_ONE:-3}"; backoff="${BACKOFF:-600}"; maxload="${MAXLOAD:-3}"

export PADDLE_PDX_CACHE_HOME="$root/cache/paddlex" HF_HOME="$root/cache/hf" TORCH_HOME="$root/cache/torch" \
       DOCTR_CACHE_DIR="$root/cache/doctr" PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
configs=(${CONFIGS:-paddle:v5-mobile paddle:v5-server doctr:default doctr:mobile})
sets=(${SETS:-clean300-v1 degraded-v1 screenshot-v1 layout-v1 phone-v1 receipts-v1})

chunk() {  # engine variant set
  local engine="$1" variant="$2" set="$3" extra=()
  [[ "$engine" == paddle ]] && extra=(--det-side 1920)
  ( set -o noclobber; echo "oneocr competitor timing $engine:$variant $set $(date -Is) expected_end=+15min" > "$lock" ) 2>/dev/null || return 3
  trap 'rm -f "$lock"' EXIT
  local rc=0
  if awk -v m="$maxload" '{exit !($1 < m)}' /proc/loadavg; then
    local py="$root/venv-$engine/bin/python"
    "$py" "$here/run.py" --engine "$engine" --variant "$variant" --pages "$bench/$set/pages" --out "$root/out-timing/$engine-$variant-default/$set" \
        --threads 0 --limit "$n_default" "${extra[@]}" 2>&1 | tail -1 | cut -c1-200
    "$py" "$here/run.py" --engine "$engine" --variant "$variant" --pages "$bench/$set/pages" --out "$root/out-timing/$engine-$variant-1t/$set" \
        --threads 1 --limit "$n_one" "${extra[@]}" 2>&1 | tail -1 | cut -c1-200
  else
    rc=3
  fi
  rm -f "$lock"; trap - EXIT
  return $rc
}

for c in "${configs[@]}"; do
  for set in "${sets[@]}"; do
    until chunk "${c%%:*}" "${c##*:}" "$set"; do sleep "$backoff"; done
  done
done
