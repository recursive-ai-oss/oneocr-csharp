#!/usr/bin/env bash
# Accuracy-only PaddleOCR runs (text and word boxes), niced and without the machine lock.
#
#   bash competitors/run_accuracy.sh OUT [VARIANT...]     # default variants: v6-default v5-mobile-en v5-server-en v5-mobile v5-server
#
# Inputs: PAGES_ROOT (default ~/oneocr/eval/bench-v2) with <set>/pages, ROOT (default ~/oneocr/bench/competitors) with the engine's
# virtual environment and model cache. Output: OUT/paddle-<variant>-default/<set>/. A (variant, set) pair that already has a
# timing.json is skipped, so the script resumes. Timings from this script are not publishable (niced, shared machine); only
# the text and boxes are used. Before every pair it checks that no machine lock file exists and exits (status 75) if one does,
# so it never starts beside a locked job; rerun it later to continue. THREADS (default 8) bounds the threads.
set -uo pipefail
out="${1:?output directory}"; shift
variants=("$@"); [[ ${#variants[@]} -eq 0 ]] && variants=(v6-default v5-mobile-en v5-server-en v5-mobile v5-server)
root="${ROOT:-$HOME/oneocr/bench/competitors}"
pages="${PAGES_ROOT:-$HOME/oneocr/eval/bench-v2}"
threads="${THREADS:-8}"
lock="$HOME/WORKER-BENCH.lock"
here="$(cd "$(dirname "$0")" && pwd)"
export PADDLE_PDX_CACHE_HOME="$root/cache/paddlex" PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
mkdir -p "$out"
for v in "${variants[@]}"; do
  for set in ${SETS:-clean300-v1 degraded-v1 screenshot-v1 layout-v1 phone-v1 receipts-v1}; do
    d="$out/paddle-$v-default/$set"
    [[ -e "$d/timing.json" ]] && continue
    # never run beside a locked job: stop here, rerun later (finished pairs are skipped)
    [[ -e "$lock" ]] && { echo "stopping: machine lock held: $(cat "$lock")"; exit 75; }
    nice -n 10 "$root/venv-paddle/bin/python" "$here/run.py" --engine paddle --variant "$v" --pages "$pages/$set/pages" \
        --out "$d" --threads "$threads" --det-side 1920 ${LIMIT:+--limit $LIMIT} 2>&1 | tail -1 | cut -c1-300
  done
done
