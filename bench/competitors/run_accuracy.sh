#!/usr/bin/env bash
# Accuracy-only PaddleOCR runs (text and word boxes), niced and without the machine lock.
#
#   bash competitors/run_accuracy.sh OUT [VARIANT...]     # default variants: v6-default v5-mobile-en v5-server-en v5-mobile v5-server
#   A variant written doctr:default or doctr:mobile runs docTR (output OUT/doctr-<variant>-default); any other is a PaddleOCR variant.
#   SETS (names under PAGES_ROOT; default: every directory there with a pages/ subdirectory), PAGES_ROOT, ROOT, THREADS, LIMIT are environment.
#   HOLD_LOCK_OK=1: the caller holds the machine lock itself (a locked job), so the lock file is not a reason to stop.
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
sets="${SETS:-}"
if [[ -z "$sets" ]]; then sets=$(for d in "$pages"/*/; do [[ -d "$d/pages" ]] && basename "$d"; done); fi
export PADDLE_PDX_CACHE_HOME="$root/cache/paddlex" PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
       HF_HOME="$root/cache/hf" TORCH_HOME="$root/cache/torch" DOCTR_CACHE_DIR="$root/cache/doctr"
mkdir -p "$out"
for v in "${variants[@]}"; do
  for set in $sets; do
    case "$v" in doctr:*) engine=doctr; variant="${v#doctr:}"; extra=();; *) engine=paddle; variant="$v"; extra=(--det-side 1920);; esac
    d="$out/$engine-$variant-default/$set"
    [[ -e "$d/timing.json" ]] && continue
    # never run beside a locked job: stop here, rerun later (finished pairs are skipped)
    [[ -z "${HOLD_LOCK_OK:-}" && -e "$lock" ]] && { echo "stopping: machine lock held: $(cat "$lock")"; exit 75; }
    nice -n 10 "$root/venv-$engine/bin/python" "$here/run.py" --engine "$engine" --variant "$variant" --pages "$pages/$set/pages" \
        --out "$d" --threads "$threads" "${extra[@]}" ${LIMIT:+--limit $LIMIT} 2>&1 | tail -1 | cut -c1-300
  done
done
