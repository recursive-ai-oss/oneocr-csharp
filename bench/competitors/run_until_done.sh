#!/usr/bin/env bash
# Runs run_accuracy.sh for one engine variant in resumable chunks of at most 30 minutes through the machine's job queue, until every set
# has finished. The scripts skip finished pages, so a chunk that hits its time limit loses nothing.
#
#   bash competitors/run_until_done.sh OUT VARIANT [CORES]       (PAGES_ROOT, THREADS and the other variables of run_accuracy.sh apply)
set -uo pipefail
out="${1:?output directory}"; variant="${2:?variant}"; cores="${3:-8}"
here="$(cd "$(dirname "$0")" && pwd)"
pages="${PAGES_ROOT:?PAGES_ROOT}"
engine=paddle; name="$variant"; case "$variant" in doctr:*) engine=doctr; name="${variant#doctr:}";; esac
for _ in $(seq 40); do
  left=0
  for d in "$pages"/*/; do [[ -d "$d/pages" ]] && [[ ! -e "$out/$engine-$name-default/$(basename "$d")/timing.json" ]] && left=$((left + 1)); done
  [[ $left -eq 0 ]] && { echo "$variant: all sets finished"; exit 0; }
  wq run --class heavy --cores "$cores" --max 30m --label "oneocr fin dev: $variant ($left sets left)" -- bash "$here/run_accuracy.sh" "$out" "$variant" 2>&1 | tail -2
done
[[ $left -eq 0 ]]
