#!/usr/bin/env bash
# Apple Vision rows for every evaluation set: builds the CLI, then runs four configurations
# (accurate / fast x language correction off / on) over PAGES/<set>/*.{png,jpg}.
#
#   bash vision/run.sh PAGES OUT [cpu]     # "cpu" adds the CPU-only configurations
#
# Output: OUT/<config>/<set>/<id>.txt, <id>.json, timing.json. Score with
#   python -m oneocr_bench score --hyp-root OUT/<config>
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
pages="$1"; out="$2"; cpu="${3:-}"
mkdir -p "$out"
swiftc -O "$here/main.swift" -o "$out/vision-ocr"
{ sw_vers; sysctl -n hw.model machdep.cpu.brand_string hw.ncpu; uptime; } > "$out/environment.txt"
configs=("accurate off" "accurate on" "fast off" "fast on")
for c in "${configs[@]}"; do
  read -r level corr <<<"$c"
  for flag in "" ${cpu:+--cpu-only}; do
    name="$level-corr$corr${flag:+-cpu}"
    for set in clean300-v1 degraded-v1 screenshot-v1 layout-v1 phone-v1 receipts-v1; do
      "$out/vision-ocr" --pages "$pages/$set" --out "$out/$name/$set" --level "$level" --correction "$corr" ${flag:+$flag}
    done
    uptime >> "$out/environment.txt"
  done
done
