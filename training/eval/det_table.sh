#!/usr/bin/env bash
# Runs the bench engine (degree 1, all bundled pins but the detector) over the dev, real-dev, tune and receipt-photo sets with one
# detector file, then scores every set (scorer v3.1) and counts extra words.
#
#   eval/det_table.sh NAME DETECTOR.onm OUT_ROOT [GROUP ...]
#
# Groups: b4 (dev sets), v5 (real-PDF dev sets), rg (receipt photos, dev and calib), t1 t2 t3 (held-out tune sets); default all.
# Environment: MODEL (recognizer .onm), CHARLM, WORDFREQ, B4_ROOT, V5_ROOT, RG_ROOT, T1_ROOT, T2_ROOT, T3_ROOT (set roots; each holds
# <set>/pages and <set>/truth). Light (4 engine processes at a time); run it under `wq run --class heavy --cores 4`.
# Writes OUT_ROOT/NAME/<group>/<set>/ (engine output), scores.txt and extra.txt next to them.
set -euo pipefail
name=$1 det=$2 out=$3
shift 3
cd "$(dirname "$0")/../.."
repo=$PWD
declare -A ROOT SETS
ROOT[b4]=${B4_ROOT:?} SETS[b4]="clean300-v4 clean-real degraded-real degraded-v4 layout-real layout-v4 phone-v4 receipts-v4 screenshot-real screenshot-v4"
ROOT[v5]=${V5_ROOT:?} SETS[v5]="clean-real degraded-real layout-real"
ROOT[rg]=${RG_ROOT:?} SETS[rg]="receipts-gen-dev receipts-gen-calib"
ROOT[t1]=${T1_ROOT:?} SETS[t1]="clean300-tune2 degraded-tune2 layout-tune2 phone-tune2 screenshot-tune2 screenshot-real-tune"
ROOT[t2]=${T2_ROOT:?} SETS[t2]="clean300-tune degraded-tune layout-tune phone-tune screenshot-tune"
ROOT[t3]=${T3_ROOT:?} SETS[t3]="layout-real-tune degraded-real-tune"
groups=("$@")
[ ${#groups[@]} -gt 0 ] || groups=(b4 v5 rg t1 t2 t3)
dotnet build -c Release bench/OneOcr.Bench -v quiet -clp:NoSummary >/dev/null
bin=$(find "${ONEOCR_ARTIFACTS:-$repo/../$(basename "$repo").artifacts}/bin/OneOcr.Bench" -name OneOcr.Bench.dll -path '*release*' | head -1)
M="--model ${MODEL:?} --detector $det --wordlm ${CHARLM:?} --wordfreq ${WORDFREQ:?}"
base=$out/$name
mkdir -p "$base"
i=0
for g in "${groups[@]}"; do
  for s in ${SETS[$g]}; do
    mkdir -p "$base/$g/$s"
    [ -e "$base/$g/$s.done" ] && continue
    (nice -n 10 dotnet "$bin" run $M --pages "${ROOT[$g]}/$s/pages" --out "$base/$g/$s" --degree 1 --words 1 >"$base/$g/$s.log" 2>&1 && touch "$base/$g/$s.done") &
    i=$((i + 1))
    [ $((i % 4)) = 0 ] && wait
  done
done
wait
cd bench
: >"$base/scores.txt"
for g in "${groups[@]}"; do
  for s in ${SETS[$g]}; do
    printf '%-4s %-22s ' "$g" "$s" >>"$base/scores.txt"
    uv run python -m oneocr_bench score --scorer v3.1 --truth "${ROOT[$g]}/$s/truth" --hyp "$base/$g/$s" 2>&1 | grep -E '^pages' |
      sed -E 's/pages ([0-9]+).*CER ([0-9.]+)%  WER ([0-9.]+)%  order ([0-9.]+)%  recall-CER ([0-9.]+)%  recall-WER ([0-9.]+)%.*ECE ([0-9.]+) \(.*/n=\1 \2 \3 \4 | \5 \6 | ECE \7/' >>"$base/scores.txt"
  done
done
cat "$base/scores.txt"
