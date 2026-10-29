#!/usr/bin/env bash
# One command from an eval root to the quality table against every competitor (accuracy only, no timing).
#
#   MODEL=… DETECTOR=… WORDLM=… bash bench/scripts/full_run.sh ROOT OUT          # run (or resume)
#   bash bench/scripts/full_run.sh --plan ROOT OUT                               # print what would run, the Apple Vision commands, and exit
#
# ROOT is an eval root: <set>/pages and <set>/truth for every set, any names (the six frozen sets and `*-real` sets alike).
# ROOT is only read. OUT receives everything else:
#   OUT/tess/<set>/          Tesseract best and fast, PSM sweep (tesseract.json + hypotheses)
#   OUT/competitors/         PaddleOCR configurations and docTR default/mobile (<config>-default/<set>/)
#   OUT/vision/              Apple Vision output, copied back from the Mac (optional, see the commands --plan prints)
#   OUT/frozen/              `competitors-freeze` of the two directories above: scores.json, SHA256SUMS
#   OUT/work/                our engine's outputs for the gate and the bench build
#   OUT/gate/                gate.md, gate.json, accuracy.json: the table
#   OUT/steps.tsv, OUT/load.log, OUT/steps/*.done
#
# Steps, in order: check the root, Tesseract per set, each PaddleOCR variant, each docTR variant, freeze, gate accuracy with the
# frozen directory as the competitor baseline. The job takes ~/WORKER-BENCH.lock first (noclobber; exits 75 if it is held) and
# releases it however it ends, runs niced, and stops at one overall timeout. Every step leaves OUT/steps/<name>.done when it is
# complete, and a rerun skips completed steps (Tesseract also skips finished (model, PSM) pairs, PaddleOCR/docTR finished
# (variant, set) pairs, the gate finished sets), so a stopped run continues where it stopped. To add the Apple Vision rows later,
# copy its output to OUT/vision and run the same command again: only freeze and gate rerun.
#
# Environment (defaults in brackets):
#   MODEL DETECTOR WORDLM   recognizer, detector and character-model files (required; their SHA-256s are in the gate report)
#   WORDFREQ  [none]  word frequency table (the shipped pipeline has one; pass it to measure what ships)
#   TESS      [~/tools/tesseract/tess]               Tesseract wrapper: `tess MODEL IMAGE OUTBASE ...`, stock tessdata_best/fast
#   COMP_ROOT [~/oneocr/bench/competitors]           PaddleOCR / docTR virtual environments and model caches (competitors/setup.sh)
#   PSMS [3 4 6 11]  PADDLE_VARIANTS [v6-default v5-mobile-en v5-server-en v5-mobile v5-server]  DOCTR_VARIANTS [default mobile]  (set empty to skip an engine)
#   JOBS [8]  processes/threads for Tesseract pages and competitor engines (accuracy does not depend on it)
#   COMP_PAR [1]  competitor variants run at the same time, each with JOBS threads
#   SETS      [every set found under ROOT]
#   TIMEOUT   [14400] overall seconds        LOCK [~/WORKER-BENCH.lock]
#   DOTNET_DIR [~/dotnet10, when it exists]
#
# Estimated worker time: the table printed by --plan (about 12.5 h for ~520 pages with COMP_PAR=1, 4-5 h with COMP_PAR=5).
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
bench="$(cd "$here/.." && pwd)"

plan=0
if [[ "${1:-}" == "--plan" ]]; then plan=1; shift; fi
ROOT="${1:?usage: full_run.sh [--plan] ROOT OUT}"; OUT="${2:?usage: full_run.sh [--plan] ROOT OUT}"
ROOT="$(cd "$ROOT" && pwd)"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

TESS="${TESS:-$HOME/tools/tesseract/tess}"
COMP_ROOT="${COMP_ROOT:-$HOME/oneocr/bench/competitors}"
PSMS="${PSMS:-3 4 6 11}"
PADDLE_VARIANTS="${PADDLE_VARIANTS-v6-default v5-mobile-en v5-server-en v5-mobile v5-server}"
DOCTR_VARIANTS="${DOCTR_VARIANTS-default mobile}"
JOBS="${JOBS:-8}"
COMP_PAR="${COMP_PAR:-1}"
TIMEOUT="${TIMEOUT:-14400}"
LOCK="${LOCK:-$HOME/WORKER-BENCH.lock}"
if [[ -z "${SETS:-}" ]]; then
  SETS=$(for d in "$ROOT"/*/; do [[ -d "$d/pages" && -d "$d/truth" ]] && basename "$d"; done | sort | tr '\n' ' ')
fi
SETS="${SETS% }"

# ---- plan -----------------------------------------------------------------------------------------------------------------
vision_commands() {
  local mac="${MAC:-mac}" dir="${MAC_DIR:-oneocr/fullrun}"
  cat <<EOF
# Apple Vision (on the Mac; only the page images leave this machine, truth stays here):
rsync -a --delete $bench/vision/ $mac:$dir/vision/
$(for s in $SETS; do echo "rsync -a --delete $ROOT/$s/pages/ $mac:$dir/pages/$s/"; done)
ssh $mac 'cd $dir && bash vision/run.sh pages vout'
rsync -a $mac:$dir/vout/ $OUT/vision/
# then fold it in: the same command again reruns only freeze and gate
MODEL=\$MODEL DETECTOR=\$DETECTOR WORDLM=\$WORDLM bash $here/full_run.sh $ROOT $OUT
EOF
}

count_pages() { find "$ROOT/$1/pages" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l; }

print_plan() {
  echo "eval root: $ROOT"; echo "out:       $OUT"
  local total=0 n
  for s in $SETS; do n=$(count_pages "$s"); total=$((total + n)); echo "  set $s: $n pages, $(ls "$ROOT/$s/truth"/*.txt 2>/dev/null | wc -l) truth files"; done
  echo "  total $total pages"
  echo "steps: check; tesseract best+fast PSM {$PSMS} per set; paddle {$PADDLE_VARIANTS}; doctr {$DOCTR_VARIANTS}; freeze; gate"
  echo "lock $LOCK, timeout ${TIMEOUT}s, niced, JOBS=$JOBS"
  echo
  vision_commands
  echo
  cat <<EOT
estimate, measured 2026-10-25 on the 32-thread worker (load 5-7 from the run itself), seconds per page:
                        A4-size page (300 DPI)   screenshot / receipt-size page
  Paddle v6-default      30                       10
  Paddle v5-mobile-en    11                       3.1
  Paddle v5-mobile       12                       3.3
  Paddle v5-server-en    34                       16
  Paddle v5-server       37                       17
  docTR default / mobile 2.5 / 0.6                0.9 / 0.45
  Tesseract, 8 runs      1.2 (wall, JOBS=8)       0.3
  ours (gate)            about 1                  about 0.5   (not measured here)
For ~520 pages (about 220 A4-size, 300 small): Paddle one variant after another ~11.8 h, docTR ~0.3 h, Tesseract ~0.2 h, gate ~0.2 h:
  about 12.5 h with COMP_PAR=1 (needs TIMEOUT above the default 14400), about 4-5 h with COMP_PAR=5 (the v5-server variants bound it, ~3.7 h each).
  Per set, multiply its pages by the row sums: A4-size ~125 s/page for Paddle, small ~50 s/page; plus ~5 s model load per variant and set.
EOT
}

if [[ $plan == 1 ]]; then print_plan; exit 0; fi

: "${MODEL:?MODEL (recognizer .onm) is required}" "${DETECTOR:?DETECTOR is required}" "${WORDLM:?WORDLM (charlm-v1.bin) is required}"
[[ -n "$SETS" ]] || { echo "no sets under $ROOT"; exit 2; }

# ---- lock, timeout, load sampling (outer shell) ----------------------------------------------------------------------------
if [[ -z "${FULL_RUN_INNER:-}" ]]; then
  ( set -o noclobber; echo "oneocr full run (accuracy) $(date -Is) root=$ROOT expected_end=$(date -Is -d "+${TIMEOUT} seconds")" > "$LOCK" ) 2>/dev/null \
    || { echo "worker lock busy: $(cat "$LOCK" 2>/dev/null)"; exit 75; }
  mkdir -p "$OUT/steps"
  sampler=""
  cleanup() { [[ -n "$sampler" ]] && kill "$sampler" 2>/dev/null; echo "$(date -Is) after $(cat /proc/loadavg)" >> "$OUT/load.log"; rm -f "$LOCK"; }
  trap cleanup EXIT
  trap 'exit 143' TERM INT
  echo "$(date -Is) before $(cat /proc/loadavg)" >> "$OUT/load.log"
  ( while :; do echo "$(date -Is) during $(cat /proc/loadavg)" >> "$OUT/load.log"; sleep 60; done ) &
  sampler=$!
  FULL_RUN_INNER=1 nice -n 10 timeout "$TIMEOUT" bash "$0" "$ROOT" "$OUT"
  rc=$?
  [[ $rc == 124 ]] && echo "stopped at the ${TIMEOUT}s timeout; run the same command again to continue"
  exit $rc
fi

# ---- inner ------------------------------------------------------------------------------------------------------------------
[[ -d "$HOME/dotnet10" || -n "${DOTNET_DIR:-}" ]] && { d="${DOTNET_DIR:-$HOME/dotnet10}"; export PATH="$d:$PATH" DOTNET_ROOT="$d"; }
export PATH="$HOME/.local/bin:$PATH" DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 MSBUILDDISABLENODEREUSE=1
export ONEOCR_ARTIFACTS="${ONEOCR_ARTIFACTS:-$OUT/artifacts}"
cd "$bench" || exit 2
py=(uv run python -m oneocr_bench)

step() { # step NAME COMMAND...  (skips a completed step, records seconds)
  local name="$1"; shift
  if [[ -e "$OUT/steps/$name.done" ]]; then echo "skip $name (done)"; return 0; fi
  echo "== $name  $(date -Is)  load $(cut -d' ' -f1-3 /proc/loadavg)"
  local t0=$SECONDS
  "$@" || { echo "FAILED $name"; exit 1; }
  printf '%s\t%s\n' "$name" "$((SECONDS - t0))" >> "$OUT/steps.tsv"
  touch "$OUT/steps/$name.done"
}

check_root() {
  "${py[@]}" gate --dry-run --eval "$ROOT" --sets $SETS | tee "$OUT/check.txt"
  ! grep -q MISMATCH "$OUT/check.txt"
}

tess_set() { # all (model, PSM) pairs of one set; finished pairs are reused
  "${py[@]}" tess --set "$1" --root "$ROOT" --results "$OUT/tess/$1" --psms $PSMS --tess "$TESS" --jobs "$JOBS" --no-timing-runs > "$OUT/logs/tess-$1.log" 2>&1
}

competitor() { # competitor VARIANT   (paddle variant, or doctr:NAME)
  HOLD_LOCK_OK=1 PAGES_ROOT="$ROOT" SETS="$SETS" ROOT="$COMP_ROOT" THREADS="$JOBS" \
    bash "$bench/competitors/run_accuracy.sh" "$OUT/competitors" "$1" > "$OUT/logs/comp-${1//:/-}.log" 2>&1
  local engine variant s
  case "$1" in doctr:*) engine=doctr; variant="${1#doctr:}";; *) engine=paddle; variant="$1";; esac
  for s in $SETS; do [[ -e "$OUT/competitors/$engine-$variant-default/$s/timing.json" ]] || { echo "incomplete: $engine-$variant $s (see $OUT/logs)"; return 1; }; done
}

freeze() {
  rm -rf "$OUT/frozen"
  local srcs=(--src "$OUT/competitors")
  [[ -d "$OUT/vision" ]] && srcs+=(--src "vision-=$OUT/vision")
  "${py[@]}" competitors-freeze "${srcs[@]}" --eval "$ROOT" --out "$OUT/frozen" > "$OUT/logs/freeze.log" 2>&1
}

gate() {
  local reuse=() s
  local all=1
  for s in $SETS; do [[ -e "$OUT/work/out/$s/timing.json" ]] || all=0; done
  [[ $all == 1 ]] && reuse=(--reuse)
  local wf=()
  [[ -n "${WORDFREQ:-}" ]] && wf=(--wordfreq "$WORDFREQ")
  "${py[@]}" gate --model "$MODEL" --detector "$DETECTOR" --wordlm "$WORDLM" "${wf[@]}" --eval "$ROOT" --sets $SETS \
      --competitors "$OUT/frozen" --tess-results "$OUT/tess" --work "$OUT/work" --out "$OUT/gate" "${reuse[@]}" > "$OUT/logs/gate.log" 2>&1
  local rc=$?
  tail -n 5 "$OUT/logs/gate.log"
  return $rc
}

mkdir -p "$OUT/logs" "$OUT/steps" "$OUT/competitors"
{ echo "start $(date -Is)"; lscpu | grep -E 'Model name|^CPU\(s\)'; "$TESS" best --version 2>&1 | head -1; dotnet --version; echo "sets: $SETS"; } > "$OUT/environment.txt" 2>&1

step check check_root
for s in $SETS; do step "tess-$s" tess_set "$s"; done
# Competitor variants are independent: COMP_PAR of them run at once (each with THREADS=JOBS); accuracy does not depend on it.
names=()
for v in $PADDLE_VARIANTS; do
  names+=("paddle-$v")
  ( step "paddle-$v" competitor "$v" ) &
  while (( $(jobs -rp | wc -l) >= COMP_PAR )); do wait -n; done
done
for v in $DOCTR_VARIANTS; do
  names+=("doctr-$v")
  ( step "doctr-$v" competitor "doctr:$v" ) &
  while (( $(jobs -rp | wc -l) >= COMP_PAR )); do wait -n; done
done
wait
for n in "${names[@]}"; do [[ -e "$OUT/steps/$n.done" ]] || { echo "FAILED $n"; exit 1; }; done
fsig=$(echo "$(ls "$OUT/competitors" 2>/dev/null) $(ls "$OUT/vision" 2>/dev/null)" | md5sum | cut -c1-8)
rm -f $(ls "$OUT"/steps/freeze-*.done "$OUT"/steps/gate-*.done 2>/dev/null | grep -v -e "freeze-$fsig" -e "gate-$fsig")
step "freeze-$fsig" freeze
step "gate-$fsig" gate

echo "done $(date -Is)" >> "$OUT/environment.txt"
echo "--- steps (seconds)"; cat "$OUT/steps.tsv"
echo "--- load (1 min): $(awk '$2=="during"{n++; s+=$3; if($3>m)m=$3} END{if(n)printf "mean %.1f max %.1f over %d samples", s/n, m, n}' "$OUT/load.log")"
echo "table: $OUT/gate/gate.md"
[[ -d "$OUT/vision" ]] || { echo; echo "Apple Vision rows are not in this table yet:"; vision_commands; }
