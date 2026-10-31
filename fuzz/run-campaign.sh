#!/usr/bin/env bash
# Coverage-guided fuzz campaign: builds the fuzz program, instruments our assemblies (and the PDF and image libraries they call)
# with SharpFuzz, and runs each target under libFuzzer through libfuzzer-dotnet for a time budget, starting from its seed corpus.
#
#   fuzz/run-campaign.sh run [targets...]     build, instrument and run (default: every target)
#   fuzz/run-campaign.sh build                build and instrument only (then `FUZZ_NO_BUILD=1 ... run <target>` in parallel processes)
#   fuzz/run-campaign.sh promote [targets...] copy the minimized corpus of the last run into fuzz/corpus/<target>
#   fuzz/run-campaign.sh repro <target> <file> replay one saved input (crash, timeout, oom) through the instrumented build
#
# Environment:
#   FUZZ_TOOLS    tool directory made by fuzz/setup-tools.sh            (default $HOME/oneocr-fuzz-tools)
#   FUZZ_OUT      output directory: build, corpora, artifacts, logs     (default $PWD/fuzz-out; keep it outside the checkout)
#   FUZZ_BUILD_ARGS  extra dotnet arguments for the build; image-real needs the model files:
#                 "-p:OneOcrModelFile=F -p:OneOcrDetectorFile=F -p:OneOcrLanguageModelFile=F -p:OneOcrWordFrequencyFile=F"
#   FUZZ_SECONDS  time budget per target                                (default 600)
#   FUZZ_TIMEOUT  seconds one input may run before it is a timeout      (default 20)
#   FUZZ_HEAP_MB  managed heap limit of the target process              (default 2048; exceeding it is an OutOfMemoryException finding)
#   PROMOTE_MAX_KB  most new corpus bytes `promote` adds per target (default 256)
#   FUZZ_COMMAND_PREFIX  optional, e.g. "nice -n 10"
# Per target, FUZZ_OUT/<target>/ gets: corpus/ (grown), min/ (minimized), artifacts/ (crash-*, timeout-*, oom-*), run.log, and
# FUZZ_OUT/report.txt a summary row per target.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FUZZ_TOOLS="${FUZZ_TOOLS:-$HOME/oneocr-fuzz-tools}"
FUZZ_OUT="${FUZZ_OUT:-$PWD/fuzz-out}"
FUZZ_SECONDS="${FUZZ_SECONDS:-600}"
FUZZ_TIMEOUT="${FUZZ_TIMEOUT:-20}"
FUZZ_HEAP_MB="${FUZZ_HEAP_MB:-2048}"
ALL_TARGETS="container docextract charlm lexicon wordfreq pagejson review pdf image image-real density normalization llm-anthropic llm-openai llm-gemini llm-schema llm-answer training-files"
INSTRUMENTED="OneOcr.dll OneOcr.Llm.dll OneOcr.Training.dll OneOcr.Pdf.dll OnePdf.dll OneImage.dll OneImage.Pdf.dll"

maxlen() {
  case "$1" in
    container|docextract|charlm) echo 65536 ;;
    pdf) echo 262144 ;;
    image|image-real) echo 65536 ;;
    *) echo 32768 ;;
  esac
}

build() {
  local app="$FUZZ_OUT/app"
  rm -rf "$app"
  mkdir -p "$FUZZ_OUT"
  ( cd "$ROOT" && dotnet publish fuzz/OneOcr.Fuzz -c Release -o "$app" -v quiet -clp:NoSummary ${FUZZ_BUILD_ARGS:-} )
  for dll in $INSTRUMENTED; do
    # sharpfuzz prints and skips an assembly it cannot instrument; one such assembly must not stop the campaign.
    DOTNET_ROLL_FORWARD=Major "$FUZZ_TOOLS/sharpfuzz/sharpfuzz" "$app/$dll" >"$FUZZ_OUT/instrument-$dll.log" 2>&1 \
      || echo "warning: could not instrument $dll (see $FUZZ_OUT/instrument-$dll.log)"
  done
}

run_target() {
  local t="$1" dir="$FUZZ_OUT/$1" start now left status=0 round=0
  rm -rf "$dir"
  mkdir -p "$dir/corpus" "$dir/artifacts"
  cp -r "$ROOT/fuzz/corpus/$t/." "$dir/corpus/"
  start=$(date +%s)
  : >"$dir/run.log"
  while :; do
    now=$(date +%s); left=$(( FUZZ_SECONDS - (now - start) ))
    [ "$left" -ge 5 ] || break
    round=$((round + 1)); [ "$round" -le 50 ] || break
    status=0
    # libFuzzer stops at the first crash, timeout or out-of-memory. The finding is kept in artifacts/ and the run goes on for the
    # time that is left, so one early finding does not hide the others.
    ONEOCR_FUZZ_TARGET="$t" DOTNET_TieredPGO=0 DOTNET_GCHeapHardLimit="$(printf '0x%x' $((FUZZ_HEAP_MB * 1048576)))" \
      ${FUZZ_COMMAND_PREFIX:-} "$FUZZ_TOOLS/libfuzzer-dotnet" \
      --target_path="$(command -v dotnet)" --target_arg="$FUZZ_OUT/app/OneOcr.Fuzz.dll" \
      -max_total_time="$left" -timeout="$FUZZ_TIMEOUT" -rss_limit_mb=4096 -max_len="$(maxlen "$t")" \
      -artifact_prefix="$dir/artifacts/" -print_final_stats=1 "$dir/corpus" >>"$dir/run.log" 2>&1 || status=$?
    [ "$status" -ne 0 ] || break
    # A run that dies at once without leaving an input behind is a setup problem, not a finding.
    if [ $(( $(date +%s) - now )) -lt 5 ] && [ -z "$(ls -A "$dir/artifacts")" ]; then
      echo "target $t: libFuzzer failed to start (status $status); see $dir/run.log" >&2
      return 1
    fi
    echo "target $t: libFuzzer exited with status $status (round $round); finding kept in $dir/artifacts" | tee -a "$dir/run.log"
  done

  # Minimized corpus: the smallest subset of the grown corpus with the same coverage.
  mkdir -p "$dir/min"
  ONEOCR_FUZZ_TARGET="$t" DOTNET_TieredPGO=0 "$FUZZ_TOOLS/libfuzzer-dotnet" \
    --target_path="$(command -v dotnet)" --target_arg="$FUZZ_OUT/app/OneOcr.Fuzz.dll" \
    -merge=1 -timeout="$FUZZ_TIMEOUT" -max_len="$(maxlen "$t")" "$dir/min" "$dir/corpus" >"$dir/merge.log" 2>&1 || true

  local execs ft crashes
  execs=$(grep -oE 'stat::number_of_executed_units: *[0-9]+' "$dir/run.log" | awk '{s+=$2} END{print s+0}')
  ft=$(grep -oE 'ft: [0-9]+' "$dir/run.log" | tail -1 | awk '{print $2}')
  crashes=$(find "$dir/artifacts" -type f | wc -l)
  printf '%-14s executions=%s features=%s seeds=%s corpus=%s(%sKB) min=%s(%sKB) findings=%s\n' \
    "$t" "$execs" "${ft:-?}" "$(find "$ROOT/fuzz/corpus/$t" -type f | wc -l)" \
    "$(find "$dir/corpus" -type f | wc -l)" "$(du -sk "$dir/corpus" | cut -f1)" \
    "$(find "$dir/min" -type f | wc -l)" "$(du -sk "$dir/min" | cut -f1)" "$crashes" | tee -a "$FUZZ_OUT/report.txt"
}

cmd="${1:-run}"; shift || true
case "$cmd" in
  run)
    targets="${*:-$ALL_TARGETS}"
    # FUZZ_NO_BUILD=1 reuses the build and the report of an earlier `build`, so several targets can run side by side (one process each).
    if [ -z "${FUZZ_NO_BUILD:-}" ]; then build; : >"$FUZZ_OUT/report.txt"; fi
    for t in $targets; do run_target "$t"; done
    echo "report: $FUZZ_OUT/report.txt"
    ;;
  build)
    build
    : >"$FUZZ_OUT/report.txt"
    ;;
  promote)
    targets="${*:-$ALL_TARGETS}"
    for t in $targets; do
      [ -d "$FUZZ_OUT/$t/min" ] || continue
      # Names are content hashes, so a promoted input never overwrites a different seed.
      have=$(find "$ROOT/fuzz/corpus/$t" -type f -exec sha1sum {} + | cut -d' ' -f1)
      budget=$(( ${PROMOTE_MAX_KB:-256} * 1024 )); used=0
      # Smallest inputs first, so the committed corpus stays small: at most PROMOTE_MAX_KB (default 256) of new files per target.
      for f in $(ls -Sr "$FUZZ_OUT/$t/min"/* 2>/dev/null); do
        [ -f "$f" ] || continue
        size=$(stat -c %s "$f"); [ $((used + size)) -le "$budget" ] || continue
        full=$(sha1sum "$f" | cut -d' ' -f1)
        grep -q "$full" <<<"$have" || { cp "$f" "$ROOT/fuzz/corpus/$t/${full:0:16}"; used=$((used + size)); }
      done
    done
    ;;
  repro)
    # Instrumented code needs the shared memory libFuzzer sets up, so the input runs under libFuzzer itself (one execution, no mutation).
    ONEOCR_FUZZ_TARGET="$1" "$FUZZ_TOOLS/libfuzzer-dotnet" --target_path="$(command -v dotnet)" --target_arg="$FUZZ_OUT/app/OneOcr.Fuzz.dll" \
      -timeout="$FUZZ_TIMEOUT" "$2" && echo "repro: $2 ran clean"
    ;;
  *)
    sed -n 2,12p "$0"
    exit 2
    ;;
esac
