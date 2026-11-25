#!/usr/bin/env bash
# Reads the development page sets with a built OneOcr.Bench and one recognizer container (detector, character model and word
# frequencies are the bundled ones), then scores every set: scorer v3.2 on the financial sets, v3.1 on the others.
#   eval/page_eval.sh BIN_DIR OUT_ROOT RECOGNIZER.onm DEGREE
# Set roots come from the environment: ONEOCR_FINDEV (fin-dev), ONEOCR_V4 (bench-v4-dev), ONEOCR_V5 (bench-v5-real-dev), ONEOCR_RG (a
# directory holding receipts-gen-dev); ONEOCR_PARTS (detector, charlm-v1.bin, wordfreq-v2.bin directory) names the other model files.
# Outputs: OUT_ROOT/<root>/<set>/<id>.txt|json, OUT_ROOT/scores/<set>.json. Sets with a score file are skipped, so a killed job resumes.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
bin=$1 out=$2 model=$3 degree=$4
: "${ONEOCR_PARTS:?}" "${ONEOCR_FINDEV:?}" "${ONEOCR_V4:?}" "${ONEOCR_V5:?}" "${ONEOCR_RG:?}"
det=$(ls "$ONEOCR_PARTS"/dbnet-lite-v1-*.onm | head -1)
export PATH=~/dotnet10:$PATH DOTNET_ROOT=~/dotnet10
mkdir -p "$out/scores"
run_set() { # ROOT SET
  local root=$1 s=$2
  [[ -f "$out/$s/run.log" ]] && return 0
  mkdir -p "$out/$s"
  dotnet "$bin/OneOcr.Bench.dll" run --model "$model" --detector "$det" --wordlm "$ONEOCR_PARTS/charlm-v1.bin" \
    --wordfreq "$ONEOCR_PARTS/wordfreq-v2.bin" --pages "$root/$s/pages" --out "$out/$s" --degree "$degree" --words 1 \
    >"$out/$s/run.log.tmp" 2>&1
  mv "$out/$s/run.log.tmp" "$out/$s/run.log"
}
score_set() { # ROOT SET SCORER
  [[ -f "$out/scores/$2.json" ]] && return 0
  (cd "$here/bench" && uv run python -m oneocr_bench score "$1/$2/truth" "$out/$2" --scorer "$3" --json "$out/scores/$2.json" >"$out/scores/$2.txt")
}
jobs_list=()
for s in fin-scan fin-fax fin-phone fin-tables fin-scan-real fin-fax-real fin-phone-real fin-tables-real; do jobs_list+=("$ONEOCR_FINDEV $s v3.2"); done
for s in clean300-v4 degraded-v4 layout-v4 phone-v4 receipts-v4 screenshot-v4 clean-real degraded-real layout-real screenshot-real; do jobs_list+=("$ONEOCR_V4 $s v3.1"); done
for s in clean-real degraded-real layout-real; do jobs_list+=("$ONEOCR_V5 $s v3.1"); done
jobs_list+=("$ONEOCR_RG receipts-gen-dev v3.1")
# the v5 sets share names with v4 sets; their output directories carry a prefix
for j in "${jobs_list[@]}"; do
  read -r root s sc <<<"$j"
  name=$s
  [[ $root == "$ONEOCR_V5" ]] && name=v5-$s
  if [[ $root == "$ONEOCR_V5" ]]; then
    mkdir -p "$out/v5-$s"
    [[ -f "$out/v5-$s/run.log" ]] || { dotnet "$bin/OneOcr.Bench.dll" run --model "$model" --detector "$det" --wordlm "$ONEOCR_PARTS/charlm-v1.bin" \
      --wordfreq "$ONEOCR_PARTS/wordfreq-v2.bin" --pages "$root/$s/pages" --out "$out/v5-$s" --degree "$degree" --words 1 >"$out/v5-$s/run.log.tmp" 2>&1 && mv "$out/v5-$s/run.log.tmp" "$out/v5-$s/run.log"; }
    [[ -f "$out/scores/$name.json" ]] || (cd "$here/bench" && uv run python -m oneocr_bench score "$root/$s/truth" "$out/$name" --scorer "$sc" --json "$out/scores/$name.json" >"$out/scores/$name.txt")
  else
    run_set "$root" "$s"
    score_set "$root" "$s" "$sc"
  fi
done
echo done
