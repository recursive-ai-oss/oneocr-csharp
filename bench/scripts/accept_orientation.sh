#!/usr/bin/env bash
# Orientation and deskew acceptance on the benchmark host, from the repository root.
#   accept_orientation.sh <label> upright    text and orientation of every set, diffed against a reference text directory (REF=<dir>/<set>/pNNN.txt)
#   accept_orientation.sh <label> rotated    angle and orientation on the rotated copies (ROT=<dir>), and their page CER against the upright truth
# Rotated copies come from make_rotated.py. Outputs go to ~/oneocr/u27/acc/<label>/.
set -eu
label=$1; mode=$2
export PATH=~/dotnet10:$PATH DOTNET_ROOT=~/dotnet10 DOTNET_CLI_TELEMETRY_OPTOUT=1
model=${MODEL:-~/oneocr/u9/exports/crnn-v2/model.onm}
eval_dir=~/oneocr/eval/bench-v1
out=~/oneocr/u27/acc/$label
mkdir -p "$out"
bin=$out/bin
if [ ! -f "$bin/OneOcr.Bench.dll" ]; then
  dotnet build bench/OneOcr.Bench -c Release -v quiet -clp:NoSummary -o "$bin" >"$out/build.log" 2>&1 || { tail -20 "$out/build.log"; exit 1; }
fi
run() { # pages-dir out-dir degree
  mkdir -p "$2"
  dotnet "$bin/OneOcr.Bench.dll" run --model "$model" --pages "$1" --out "$2" --degree "$3" --repeat 1 --meta 1 >/dev/null
}
export -f run; export bin model

if [ "$mode" = upright ]; then
  for s in clean300-v1 degraded-v1 layout-v1 screenshot-v1 phone-v1 receipts-v1; do
    run "$eval_dir/$s/pages" "$out/new/$s" 4
    n=$(ls "$eval_dir/$s/pages" | wc -l)
    changed=$(for f in "$out/new/$s"/p*.txt "$out/new/$s"/r*.txt; do [ -f "$f" ] || continue; b=$(basename "$f"); cmp -s "$f" "${REF:?}/$s/$b" || echo "$b"; done | tr '\n' ' ')
    turned=$(grep -L '"orientation":0,"skew":0}' "$out/new/$s"/*.meta.json 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')
    echo "$s: pages $n; text differs from reference: ${changed:-none}; orientation/skew not zero: ${turned:-none}"
  done
else
  rot=${ROT:-~/oneocr/u27/rot}
  jobs=()
  for v in $(ls "$rot" | grep -v expected.json); do
    for s in $(ls "$rot/$v"); do jobs+=("$v/$s"); done
  done
  printf '%s\n' "${jobs[@]}" | xargs -P 6 -I{} bash -c 'v=$(dirname {}); s=$(basename {}); nice bash -c "run \"'"$rot"'/{}\" \"'"$out"'/rot/{}\" 1"'
  python3 bench/scripts/check_rotated.py "$out/rot" "$rot/expected.json" | tail -40
  # Page CER of every rotated copy against the upright truth, next to the upright pages' own CER.
  (cd bench && for s in clean300-v1 layout-v1; do
    echo -n "upright $s: "; uv run python -m oneocr_bench score "$eval_dir/$s" "${UPRIGHT:?}/$s" 2>&1 | tail -1 | cut -c1-90
    for v in $(ls "$rot" | grep -v expected.json); do echo -n "$v $s: "; uv run python -m oneocr_bench score "$eval_dir/$s" "$out/rot/$v/$s" 2>&1 | tail -1 | cut -c1-90; done
  done)
fi
