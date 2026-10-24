#!/usr/bin/env bash
# Peak RSS of `OneOcr.Bench run` per set and degree, one process each.
# usage: rss_rows.sh <label> [sets...]   (cwd: repository root; model and eval sets as on the benchmark host)
set -eu
label=$1; shift
sets=${*:-clean300-v1 phone-v1}
export PATH=~/dotnet10:$PATH DOTNET_ROOT=~/dotnet10 DOTNET_CLI_TELEMETRY_OPTOUT=1
out=~/oneocr/u27/mem/$label
mkdir -p "$out"
dotnet build bench/OneOcr.Bench -c Release -v quiet -clp:NoSummary -o "$out/bin" >"$out/build.log" 2>&1 || { tail -20 "$out/build.log"; exit 1; }
for s in $sets; do
  for d in 1 4; do
    mkdir -p "$out/$s-d$d"
    dotnet "$out/bin/OneOcr.Bench.dll" run --model ~/oneocr/u9/exports/crnn-v2/model.onm \
      --pages ~/oneocr/eval/bench-v1/$s/pages --out "$out/$s-d$d" --degree $d --repeat 1 ${EXTRA:-} >/dev/null
    python3 - "$out/$s-d$d/timing.json" "$label" "$s" "$d" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))
print(f"{sys.argv[2]} {sys.argv[3]} degree {sys.argv[4]}: peak RSS {t['peak_rss_mb']} MB, mean page {t['mean_page_ms']} ms, alloc {t.get('allocated_mb_per_page')} MB/page, gen2 {t.get('gen2_collections')}")
EOF
  done
done
