#!/usr/bin/env bash
# Cuts the development set `bench-v4-dev`: new pages in the six categories from fresh public seeds (dev-v4-seeds.json), plus an optional
# real-sourced slice, into one eval root with a checksum file. Nothing here touches the pinned manifests of the earlier sets.
#
#   ONEOCR_BENCH_DIR=<fonts, text, assets> AVOID=<earlier eval root> [LISTS=<dir>] bash scripts/cut_dev.sh OUT [synthetic|real|freeze]
#
# OUT/<set>/{pages,truth,MANIFEST.json}: clean300-v4 (30), degraded-v4 (40), screenshot-v4 (40), layout-v4 (40), phone-v4 (40) and
# receipts-v4 (100, from the synthetic receipt generator). With LISTS (a directory holding pages-list.json and pdfs-list.json) the real
# slice adds screenshot-real, clean-real, layout-real and degraded-real. `freeze` writes DEV.json and SHA256SUMS and removes write
# permission. Stages are separate so a failed capture can be repeated without redrawing the synthetic pages; an existing set directory
# is never overwritten.
# Prose of the earlier root AVOID (eight-word runs) is left out of the new pages where the generators draw text from books.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
bench="$(cd "$here/.." && pwd)"
OUT="${1:?usage: cut_dev.sh OUT [synthetic|real|freeze]}"; stage="${2:-all}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
: "${ONEOCR_BENCH_DIR:?set ONEOCR_BENCH_DIR (fonts and texts)}"
AVOID="${AVOID:-$HOME/oneocr/eval/bench-v2}"
export PATH="$HOME/.local/bin:$HOME/dotnet10:$PATH" DOTNET_ROOT="$HOME/dotnet10" DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
cd "$bench"
py=(uv run python -m oneocr_bench)
seed() { python3 -c "import json,sys;print(json.load(open('dev-v4-seeds.json'))[sys.argv[1]])" "$1"; }

synthetic() {
  local src name flags
  for pair in clean300-v1:clean300-v4 degraded-v1:degraded-v4 screenshot-v1:screenshot-v4 layout-v1:layout-v4 phone-v1:phone-v4 receipts-v1:receipts-v4; do
    src=${pair%%:*}; name=${pair##*:}
    [[ -e "$OUT/$name" ]] && { echo "skip $name (exists)"; continue; }
    flags=(); [[ $src == receipts-v1 ]] && flags=(--synthetic-receipts --count 100)
    "${py[@]}" gen --set "$src" --seed "$(seed "$src")" --out "$OUT/$name" --avoid-text "$AVOID" "${flags[@]}"
  done
}

real() {
  : "${LISTS:?LISTS (directory with pages-list.json and pdfs-list.json)}"
  local stage="$OUT/.real"; mkdir -p "$stage"
  if [[ ! -e "$OUT/screenshot-real" ]]; then
    "${py[@]}" realweb capture --list "$LISTS/pages-list.json" --out "$stage/web" --target 40
    cp -r "$stage/web" "$OUT/screenshot-real"
  fi
  if [[ ! -e "$OUT/layout-real" ]]; then
    dotnet build PdfTruth -c Release -o "$stage/pdftruth" 2>&1 | grep -E 'error|Warn|Elapsed' || true
    "${py[@]}" realpdf fetch --list "$LISTS/pdfs-list.json" --out "$stage/pdfs"
    "${py[@]}" realpdf cut --pdfs "$stage/pdfs" --out "$stage/cut" --seeds dev-v4-seeds.json --pdftruth "$stage/pdftruth/PdfTruth.dll" \
      --clean 30 --layout 40 --degraded 40
    for s in clean-real layout-real degraded-real; do cp -r "$stage/cut/$s" "$OUT/$s"; done
    cp "$stage/cut/CUT-LOG.json" "$OUT/CUT-LOG.real.json"
  fi
  find "$OUT" -name page.html -delete
}

freeze() {
  [[ -e "$OUT/DEV.json" ]] && { echo "already frozen"; return; }
  chmod -R u+w "$OUT"; rm -rf "$OUT/.real"
  python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
seeds = json.loads(Path("dev-v4-seeds.json").read_text())
sets = {d.name: len(list((d / "pages").iterdir())) for d in sorted(root.iterdir()) if (d / "pages").is_dir()}
(root / "DEV.json").write_text(json.dumps({"name": "bench-v4-dev", "development set": True, "seeds": seeds, "sets": sets}, indent=1) + "\n")
PY
  (cd "$OUT" && LC_ALL=C find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
  chmod -R a-w "$OUT"
  echo "froze $(wc -l < "$OUT/SHA256SUMS") files; digest $(sha256sum "$OUT/SHA256SUMS" | cut -c1-16)"
}

case "$stage" in
  synthetic) synthetic ;;
  real) real ;;
  freeze) freeze ;;
  all) synthetic; [[ -n "${LISTS:-}" ]] && real; freeze ;;
esac
