#!/usr/bin/env bash
# Receipt photo sets and engine-crop line sets from a local directory of generated receipt photos with transcriptions.
#   gen/make_receipt_lines.sh PHOTOS LABELS ENGINE OUT
# PHOTOS: images; LABELS: one JSON per image ({"readable", "lines": [{"text", "sure"}]}); ENGINE: `OneOcr.Bench run --pagejson 1` output on
# PHOTOS. Writes OUT/set/{split.json, qa.json, receipts-gen-dev, receipts-gen-calib} (bench page sets) and OUT/crops/{train,dev,calib}/finder
# (line crops cut at the engine's line boxes; train is the training part, dev is held out, calib fits the confidence map). Needs dotnet and uv.
set -euo pipefail
photos=$(realpath "$1") labels=$(realpath "$2") engine=$(realpath "$3") out=$(realpath -m "$4")
cd "$(dirname "$0")/.."
dotnet build gen/LineGen -c Release 2>&1 | grep -E ' error ' || true
mkdir -p "$out/set"
uv run python -m gen.receipt_photos qa --labels "$labels" --engine "$engine" --out "$out/set/qa.json"
uv run python -m gen.receipt_photos split --images "$photos" --labels "$labels" --out "$out/set"
for p in train calib dev; do
  uv run python -m gen.receipt_photos align --images "$photos" --labels "$labels" --split "$out/set/split.json" --engine "$engine" --part $p --out "$out/set/align-$p"
  dotnet out/dotnet/bin/LineGen/release/LineGen.dll --page-lines "$out/set/align-$p/lines.jsonl" --out "$out/crops/$p"
done
for p in dev calib; do
  uv run python -m gen.receipt_photos pageset --images "$photos" --labels "$labels" --split "$out/set/split.json" --part $p --out "$out/set/receipts-gen-$p"
done
