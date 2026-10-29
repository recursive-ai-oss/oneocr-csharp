#!/usr/bin/env bash
# Cut receipt line crops again with a different engine run, keeping an existing photo split.
#   gen/recut_receipt_crops.sh PHOTOS LABELS ENGINE SPLIT_JSON OUT
# Same inputs as make_receipt_lines.sh, but the train / calibration / dev assignment is taken from SPLIT_JSON (written by `gen.receipt_photos
# split`), so no photo changes part when the engine, and with it the coverage filter, changes. Writes OUT/set/align-<part> and
# OUT/crops/<part>/finder. Needs dotnet and uv; run from training/.
set -euo pipefail
photos=$(realpath "$1") labels=$(realpath "$2") engine=$(realpath "$3") split=$(realpath "$4") out=$(realpath -m "$5")
cd "$(dirname "$0")/.."
dotnet build gen/LineGen -c Release 2>&1 | grep -E ' error ' || true
mkdir -p "$out/set"
cp "$split" "$out/set/split.json"
for p in train calib dev; do
  uv run python -m gen.receipt_photos align --images "$photos" --labels "$labels" --split "$out/set/split.json" --engine "$engine" --part $p --out "$out/set/align-$p"
  dotnet out/dotnet/bin/LineGen/release/LineGen.dll --page-lines "$out/set/align-$p/lines.jsonl" --out "$out/crops/$p"
done
