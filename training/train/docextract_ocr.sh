#!/usr/bin/env bash
# Reads page images with the library's own OCR and writes the page model of each: OUT/<id>.page.json (oneocr.page v1).
#   train/docextract_ocr.sh IMAGES OUT MODEL [SHARDS] [LIMIT]
# IMAGES holds .png/.jpg files; the images are split over SHARDS processes (degree 1 each). Needs .NET 10 on PATH (DOTNET_ROOT set) and
# the repository's bench project; ONEOCR_ARTIFACTS (optional) says where the build output goes. The OCR only supplies words and boxes:
# labels come from the generator's truth or CORD (train/docextract_data.py).
set -euo pipefail
images=$1 out=$2 model=$3 shards=${4:-8} limit=${5:-0}
root="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$out"
art=${ONEOCR_ARTIFACTS:-$root/../oneocr.artifacts}
export ONEOCR_ARTIFACTS=$art
dotnet build "$root/bench/OneOcr.Bench" -c Release -v q --nologo >/dev/null
dll=$(find "$art/bin" -name OneOcr.Bench.dll -path '*elease*' | head -1)
work=$(mktemp -d "$(dirname "$out")/.ocr-work.XXXXXX")  # same filesystem as the images, so they can be hard linked
trap 'rm -rf "$work"' EXIT
i=0
ls "$images" | grep -E '\.(png|jpg|jpeg)$' | sort | { if [ "$limit" -gt 0 ]; then head -n "$limit"; else cat; fi; } | while read -r f; do
  s=$((i % shards)); mkdir -p "$work/$s"; ln "$images/$f" "$work/$s/$f" 2>/dev/null || cp "$images/$f" "$work/$s/$f"; i=$((i + 1))
done
# The reader process stops at the first image it cannot decode (seen on a few large grayscale PNGs). Pages already done are removed
# from the shard; the failing one is re-encoded as a lossless RGB PNG and tried again, then dropped if it still fails.
run_shard() {
  s=$1; tries=0
  while :; do
    for f in "$work/$s"/*; do [ -e "$work/out$s/$(basename "${f%.*}").page.json" ] && rm -f "$f"; done
    ls "$work/$s" | grep -qE '\.(png|jpg|jpeg)$' || return 0
    dotnet "$dll" run --model "$model" --pages "$work/$s" --out "$work/out$s" --degree 1 --metaonly 1 --pagejson 1 >>"$work/log$s" 2>&1 && return 0
    first=$(ls "$work/$s" | grep -E '\.(png|jpg|jpeg)$' | head -1); tries=$((tries + 1))
    if [ "$tries" -gt 40 ]; then echo "giving up on shard $s" >&2; return 1; fi
    if [ "${first%.*}.png" = "$first" ] && [ -e "$work/$s/$first.recoded" ]; then rm -f "$work/$s/$first" "$work/$s/$first.recoded"; echo "dropped $first" >>"$work/dropped"; continue; fi
    python3 -c "import sys; from PIL import Image; im = Image.open(sys.argv[1]).convert('RGB'); im.save(sys.argv[2], compress_level=3)" "$work/$s/$first" "$work/$s/.${first%.*}.tmp.png" \
      && { [ "${first%.*}.png" = "$first" ] || rm -f "$work/$s/$first"; mv "$work/$s/.${first%.*}.tmp.png" "$work/$s/${first%.*}.png"; touch "$work/$s/${first%.*}.png.recoded"; } \
      || { rm -f "$work/$s/$first"; echo "dropped $first" >>"$work/dropped"; }
  done
}
pids=()
for s in $(seq 0 $((shards - 1))); do
  [ -d "$work/$s" ] || continue
  run_shard "$s" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for s in $(seq 0 $((shards - 1))); do [ -d "$work/out$s" ] && cp "$work/out$s"/*.page.json "$out"/ || true; done
[ -e "$work/dropped" ] && { echo "dropped images:"; cat "$work/dropped"; } || true
[ $fail = 0 ] || { tail -n 20 "$work"/log*; exit 1; }
echo "wrote $(ls "$out"/*.page.json | wc -l) page models to $out"
