#!/usr/bin/env bash
# Reads page images with the library's own OCR and writes the page model of each: OUT/<id>.page.json (oneocr.page v1).
#   train/docextract_ocr.sh IMAGES OUT MODEL [SHARDS] [LIMIT]
# IMAGES holds .png/.jpg files; the images are split over SHARDS processes (degree 1 each). Needs .NET 10 on PATH (DOTNET_ROOT set) and
# the repository's bench project; ONEOCR_ARTIFACTS (optional) says where the build output goes. The OCR only supplies words and boxes:
# ONEOCR_DETECTOR (optional) names a detector model file to read with. Labels come from the generator's truth or CORD (train/docextract_data.py).
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
trap 'kill 0' TERM INT  # a timeout also stops the reader processes
i=0
ls "$images" | grep -E '\.(png|jpg|jpeg)$' | sort | { if [ "$limit" -gt 0 ]; then head -n "$limit"; else cat; fi; } | while read -r f; do
  [ -e "$out/${f%.*}.page.json" ] && continue  # resume: pages read by an earlier run are kept
  s=$((i % shards)); mkdir -p "$work/$s"; ln "$images/$f" "$work/$s/$f" 2>/dev/null || cp "$images/$f" "$work/$s/$f"; i=$((i + 1))
done
# The reader process stops at the first image it fails to decode (seen on large PNGs, not repeatable on the same file). Pages already
# done are removed from the shard and the failing image is tried again as it is, then re-encoded as a lossless RGB PNG, then dropped.
run_shard() {
  s=$1; tries=0
  while :; do
    for f in "$work/$s"/*; do [ -e "$work/out$s/$(basename "${f%.*}").page.json" ] && rm -f "$f"; done
    [ -n "$(ls "$work/$s" | grep -E '\.(png|jpg|jpeg)$' | head -1 || true)" ] || return 0  # (pipefail: head closes the pipe early)
    dotnet "$dll" run --model "$model" --pages "$work/$s" --out "$work/out$s" --degree 1 --metaonly 1 --pagejson 1 ${ONEOCR_DETECTOR:+--detector "$ONEOCR_DETECTOR"} >>"$work/log$s" 2>&1 && return 0
    first=$(ls "$work/$s" | grep -E '\.(png|jpg|jpeg)$' | head -1 || true); tries=$((tries + 1))
    [ "$tries" -le 2000 ] || { echo "giving up on shard $s" >&2; return 1; }
    n=$(( $(cat "$work/$s/$first.fail" 2>/dev/null || echo 0) + 1 )); echo $n >"$work/$s/$first.fail"
    if [ "$n" -eq 2 ]; then
      python3 -c "import sys; from PIL import Image; Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2], compress_level=3)" "$work/$s/$first" "$work/$s/.recode.png" \
        && { [ "${first%.*}.png" = "$first" ] || { rm -f "$work/$s/$first"; cp "$work/$s/$first.fail" "$work/$s/${first%.*}.png.fail"; first=${first%.*}.png; }; mv "$work/$s/.recode.png" "$work/$s/$first"; } || true
    elif [ "$n" -ge 3 ]; then
      rm -f "$work/$s/$first"; echo "dropped $first" >>"$work/dropped"
    fi
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
for s in $(seq 0 $((shards - 1))); do [ -d "$work/out$s" ] && cp "$work/out$s"/*.page.json "$out"/ 2>/dev/null || true; done
# stacks of pages the reader failed on (the reader writes <id>.error.txt) are kept next to the output for inspection
mkdir -p "$out/.errors"; cp "$work"/out*/*.error.txt "$out/.errors"/ 2>/dev/null || true
[ -e "$work/dropped" ] && { echo "dropped images:"; cat "$work/dropped"; } || true
[ $fail = 0 ] || { tail -n 20 "$work"/log*; exit 1; }
echo "wrote $(ls "$out"/*.page.json | wc -l) page models to $out"
