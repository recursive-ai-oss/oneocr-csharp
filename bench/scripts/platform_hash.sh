#!/usr/bin/env bash
# Hashes every page of a sample directory with the bundled pins at the given degrees and writes one file per degree,
# for comparing platforms byte for byte (diff the files).
#
#   BIN=<dir with OneOcr.Bench.dll> MODELS=<dir with model-0.7.3.onm det032.onm charlm-v1.bin wordfreq-v2.bin> \
#     bash bench/scripts/platform_hash.sh SAMPLE_DIR OUT_DIR [DEGREES...] [-- extra ocrhash args]
#
# Output: OUT_DIR/hash-d<N>.txt, one `<page> <16 hex>` line per page and an `ALL` line.
set -euo pipefail
sample="${1:?usage: platform_hash.sh SAMPLE_DIR OUT_DIR [DEGREES...]}"; out="${2:?}"; shift 2
degrees=(); extra=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then shift; extra=("$@"); break; fi
  degrees+=("$1"); shift
done
[[ ${#degrees[@]} -eq 0 ]] && degrees=(1 4)
mkdir -p "$out"
for d in "${degrees[@]}"; do
  dotnet "$BIN/OneOcr.Bench.dll" ocrhash --pages "$sample" --degree "$d" --out "$out/hash-d$d.txt" \
    --model "$MODELS/model-0.7.3.onm" --detector "$MODELS/det032.onm" \
    --wordlm "$MODELS/charlm-v1.bin" --wordfreq "$MODELS/wordfreq-v2.bin" ${extra[@]+"${extra[@]}"}
done
