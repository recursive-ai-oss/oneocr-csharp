#!/usr/bin/env bash
# Run the shipped page pipeline (recognizer, detector, character model, word frequencies) over a directory of receipt photos and keep
# the page models, so `gen.receipt_photos align` cuts crops exactly where the current engine cuts them.
#   gen/receipt_engine_run.sh MODEL DETECTOR WORDLM WORDFREQ PHOTOS OUT [DEGREE]
# Run from training/. Needs dotnet (DOTNET_ROOT on PATH). Light: nice, at most DEGREE (default 4) threads.
set -euo pipefail
model=$(realpath "$1") det=$(realpath "$2") lm=$(realpath "$3") wf=$(realpath "$4") photos=$(realpath "$5") out=$(realpath -m "$6")
degree=${7:-4}
bench=$(cd .. && pwd)/bench/OneOcr.Bench
mkdir -p "$out"
nice -n 10 dotnet run -c Release --project "$bench" -- run --model "$model" --detector "$det" --wordlm "$lm" --wordfreq "$wf" \
  --pages "$photos" --out "$out" --degree "$degree" --repeat 1 --words 1 --pagejson 1
