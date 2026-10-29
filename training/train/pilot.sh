#!/usr/bin/env bash
# Trains and evaluates capacity-pilot designs one after another (configs/pilot-<name>.yaml), then writes for each the line gate against the
# stored Tesseract outputs, the per-error-kind breakdown and the single-thread CPU latency table.
#   train/pilot.sh p0 a1 ...      names of the pilot configs to run
# Environment: ONEOCR_HOME (state: runs/ and eval-pilot-<name>/), ONEOCR_PREV (production lines-v4 data), ONEOCR_FT (lines-v5 sets, CORD crops),
# ONEOCR_V6 (lines-v6 sets), BENCH_LINES (page-set crops), TESS_DIR (stored Tesseract outputs), ONEOCR_GIT_COMMIT, STEPS_LIMIT (optional --until).
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
uptime
for n in "$@"; do
  o=$ONEOCR_HOME/eval-pilot-$n
  mkdir -p "$o"
  uv run python -m train.run "configs/pilot-$n.yaml" ${STEPS_LIMIT:+--until "$STEPS_LIMIT"} || { echo "training $n failed"; continue; }
  ck=$ONEOCR_HOME/runs/pilot-$n/final.pt
  [ -e "$ck" ] || { echo "no final.pt for $n (partial run)"; continue; }
  extra=()
  for s in lines-v5/gaps_val:gaps_val cord-lines/val/finder:cord_dev lines-v5/val:receipt_synth_val lines-v5/ui_val:ui_val lines-v5/tight_val:tight_val; do
    extra+=(--extra "${s#*:}=$ONEOCR_FT/data/${s%%:*}")
  done
  for s in figures_val small_val reader_val shot_val; do extra+=(--extra "${s}6=$ONEOCR_V6/data/lines-v6/$s"); done
  # the shipped-form evaluation (batch norm folded, fp16-rounded weights), on the GPU for speed: same hypotheses as the CPU
  uv run python -m eval.model_hyps --checkpoint "$ck" --sets-dir "$ONEOCR_PREV/data/lines-v4" --bench-dir "$BENCH_LINES" --out "$o/hyps" "${extra[@]}" | tail -2
  uv run python -m eval.gate --ours "$o/hyps" --tess "$TESS_DIR" --out "$o/gate.md" --json "$o/gate.json" > /dev/null
  uv run python -m eval.capacity_diag --hyps "$o/hyps" --out "$o/diag.md" \
    --sets val,diag_fonts,eval,clean300-v1,degraded-v1,screenshot-v1,layout-v1,phone-v1,receipts-v1,ui_val,tight_val,shot_val6,small_val6,figures_val6,reader_val6
done
