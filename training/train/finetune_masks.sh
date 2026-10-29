#!/usr/bin/env bash
# Fine-tune on masked card numbers and item-count prefixes (configs/crnn-masks.yaml): degrades the held-out mask/quantity sets once on the
# GPU, then trains in chunks (UNTIL=step stops there) (the run resumes from its latest checkpoint, so call it again for the next chunk).
#   train/finetune_masks.sh
# Environment: ONEOCR_MASKS (rendered gen/make_lines_r1.sh output of configs/lines-masks.yaml, under data/lines-masks), ONEOCR_HOME (run
# state; data/phone-lines/phone-tune/finder is read from here), ONEOCR_BASE (checkpoint to start from), ONEOCR_PREV, ONEOCR_FT,
# ONEOCR_V6, ONEOCR_GEN (data of the earlier stages, as in configs/crnn-r3.yaml), ONEOCR_GIT_COMMIT.
set -euo pipefail
cd "$(dirname "$0")/.."
d=$ONEOCR_MASKS/data/lines-masks
if [ ! -d "$d/qty_faint_val" ]; then
  uv run python -m degrade.materialize configs/lines-masks.yaml "$d" mask_val qty_val mask_faint_val qty_faint_val
fi
uv run python -m train.run configs/crnn-masks.yaml ${UNTIL:+--until $UNTIL}
