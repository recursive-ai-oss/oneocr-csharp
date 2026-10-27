#!/usr/bin/env bash
# Fine-tune on web text lines (configs/crnn-web.yaml). Runs in chunks: UNTIL=step stops there, and the run resumes from its latest checkpoint
# when called again.
#   train/finetune_web.sh
# Environment: ONEOCR_WEB (rendered gen/web_lines.py output with train/ and heldout/), ONEOCR_HOME (run state; data/phone-lines/phone-tune/finder is
# read from here), ONEOCR_BASE (checkpoint to start from), ONEOCR_PREV, ONEOCR_FT, ONEOCR_V6, ONEOCR_GEN (data of the earlier stages, as in
# configs/crnn-r3.yaml), ONEOCR_GIT_COMMIT.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m train.run configs/crnn-web.yaml ${UNTIL:+--until $UNTIL}
