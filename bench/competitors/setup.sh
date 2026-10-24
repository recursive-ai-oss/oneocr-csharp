#!/usr/bin/env bash
# Install PaddleOCR and docTR (released pip packages, stock models) into their own virtual environments, apart from
# the benchmark's. Nothing here is imported by our code; the engines are only run as black boxes by run.py.
#
#   bash competitors/setup.sh [ROOT]        # default ROOT: ~/oneocr/bench/competitors
#
# Writes ROOT/versions.txt (pinned versions of every installed package). Model files are downloaded by the packages
# themselves on first use and stay under ROOT (PADDLE_PDX_CACHE_HOME, DOCTR_CACHE_DIR, HF_HOME, TORCH_HOME).
set -euo pipefail
root="${1:-$HOME/oneocr/bench/competitors}"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$root"
cd "$root"

uv venv --python 3.13 venv-paddle
uv pip install --python venv-paddle/bin/python paddlepaddle paddleocr

uv venv --python 3.13 venv-doctr
uv pip install --python venv-doctr/bin/python --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple torch torchvision
uv pip install --python venv-doctr/bin/python "python-doctr[torch]"

{
  echo "# paddle venv"; uv pip freeze --python venv-paddle/bin/python | grep -iE '^(paddle|paddlepaddle|paddleocr|paddlex|numpy|opencv)'
  echo "# doctr venv";  uv pip freeze --python venv-doctr/bin/python  | grep -iE '^(python-doctr|torch|torchvision|numpy|onnxruntime|opencv)'
} > versions.txt
cat versions.txt
