from __future__ import annotations

import os
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
MANIFESTS = BENCH / "manifests"


def data_dir() -> Path:
    """Where fonts, text, pages and results live (never inside git)."""
    return Path(os.environ.get("ONEOCR_BENCH_DIR", BENCH / ".data")).expanduser()
