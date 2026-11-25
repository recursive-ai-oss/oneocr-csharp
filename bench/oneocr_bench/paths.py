from __future__ import annotations

import os
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
MANIFESTS = BENCH / "manifests"
# Page image formats the runners read (the fax sets are CCITT G3 TIFF).
PAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def data_dir() -> Path:
    """Where fonts, text, pages and results live (never inside git)."""
    return Path(os.environ.get("ONEOCR_BENCH_DIR", BENCH / ".data")).expanduser()
