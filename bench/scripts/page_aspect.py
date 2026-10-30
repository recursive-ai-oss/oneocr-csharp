"""Aspect ratio (height / width) of the pages of some sets: python scripts/page_aspect.py NAME=ROOT [...]; ROOT holds pages/."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

for arg in sys.argv[1:]:
    name, root = arg.split("=", 1)
    a = sorted(Image.open(f).size[1] / Image.open(f).size[0] for f in sorted((Path(root).expanduser() / "pages").glob("*")))
    print(f"{name}: {len(a)} pages, h/w min {a[0]:.2f} p10 {a[len(a) // 10]:.2f} median {a[len(a) // 2]:.2f} p90 {a[9 * len(a) // 10]:.2f} max {a[-1]:.2f}")
