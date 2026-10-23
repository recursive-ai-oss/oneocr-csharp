"""Font manifest tooling: verifies (or with --write fills in) the SHA-256 of every font file.

Fonts are open-licensed only (OFL, Ubuntu Font Licence). The manifest pins each file by hash so a
dataset can be reproduced from the same font bytes.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import yaml

MANIFEST = Path(__file__).with_name("fonts.yaml")


def font_root() -> Path:
    return Path(os.environ.get("ONEOCR_FONT_ROOT", "/usr/share/fonts/truetype"))


def load_manifest() -> list[dict]:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["fonts"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    write = "--write" in sys.argv
    fonts = load_manifest()
    root = font_root()
    bad = 0
    text = MANIFEST.read_text(encoding="utf-8")
    for f in fonts:
        digest = sha256(root / f["file"])
        if f["sha256"] and f["sha256"] != digest:
            print(f"MISMATCH {f['file']}: {digest}")
            bad += 1
        if write and not f["sha256"]:
            # the entry's file line is unique, so the empty hash after it is the one to fill
            head, _, tail = text.partition(f"    file: {f['file']}\n")
            tail = tail.replace('sha256: ""', f'sha256: "{digest}"', 1)
            text = head + f"    file: {f['file']}\n" + tail
    if write:
        MANIFEST.write_text(text, encoding="utf-8")
    print(f"{len(fonts)} fonts, {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
