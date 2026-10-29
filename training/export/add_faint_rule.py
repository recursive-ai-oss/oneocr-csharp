"""Adds the faint-text recovery members to a detector container's manifest (config.postprocess) and copies every tensor byte unchanged.

The members are optional; a container without them behaves as before. They are written together (docs/spec/detector-v1.md,
"Faint text on dark backgrounds"). Passing --remove drops them again.

    python3 -m export.add_faint_rule IN.onm OUT.onm --threshold 0.1 --bg-max 100 --min-cells 6 --min-aspect 1.5 --min-confidence 0.9 [--version 0.3.1]
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

MAGIC = b"ONEOCRM\x00"
ALIGN = 64
KEYS = ("faint_text_threshold", "faint_bg_max", "faint_min_cells", "faint_min_aspect", "faint_min_confidence")


def _pad(n: int) -> int:
    return -n % ALIGN


def rewrite(src: Path, dst: Path, faint: dict | None, version: str | None = None) -> None:
    data = src.read_bytes()
    if data[:8] != MAGIC:
        raise ValueError("bad magic")
    fmt, mlen = struct.unpack_from("<II", data, 8)
    manifest = json.loads(data[16 : 16 + mlen].decode("utf-8"))
    start = 16 + mlen
    body = data[start + _pad(start) :]
    if version is not None:
        manifest["version"] = version
    post = manifest["config"].setdefault("postprocess", {})
    for key in KEYS:
        post.pop(key, None)
    if faint is not None:
        post.update(faint)
    new = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out = bytearray(MAGIC + struct.pack("<II", fmt, len(new)) + new)
    out += bytes(_pad(len(out)))
    out += body
    dst.write_bytes(bytes(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--bg-max", type=int, default=100)
    ap.add_argument("--min-cells", type=int, default=6)
    ap.add_argument("--min-aspect", type=float, default=1.5)
    ap.add_argument("--min-confidence", type=float, default=0.9)
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--version")
    a = ap.parse_args()
    faint = None if a.remove else dict(zip(KEYS, (a.threshold, a.bg_max, a.min_cells, a.min_aspect, a.min_confidence)))
    rewrite(a.src, a.dst, faint, a.version)


if __name__ == "__main__":
    main()
