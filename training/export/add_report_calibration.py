"""Adds a `report_calibration` member to a recognizer container (version 1) and leaves every other byte of the weights alone.

The member has the shape of `calibration` and is the curve a page reports its word confidence with; the runtime's own decisions (noise
filter, word joiner, angle search) keep reading `calibration`. The curve is taken from the `calibration` member of another container
(for example one refitted with eval/calibrate.py) or from a calibration JSON file. Tensor offsets are relative to the data region, so
only the region's start moves. The output is deterministic for given inputs.

    uv run python -m export.add_report_calibration IN.onm OUT.onm (--from-container OTHER.onm | --from-json cal.json) [--version 0.7.3]
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

MAGIC = b"ONEOCRM\x00"
ALIGN = 64
RAW = "word-mean-max-run"


def _pad(n: int) -> int:
    return -n % ALIGN


def _split(data: bytes) -> tuple[int, dict, bytes]:
    if data[:8] != MAGIC:
        raise ValueError("bad magic")
    fmt, mlen = struct.unpack_from("<II", data, 8)
    manifest = json.loads(data[16 : 16 + mlen].decode("utf-8"))
    start = 16 + mlen
    return fmt, manifest, data[start + _pad(start) :]


def add_report_calibration(src: Path, dst: Path, curve: dict, version: str | None = None) -> None:
    if curve.get("raw") != RAW:
        raise ValueError(f"the curve must name raw '{RAW}'")
    fmt, manifest, body = _split(src.read_bytes())
    if fmt != 1:
        raise ValueError("report_calibration belongs to container version 1")
    manifest["report_calibration"] = curve
    if version is not None:
        manifest["version"] = version
    new = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out = bytearray(MAGIC + struct.pack("<II", fmt, len(new)) + new)
    out += bytes(_pad(len(out)))
    out += body
    dst.write_bytes(bytes(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-container", type=Path)
    g.add_argument("--from-json", type=Path)
    ap.add_argument("--version", help="manifest version to write (default: keep)")
    a = ap.parse_args()
    curve = _split(a.from_container.read_bytes())[1]["calibration"] if a.from_container else json.loads(a.from_json.read_text(encoding="utf-8"))
    add_report_calibration(a.src, a.dst, curve, a.version)


if __name__ == "__main__":
    main()
