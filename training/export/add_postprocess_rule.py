"""Adds the page-conditional text threshold to a detector container's manifest and leaves every other byte of the weights alone.

The container (docs/spec/model-container-v1.md) puts tensor offsets relative to the data region, so changing the manifest only moves
that region; the tensor bytes are copied unchanged. The output is deterministic for given inputs.

    uv run python -m export.add_postprocess_rule IN.onm OUT.onm [--sigma-min 0.1235 --junk-max 0.6358 --text-threshold 0.5]
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

MAGIC = b"ONEOCRM\x00"
ALIGN = 64


def _pad(n: int) -> int:
    return -n % ALIGN


def add_rule(src: Path, dst: Path, sigma_min: float, junk_max: float, text_threshold: float) -> None:
    data = src.read_bytes()
    if data[:8] != MAGIC:
        raise ValueError("bad magic")
    version, mlen = struct.unpack_from("<II", data, 8)
    manifest = json.loads(data[16 : 16 + mlen].decode("utf-8"))
    start = 16 + mlen
    body = data[start + _pad(start) :]
    post = manifest["config"].setdefault("postprocess", {})
    post["rule_sigma_min"] = sigma_min
    post["rule_junk_max"] = junk_max
    post["rule_text_threshold"] = text_threshold
    new = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out = bytearray(MAGIC + struct.pack("<II", version, len(new)) + new)
    out += bytes(_pad(len(out)))
    out += body
    dst.write_bytes(bytes(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--sigma-min", type=float, default=0.1235)
    ap.add_argument("--junk-max", type=float, default=0.6358)
    ap.add_argument("--text-threshold", type=float, default=0.5)
    a = ap.parse_args()
    add_rule(a.src, a.dst, a.sigma_min, a.junk_max, a.text_threshold)


if __name__ == "__main__":
    main()
