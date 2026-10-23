"""Model container v0: writer and reader (see docs/spec/model-container-v0.md).

Layout: magic `ONEOCRM\\0`, u32 version 0, u32 manifest length, UTF-8 JSON manifest, zero padding to a
multiple of 64, then tensors (fp16, row-major) each starting at a multiple of 64 from the data region.
Nothing follows the last tensor. All integers are little-endian.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"ONEOCRM\x00"
VERSION = 0
ALIGN = 64


def _pad(n: int) -> int:
    return -n % ALIGN


def write_container(path: str | Path, manifest: dict, tensors: dict[str, np.ndarray]) -> None:
    """tensors: name -> array (any float dtype; stored as fp16 in the order given). `manifest` must not hold `tensors`."""
    entries = []
    blobs = []
    offset = 0
    for name, arr in tensors.items():
        h = np.ascontiguousarray(arr, dtype="<f2")
        entries.append({"name": name, "dtype": "f16", "shape": list(h.shape), "offset": offset, "length": h.size * 2})
        blobs.append(h.tobytes())
        offset += h.size * 2 + _pad(h.size * 2)
    keys = ["architecture", "name", "version", "license", "input", "charset", "config"]
    full = {k: manifest[k] for k in keys}
    full["tensors"] = entries
    full["provenance"] = manifest.get("provenance", {})
    body = json.dumps(full, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out = bytearray(MAGIC + struct.pack("<II", VERSION, len(body)) + body)
    out += bytes(_pad(len(out)))
    for blob in blobs:
        out += blob
        if blob is not blobs[-1]:
            out += bytes(_pad(len(blob)))
    Path(path).write_bytes(bytes(out))


def read_container(path: str | Path, max_manifest: int = 1 << 24) -> tuple[dict, dict[str, np.ndarray]]:
    """Reads and validates a container; returns (manifest, name -> fp32 array widened from fp16)."""
    data = Path(path).read_bytes()
    if len(data) < 16 or data[:8] != MAGIC:
        raise ValueError("bad magic")
    version, mlen = struct.unpack_from("<II", data, 8)
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    if mlen > max_manifest or 16 + mlen > len(data):
        raise ValueError("manifest length out of range")
    manifest = json.loads(data[16 : 16 + mlen].decode("utf-8"))
    start = 16 + mlen
    start += _pad(start)
    spans = []
    tensors: dict[str, np.ndarray] = {}
    for t in manifest["tensors"]:
        n = int(np.prod(t["shape"], dtype=np.int64)) if t["shape"] else 1
        if t["dtype"] != "f16" or t["length"] != n * 2 or t["offset"] % ALIGN:
            raise ValueError(f"bad tensor entry {t['name']}")
        a, b = start + t["offset"], start + t["offset"] + t["length"]
        if b > len(data):
            raise ValueError(f"tensor {t['name']} past end of file")
        spans.append((a, b))
        tensors[t["name"]] = np.frombuffer(data, dtype="<f2", count=n, offset=a).reshape(t["shape"]).astype(np.float32)
    spans.sort()
    for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        if s2 < e1:
            raise ValueError("overlapping tensors")
    if spans and spans[-1][1] != len(data):
        raise ValueError("bytes after the last tensor")
    if len(set(manifest["charset"])) != len(manifest["charset"]) or manifest["charset"][0] != "<blank>":
        raise ValueError("bad charset")
    return manifest, tensors
