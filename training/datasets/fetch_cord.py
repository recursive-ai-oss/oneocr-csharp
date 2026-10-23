#!/usr/bin/env python3
"""Fetch CORD v2 (receipts) with its official train/dev/test splits.

Licence: CC BY 4.0 (https://github.com/clovaai/cord, README). Attribution:
Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing".
Data comes from the pinned revision of the official dataset repo on the
Hugging Face Hub: dataset files only.

Reading parquet needs pyarrow, which is not in the training project. When it is
missing this script re-runs itself under `uv run --with pyarrow`; nothing is
installed into any project or system Python.

    python3 fetch_cord.py [--root DIR] [--keep-raw] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

try:
    import pyarrow.parquet as pq
except ImportError:
    if os.environ.get("_CORD_REEXEC"):
        raise
    os.environ["_CORD_REEXEC"] = "1"
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    sys.exit(subprocess.call([uv, "run", "--no-project", "--python", "3.13", "--with", "pyarrow", "python", *sys.argv]))


def image_ext(b: bytes) -> str:
    if b[:3] == b"\xff\xd8\xff":
        return "jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    raise ValueError("unknown image format")


def main() -> None:
    ap = argparse.ArgumentParser()
    c.common_args(ap)
    a = ap.parse_args()
    m = c.load_manifest("cord")
    ds = c.data_root(a.root) / m["name"] / m["version"]
    ds.mkdir(parents=True, exist_ok=True)
    done = None if a.force else c.check_done(ds, m)
    if done:
        c.log("already complete (no-op)")
        c.report(done)
        return

    archives = {}
    raw = ds / "raw"
    for f in m["files"]:
        archives[f["id"]] = c.download(f["url"], raw / f["url"].rsplit("/", 1)[1], f["size"], f["sha256"])
    c.log("archives verified")

    index: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    (ds / "annotations").mkdir(exist_ok=True)
    ann = {s: open(ds / "annotations" / f"{s}.jsonl", "w", encoding="utf-8") for s in index}
    for f in m["files"]:
        path = raw / f["url"].rsplit("/", 1)[1]
        split = {"validation": "dev"}.get(path.name.split("-")[0], path.name.split("-")[0])
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        t = pq.read_table(path)
        for row in t.to_pylist():
            gt = row["ground_truth"]
            img = row["image"]["bytes"]
            meta = json.loads(gt)["meta"]
            if meta.get("split") not in (None, "train", "dev", "valid", "test", "validation"):
                raise SystemExit(f"unexpected split label {meta.get('split')!r}")
            rid = f"{meta['image_id']:04d}"
            name = f"{rid}.{image_ext(img)}"
            (ds / "images" / split / name).write_bytes(img)
            rec = {"id": f"{split}/{rid}", "file": f"images/{split}/{name}"}
            index[split].append(rec | {"sha256": hashlib.sha256(img).hexdigest(), "bytes": len(img)})
            ann[split].write(json.dumps({"id": f"{split}/{rid}", "ground_truth": gt}, ensure_ascii=False) + "\n")
    for fh in ann.values():
        fh.close()

    want = m["splits"]
    got = {s: len(v) for s, v in index.items()}
    if got != want:
        raise SystemExit(f"split counts {got} != official {want}")
    summary = c.finish(ds, m, archives, index, {"licence": m["licence"], "revision": m["revision"]})
    if not a.keep_raw:
        shutil.rmtree(raw)
    c.report(summary)


if __name__ == "__main__":
    main()
