#!/usr/bin/env python3
"""Fetch the DocLayNet "extra" archive and keep only its per-page JSON (text cells).

Licence: CDLA-Permissive-1.0 (https://github.com/DS4SD/DocLayNet, LICENSE), same
dataset as fetch_doclaynet.py. The cells are the PDF's own programmatic text
boxes, shipped with the dataset; they are the word-detection labels for
DocLayNet pages. The PDFs inside the archive are not extracted.

    python3 fetch_doclaynet_extra.py [--root DIR] [--keep-raw] [--force]

Output: <root>/doclaynet/1.0.0/cells/<page-hash>.json plus cells/DONE.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    c.common_args(ap)
    a = ap.parse_args()
    m = c.load_manifest("doclaynet_extra")
    ds = c.data_root(a.root) / "doclaynet" / m["version"]
    out = ds / "cells"
    if (out / "DONE.json").exists() and not a.force:
        c.log("already complete (no-op)")
        return
    f = m["files"][0]
    zpath = ds / "raw" / "DocLayNet_extra.zip"
    digest = c.download(f["url"], zpath, f["size"], f["sha256"])
    c.log(f"archive verified, sha256 {digest}")
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.endswith(".json"):
                continue
            (out / Path(info.filename).name).write_bytes(z.read(info))
            n += 1
    (out / "DONE.json").write_text(json.dumps({"archive_sha256": digest, "json_files": n}))
    if not a.keep_raw:
        shutil.rmtree(ds / "raw")
    c.log(f"{n} cell files")


if __name__ == "__main__":
    main()
