#!/usr/bin/env python3
"""Fetch the DocLayNet core dataset (page PNGs + COCO layout labels).

Licence: CDLA-Permissive-1.0 (https://github.com/DS4SD/DocLayNet, LICENSE).
Attribution: Pfitzmann et al., "DocLayNet: A Large Human-Annotated Dataset for
Document-Layout Segmentation", KDD 2022. Official train/val/test splits come
from the COCO files inside the archive. The "extra" archive (PDFs and text
cells) is not fetched.

    python3 fetch_doclaynet.py [--root DIR] [--keep-raw] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
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
    m = c.load_manifest("doclaynet")
    ds = c.data_root(a.root) / m["name"] / m["version"]
    ds.mkdir(parents=True, exist_ok=True)
    done = None if a.force else c.check_done(ds, m)
    if done:
        c.log("already complete (no-op)")
        c.report(done)
        return

    f = m["files"][0]
    zpath = ds / "raw" / "DocLayNet_core.zip"
    archives = {f["id"]: c.download(f["url"], zpath, f["size"], f["sha256"])}
    c.log(f"archive verified, sha256 {archives[f['id']]}")

    (ds / "annotations").mkdir(exist_ok=True)
    owner: dict[str, str] = {}  # PNG file name -> split
    n_images = {}
    with zipfile.ZipFile(zpath) as z:
        names = {Path(n).name: n for n in z.namelist() if not n.endswith("/")}
        for split in ("train", "val", "test"):
            coco_bytes = z.read(names[f"{split}.json"])
            (ds / "annotations" / f"{split}.json").write_bytes(coco_bytes)
            coco = json.loads(coco_bytes)
            n_images[split] = len(coco["images"])
            for im in coco["images"]:
                owner[im["file_name"]] = split
        missing = sorted(set(owner) - set(names))
        if missing:
            raise SystemExit(f"{len(missing)} images in the COCO files are missing from the archive, e.g. {missing[:3]}")
        index: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for k, fn in enumerate(sorted(owner)):
            split = owner[fn]
            data = z.read(names[fn])
            out = ds / "images" / split / fn
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            index[split].append({"id": fn, "file": f"images/{split}/{fn}", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            if k % 10000 == 0:
                c.log(f"  extracted {k}/{len(owner)}")
        stray = len([n for n in names if n.endswith(".png")]) - len(owner)

    # COCO "images" lists unique pages; redundant annotations are extra annotation records.
    counts = {s: len(v) for s, v in index.items()}
    if counts != m["splits"]:
        raise SystemExit(f"split counts {counts} != official {m['splits']}")
    summary = c.finish(ds, m, archives, index, {"licence": m["licence"], "png_not_in_any_split": stray})
    if not a.keep_raw:
        shutil.rmtree(ds / "raw")
    c.report(summary)


if __name__ == "__main__":
    main()
