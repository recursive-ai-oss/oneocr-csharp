#!/usr/bin/env python3
"""Fetch PubLayNet 1.0.0: the official val split and one official train shard.

Annotations: CDLA-Permissive-1.0 (LICENSE.txt in labels.tar.gz). Images belong
to their authors under the PubMed Central Open Access Subset terms
(commercial-use collection); see LICENCES.md before using or publishing them.
Attribution: Zhong et al., "PubLayNet: largest dataset ever for document layout
analysis", ICDAR 2019.

The full train split is seven 13.6 GB shards; a layout head does not need
them all, so train is the shard named in manifests/publaynet.json
("train-0"). Its COCO file is cut down to the images in that shard. The 1.0.0
package has no test labels, so val is the evaluation split.

    python3 fetch_publaynet.py [--root DIR] [--keep-raw] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402


def extract_images(tar_path: Path, split: str, ds: Path) -> list[dict]:
    recs = []
    out_dir = ds / "images" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r|gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith((".jpg", ".png")):
                continue
            data = tf.extractfile(member).read()
            name = Path(member.name).name
            (out_dir / name).write_bytes(data)
            recs.append({"id": name, "file": f"images/{split}/{name}", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
            if len(recs) % 10000 == 0:
                c.log(f"  {split}: {len(recs)} images")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    c.common_args(ap)
    a = ap.parse_args()
    m = c.load_manifest("publaynet")
    ds = c.data_root(a.root) / m["name"] / m["version"]
    ds.mkdir(parents=True, exist_ok=True)
    done = None if a.force else c.check_done(ds, m)
    if done:
        c.log("already complete (no-op)")
        c.report(done)
        return

    raw = ds / "raw"
    archives = {f["id"]: c.download(f["url"], raw / f["url"].rsplit("/", 1)[1], f["size"], f["sha256"]) for f in m["files"]}
    c.log("archives verified: " + json.dumps(archives))

    (ds / "annotations").mkdir(exist_ok=True)
    (ds / "licence").mkdir(exist_ok=True)
    with tarfile.open(raw / "labels.tar.gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = Path(member.name).name
            dest = ds / ("licence" if name.endswith(".txt") else "annotations") / name
            dest.write_bytes(tf.extractfile(member).read())

    index = {"val": extract_images(raw / "val.tar.gz", "val", ds), "train": extract_images(raw / "train-0.tar.gz", "train", ds)}

    # Cut train.json to the shard so every annotated image has its file.
    have = {r["id"] for r in index["train"]}
    full = json.loads((ds / "annotations" / "train.json").read_bytes())
    keep = {im["id"] for im in full["images"] if im["file_name"] in have}
    sub = {k: v for k, v in full.items() if k not in ("images", "annotations")}
    sub["images"] = [im for im in full["images"] if im["id"] in keep]
    sub["annotations"] = [an for an in full["annotations"] if an["image_id"] in keep]
    unlabelled = sorted(have - {im["file_name"] for im in sub["images"]})
    (ds / "annotations" / "train_shard0.json").write_text(json.dumps(sub))
    del full
    val = json.loads((ds / "annotations" / "val.json").read_bytes())
    val_missing = {im["file_name"] for im in val["images"]} - {r["id"] for r in index["val"]}

    def articles(recs):
        return {r["id"].split("_")[0] for r in recs}

    tr_art, va_art = articles(index["train"]), articles(index["val"])
    ids = sorted(tr_art)
    extra = {
        "licence": m["licence"],
        "train_annotated_images": len(sub["images"]),
        "train_annotations": len(sub["annotations"]),
        "train_images_without_labels": len(unlabelled),
        "val_annotated_images": len(val["images"]),
        "val_images_without_file": len(val_missing),
        "val_annotations": len(val["annotations"]),
        "train_articles": len(tr_art),
        "val_articles": len(va_art),
        "articles_in_both_splits": len(tr_art & va_art),
        "train_pmcid_range": [ids[0], ids[-1]],
    }
    # PMC ids per split, for per-article licence lookups and attribution.
    for s, arts in (("train", tr_art), ("val", va_art)):
        (ds / "licence" / f"pmcids_{s}.txt").write_text("\n".join(sorted(arts)) + "\n")
    summary = c.finish(ds, m, archives, index, extra)
    if not a.keep_raw:
        shutil.rmtree(raw)
    c.report(summary)
    c.log(json.dumps(extra))


if __name__ == "__main__":
    main()
