#!/usr/bin/env python3
"""CORD v2 word-crop training data: pages, truth files and a split manifest, ready for `eval.page_lines` and `LineGen --page-lines`.

Labels are CORD's own (CC BY 4.0, Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing"): the word text and quad
of every `valid_line` word, in the order of the fetch (`datasets/fetch_cord.py`). Only the official train and dev splits are read;
the official test split is the receipts evaluation set and this script refuses to touch it.

Splits (all by image, none shares an image with another or with test):
  train  the official train images except the calibration ones
  calib  every 10th train image in SHA-256 order (80 images): fits the confidence map, never trains or selects
  val    the official dev split: selects the checkpoint
Images that `dedup.json` lists in its `drop` lists are left out, and any image whose file hash equals a test image is dropped.

    python3 cord_crops.py --out DIR [--root DATASETS_ROOT]

Writes DIR/<split>/{pages,truth}/ (symlinks to the fetched images, one truth JSON per page in the evaluation-set format) and
the committed-size split manifest DIR/split.json (ids and image hashes). Words whose text has a character outside charset-v1 are
dropped and counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from pathlib import Path

CHARSET = Path(__file__).resolve().parents[1] / "data" / "charset-v1.txt"


def quad_pts(q: dict) -> list[float]:
    return [q["x1"], q["y1"], q["x2"], q["y2"], q["x3"], q["y3"], q["x4"], q["y4"]]


def words_of(gt: dict) -> list[dict]:
    """CORD words in reading order: rows by mean y, words by x (the order of the evaluation truth)."""
    rows: dict[int, list[dict]] = {}
    for line in gt["valid_line"]:
        for w in line["words"]:
            text = unicodedata.normalize("NFC", w["text"].strip())
            if not text:
                continue
            p = quad_pts(w["quad"])
            xs, ys = p[0::2], p[1::2]
            box = [min(xs), min(ys), max(xs), max(ys)]
            rows.setdefault(w["row_id"], []).append(
                {"text": text, "quad": [float(v) for v in p], "box": box, "cx": sum(xs) / 4, "cy": sum(ys) / 4}
            )
    ordered = sorted(rows.values(), key=lambda r: sum(w["cy"] for w in r) / len(r))
    return [{"text": w["text"], "box": [round(v, 1) for v in w["box"]], "quad": w["quad"]}
            for r in ordered for w in sorted(r, key=lambda w: w["cx"])]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=os.environ.get("ONEOCR_DATASETS", str(Path.home() / "oneocr" / "datasets")))
    a = ap.parse_args()
    root, out = Path(a.root) / "cord" / "v2", Path(a.out)
    charset = set(CHARSET.read_text(encoding="utf-8").split("\n"))
    dedup = json.loads((root / "dedup.json").read_text(encoding="utf-8"))
    dropped = dedup.get("drop", {})
    drop = set(dropped) if isinstance(dropped, dict) else set(dropped)
    index = {s: [json.loads(x) for x in (root / "index" / f"{s}.jsonl").read_text().splitlines()] for s in ("train", "dev", "test")}
    test_hashes = {r["sha256"] for r in index["test"]}
    ann = {}
    for s in ("train", "dev"):
        ann[s] = {r["id"]: r for r in map(json.loads, (root / "annotations" / f"{s}.jsonl").read_text().splitlines())}

    def keep(rows: list[dict]) -> list[dict]:
        seen, res = set(), []
        for r in rows:
            if r["id"] in drop or r["sha256"] in test_hashes or r["sha256"] in seen:
                continue
            seen.add(r["sha256"])
            res.append(r)
        return res

    train_rows = keep(index["train"])
    by_hash = sorted(train_rows, key=lambda r: r["sha256"])
    calib_ids = {r["id"] for r in by_hash[::10]}
    split_rows = {"train": [r for r in train_rows if r["id"] not in calib_ids], "calib": [r for r in train_rows if r["id"] in calib_ids],
                  "val": keep(index["dev"])}  # fmt: skip
    manifest = {"dataset": "cord v2 (CC BY 4.0)", "label_source": "CORD annotations (valid_line words and quads)", "splits": {}}
    for split, rows in split_rows.items():
        src = "dev" if split == "val" else "train"
        (out / split / "pages").mkdir(parents=True, exist_ok=True)
        (out / split / "truth").mkdir(parents=True, exist_ok=True)
        n_words = n_drop = 0
        for r in rows:
            pid = r["id"].replace("/", "-")
            gt = json.loads(ann[src][r["id"]]["ground_truth"])
            ws = words_of(gt)
            good = [w for w in ws if set(w["text"]) <= charset]
            n_words += len(good)
            n_drop += len(ws) - len(good)
            link = out / split / "pages" / (pid + Path(r["file"]).suffix)
            link.unlink(missing_ok=True)
            link.symlink_to(root / r["file"])
            truth = {"id": pid, "set": f"cord-{split}", "source": f"cord-v2 {r['id']}", "licence": "CC BY 4.0", "words": good}
            (out / split / "truth" / (pid + ".json")).write_text(json.dumps(truth, ensure_ascii=False), encoding="utf-8")
        digest = hashlib.sha256("".join(f"{r['id']} {r['sha256']}\n" for r in rows).encode()).hexdigest()
        manifest["splits"][split] = {"images": len(rows), "words": n_words, "words_dropped_charset": n_drop,
                                     "ids": [r["id"] for r in rows],
                                     "digest": digest}  # fmt: skip
        print(split, len(rows), "images", n_words, "words", n_drop, "dropped for charset")
    (out / "split.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
