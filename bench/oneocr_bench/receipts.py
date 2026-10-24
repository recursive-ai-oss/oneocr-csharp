"""receipts-v1: the CORD v2 test split (100 phone-photographed receipts), truth from CORD's own annotations.

Licence: CC BY 4.0, Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing" (see README).
Images and annotations come from the pinned fetch in training/datasets (`fetch_cord.py`); this module only derives
truth files from them and copies the images.

Truth. CORD annotates words in `valid_line` groups (one group per semantic item) with a quad and a row id. Words
are placed in reading order like this: rows are the words sharing a `row_id`; rows are sorted top to bottom by the
mean y of their word centres; inside a row words go left to right by x centre. Words with empty text are dropped.
The word box is the axis-aligned box around the quad; the quad is kept.

Coverage caveat. CORD publishes only part of each receipt: store and payment header blocks were removed for legal
reasons, and `dontcare`/`repeating_symbol` regions are not text. An engine that reads the whole receipt therefore
produces words that are not in the truth. Read the `*_recall` scores (errors on truth words only) next to the
standard ones for this set, and treat detection precision as a lower bound.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from PIL import Image

from .render import sha256, text_of, write_page


def cord_root() -> Path:
    return Path(os.environ.get("ONEOCR_DATASETS", Path.home() / "oneocr" / "datasets")).expanduser() / "cord" / "v2"


def quad_pts(q: dict) -> list[float]:
    return [q["x1"], q["y1"], q["x2"], q["y2"], q["x3"], q["y3"], q["x4"], q["y4"]]


def derive_words(gt: dict) -> list[dict]:
    rows: dict[int, list[dict]] = {}
    for line in gt["valid_line"]:
        for w in line["words"]:
            if not w["text"].strip():
                continue
            p = quad_pts(w["quad"])
            xs, ys = p[0::2], p[1::2]
            rows.setdefault(w["row_id"], []).append(
                {"text": w["text"].strip(), "quad": [float(v) for v in p], "box": [min(xs), min(ys), max(xs), max(ys)],
                 "cx": sum(xs) / 4, "cy": sum(ys) / 4, "category": line["category"]})
    ordered = sorted(rows.values(), key=lambda r: sum(w["cy"] for w in r) / len(r))
    out = []
    for bi, r in enumerate(ordered):
        for w in sorted(r, key=lambda w: w["cx"]):
            out.append({"text": w["text"], "box": [round(v, 1) for v in w["box"]], "quad": w["quad"], "block": bi,
                        "category": w["category"]})
    return out


def render(out: Path, set_name: str) -> list[dict]:
    root = cord_root()
    ann = [json.loads(line) for line in (root / "annotations" / "test.jsonl").read_text().splitlines()]
    idx = [json.loads(line) for line in (root / "index" / "test.jsonl").read_text().splitlines()]
    entries = []
    for i, (a, ix) in enumerate(zip(ann, idx, strict=True)):
        if a["id"] != ix["id"]:
            raise SystemExit("annotation and index order differ")
        src = root / ix["file"]
        data = src.read_bytes()
        if sha256(data) != ix["sha256"]:
            raise SystemExit(f"{src}: hash differs from the dataset index")
        gt = json.loads(a["ground_truth"])
        with Image.open(src) as im:
            w, h = im.size
        if (w, h) != (gt["meta"]["image_size"]["width"], gt["meta"]["image_size"]["height"]):
            raise SystemExit(f"{ix['id']}: annotation size differs from the image")
        tw = derive_words(gt)
        for t in tw:  # truth sanity: box inside the image, non-empty text
            t["ink"] = list(t["box"])  # CORD's word quads already enclose the printed word closely; no pixel pass
            x0, y0, x1, y1 = t["box"]
            if not (t["text"] and -2 <= x0 < x1 <= w + 2 and -2 <= y0 < y1 <= h + 2):
                raise SystemExit(f"{ix['id']}: bad word {t}")
        pid = f"r{i + 1:03d}"
        truth = {"id": pid, "set": set_name, "source": f"cord-v2 {ix['id']}", "licence": "CC BY 4.0",
                 "image": {"width": w, "height": h}, "words": tw, "text": text_of(tw)}
        ent = write_page(out, pid, data, src.suffix.lstrip("."), truth)
        ent["source"] = ix["id"]
        entries.append(ent)
    return entries


def clean(out: Path) -> None:
    shutil.rmtree(out, ignore_errors=True)
