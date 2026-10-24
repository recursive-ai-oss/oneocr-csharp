"""Training data for the text detector + layout model.

Sources (label provenance is recorded per record in the manifest):
  doclaynet  layout boxes and text cells shipped with DocLayNet 1.0.0 (CDLA-Permissive-1.0)
  publaynet  layout boxes shipped with PubLayNet 1.0.0, CC0 / CC BY articles only
  pages      exact-truth synthetic pages (line boxes and region boxes from the renderer)

A record is a dict:
  path    image file
  kind    source name
  layout  [[class_index, x0, y0, x1, y1], ...] in source pixels (our 8 classes)
  lines   [[x0, y0, x1, y1], ...] text-line boxes, or None when the source has none
  cmask   8 flags: which layout classes this source labels (unlabelled classes are not trained as negatives)

Text detection is line-level: DocLayNet's cells are text runs (one per line segment) and the recognizer reads line crops.
Targets follow the shrink-and-recover rule documented in
export/dbnet-lite-architecture.md: a line box with short side m is shrunk by 0.3 m on
every side to make the probability target (stride 2); a layout box is shrunk
by min(0.2 m, 20 px) on every side (stride 8).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

CLASSES = ["text", "title", "list", "table", "figure", "header_footer", "form_field", "checkbox"]

# DocLayNet (11 classes) -> ours. Granularity stays the source's: DocLayNet boxes
# one list item at a time, PubLayNet a whole list.
DOCLAYNET_MAP = {
    "Caption": "text",
    "Footnote": "text",
    "Formula": "text",
    "List-item": "list",
    "Page-footer": "header_footer",
    "Page-header": "header_footer",
    "Picture": "figure",
    "Section-header": "title",
    "Table": "table",
    "Text": "text",
    "Title": "title",
}
PUBLAYNET_MAP = {"text": "text", "title": "title", "list": "list", "table": "table", "figure": "figure"}
CMASK = {
    "doclaynet": [1, 1, 1, 1, 1, 1, 0, 0],
    "publaynet": [1, 1, 1, 1, 1, 0, 0, 0],
}

TEXT_SHRINK = 0.3  # of the short side, per side
LAYOUT_SHRINK = 0.2
LAYOUT_SHRINK_CAP = 20.0
TEXT_STRIDE = 2
LAYOUT_STRIDE = 8


def text_core_offset(m: float) -> float:
    return TEXT_SHRINK * m


def layout_core_offset(m: float) -> float:
    return min(LAYOUT_SHRINK * m, LAYOUT_SHRINK_CAP)


def _cells(lo: float, hi: float, stride: int, n: int) -> tuple[int, int]:
    """Cell index range whose centres (i + 0.5) * stride fall inside [lo, hi]."""
    a = max(0, int(np.ceil(lo / stride - 0.5)))
    b = min(n - 1, int(np.floor(hi / stride - 0.5)))
    return a, b


def text_targets(lines: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probability core, threshold ground truth and threshold mask at stride 2 (DB, Liao et al. 2020)."""
    H, W = h // TEXT_STRIDE, w // TEXT_STRIDE
    prob = np.zeros((H, W), np.float32)
    thr = np.zeros((H, W), np.float32)
    mask = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1 in lines:
        bw, bh = x1 - x0, y1 - y0
        m = min(bw, bh)
        if m < 2.0:
            continue
        d = text_core_offset(m)
        ia, ib = _cells(x0 + d, x1 - d, TEXT_STRIDE, W)
        ja, jb = _cells(y0 + d, y1 - d, TEXT_STRIDE, H)
        if ib < ia or jb < ja:  # box too small for a core at this stride: keep its centre cell
            ci, cj = int((x0 + x1) / 2 // TEXT_STRIDE), int((y0 + y1) / 2 // TEXT_STRIDE)
            if 0 <= ci < W and 0 <= cj < H:
                prob[cj, ci] = 1.0
            continue
        prob[ja : jb + 1, ia : ib + 1] = 1.0
        # threshold band: within d of the box border, inside or outside
        ia2, ib2 = _cells(x0 - d, x1 + d, TEXT_STRIDE, W)
        ja2, jb2 = _cells(y0 - d, y1 + d, TEXT_STRIDE, H)
        if ib2 < ia2 or jb2 < ja2:
            continue
        xs = (np.arange(ia2, ib2 + 1) + 0.5) * TEXT_STRIDE
        ys = (np.arange(ja2, jb2 + 1) + 0.5) * TEXT_STRIDE
        dx = np.maximum(np.maximum(x0 - xs, xs - x1), 0.0)[None, :]
        dy = np.maximum(np.maximum(y0 - ys, ys - y1), 0.0)[:, None]
        outside = np.hypot(dx, dy)
        inside = np.minimum(np.minimum(xs - x0, x1 - xs)[None, :], np.minimum(ys - y0, y1 - ys)[:, None])
        dist = np.where((dx > 0) | (dy > 0), outside, np.maximum(inside, 0.0))
        band = dist <= d
        val = np.where(band, 1.0 - dist / d, 0.0).astype(np.float32)
        win = (slice(ja2, jb2 + 1), slice(ia2, ib2 + 1))
        thr[win] = np.maximum(thr[win], val)
        mask[win] = np.maximum(mask[win], band.astype(np.float32))
    mask = np.where(prob > 0, 0.0, mask)  # the core itself is not part of the threshold loss
    return prob, 0.3 + 0.4 * thr, mask


def layout_targets(boxes: np.ndarray, classes: np.ndarray, h: int, w: int, n_classes: int = len(CLASSES)) -> np.ndarray:
    H, W = h // LAYOUT_STRIDE, w // LAYOUT_STRIDE
    out = np.zeros((n_classes, H, W), np.float32)
    for (x0, y0, x1, y1), c in zip(boxes, classes, strict=False):
        m = min(x1 - x0, y1 - y0)
        if m < 4.0:
            continue
        d = layout_core_offset(m)
        ia, ib = _cells(x0 + d, x1 - d, LAYOUT_STRIDE, W)
        ja, jb = _cells(y0 + d, y1 - d, LAYOUT_STRIDE, H)
        if ib < ia or jb < ja:
            ci, cj = int((x0 + x1) / 2 // LAYOUT_STRIDE), int((y0 + y1) / 2 // LAYOUT_STRIDE)
            if 0 <= ci < W and 0 <= cj < H:
                out[c, cj, ci] = 1.0
            continue
        out[c, ja : jb + 1, ia : ib + 1] = 1.0
    return out


def rot90_boxes(b: np.ndarray, k: int, w: int, h: int) -> tuple[np.ndarray, int, int]:
    """Rotate boxes with np.rot90(img, k) on an image of width w, height h (counter-clockwise)."""
    for _ in range(k % 4):
        # one CCW quarter turn: (x, y) -> (y, w - x); width and height swap
        x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        b = np.stack([y0, w - x1, y1, w - x0], 1)
        w, h = h, w
    return b, w, h


class DetDataset(Dataset):
    def __init__(self, records: list[dict], size: int = 1024, train: bool = True, seed: int = 0, scale=(0.7, 1.15), rot_p: float = 0.25):
        self.recs, self.size, self.train, self.seed, self.scale, self.rot_p = records, size, train, seed, scale, rot_p

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, key):
        # key = (record index, nonce); the nonce (step * batch + slot) makes augmentation reproducible after a resume
        i, nonce = key if isinstance(key, tuple) else (key, 0)
        rec = self.recs[i]
        rng = random.Random(f"{self.seed}-{i}-{nonce}")
        img = Image.open(rec["path"]).convert("L")
        w0, h0 = img.size
        s = self.size / max(w0, h0)
        if self.train:
            s *= rng.uniform(*self.scale)
        nw, nh = max(32, round(w0 * s)), max(32, round(h0 * s))
        img = img.resize((nw, nh), Image.BILINEAR, reducing_gap=2.0)
        sx, sy = nw / w0, nh / h0
        lay = np.array(rec["layout"], np.float32).reshape(-1, 5)
        lb = lay[:, 1:] * np.array([sx, sy, sx, sy], np.float32)
        lc = lay[:, 0].astype(np.int64)
        wb = None if rec["lines"] is None else np.array(rec["lines"], np.float32).reshape(-1, 4) * np.array([sx, sy, sx, sy], np.float32)
        arr = np.asarray(img)
        k = 0
        if self.train and rng.random() < self.rot_p:
            k = rng.choice([1, 2, 3])
            arr = np.rot90(arr, k)
            lb, _, _ = rot90_boxes(lb, k, nw, nh)
            if wb is not None:
                wb, _, _ = rot90_boxes(wb, k, nw, nh)
        ah, aw = arr.shape
        S = self.size
        canvas = np.full((S, S), 255, np.uint8)
        # crop offset (source) and paste offset (canvas)
        ox = rng.randint(0, aw - S) if aw > S else 0
        oy = rng.randint(0, ah - S) if ah > S else 0
        px = py = 0
        if self.train and rng.random() < 0.2:
            px = rng.randint(0, max(0, S - min(aw, S)))
            py = rng.randint(0, max(0, S - min(ah, S)))
        cw, ch = min(aw, S), min(ah, S)
        canvas[py : py + ch, px : px + cw] = arr[oy : oy + ch, ox : ox + cw]
        shift = np.array([px - ox, py - oy, px - ox, py - oy], np.float32)

        def clip(b, keep_frac):
            if len(b) == 0:
                return b, np.zeros(0, bool)
            b = b + shift
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            c = np.clip(b, 0, S)
            carea = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
            ok = (area > 0) & (carea >= keep_frac * area) & (c[:, 2] - c[:, 0] > 1) & (c[:, 3] - c[:, 1] > 1)
            return c, ok

        lb, ok = clip(lb, 0.5)
        lb, lc = lb[ok], lc[ok]
        if self.train:
            canvas = self._photometric(canvas, rng)
        out = {
            "image": torch.from_numpy(canvas)[None],
            "layout": torch.from_numpy(layout_targets(lb, lc, S, S)),
            "cmask": torch.tensor(rec["cmask"], dtype=torch.float32),
            "orient": torch.tensor(k),
            "text_valid": torch.tensor(float(wb is not None)),
        }
        if wb is not None:
            wb, ok = clip(wb, 0.6)
            p, t, m = text_targets(wb[ok], S, S)
        else:
            H = S // TEXT_STRIDE
            p = t = m = np.zeros((H, H), np.float32)
        out["prob"], out["thr"], out["thr_mask"] = (torch.from_numpy(np.ascontiguousarray(a))[None] for a in (p, t, m))
        return out

    @staticmethod
    def _photometric(a: np.ndarray, rng: random.Random) -> np.ndarray:
        if rng.random() < 0.3:  # lost resolution
            img = Image.fromarray(a)
            f = rng.uniform(0.5, 0.85)
            img = img.resize((int(a.shape[1] * f), int(a.shape[0] * f)), Image.BILINEAR).resize(img.size, Image.BILINEAR)
            a = np.asarray(img)
        elif rng.random() < 0.2:
            a = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.2))))
        x = a.astype(np.float32)
        if rng.random() < 0.5:  # contrast / brightness / faint print
            c = rng.uniform(0.45, 1.1)
            x = 255 - (255 - x) * c + rng.uniform(-25, 15)
        if rng.random() < 0.3:
            x = x + np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(2, 9), x.shape)
        return np.clip(x, 0, 255).astype(np.uint8)


def load_doclaynet_records(root: Path, split: str, cells_dir: Path | None = None) -> list[dict]:
    """DocLayNet pages with layout boxes mapped to our classes; line boxes come from the extra archive's cells."""
    ds = root / "doclaynet" / "1.0.0"
    coco = json.loads((ds / "annotations" / f"{split}.json").read_text())
    cat = {c["id"]: DOCLAYNET_MAP[c["name"]] for c in coco["categories"]}
    drop = set(json.loads((ds / "dedup.json").read_text())["drop"].get(split, []))
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append([CLASSES.index(cat[a["category_id"]]), x, y, x + w, y + h])
    recs = []
    for im in sorted(coco["images"], key=lambda im: im["file_name"]):
        if im["file_name"] in drop:
            continue
        rec = {
            "path": str(ds / "images" / split / im["file_name"]),
            "kind": "doclaynet",
            "id": im["file_name"],
            "layout": by_img.get(im["id"], []),
            "lines": None,
            "cmask": CMASK["doclaynet"],
        }
        if cells_dir is not None:
            rec["lines"] = load_cells(cells_dir / (im["file_name"][:-4] + ".json"))
        recs.append(rec)
    return recs


def merge_runs(runs: list[list[float]], gap: float = 0.8, overlap: float = 0.6) -> list[list[float]]:
    """Join text runs that sit on one baseline into line boxes.

    DocLayNet cells split a line wherever the font changes (code inside prose, bold labels), but the recognizer
    reads whole lines. Two runs join when their vertical overlap is at least `overlap` of the shorter one and the
    horizontal gap is at most `gap` times the shorter height; wide gaps (table columns, margin notes) stay apart.
    """
    runs = sorted(runs, key=lambda b: (b[1], b[0]))
    lines: list[list[float]] = []
    for b in runs:
        for ln in lines:
            h = min(ln[3] - ln[1], b[3] - b[1])
            v = min(ln[3], b[3]) - max(ln[1], b[1])
            g = max(b[0] - ln[2], ln[0] - b[2])
            if v >= overlap * h and g <= gap * h:
                ln[0], ln[1], ln[2], ln[3] = min(ln[0], b[0]), min(ln[1], b[1]), max(ln[2], b[2]), max(ln[3], b[3])
                break
        else:
            lines.append(list(b))
    return lines


def load_cells(path: Path) -> list[list[float]] | None:
    """Line boxes of a DocLayNet page from its PDF text cells (already in the 1025 px page frame); blank runs are dropped."""
    if not path.exists():
        return None
    out = []
    for c in json.loads(path.read_text())["cells"]:
        x, y, w, h = c["bbox"]
        if c.get("text", "").strip() and w > 1 and h > 1:
            out.append([x, y, x + w, y + h])
    return merge_runs(out)


def load_records(name: str, root: Path, split: str, src: dict | None = None) -> list[dict]:
    src = src or {}
    if name == "doclaynet":
        recs = load_doclaynet_records(root, split, root / "doclaynet" / "1.0.0" / "cells")
        return [r for r in recs if r["lines"] is not None]  # pages without cells would train "no text" by mistake
    if name == "publaynet":
        return load_publaynet_records(root, split)
    if name == "pages":
        return load_page_records(Path(src["dir"]), split)
    raise ValueError(name)


def load_page_records(path: Path, split: str) -> list[dict]:
    raise NotImplementedError("synthetic page set not available yet")


def load_publaynet_records(root: Path, split: str) -> list[dict]:
    """PubLayNet pages from CC0 / CC BY articles only (see datasets/filter_publaynet_licence.py)."""
    ds = root / "publaynet" / "1.0.0"
    allowed = set((ds / "licence" / f"allowed_{split}.txt").read_text().split())
    coco = json.loads((ds / "annotations" / ("val.json" if split == "val" else "train_shard0.json")).read_text())
    cat = {c["id"]: PUBLAYNET_MAP[c["name"]] for c in coco["categories"]}
    drop = set(json.loads((ds / "dedup.json").read_text())["drop"].get(split, []))
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append([CLASSES.index(cat[a["category_id"]]), x, y, x + w, y + h])
    recs = []
    for im in sorted(coco["images"], key=lambda im: im["file_name"]):
        if im["file_name"].split("_")[0] not in allowed or im["file_name"] in drop:
            continue
        recs.append(
            {
                "path": str(ds / "images" / split / im["file_name"]),
                "kind": "publaynet",
                "id": im["file_name"],
                "layout": by_img.get(im["id"], []),
                "lines": None,
                "cmask": CMASK["publaynet"],
            }
        )
    return recs
