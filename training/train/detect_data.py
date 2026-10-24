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
    # PubLayNet boxes a whole list, DocLayNet one item at a time: its list channel is left unlabelled so the class keeps one meaning
    "publaynet": [1, 1, 0, 1, 1, 0, 0, 0],
    "pages": [1, 1, 1, 1, 1, 1, 1, 1],
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


def _oriented(q: np.ndarray) -> tuple[float, float, float, float, float]:
    """Centre, width, height and angle of a quad given as 4 corners in order (TL, TR, BR, BL up to rotation of the page)."""
    cx, cy = q[:, 0].mean(), q[:, 1].mean()
    e1, e2 = q[1] - q[0], q[3] - q[0]
    w, h = float(np.hypot(*e1)), float(np.hypot(*e2))
    return float(cx), float(cy), w, h, float(np.arctan2(e1[1], e1[0]))


def text_targets(quads: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probability core, threshold ground truth and threshold mask at stride 2 (DB, Liao et al. 2020).

    Each line is an oriented rectangle (a quad from a rotated or warped page is replaced by the rectangle with its edge lengths and
    first-edge angle); distances are measured in the rectangle's own frame, so a tilted line gets a tilted core.
    """
    H, W = h // TEXT_STRIDE, w // TEXT_STRIDE
    prob = np.zeros((H, W), np.float32)
    thr = np.zeros((H, W), np.float32)
    mask = np.zeros((H, W), np.float32)
    for q in quads:
        cx, cy, bw, bh, ang = _oriented(q)
        m = min(bw, bh)
        if m < 2.0:
            continue
        d = text_core_offset(m)
        reach = np.hypot(bw, bh) / 2 + d
        ia, ib = _cells(cx - reach, cx + reach, TEXT_STRIDE, W)
        ja, jb = _cells(cy - reach, cy + reach, TEXT_STRIDE, H)
        if ib < ia or jb < ja:
            continue
        xs = (np.arange(ia, ib + 1) + 0.5) * TEXT_STRIDE - cx
        ys = (np.arange(ja, jb + 1) + 0.5) * TEXT_STRIDE - cy
        c, s = np.cos(ang), np.sin(ang)
        a = xs[None, :] * c + ys[:, None] * s  # coordinate along the first edge
        b = -xs[None, :] * s + ys[:, None] * c
        dx, dy = np.abs(a) - bw / 2, np.abs(b) - bh / 2
        win = (slice(ja, jb + 1), slice(ia, ib + 1))
        core = (dx <= -d) & (dy <= -d)
        if core.any():
            prob[win] = np.maximum(prob[win], core.astype(np.float32))
        else:  # too small for a core at this stride: keep the cell nearest the centre
            ci, cj = int(cx // TEXT_STRIDE), int(cy // TEXT_STRIDE)
            if 0 <= ci < W and 0 <= cj < H:
                prob[cj, ci] = 1.0
            continue
        outside = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
        inside = np.maximum(np.minimum(-dx, -dy), 0.0)
        dist = np.where((dx > 0) | (dy > 0), outside, inside)
        band = dist <= d
        val = np.where(band, 1.0 - dist / d, 0.0).astype(np.float32)
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


def box_quads(b: np.ndarray) -> np.ndarray:
    """(N,4) boxes -> (N,4,2) quads, corners clockwise from top-left."""
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([np.stack([x0, y0], 1), np.stack([x1, y0], 1), np.stack([x1, y1], 1), np.stack([x0, y1], 1)], 1)


def as_quads(lines: list) -> np.ndarray:
    """Lines given as boxes (4 numbers) or quads (8 numbers) -> (N,4,2)."""
    if not lines:
        return np.zeros((0, 4, 2), np.float32)
    out = [np.array(ln, np.float32).reshape(4, 2) if len(ln) == 8 else box_quads(np.array([ln], np.float32))[0] for ln in lines]
    return np.stack(out)


def rot90_points(p: np.ndarray, k: int, w: int, h: int) -> tuple[np.ndarray, int, int]:
    """Rotate points (..., 2) with np.rot90(img, k) on an image of width w, height h (counter-clockwise)."""
    for _ in range(k % 4):
        # one CCW quarter turn: (x, y) -> (y, w - x); width and height swap
        p = np.stack([p[..., 1], w - p[..., 0]], -1)
        w, h = h, w
    return p, w, h


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
        lq = box_quads(lay[:, 1:] * np.array([sx, sy, sx, sy], np.float32))
        lc = lay[:, 0].astype(np.int64)
        wq = None if rec["lines"] is None else as_quads(rec["lines"]) * np.array([sx, sy], np.float32)
        arr = np.asarray(img)
        k = 0
        if self.train and rng.random() < self.rot_p:
            k = rng.choice([1, 2, 3])
            arr = np.rot90(arr, k)
            lq, _, _ = rot90_points(lq, k, nw, nh)
            if wq is not None:
                wq, _, _ = rot90_points(wq, k, nw, nh)
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
        shift = np.array([px - ox, py - oy], np.float32)

        def clip(q, keep_frac):
            """Shift quads into the canvas; keep those whose bounding box is mostly inside."""
            if len(q) == 0:
                return q, np.zeros(0, bool)
            q = q + shift
            lo, hi = q.min(1), q.max(1)
            area = (hi[:, 0] - lo[:, 0]) * (hi[:, 1] - lo[:, 1])
            clo, chi = np.clip(lo, 0, S), np.clip(hi, 0, S)
            carea = (chi[:, 0] - clo[:, 0]) * (chi[:, 1] - clo[:, 1])
            ok = (area > 0) & (carea >= keep_frac * area) & (chi[:, 0] - clo[:, 0] > 1) & (chi[:, 1] - clo[:, 1] > 1)
            return q, ok

        lq, ok = clip(lq, 0.5)
        lq, lc = lq[ok], lc[ok]
        lb = np.clip(np.concatenate([lq.min(1), lq.max(1)], 1), 0, S)
        if self.train:
            canvas = self._photometric(canvas, rng)
        out = {
            "image": torch.from_numpy(canvas)[None],
            "layout": torch.from_numpy(layout_targets(lb, lc, S, S)),
            "cmask": torch.tensor(rec["cmask"], dtype=torch.float32),
            "orient": torch.tensor(k),
            "text_valid": torch.tensor(float(wq is not None)),
        }
        if wq is not None:
            wq, ok = clip(wq, 0.6)
            p, t, m = text_targets(wq[ok], S, S)
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
        cache = Path(src["cache"].format(split=split)) if src.get("cache") else None
        if src.get("holdout_every"):
            # no separate validation set yet: every n-th page (by id order) is held out of training. Pages share templates with
            # training, so this estimate is optimistic; final numbers use the generator's own validation split.
            recs = load_page_records(Path(src["dir"]), "train", cache)
            k = src["holdout_every"]
            return [r for i, r in enumerate(recs) if (i % k == 0) == (split != "train")]
        return load_page_records(Path(src["dir"]), src.get("splits", {}).get(split, split), cache)
    raise ValueError(name)


PAGE_ROLES = {
    "text": "text", "title": "title", "list": "list", "table": "table", "figure": "figure",
    "header": "header_footer", "footer": "header_footer", "formField": "form_field", "checkbox": "checkbox",
}  # fmt: skip


def split_line(ln: dict, gap: float = 1.2) -> list[list[float]]:
    """Cut a generated line (a DOM line, possibly spanning table cells or form columns) wherever two words are more than
    `gap` line heights apart, so each piece is one text run the recognizer reads as a line. Boxes for upright pages, quads
    (cut along the line's own direction) for rotated or warped ones."""
    words = ln["words"]
    quad = ln.get("quad")
    if quad is not None:
        q = np.array(quad, np.float32).reshape(4, 2)
    else:
        x0, y0, x1, y1 = ln["bounds"]
        q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    e1, e2 = q[1] - q[0], q[3] - q[0]
    length, height = float(np.hypot(*e1)), float(np.hypot(*e2))
    if len(words) < 2 or length <= 0:
        return [[round(float(v), 1) for v in q.reshape(-1)]] if quad is not None else [[round(v, 1) for v in ln["bounds"]]]
    u = e1 / length
    spans = []
    for w in words:
        pts = (
            np.array(w["quad"], np.float32).reshape(4, 2)
            if "quad" in w
            else np.array(
                [
                    [w["bounds"][0], w["bounds"][1]],
                    [w["bounds"][2], w["bounds"][1]],
                    [w["bounds"][2], w["bounds"][3]],
                    [w["bounds"][0], w["bounds"][3]],
                ],
                np.float32,
            )
        )
        t = (pts - q[0]) @ u
        spans.append((float(t.min()), float(t.max())))
    spans.sort()
    groups = [[spans[0][0], spans[0][1]]]
    for a, b in spans[1:]:
        if a - groups[-1][1] > gap * height:
            groups.append([a, b])
        else:
            groups[-1][1] = max(groups[-1][1], b)
    if len(groups) == 1:
        return [[round(float(v), 1) for v in q.reshape(-1)]] if quad is not None else [[round(v, 1) for v in ln["bounds"]]]
    out = []
    for a, b in groups:
        a, b = max(a, 0.0), min(b, length)
        tl, tr = q[0] + u * a, q[0] + u * b
        poly = np.array([tl, tr, tr + e2, tl + e2], np.float32)
        if quad is None:
            out.append([round(float(v), 1) for v in (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())])
        else:
            out.append([round(float(v), 1) for v in poly.reshape(-1)])
    return out


def load_page_records(path: Path, split: str, cache: Path | None = None) -> list[dict]:
    """Synthetic pages (exact truth from the renderer's own layout): line boxes or quads, region boxes by role.

    `path` is the generator's output root holding `<split>/images` and `<split>/truth`. Parsing thousands of truth files takes a
    while, so the parsed records are cached as one JSON next to the run state.
    """
    root = path / split
    if cache is not None and cache.exists():
        c = json.loads(cache.read_text())
        if isinstance(c, dict) and c.get("root") == str(root) and c.get("v") == 2:  # a cache written for another page set is rebuilt
            return c["recs"]
    recs = []
    for tf in sorted((root / "truth").glob("*.json")):
        d = json.loads(tf.read_text())
        pg = d["pages"][0]
        img = root / d["synthetic"]["image"]["file"]
        if not img.exists():
            continue
        lines = []
        for blk in pg["blocks"]:
            for ln in blk["lines"]:
                lines += split_line(ln)
        layout = [
            [CLASSES.index(PAGE_ROLES[r["role"]]), *[round(v, 1) for v in r["bounds"]]] for r in pg["regions"] if r["role"] in PAGE_ROLES
        ]
        recs.append(
            {"path": str(img), "kind": "pages", "id": tf.stem, "layout": layout, "lines": lines, "cmask": CMASK["pages"],
             "doc": pg["documentType"], "profile": d["synthetic"]["degradation"]["profile"]}
        )  # fmt: skip
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"root": str(root), "v": 2, "recs": recs}))
    return recs


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
