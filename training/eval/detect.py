"""Inference, post-processing and metrics for the text detector + layout model (models/dbnet.py).

Post-processing is the rule the architecture spec gives the C# side:
  text    threshold the probability map, label 4-connected components, take each component's cell
          extent as the shrunk core and grow it back by the inverse of the training shrink.
  layout  the same per class on the stride-8 maps; no non-maximum suppression.
Scores are the mean probability over the component.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from train.detect_data import CLASSES, LAYOUT_SHRINK_CAP, LAYOUT_STRIDE, TEXT_SHRINK, TEXT_STRIDE

TEXT_THRESHOLD = 0.3
LAYOUT_THRESHOLD = 0.6  # picked on DocLayNet val (sweep in eval.detect_eval)
EXPAND_LAYOUT = 0.85
ORIENT_CONFIDENCE = 0.9  # below this the page is left unrotated
LONG_SIDE = 1024
MULTIPLE = 32


def _components(prob: np.ndarray, thr: float, min_cells: int):
    lab, n = ndimage.label(prob > thr)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == k
        if m.sum() < min_cells:
            continue
        yield sl, float(prob[sl][m].mean())


def text_boxes(prob: np.ndarray, thr: float = TEXT_THRESHOLD, expand: float = 1.0, min_cells: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Boxes [x0, y0, x1, y1] in input pixels from a stride-2 probability map."""
    s = TEXT_STRIDE
    boxes, scores = [], []
    k = TEXT_SHRINK / (1 - 2 * TEXT_SHRINK)  # core short side c = (1 - 2 r) m, offset r m = k c
    for sl, sc in _components(prob, thr, min_cells):
        y0, y1, x0, x1 = sl[0].start * s, sl[0].stop * s, sl[1].start * s, sl[1].stop * s
        d = k * min(x1 - x0, y1 - y0) * expand
        boxes.append([x0 - d, y0 - d, x1 + d, y1 + d])
        scores.append(sc)
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(scores, np.float32)


def layout_boxes(maps: np.ndarray, thr: float = LAYOUT_THRESHOLD, expand: float = 1.0) -> list[tuple[int, float, list[float]]]:
    s = LAYOUT_STRIDE
    out = []
    for c in range(maps.shape[0]):
        for sl, sc in _components(maps[c], thr, 1):
            y0, y1, x0, x1 = sl[0].start * s, sl[0].stop * s, sl[1].start * s, sl[1].stop * s
            core = min(x1 - x0, y1 - y0)
            d = (core / 3.0 if core <= 60 else LAYOUT_SHRINK_CAP) * expand
            out.append((c, sc, [x0 - d, y0 - d, x1 + d, y1 + d]))
    return out


def to_input(gray: np.ndarray) -> torch.Tensor:
    """uint8 gray -> float, (p / 255 - 0.5) / 0.5, the recognizer's normalisation."""
    return torch.from_numpy(gray).float().div(127.5).sub(1.0)[None, None]


def prepare(img: Image.Image, long_side: int = LONG_SIDE) -> tuple[np.ndarray, float]:
    """Resize so the long side is `long_side`, pad right/bottom with white to a multiple of 32."""
    g = img.convert("L")
    w, h = g.size
    s = long_side / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    g = g.resize((nw, nh), Image.BILINEAR, reducing_gap=2.0)
    W, H = -(-nw // MULTIPLE) * MULTIPLE, -(-nh // MULTIPLE) * MULTIPLE
    canvas = np.full((H, W), 255, np.uint8)
    canvas[:nh, :nw] = np.asarray(g)
    return canvas, s


@torch.no_grad()
def raw_maps(model, img: Image.Image, device="cuda", long_side: int = LONG_SIDE) -> dict:
    """Network outputs for one page: text probability (stride 2), layout probabilities (stride 8), orientation softmax, resize scale."""
    canvas, s = prepare(img, long_side)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(to_input(canvas).to(device), train_heads=False)
    return {
        "prob": out["prob"][0, 0].float().sigmoid().cpu().numpy(),
        "layout": out["layout"][0].float().sigmoid().cpu().numpy(),
        "orient": out["orient"][0].float().softmax(-1).cpu().numpy(),
        "scale": s,
    }


SNAP_SKIP = (CLASSES.index("table"), CLASSES.index("figure"))
SNAP_TOLERANCE = 8.0  # input pixels: how far a snapped box may leave the stride-8 box


def snap_to_lines(layout: list, lines: np.ndarray, tol: float = SNAP_TOLERANCE) -> list:
    """Replace a text-like region's box by the union of the text lines whose centres fall inside it.

    The stride-8 layout map locates a region to within a cell (8 px), too coarse for IoU 0.75+ on one-line titles and headers; the
    stride-2 line boxes are ink-accurate. The union is limited to the region grown by `tol`. Tables and figures keep their box
    (their content is not text lines); a region with no line inside keeps its box.
    """
    if len(lines) == 0:
        return layout
    cx, cy = (lines[:, 0] + lines[:, 2]) / 2, (lines[:, 1] + lines[:, 3]) / 2
    out = []
    for c, sc, b in layout:
        if c not in SNAP_SKIP:
            inside = (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])
            if inside.any():
                u = lines[inside]
                nb = [
                    max(u[:, 0].min(), b[0] - tol),
                    max(u[:, 1].min(), b[1] - tol),
                    min(u[:, 2].max(), b[2] + tol),
                    min(u[:, 3].max(), b[3] + tol),
                ]
                if nb[2] > nb[0] and nb[3] > nb[1]:
                    b = nb
        out.append((c, sc, b))
    return out


def decode(m: dict, text_thr=TEXT_THRESHOLD, lay_thr=LAYOUT_THRESHOLD, expand_text=1.0, expand_layout=EXPAND_LAYOUT, snap=True) -> dict:
    s = m["scale"]
    tb, ts = text_boxes(m["prob"], text_thr, expand_text)
    lb = layout_boxes(m["layout"], lay_thr, expand_layout)
    if snap:  # in input pixels, before the resize back to the page
        lb = snap_to_lines(lb, tb)
    return {"text": tb / s, "text_scores": ts, "layout": [(c, sc, [v / s for v in b]) for c, sc, b in lb], "orient": m["orient"]}


def infer(model, img: Image.Image, device="cuda", long_side: int = LONG_SIDE, **post):
    return decode(raw_maps(model, img, device, long_side), **post)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def match_counts(pred: np.ndarray, gt: np.ndarray, iou: float = 0.5) -> tuple[int, int, int]:
    """One-to-one greedy matching by descending IoU: (true positives, predictions, ground truths)."""
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt)
    m = iou_matrix(pred, gt)
    tp = 0
    used_p, used_g = set(), set()
    for flat in np.argsort(-m, axis=None):
        i, j = divmod(int(flat), m.shape[1])
        if m[i, j] < iou:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp, len(pred), len(gt)


def match_flags(pred: np.ndarray, gt: np.ndarray, iou: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one greedy matching by descending IoU: which predictions and which truths found a partner."""
    mp, mg = np.zeros(len(pred), bool), np.zeros(len(gt), bool)
    if len(pred) == 0 or len(gt) == 0:
        return mp, mg
    m = iou_matrix(pred, gt)
    for flat in np.argsort(-m, axis=None):
        i, j = divmod(int(flat), m.shape[1])
        if m[i, j] < iou:
            break
        if mp[i] or mg[j]:
            continue
        mp[i] = mg[j] = True
    return mp, mg


def in_boxes(centres: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """For each (x, y) centre: does it lie inside any of the boxes."""
    if len(boxes) == 0 or len(centres) == 0:
        return np.zeros(len(centres), bool)
    x, y = centres[:, 0:1], centres[:, 1:2]
    return ((x >= boxes[None, :, 0]) & (x <= boxes[None, :, 2]) & (y >= boxes[None, :, 1]) & (y <= boxes[None, :, 3])).any(1)


def match_counts_ignoring(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray, iou: float = 0.5) -> tuple[int, int, int]:
    """Like match_counts, but a prediction that found no truth and sits (by its centre) in an ignore region is not counted.

    Ignore regions are text the source could not verify (see train.detect_data.load_web_records): finding it is neither right nor wrong.
    """
    mp, mg = match_flags(pred, gt, iou)
    free = ~mp & in_boxes(np.stack([(pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2], 1), ignore) if len(pred) else mp
    return int(mp.sum()), int(len(pred) - free.sum()), len(gt)


def prf(tp: int, npred: int, ngt: int) -> dict:
    p = tp / max(npred, 1)
    r = tp / max(ngt, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "pred": npred, "gt": ngt}


def layout_ap(gt_images: list[dict], gt_anns: list[dict], dets: list[dict], class_names=CLASSES) -> dict:
    """COCO AP (IoU .50:.95) overall and per class. Classes with no ground truth are reported as None and left out of the mean."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO()
    coco.dataset = {"images": gt_images, "annotations": gt_anns, "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)]}
    coco.createIndex()
    if not dets:
        return {"mAP": 0.0, "AP50": 0.0, "per_class": {}}
    cd = coco.loadRes(dets)
    ev = COCOeval(coco, cd, "bbox")
    ev.params.maxDets = [1, 10, 500]
    ev.evaluate()
    ev.accumulate()
    prec = ev.eval["precision"]  # T x R x K x A x M
    per, per50 = {}, {}
    for k, name in enumerate(class_names):
        p = prec[:, :, k, 0, 2]
        per[name] = None if (p < 0).all() else float(p[p > -1].mean())
        p50 = prec[0, :, k, 0, 2]
        per50[name] = None if (p50 < 0).all() else float(p50[p50 > -1].mean())
    vals = [v for v in per.values() if v is not None]
    v50 = [v for v in per50.values() if v is not None]
    return {
        "mAP": float(np.mean(vals)) if vals else 0.0,
        "AP50": float(np.mean(v50)) if v50 else 0.0,
        "per_class": per,
        "per_class_AP50": per50,
    }


def coco_layout_gt(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """COCO ground truth; classes a source does not label (its `cmask`) are left out of truth and detections alike."""
    images, anns = [], []
    for i, r in enumerate(records):
        images.append({"id": i})
        for c, x0, y0, x1, y1 in r["layout"]:
            if r.get("cmask", [1] * 8)[int(c)]:
                anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(c), "bbox": [x0, y0, x1 - x0, y1 - y0],
                             "area": (x1 - x0) * (y1 - y0), "iscrowd": 0})  # fmt: skip
    return images, anns


def merge_split(pred: list, gt: list) -> tuple[int, int, int, int]:
    """(merged predictions, predictions, split truths, truths) for one page and class.

    A prediction is merged when it holds the bulk (>= 70 % of the area) of two or more truth boxes; a truth is split when
    two or more predictions each hold at least 30 % of it. Both count adjacent same-class regions the model cannot tell apart.
    """
    if not pred or not gt:
        return 0, len(pred), 0, len(gt)
    P, G = np.array(pred, np.float32), np.array(gt, np.float32)
    ix = np.clip(np.minimum(P[:, None, 2], G[None, :, 2]) - np.maximum(P[:, None, 0], G[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(P[:, None, 3], G[None, :, 3]) - np.maximum(P[:, None, 1], G[None, :, 1]), 0, None)
    inter = ix * iy
    ga = np.maximum((G[:, 2] - G[:, 0]) * (G[:, 3] - G[:, 1]), 1e-9)
    frac = inter / ga[None, :]
    return int(((frac >= 0.7).sum(1) >= 2).sum()), len(pred), int(((frac >= 0.3).sum(0) >= 2).sum()), len(gt)


def eval_boxes(lines: list) -> np.ndarray:
    """Truth lines (boxes or quads) as axis-aligned boxes."""
    out = [(ln[0], ln[1], ln[2], ln[3]) if len(ln) == 4 else (min(ln[0::2]), min(ln[1::2]), max(ln[0::2]), max(ln[1::2])) for ln in lines]
    return np.array(out, np.float32).reshape(-1, 4)


def score_maps(maps: list[dict], recs: list[dict], **post) -> dict:
    """Layout AP on all pages, text P/R/F1 on those that carry line boxes, for one post-processing setting."""
    dets, tp = [], np.zeros(3, np.int64)
    orient_ok = turned = 0
    ms = np.zeros(4, np.int64)
    for i, (m, r) in enumerate(zip(maps, recs, strict=True)):
        res = decode(m, **post)
        cm = r.get("cmask", [1] * 8)
        for c in range(len(cm)):
            if cm[c]:
                ms += merge_split([b for cc, _, b in res["layout"] if cc == c], [g[1:] for g in r["layout"] if int(g[0]) == c])
        for c, sc, b in res["layout"]:
            if not cm[int(c)]:
                continue
            dets.append({"image_id": i, "category_id": int(c), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": sc})
        orient_ok += int(res["orient"].argmax() == 0)
        turned += int(res["orient"].argmax() != 0 and res["orient"].max() >= ORIENT_CONFIDENCE)
        if r.get("lines") is not None:
            if r.get("ignore"):
                tp += match_counts_ignoring(res["text"], eval_boxes(r["lines"]), eval_boxes(r["ignore"]))
            else:
                tp += match_counts(res["text"], eval_boxes(r["lines"]))
    images, anns = coco_layout_gt(recs)
    out = {
        "pages": len(recs),
        "layout": layout_ap(images, anns, dets),
        "orient_upright_acc": orient_ok / max(len(recs), 1),
        "orient_turned_confident": turned / max(len(recs), 1),
        "merged_pred_rate": ms[0] / max(ms[1], 1),
        "split_truth_rate": ms[2] / max(ms[3], 1),
    }
    if tp[2] > 0:
        out["text"] = prf(*[int(v) for v in tp])
    return out


def evaluate_records(model, records: list[dict], device="cuda", limit: int | None = None, **post) -> dict:
    recs = records[:limit] if limit else records
    maps = [raw_maps(model, Image.open(r["path"]), device) for r in recs]
    return score_maps(maps, recs, **post)


if __name__ == "__main__":
    print(json.dumps({"hint": "see train/detect.py and export/ for entry points"}))
