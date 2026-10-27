"""Text-line recall and precision of a detector on held-out web windows (gen/web_lines.py --windows-dir), sliced by text polarity and size.

Usage: uv run python -m eval.detect_web --dir WINDOWS_DIR --model model.onm [--publishers a,b] [--text-thr 0.3 --expand 0.4] [--out r.json]

Truth is the DOM line box of every line that passed the pixel checks. DOM lines that were not verified are ignore regions: a
prediction with no truth partner whose centre falls in one is not counted as a false positive. Recall and precision use IoU 0.5 and
one-to-one matching, like the other detector evaluations. The model runs as exported (fp16-rounded weights, folded batch norm).
Slices: `dark` = light text on a darker background; `small` = em height under 14 device pixels (the page is then scaled to 1024 px
on the long side, so this is the size the network sees times the page scale, reported separately as `em_seen`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval.detect import decode, eval_boxes, in_boxes, match_flags, raw_maps
from eval.detect_fp16 import ContainerModel
from export.dbnet import read_tensors


def run(maps: list[dict], recs: list[dict], **post) -> dict:
    acc: dict[str, list[int]] = {}

    def add(key: str, tp: int, n: int) -> None:
        a = acc.setdefault(key, [0, 0])
        a[0] += tp
        a[1] += n

    preds = ign_free = 0
    tp_all = 0
    for m, r in zip(maps, recs, strict=True):
        res = decode(m, **post)
        gt, ign = eval_boxes(r["lines"]), eval_boxes(r["ignore"])
        mp, mg = match_flags(res["text"], gt)
        pc = np.stack([(res["text"][:, 0] + res["text"][:, 2]) / 2, (res["text"][:, 1] + res["text"][:, 3]) / 2], 1)
        free = ~mp & in_boxes(pc, ign) if len(pc) else mp
        preds += len(mp) - int(free.sum())
        tp_all += int(mg.sum())
        longside = max(r["_w"], r["_h"])
        for k, hit in enumerate(mg):
            dark, em = bool(r["dark"][k]), float(r["em_px"][k])
            seen = em * 1024.0 / longside
            for key, on in (("all", True), ("dark", dark), ("light", not dark), ("small", em < 14), ("dark_small", dark and em < 14),
                            ("seen_lt8", seen < 8), ("dark_seen_lt8", dark and seen < 8)):  # fmt: skip
                if on:
                    add(key, int(hit), 1)
    out = {k: {"recall": round(v[0] / max(v[1], 1), 4), "lines": v[1]} for k, v in acc.items()}
    out["precision"] = round(tp_all / max(preds, 1), 4)
    out["predictions"] = preds
    out["windows"] = len(recs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--publishers")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--text-thr", type=float)
    ap.add_argument("--expand", type=float)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    a = ap.parse_args()
    root = Path(a.dir) / a.split
    recs = [json.loads(x) for x in (root / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
    if a.publishers:
        recs = [r for r in recs if r["publisher"] in a.publishers.split(",")]
    model = ContainerModel(Path(a.model))
    pp = read_tensors(Path(a.model))[0]["config"]["postprocess"]
    post = {"text_thr": a.text_thr or pp["text_threshold"], "expand_text": a.expand or pp["text_expand"]}
    maps = []
    for r in recs:
        img = Image.open(root / "images" / f"{r['id']}.png")
        r["_w"], r["_h"] = img.size
        maps.append(raw_maps(model, img, a.device))
    res = {"model": a.model, "post": post, "publishers": a.publishers, **run(maps, recs, **post)}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
