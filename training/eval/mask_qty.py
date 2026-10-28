"""Line CER of a checkpoint on card-mask and item-count lines, read from truth crops and from engine crops.

    uv run python -m eval.mask_qty --checkpoint final.pt --sets truth=DIR,engine=DIR [--errors DIR]

Each DIR is a materialised crop set (images/, labels.jsonl). A line is a *mask* line when its truth holds a run of three or more
mask marks (`*`, `X`, `x`, `•`, `#`) and a *quantity* line when it starts with a count mark (`2x`, `2 x`, `x2`, `2 @`, `QTY 2`) or
holds an `N @ price` form. Prints CER, WER and line accuracy per class and set, and the share of mask lines whose reading has
no mark at all (the dots failure). Pairing truth and engine sets on the same ids shows what the crop costs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from models import build
from models.crnn import greedy_decode
from train.data import LineSet, edit_distance, to_input

DEVICE = "cpu"
MASK = re.compile(r"[*Xx•#]{3,}")
QTY = re.compile(r"^(\d{1,3}\s?[xX@]|[xX]\s?\d{1,3}\b|(?i:qty|quantity)[:\s]*\d)|\b\d{1,3}\s?@\s?\S*\d")


@torch.no_grad()
def read(model, data: LineSet, charset: list[str], idx: list[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    order = sorted(idx, key=lambda i: data.widths[i])
    for s in range(0, len(order), 128):
        b = order[s : s + 128]
        x, widths = data.batch(b)
        logits = model(to_input(x, DEVICE), widths).float().cpu()
        for k, i in enumerate(b):
            out[i] = greedy_decode(logits[k, : model.out_len(int(widths[k]))], charset)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sets", required=True, help="name=DIR,name=DIR")
    ap.add_argument("--errors")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    global DEVICE
    DEVICE = a.device
    ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False)
    model = build(ck["config"]).to(a.device).eval()
    model.load_state_dict(ck["model"])
    for spec in a.sets.split(","):
        name, d = spec.split("=", 1)
        data = LineSet(Path(d), ck["charset"])
        classes = {"mask": [i for i, t in enumerate(data.texts) if MASK.search(t)],
                   "qty": [i for i, t in enumerate(data.texts) if QTY.search(t) and not MASK.search(t)]}  # fmt: skip
        for cls, idx in classes.items():
            if not idx:
                print(json.dumps({name: {cls: {"lines": 0}}}))
                continue
            hyp = read(model, data, ck["charset"], idx)
            ce = sum(edit_distance(hyp[i], data.texts[i]) for i in idx)
            ct = sum(len(data.texts[i]) for i in idx)
            we = sum(edit_distance(hyp[i].split(), data.texts[i].split()) for i in idx)
            wt = sum(len(data.texts[i].split()) for i in idx)
            r = {"lines": len(idx), "cer": round(100 * ce / ct, 2), "wer": round(100 * we / wt, 2),
                 "line_acc": round(sum(hyp[i] == data.texts[i] for i in idx) / len(idx), 4)}  # fmt: skip
            if cls == "mask":
                r["no_mark_read"] = round(sum(not MASK.search(hyp[i]) for i in idx) / len(idx), 4)
            print(json.dumps({name: {cls: r}}))
            if a.errors:
                Path(a.errors).mkdir(parents=True, exist_ok=True)
                with open(Path(a.errors) / f"{name}.{cls}.jsonl", "w", encoding="utf-8") as f:
                    for i in idx:
                        if hyp[i] != data.texts[i]:
                            f.write(json.dumps({"id": data.ids[i], "ref": data.texts[i], "hyp": hyp[i]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
