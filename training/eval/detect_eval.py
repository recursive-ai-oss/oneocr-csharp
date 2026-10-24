"""Layout AP and text-line P/R/F1 on DocLayNet (layout also on PubLayNet val), with an optional post-processing sweep on val.

Usage: uv run python -m eval.detect_eval --checkpoint best.pt --split val|test [--sweep] [--limit N] [--out r.json]
       [--expand 1.0 --text-thr 0.3 --lay-thr 0.4 --expand-layout 1.0]

DocLayNet's 11 classes are mapped onto ours (train.detect_data.DOCLAYNET_MAP) for truth and predictions alike, so AP is per mapped
class; `form_field` and `checkbox` have no truth in these datasets and are reported as null. The sweep picks the text and layout
post-processing values separately on val; test is evaluated once with the values given on the command line.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import torch
from PIL import Image

from eval.detect import raw_maps, score_maps
from models.dbnet import DBNetLite
from train.detect_data import load_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--expand", type=float, default=1.0)
    ap.add_argument("--text-thr", type=float, default=0.3)
    ap.add_argument("--lay-thr", type=float, default=0.6)
    ap.add_argument("--expand-layout", type=float, default=0.85)
    ap.add_argument("--sets", default="doclaynet,publaynet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    a = ap.parse_args()
    root = Path(os.environ.get("ONEOCR_DATASETS", Path.home() / "oneocr" / "datasets"))
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = DBNetLite(ck["config"])
    model.load_state_dict(ck["model"])
    model = model.to(a.device).eval()
    result: dict = {"checkpoint_step": ck["step"], "split": a.split}
    for name in a.sets.split(","):
        recs = load_records(name, root, a.split)
        step = max(1, len(recs) // (a.limit or len(recs)))
        recs = recs[::step][: a.limit]
        maps = [raw_maps(model, Image.open(r["path"]), a.device) for r in recs]
        post = {"expand_text": a.expand, "text_thr": a.text_thr, "lay_thr": a.lay_thr, "expand_layout": a.expand_layout}
        if a.sweep:
            best_t = (
                max(
                    (
                        (e, t, score_maps(maps, recs, expand_text=e, text_thr=t).get("text", {}).get("f1", 0.0))
                        for e, t in itertools.product([0.7, 0.85, 1.0, 1.15, 1.3], [0.2, 0.3, 0.4, 0.5])
                    ),
                    key=lambda v: v[2],
                )
                if name == "doclaynet"
                else (post["expand_text"], post["text_thr"], 0)
            )
            best_l = max(
                (
                    (t, e, score_maps(maps, recs, lay_thr=t, expand_layout=e)["layout"]["mAP"])
                    for t, e in itertools.product([0.3, 0.4, 0.5, 0.6], [0.7, 0.85, 1.0, 1.15])
                ),
                key=lambda v: v[2],
            )
            post = {"expand_text": best_t[0], "text_thr": best_t[1], "lay_thr": best_l[0], "expand_layout": best_l[1]}
            print(name, "sweep picked", json.dumps(post), flush=True)
        r = score_maps(maps, recs, **post)
        r["post"] = post
        result[name] = r
        print(name, json.dumps(r), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
