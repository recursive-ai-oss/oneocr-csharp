"""Evaluates the exported container (fp16-rounded, batch-norm-folded weights, fp32 compute) with the same scoring as eval.detect_eval.

Usage: uv run python -m eval.detect_fp16 --model model.onm --split val --limit 400 [--bench DIR] [--out r.json]
Post-processing values are the defaults in eval.detect.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.detect import infer, match_counts, prf, raw_maps, score_maps
from eval.detect_bench import SETS, truth_lines
from export.dbnet import FoldedReference, read_tensors
from train.detect_data import load_records


class ContainerModel:
    """Callable with the training model's interface, running the exported graph."""

    def __init__(self, path: Path):
        manifest, t = read_tensors(path)
        self.ref = FoldedReference(manifest["config"], {k: torch.from_numpy(v) for k, v in t.items()})

    def __call__(self, x, train_heads=False):
        return self.ref.forward(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--bench")
    ap.add_argument("--out")
    a = ap.parse_args()
    root = Path(os.environ.get("ONEOCR_DATASETS", Path.home() / "oneocr" / "datasets"))
    model = ContainerModel(Path(a.model))
    result: dict = {"model": a.model}
    recs = load_records("doclaynet", root, a.split)
    recs = recs[:: max(1, len(recs) // a.limit)][: a.limit]
    maps = [raw_maps(model, Image.open(r["path"]), "cpu") for r in recs]
    pp = read_tensors(Path(a.model))[0]["config"]["postprocess"]  # the thresholds the file itself carries
    post = {
        "expand_text": pp["text_expand"],
        "text_thr": pp["text_threshold"],
        "lay_thr": pp["layout_threshold"],
        "expand_layout": pp["layout_expand"],
    }
    result["doclaynet"] = score_maps(maps, recs, **post)
    print("doclaynet", json.dumps(result["doclaynet"]), flush=True)
    if a.bench:
        result["bench"] = {}
        for s in SETS[:1] + SETS[2:3]:  # clean and layout, a spot check of the rounded weights
            tp = np.zeros(3, np.int64)
            for f in sorted((Path(a.bench) / s / "truth").glob("*.json")):
                t = json.loads(f.read_text())
                res = infer(model, Image.open(next((Path(a.bench) / s / "pages").glob(f.stem + ".*"))), "cpu", **post)
                tp += match_counts(res["text"], truth_lines(t["words"]))
            result["bench"][s] = prf(*[int(v) for v in tp])
            print(s, json.dumps(result["bench"][s]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
