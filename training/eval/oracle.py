"""Upper bound of the decoding rule: decode the training targets themselves and score them against the truth.

Usage: uv run python -m eval.oracle [--limit 100]

If the shrink (train.detect_data) and the recovery (eval.detect) were exact inverses and the maps had infinite resolution, F1 and AP
would be 1.0. The gap is the quantisation cost of the stride-2 / stride-8 maps, so a trained model can't be judged against 1.0.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from eval.detect import score_maps
from train.detect_data import DetDataset, load_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    root = Path(os.environ.get("ONEOCR_DATASETS", Path.home() / "oneocr" / "datasets"))
    for name in ("doclaynet", "publaynet"):
        recs = load_records(name, root, "val")
        recs = recs[:: max(1, len(recs) // a.limit)][: a.limit]
        ds = DetDataset(recs, 1024, train=False)
        maps = []
        for i, r in enumerate(recs):
            s = ds[i]
            w, h = Image.open(r["path"]).size
            maps.append(
                {"prob": s["prob"][0].numpy(), "layout": s["layout"].numpy(), "orient": s["layout"].new_tensor([1, 0, 0, 0]).numpy(), "scale": 1024 / max(w, h)}
            )
        res = score_maps(maps, recs)
        print(name, json.dumps({"layout": res["layout"], "text": res.get("text")}), flush=True)


if __name__ == "__main__":
    main()
