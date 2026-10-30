"""A shifted copy of a held-out line set: every crop moved up or down by 0-3 rows and scaled vertically by 0.94-1.06 (fixed seed).

Why: the page reader's crop box lands a few pixels off the ideal line box and the same word flips between right and wrong. The held-out
sets are cut at the ideal box, so they cannot show whether a model is robust to that. This writes the same crops with the same labels
under a fixed set of shifts, so a model's CER on the copy next to its CER on the original is the cost of the crop shift.

    uv run python -m gen.jitter_val --src DIR --out DIR2 --seed 129 [--limit 3000]

DIR holds images/<id>.png and labels.jsonl (the LineSet format). Output has the same layout and the same ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train.run import shift_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--dy", type=float, default=3.0)
    ap.add_argument("--scale", type=float, default=0.06)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    rows = [json.loads(line) for line in (src / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line][: a.limit]
    (out / "images").mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(a.seed)
    for r in rows:
        im = np.asarray(Image.open(src / "images" / (r["id"] + ".png")), dtype=np.float32)
        dy, s = float(g.uniform(-a.dy, a.dy)), float(1 + g.uniform(-a.scale, a.scale))
        t = shift_rows(torch.from_numpy(im).view(1, 1, *im.shape), torch.tensor([dy]), torch.tensor([s]))
        Image.fromarray(t[0, 0].round().clamp(0, 255).byte().numpy(), mode="L").save(out / "images" / (r["id"] + ".png"))
        r["jitter"] = {"dy": round(dy, 3), "scale": round(s, 4)}
    (out / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} crops -> {out}")


if __name__ == "__main__":
    main()
