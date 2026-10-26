"""A contact sheet of degraded synthetic lines with their exact-truth text, drawn on the CPU from a clean pack.

For reading a degradation profile before a pack is generated at scale: each tile is one line as the recognizer sees it (32 px high,
scaled up 3x) with the label beneath, and the sampled contrast, size and dot pitch.

    uv run python -m eval.line_sheet CONFIG CLEAN_DIR SIDE OUT.png [--n 36] [--seed 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from data import charset as cs
from degrade.gpu import degrade
from train.data import PackSet

BASE = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("clean")
    ap.add_argument("side")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    pack = PackSet(a.clean, cs.load(BASE / cfg["charset"]), limit=a.n)
    x, w = pack.batch(list(range(len(pack))))
    g = torch.Generator().manual_seed(a.seed)
    img, w32, par = degrade(x, w, cfg["gpu_degrade"][a.side], g)
    tiles = []
    for i in range(len(pack)):
        c = Image.fromarray(img[i, 0, :, : int(w32[i])].numpy(), mode="L").convert("RGB")
        c = c.resize((c.width * 3, c.height * 3), Image.NEAREST)
        t = Image.new("RGB", (max(c.width, 330), c.height + 28), "white")
        t.paste(c, (0, 0))
        dpe = float(par["dots_per_em"][i]) if "dots_per_em" in par else 0.0
        note = f"{pack.texts[i]}   em {float(par['em'][i]):.0f} contrast {float(par['contrast'][i]):.0f} dots {dpe:.0f}"
        ImageDraw.Draw(t).text((2, c.height + 2), note[:60], fill=(0, 0, 0))
        tiles.append(t)
    cols = 3
    rows = [tiles[i : i + cols] for i in range(0, len(tiles), cols)]
    sheet = Image.new("RGB", (cols * 340, sum(max(t.height for t in r) + 4 for r in rows)), "white")
    y = 0
    for r in rows:
        for k, t in enumerate(r):
            sheet.paste(t, (k * 340, y))
        y += max(t.height for t in r) + 4
    sheet.save(a.out)
    print("tiles", len(tiles), np.asarray(sheet).shape)


if __name__ == "__main__":
    main()
