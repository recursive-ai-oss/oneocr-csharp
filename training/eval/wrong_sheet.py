"""A contact sheet of wrongly read words: the page around the word, truth box green, recognised box red, truth and reading beneath.

Takes the JSON `wrong_readings` writes (`--json`) and the pages directory; looking at the sheet is how a confusion table gets checked
against what the crop actually shows (clipped ends, a neighbour's ink, faint print).

    python -m eval.wrong_sheet ROWS.json PAGES_DIR OUT.png [--n 40] [--seed 1] [--pad 14]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("pages")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pad", type=int, default=14)
    a = ap.parse_args()
    rows = [r for r in json.loads(Path(a.rows).read_text())["rows"] if r.get("box")]
    random.Random(a.seed).shuffle(rows)
    tiles = []
    for r in rows[: a.n]:
        pg = next(Path(a.pages).glob(r["page"] + ".*"), None)
        if pg is None:
            continue
        img = Image.open(pg).convert("RGB")
        x0, y0, x1, y1 = r["box"]
        c = img.crop((int(x0) - a.pad, int(y0) - a.pad, int(x1) + a.pad, int(y1) + a.pad))
        d = ImageDraw.Draw(c)
        d.rectangle((a.pad, a.pad, a.pad + x1 - x0, a.pad + y1 - y0), outline=(0, 170, 0))
        if r.get("hbox"):
            hx0, hy0, hx1, hy1 = r["hbox"]
            d.rectangle((hx0 - x0 + a.pad, hy0 - y0 + a.pad, hx1 - x0 + a.pad, hy1 - y0 + a.pad), outline=(220, 0, 0))
        s = 3 if c.width < 150 else 2
        c = c.resize((c.width * s, c.height * s), Image.NEAREST)
        t = Image.new("RGB", (max(c.width, 300), c.height + 34), "white")
        t.paste(c, (0, 0))
        ImageDraw.Draw(t).text((2, c.height + 2), f"{r['page']}  truth: {r['truth']}\nread:  {r['read']}", fill=(0, 0, 0))
        tiles.append(t)
    cols = 2
    h = sum(max(t.height for t in tiles[i : i + cols]) for i in range(0, len(tiles), cols))
    sheet = Image.new("RGB", (cols * 640, max(h, 10)), "white")
    y = 0
    for i in range(0, len(tiles), cols):
        row = tiles[i : i + cols]
        for k, t in enumerate(row):
            sheet.paste(t, (k * 640, y))
        y += max(t.height for t in row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.crop((0, 0, sheet.width, y)).save(a.out)
    print("tiles", len(tiles))


if __name__ == "__main__":
    main()
