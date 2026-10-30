"""Contact sheet of the words an engine output has that the truth lacks (the scorer's leftover hypothesis words), to look at them.

    uv run python scripts/extras_sheet.py --root EVAL_ROOT --out OUT_ROOT --set SET --page p012 --save sheet.png [--margin 40]

Words come from OUT_ROOT/SET/<page>.json; a word is listed when its text is a leftover of the bag-of-words comparison against the page's truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import normalize, truth_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--margin", type=int, default=40)
    ap.add_argument("--scorer", default="v3.1")
    a = ap.parse_args()
    truth_dir = Path(a.root) / a.set / "truth"
    ref = Counter(normalize(truth_text(truth_dir, a.page, a.scorer), a.scorer == "v3.1"))
    words = json.loads((Path(a.out) / a.set / f"{a.page}.json").read_text())["words"]
    seen: Counter = Counter()
    extra = []
    for w in words:
        t = " ".join(normalize(w["text"], a.scorer == "v3.1"))
        seen[t] += 1
        if seen[t] > ref[t]:
            extra.append(w)
    page = Image.open(next((Path(a.root) / a.set / "pages").glob(a.page + ".*"))).convert("L")
    tiles = []
    for w in extra:
        x0, y0, x1, y1 = [int(v) for v in w["box"]]
        m = a.margin
        tile = page.crop((max(0, x0 - m), max(0, y0 - m), min(page.width, x1 + m * 3), min(page.height, y1 + m))).convert("RGB")
        d = ImageDraw.Draw(tile)
        d.rectangle((x0 - max(0, x0 - m), y0 - max(0, y0 - m), x1 - max(0, x0 - m), y1 - max(0, y0 - m)), outline=(255, 0, 0))
        d.text((2, 2), f"{w['text']!r} c={w.get('conf', 0):.2f} @{x0},{y0}", fill=(255, 0, 0))
        tiles.append(tile)
        print(a.page, repr(w["text"]), [round(v) for v in w["box"]], round(w.get("conf", 0), 3))
    if not tiles:
        return
    wd = max(t.width for t in tiles)
    sheet = Image.new("RGB", (wd, sum(t.height + 4 for t in tiles)), (255, 255, 255))
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height + 4
    sheet.save(a.save)


if __name__ == "__main__":
    main()
