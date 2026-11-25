"""Stacks the first N crops of line sets into one image with their labels, for looking at a generated set before training on it.

    uv run python -m gen.contact_sheet OUT.png SET_DIR [SET_DIR ...] [--n 12] [--scale 2]
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    a = sys.argv[1:]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 12
    sc = int(a[a.index("--scale") + 1]) if "--scale" in a else 2
    dirs = [Path(x) for x in a[1:] if not x.startswith("--") and not x.isdigit()]
    rows = []
    for d in dirs:
        labs = [json.loads(ln) for ln in (d / "labels.jsonl").read_text(encoding="utf8").splitlines()[:n]]
        for r in labs:
            rows.append((d.name, r["text"], Image.open(d / "images" / (r["id"] + ".png")).convert("L")))
    w = max(im.width for _, _, im in rows) * sc
    sheet = Image.new("L", (w + 260, sum(32 * sc + 4 for _ in rows)), 255)
    dr = ImageDraw.Draw(sheet)
    y = 0
    for name, text, im in rows:
        sheet.paste(im.resize((im.width * sc, 32 * sc), Image.NEAREST), (0, y))
        dr.text((w + 6, y + 8), f"{name}: {text}"[:60], fill=0)
        y += 32 * sc + 4
    sheet.save(a[0])


if __name__ == "__main__":
    main()
