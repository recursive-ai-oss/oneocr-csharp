"""Draw a page's truth over its image, to look at it.

  uv run python -m pages.overlay RUN/train/images/train-0000001.jpg [OUT.png]

Words are green quads, regions are coloured by role with their DocLayNet class, table cells are blue, checkboxes red,
key-value pairs are joined by a line from key to value. Reading order is shown by the block number at each block's top-left.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROLE_COLOURS = {"text": "#e67e22", "title": "#c0392b", "list": "#8e44ad", "table": "#2980b9", "figure": "#16a085", "header": "#7f8c8d",
                "footer": "#7f8c8d", "formField": "#d35400", "checkbox": "#e74c3c"}  # fmt: skip


def quad_of(o: dict) -> list[float]:
    if "quad" in o:
        return o["quad"]
    l, t, r, b = o["bounds"]
    return [l, t, r, t, r, b, l, b]


def draw(img_path: Path, truth_path: Path) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    doc = json.loads(truth_path.read_text(encoding="utf-8"))
    page = doc["pages"][0]
    d = ImageDraw.Draw(img, "RGBA")
    words = []
    for bi, b in enumerate(page["blocks"]):
        for ln in b["lines"]:
            for w in ln["words"]:
                words.append(w)
                q = quad_of(w)
                d.polygon(q, outline=(0, 160, 0, 255))
        q = quad_of(b)
        d.text((q[0] + 2, q[1] - 10), str(bi), fill=(0, 0, 255, 255))
    for r in page["regions"]:
        c = ROLE_COLOURS.get(r["role"], "#000000")
        q = quad_of(r)
        d.polygon(q, outline=c, width=2)
        d.text((q[0] + 3, q[1] + 1), f"{r['role']}/{r.get('docLayNet')}", fill=c)
    for t in page["tables"]:
        for c in t["cells"]:
            l, tp, rr, bt = c["bounds"]
            d.rectangle([l, tp, rr, bt], outline=(41, 128, 185, 200))
    for kv in page["keyValues"]:
        k, v = words[kv["key"][0]], words[kv["value"][0]]
        kb, vb = k["bounds"], v["bounds"]
        d.line([(kb[2], (kb[1] + kb[3]) / 2), (vb[0], (vb[1] + vb[3]) / 2)], fill=(200, 0, 200, 255), width=2)
    for c in page.get("checkboxes", []):
        d.rectangle(c["bounds"], outline=(255, 0, 0, 255), width=3 if c["checked"] else 1)
    for f in page.get("fields", []):
        ws = [words[i]["bounds"] for i in f["words"]]
        d.rectangle(
            [min(w[0] for w in ws) - 1, min(w[1] for w in ws) - 1, max(w[2] for w in ws) + 1, max(w[3] for w in ws) + 1],
            outline=(255, 0, 255, 255),
        )
        d.text((ws[0][0], ws[0][1] - 9), f["id"], fill=(255, 0, 255, 255))
    return img


if __name__ == "__main__":
    p = Path(sys.argv[1])
    run = p.parent.parent
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else p.with_suffix(".overlay.png")
    im = draw(p, run / "truth" / (p.stem + ".json"))
    im.thumbnail((1400, 1400))  # a viewing aid, not a data product
    im.save(out)
