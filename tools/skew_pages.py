#!/usr/bin/env python3
"""Synthetic multi-column pages, upright or rotated, with exact reading-order truth.

Used to tune and check the reading order of columns on tilted scans. The text is made up here (random sentences from a small
word list), the fonts are the Liberation and DejaVu families (SIL OFL / Bitstream Vera licence), the pages are CC0.

    uv run --with pillow python tools/skew_pages.py --out DIR [--set tune|check] [--seed N]

Every page is US Letter at 300 dpi, grey, with a `manifest.json` entry holding the file name, the angle (degrees, positive
counter-clockwise), the layout name and the truth words in reading order. A page's layout is a number of columns of unequal
length, an optional title (spanning all columns, or over the first column only) and optionally a table inside one column.
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

DPI = 300
W, H = int(8.5 * DPI), int(11 * DPI)

FONT_DIRS = ["/usr/share/fonts/truetype/liberation/", "/usr/share/fonts/truetype/dejavu/"]
FONTS = {
    "liberation-serif": ("LiberationSerif-Regular.ttf", "LiberationSans-Bold.ttf"),
    "liberation-sans": ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    "dejavu-serif": ("DejaVuSerif.ttf", "DejaVuSans-Bold.ttf"),
    "dejavu-sans": ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
}

NOUNS = ("account balance deposit payment invoice statement customer supplier charge credit transfer amount period report "
         "service agreement notice tax refund quarter budget ledger receipt order delivery contract rate fee limit reserve "
         "branch address office record schedule summary entry total").split()
VERBS = ("shows lists covers includes records reports confirms states requires allows follows reflects adds removes "
         "keeps holds sets marks lowers raises matches").split()
ADJ = ("monthly annual final current previous pending separate clear short late early joint standard additional "
       "regular open closed full partial net gross").split()
LEAD = ("Each", "The", "Every", "Your", "This", "Any", "Our", "A")
TAIL = ("for the period", "on the last day", "to the account", "in the next cycle", "before the due date",
        "after the notice", "with the supplier", "at the branch", "under the agreement", "by the office")
HEADS = ("Notes on charges", "Account summary", "Payment terms", "Quarterly report", "Supplier invoice", "Statement guide",
         "Delivery schedule", "Budget overview", "Customer notice", "Fee schedule")


def sentence(rng):
    return "%s %s %s %s %s %s." % (rng.choice(LEAD), rng.choice(ADJ), rng.choice(NOUNS), rng.choice(VERBS),
                                    rng.choice(ADJ) + " " + rng.choice(NOUNS), rng.choice(TAIL))


def paragraph(rng, sentences):
    return " ".join(sentence(rng) for _ in range(sentences))


def wrap(font, text, width):
    lines, cur = [], ""
    for word in text.split():
        trial = word if not cur else cur + " " + word
        if font.getlength(trial) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def money(rng):
    return "{:,.2f}".format(rng.randint(100, 999999) / 100)


def page(rng, spec, path):
    """Draws one page; returns (manifest entry). spec keys: font, size, columns, title, lines, table, angle, noise."""
    body_name, head_name = FONTS[spec["font"]]
    fdir = next(d for d in FONT_DIRS if os.path.exists(d + body_name))
    body = ImageFont.truetype(fdir + body_name, spec["size"])
    head = ImageFont.truetype(fdir + head_name, int(spec["size"] * 1.5))
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    columns = spec["columns"]
    margin = spec["margin"]
    gutter = spec["gutter"]
    colw = (W - 2 * margin - gutter * (columns - 1)) // columns
    lead = int(spec["size"] * 1.38)
    top = 260
    truth = []

    title = spec["title"]
    y0 = top
    if title != "none":
        text = rng.choice(HEADS)
        if title == "spanning":
            tw = head.getlength(text)
            d.text(((W - tw) / 2, top), text, font=head, fill=0)
        else:
            d.text((margin, top), text, font=head, fill=0)
        truth += text.split()
        y0 = top + int(spec["size"] * 3.2)

    for c in range(columns):
        x = margin + c * (colw + gutter)
        y = y0
        budget = spec["lines"][c]
        table_at = spec["table"] if spec["table_column"] == c else None
        n = 0
        placed_table = False
        while n < budget:
            if table_at is not None and not placed_table and n >= table_at:
                placed_table = True
                rows = spec["table_rows"]
                cols = 3
                cw = [colw * 0.46, colw * 0.24, colw * 0.30]
                for r in range(rows):
                    cells = ["Item %s" % rng.choice(NOUNS) if r else "Description", money(rng) if r else "Amount",
                             money(rng) if r else "Balance"]
                    cx = x
                    for k, cell in enumerate(cells):
                        if k == 0:
                            d.text((cx, y), cell, font=body, fill=0)
                        else:
                            d.text((cx + cw[k] - body.getlength(cell), y), cell, font=body, fill=0)
                        cx += cw[k]
                    truth += " ".join(cells).split()
                    y += lead
                    n += 1
                y += int(lead * 0.7)
                continue
            for ln in wrap(body, paragraph(rng, rng.randint(2, 4)), colw):
                if n >= budget:
                    break
                d.text((x, y), ln, font=body, fill=0)
                truth += ln.split()
                y += lead
                n += 1
            y += int(lead * 0.7)

    img = img.rotate(spec["angle"], resample=Image.BICUBIC, fillcolor=255)
    if spec["noise"]:
        # A few random grey specks, as on a photocopy; the page stays readable.
        d = ImageDraw.Draw(img)
        for _ in range(60):
            px, py = rng.randrange(W), rng.randrange(H)
            d.point((px, py), fill=rng.randrange(120, 200))
    img.save(path, dpi=(DPI, DPI))
    return {"file": os.path.basename(path), "angle": spec["angle"], "layout": spec["name"], "words": truth}


def layouts(rng, kind):
    """The layout menu: (name, columns, lines per column, title, table column, table row)."""
    if kind == "check":
        names = [("2col-unequal", 2, 38, 12, 0), ("2col-spanning-title", 2, 30, 18, 0), ("3col-short-last", 3, 30, 26, 9),
                 ("3col-spanning-title", 3, 24, 24, 20), ("2col-table", 2, 34, 26, 0)]
    else:
        names = [("2col-unequal", 2, 38, 12, 0), ("2col-narrow-title", 2, 36, 14, 0), ("2col-spanning-title", 2, 30, 18, 0),
                 ("3col-short-last", 3, 30, 26, 9), ("3col-spanning-title", 3, 24, 24, 20), ("2col-table", 2, 34, 26, 0),
                 ("3col-unequal", 3, 38, 22, 12)]
    return names


def build(out, kind, seed):
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)
    angles = [0.0, 0.5, -0.5, 1.5, -2.0, 3.0, -4.0, 6.0, -8.0] if kind == "tune" else [-4.0, 3.0, -1.0, 5.0, -7.0]
    entries = []
    idx = 0
    for angle in angles:
        for name, columns, a, b, c in layouts(rng, kind):
            lines = [a, b, c][:columns]
            title = "spanning" if "spanning" in name else ("narrow" if "narrow" in name else rng.choice(["narrow", "none"]))
            table = "table" in name
            spec = {
                "name": name, "font": rng.choice(sorted(FONTS)), "size": rng.choice([38, 42, 46]), "columns": columns,
                "title": title, "lines": lines, "angle": angle, "noise": rng.random() < 0.3,
                "margin": rng.choice([240, 300]), "gutter": rng.choice([100, 130, 160]),
                "table": 4 if table else None, "table_column": rng.choice([0, 1]) if table else -1, "table_rows": 6,
            }
            entries.append(page(rng, spec, os.path.join(out, "%s-%03d.png" % (kind, idx))))
            idx += 1
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(entries, fh, indent=1)
    print("wrote", len(entries), "pages to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="tune", choices=["tune", "check"])
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    build(a.out, a.set, a.seed if a.seed is not None else (148001 if a.set == "tune" else 148002))
