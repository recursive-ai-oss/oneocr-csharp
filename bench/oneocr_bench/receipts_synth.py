"""Synthetic till receipts photographed by a simulated phone, with exact truth.

Used where no real receipt images are left unseen (every CORD v2 split has been used for training or tuning). A receipt
is HTML rendered by headless Chromium in one of the seven held-out open families; every word box is the browser's own
DOM-range geometry, as for the other generated sets. Then the same phone-camera simulation as `phone-v1` places it on a
desk (perspective, uneven light, blur, sensor noise, JPEG) and moves the boxes with the image.

A row ("2 x Rye loaf ........ 7.80") is one truth block: left text, then right text, in DOM order. Store name, address,
date and time, items, tax, total and payment lines follow the layout of an ordinary till receipt. Text is generated
(merchant, item and street names from small word lists, amounts and dates from the seed); nothing is copied from a
dataset. Receipts are narrow and tall, so the phone canvas is taller than the page sets'.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
from PIL import Image

from . import degrade
from .assets import FAMILIES
from .render import (
    WORDS_JS,
    add_ink,
    font_css,
    text_of,
    truth_words,
    wait_fonts,
    write_page,
)

SCALE = 3.0  # device pixels per CSS pixel: small thermal print needs a sharp base render
CANVAS = (1600, 2400)  # phone canvas for receipts
MAX_ASPECT = 1.95  # receipt height / width; taller strips cannot sit inside the canvas at the usual page fraction

# Block index = the enclosing row, not the span that holds the text.
ROW_JS = (
    WORDS_JS.replace("if (!ids.has(el)) ids.set(el, ids.size);",
                     "const key = el.closest('[data-row]') || el; if (!ids.has(key)) ids.set(key, ids.size);")
    .replace("b: ids.get(el)", "b: ids.get(key)")
)
assert "closest" in ROW_JS and "ids.get(key)" in ROW_JS

FIRST = ["Harbor", "Maple", "Corner", "Sunrise", "Willow", "Granite", "Juniper", "Copper", "Meadow", "Lakeside", "Alder",
         "Northgate", "Bluebird", "Cedar", "Fairview", "Redwood", "Stone", "Linden"]
KIND = {"Market": ["Rye loaf", "Whole milk", "Free-range eggs", "Cheddar", "Basmati rice", "Olive oil", "Tomatoes", "Apples",
                   "Oat flakes", "Green tea", "Butter", "Pasta", "Orange juice", "Spinach", "Chicken breast", "Yoghurt"],
        "Cafe": ["Flat white", "Espresso", "Croissant", "Chai latte", "Blueberry muffin", "Club sandwich", "Soup of the day",
                 "Iced tea", "Cinnamon roll", "Lemon tart", "Granola bowl", "Cold brew"],
        "Hardware": ["Wood screws 50pk", "Paint roller", "Masking tape", "LED bulb", "Extension cord", "Wall plugs", "Sandpaper",
                     "Work gloves", "Hinge 3in", "Spirit level", "Garden hose", "Cable ties"],
        "Pharmacy": ["Vitamin C", "Hand cream", "Bandages", "Cough syrup", "Toothpaste", "Sun lotion", "Pain relief",
                     "Lip balm", "Eye drops", "Allergy tabs", "Thermometer", "Shampoo"],
        "Books": ["Atlas of birds", "Pocket notebook", "Fountain pen", "Ink cartridge", "Crossword book", "Bookmark set",
                  "Travel guide", "Sketch pad", "Greeting card", "Paperback novel", "Desk calendar", "Pencil set"]}
STREETS = ["Elm Street", "High Road", "Station Lane", "Mill Court", "Park Avenue", "Bridge Street", "Orchard Way", "Church Row",
           "Harbour Road", "Quarry Hill", "Market Square", "Oak Terrace"]
TOWNS = ["Springfield", "Ashford", "Brookdale", "Eastbury", "Fairhaven", "Kingsport", "Larchmont", "Northam", "Oakridge", "Riverton"]
FOOTER = ["THANK YOU", "Thank you for shopping with us", "Please come again", "Have a nice day", "Keep your receipt for returns"]
PAY = ["VISA", "MASTERCARD", "CASH", "DEBIT", "CONTACTLESS"]


def _money(v: int, cur: str) -> str:
    return f"{cur}{v // 100}.{v % 100:02d}"


def _row(left: str, right: str, cls: str = "") -> str:
    return f'<div class="row {cls}" data-row><span>{left}</span><span>{right}</span></div>'


def _line(text: str, cls: str = "") -> str:
    return f'<div class="row c {cls}" data-row><span>{text}</span></div>'


def make_receipt(rng: random.Random, n_items: int) -> tuple[str, dict]:
    kind = rng.choice(sorted(KIND))
    cur = rng.choice(["$", "$", "€", "£", ""])
    name = f"{rng.choice(FIRST)} {kind}"
    street = f"{rng.randint(1, 980)} {rng.choice(STREETS)}"
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(2019, 2026)
    date = rng.choice([f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{d:02d}.{m:02d}.{y}"])
    time = f"{rng.randint(7, 21):02d}:{rng.randint(0, 59):02d}"
    rows = [_line(name.upper(), "big"), _line(street), _line(f"{rng.choice(TOWNS)} {rng.randint(10000, 99999)}"),
            _line(f"Tel {rng.randint(200, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"), '<div class="sep"></div>',
            _row(date, time), _row("Receipt", f"#{rng.randint(1000, 999999)}"), '<div class="sep"></div>']
    sub = 0
    for name_ in rng.sample(KIND[kind], n_items):
        qty, unit = rng.choice([1, 1, 1, 2, 3]), rng.randint(80, 4200)
        amt = qty * unit
        sub += amt
        rows.append(_row(f"{qty} x {name_}" if qty > 1 else name_, _money(amt, cur)))
    rate = rng.choice([5, 7, 8, 10, 19, 20])
    tax = round(sub * rate / 100)
    total = sub + tax
    rows += ['<div class="sep"></div>', _row("Subtotal", _money(sub, cur)), _row(f"Tax {rate}%", _money(tax, cur)),
             _row("TOTAL", _money(total, cur), "big b")]
    pay = rng.choice(PAY)
    rows.append(_row(pay, _money(total, cur)))
    if pay != "CASH":
        rows.append(_row("Card", f"**** {rng.randint(0, 9999):04d}"))
    rows += ['<div class="sep"></div>', _line(rng.choice(FOOTER))]
    return "\n".join(rows), {"merchant_kind": kind, "currency": cur, "items": n_items, "total": total}


def receipt_html(family: str, body: str, width: int, size: float, ink: int, lh: float) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{font_css(family)}
html,body{{margin:0;background:#fff}}
#r{{width:{width}px;box-sizing:border-box;padding:18px 16px 20px;background:#fff;color:rgb({ink},{ink},{ink});display:inline-block;
 font-family:'{family}';font-size:{size:.2f}px;line-height:{lh}}}
.row{{display:flex;justify-content:space-between;gap:10px;white-space:nowrap}} .row.c{{justify-content:center}}
.big{{font-size:1.25em;font-weight:700}} .b{{font-weight:700}}
.sep{{border-top:1.5px dashed rgb({ink},{ink},{ink});margin:7px 0}}
</style></head><body><div id="r">{body}</div></body></html>"""


def _post_phone(set_name: str, seed: int):
    def post(img, words, i):
        for attempt in range(60):
            r = random.Random(f"{set_name}:{seed}:{i}:phone:{attempt}")
            n = np.random.default_rng([seed, i, attempt])
            try:
                data, w, params = degrade.phone_capture(img, words, r, n, CANVAS)
            except ValueError:
                continue
            params["attempt"] = attempt
            return data, w, params
        raise RuntimeError(f"receipt {i}: no placement fit the canvas")
    return post


def render(count: int, seed: int, out: Path, set_name: str) -> tuple[list[dict], str]:
    from playwright.sync_api import sync_playwright

    fams = list(FAMILIES)
    post = _post_phone(set_name, seed)
    entries: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 600, "height": 1400}, device_scale_factor=SCALE)
        for i in range(count):
            pid = f"r{i + 1:03d}"
            rng = random.Random(f"{set_name}:{seed}:{i}")
            family = fams[i % len(fams)]
            width = rng.randint(290, 400)
            size = rng.uniform(11.0, 14.5)
            ink = rng.choice([20, 40, 60, 85, 110])  # thermal print is often faint
            lh = rng.choice([1.25, 1.35, 1.45])
            n_items = rng.randint(3, 11)
            for attempt in range(30):
                body, info = make_receipt(rng, n_items)
                page = ctx.new_page()
                src = out / "page.html"
                src.parent.mkdir(parents=True, exist_ok=True)
                src.write_text(receipt_html(family, body, width, size, ink, lh), encoding="utf-8")
                page.goto(src.as_uri())
                wait_fonts(page, family)
                over = page.evaluate("[...document.querySelectorAll('[data-row]')].some(r => r.scrollWidth > r.clientWidth + 1)")
                box = page.evaluate("(() => { const b = document.querySelector('#r').getBoundingClientRect(); return [b.width, b.height]; })()")
                if over or box[1] / box[0] > MAX_ASPECT:
                    page.close()
                    n_items = max(3, n_items - 1)
                    size *= 0.97 if over else 1.0
                    continue
                words = page.evaluate(ROW_JS, "#r")
                png = page.locator("#r").screenshot()
                page.close()
                break
            else:
                raise RuntimeError(f"{pid}: no receipt fit after 30 attempts")
            img = Image.open(io.BytesIO(png)).convert("L")
            tw = truth_words(words, SCALE)  # the element sits at the viewport origin, so DOM rects scale straight to pixels
            text = text_of(tw)
            add_ink(tw, img)
            data, tw, params = post(img, tw, i)
            with Image.open(io.BytesIO(data)) as im2:
                size_px = im2.size
            truth = {"id": pid, "set": set_name, "seed": seed, "page_index": i, "font": family, "font_px": round(size, 2),
                     "ink_level": ink, "line_height": lh, "width_css": width, **info, "source": "synthetic",
                     "image": {"width": size_px[0], "height": size_px[1]}, "words": tw, "text": text, "degradation": params}
            e = write_page(out, pid, data, "jpg", truth)
            e.update(font=family, source="synthetic")
            entries.append(e)
        version = browser.version
        browser.close()
    return entries, version
