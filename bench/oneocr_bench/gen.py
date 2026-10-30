"""Generate the clean A4 300 DPI page set: HTML through headless Chromium, truth from DOM geometry.

Every word box is the browser's own layout rectangle for a DOM Range over that word,
so truth never depends on OCR. Reading order is document order. One column, Latin only.

Also the page renderer the derived sets (degraded, phone) build on; see sets.py for the commands.
"""

from __future__ import annotations

import html
import io
import json
import random
import re
import unicodedata
from pathlib import Path

from PIL import Image

from .assets import FAMILIES, WORKS, font_path, gutenberg_body, sha256
from .paths import data_dir
from .render import FRAGS_JS, add_ink, text_of, truth_words

SET_NAME = "clean300-v1"
DPI = 300
SCALE = DPI / 96  # CSS px -> device px
W_CSS, H_CSS = 794, 1123  # A4 at 96 CSS px/in
W_PX, H_PX = 2480, 3508  # A4 at 300 DPI

ALLOWED = re.compile(r"^[\x20-\x7e\u2019\u2018\u201c\u201d\u2014\u2013\u00e9]+$")
SENT_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"\u201d\u2019]))\s+(?=[A-Z\u201c\"])")

NOUNS = [
    "Harbour", "Orchard", "Ledger", "Timetable", "Workshop", "Garden", "Archive", "Bridge", "Market",
    "Library", "Journey", "Season", "Station", "Meadow", "Committee", "Survey", "Budget", "Schedule",
    "Cottage", "Railway", "Almanac", "Inventory", "Harvest", "Lantern", "Compass", "Gazette",
]
HEAD_TEMPLATES = ["Notes on the {a}", "The {a} and the {b}", "{a} Report", "Annual {a} Review", "On the {a}", "{a} & {b}"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
ITEMS = ["packing list", "route map", "receipt book", "signed forms", "spare batteries", "sample cases",
         "tide tables", "meeting notes", "ink cartridges", "brass keys", "printed tickets", "field guides"]


AVOID_TEXT: list[Path] = []  # truth directories whose passages a new set must not reuse (see `avoid_text`)


def avoid_text() -> set[str]:
    """Opening words of every sentence in the truth files under AVOID_TEXT, as a key set; empty when none are set."""
    seen: set[str] = set()
    for d in AVOID_TEXT:
        for f in sorted(d.glob("*/truth/*.txt")):
            for line in f.read_text(encoding="utf-8").splitlines():
                words = line.split()
                for i in range(len(words) - 7):
                    seen.add(" ".join(words[i : i + 8]))
    return seen


def load_paragraphs() -> list[list[str]]:
    """Public-domain paragraphs split into sentences; deterministic order. Paragraphs that share an eight-word run
    with a truth under AVOID_TEXT are left out, so a new set can use fresh passages."""
    avoid = avoid_text() if AVOID_TEXT else set()
    paras: list[list[str]] = []
    for n in sorted(WORKS):
        body = gutenberg_body((data_dir() / "text" / f"pg{n}.txt").read_bytes())
        for raw in re.split(r"\n\s*\n", body):
            if raw.startswith(" ") or "[" in raw or "Chapter" in raw or "CHAPTER" in raw:
                continue
            t = unicodedata.normalize("NFC", " ".join(raw.replace("_", "").split()))
            if len(t) < 160 or not ALLOWED.match(t):
                continue
            sents = SENT_SPLIT.split(t)
            if avoid:
                w = t.split()
                if any(" ".join(w[i : i + 8]) in avoid for i in range(len(w) - 7)):
                    continue
            if len(sents) >= 2:
                paras.append(sents)
    return paras


def prose(rng: random.Random, paras: list[list[str]], lo: int, hi: int) -> str:
    target = rng.randint(lo, hi)
    while True:
        sents = rng.choice(paras)
        i = rng.randrange(len(sents))
        out, n = [], 0
        for s in sents[i:]:
            out.append(s)
            n += len(s) + 1
            if n >= target:
                break
        if n >= lo // 2:
            return " ".join(out)


def money(rng: random.Random) -> str:
    sym = rng.choice(["$", "\u00a3", "EUR ", "$"])
    v = rng.uniform(5, 99999)
    s = f"{v:,.2f}"
    return f"{sym}{s}" if sym != "EUR " else f"EUR {s}"


def date(rng: random.Random) -> str:
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1998, 2026)
    return rng.choice([f"{d} {MONTHS[m - 1]} {y}", f"{y}-{m:02d}-{d:02d}", f"{d:02d}/{m:02d}/{y}", f"{MONTHS[m - 1]} {d}, {y}"])


def figures(rng: random.Random) -> str:
    t = rng.choice([
        "Invoice {inv} dated {d1} totals {m1}, including tax at {pc}%. Payment is due by {d2}.",
        "Order {inv} shipped on {d1} with {q} units; the balance of {m1} remains open until {d2}.",
        "The meeting on {d1} began at {h}:{mi:02d} and closed with a budget of {m1}, up {pc}% from {m2}.",
        "Reference {inv}: {q} items received {d1}, valued at {m1}. Call 555-{p1:04d} before {d2}.",
    ])
    return t.format(inv=f"{rng.choice(['INV', 'PO', 'No.'])}-{rng.randint(1000, 99999)}", d1=date(rng), d2=date(rng),
                    m1=money(rng), m2=money(rng), pc=f"{rng.uniform(0.5, 29):.1f}", q=rng.randint(2, 480),
                    h=rng.randint(7, 17), mi=rng.randint(0, 59), p1=rng.randint(0, 9999))


# Line heights a page may use (a held-out tuning set for set type with tight leading replaces this before rendering).
LINE_HEIGHTS = [1.3, 1.4, 1.5]


def make_blocks(rng: random.Random, paras: list[list[str]]) -> list[tuple[str, list[str]]]:
    """(kind, lines of text); a list block has one entry per item, numbered inside the text."""
    h = lambda: rng.choice(HEAD_TEMPLATES).format(a=rng.choice(NOUNS), b=rng.choice(NOUNS))
    blocks = [("h1", [h()]), ("p", [prose(rng, paras, 300, 700)]), ("p", [prose(rng, paras, 250, 600)]), ("h2", [h()]),
              ("p", [figures(rng)]), ("p", [prose(rng, paras, 300, 650)])]
    n = rng.randint(3, 5)
    items = [f"{i + 1}. {rng.choice(ITEMS).capitalize()}, {rng.randint(2, 60)} pcs, {money(rng)}" for i in range(n)]
    blocks += [("list", items), ("p", [prose(rng, paras, 250, 600)]), ("h2", [h()]), ("p", [figures(rng)]),
               ("p", [prose(rng, paras, 400, 800)]), ("p", [prose(rng, paras, 300, 700)]),
               ("p", [prose(rng, paras, 300, 700)]), ("p", [prose(rng, paras, 300, 700)])]
    return blocks


def font_css(family: str) -> str:
    rules = []
    for name, weight, style in FAMILIES[family][1]:
        rules.append(
            f"@font-face{{font-family:'{family}';src:url('{font_path(family, name).as_uri()}');"
            f"font-weight:{weight};font-style:{style};}}"
        )
    return "\n".join(rules)


def page_html(family: str, size_pt: float, lh: float, margin_mm: float, justify: bool) -> str:
    m = margin_mm * 96 / 25.4
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{font_css(family)}
html,body{{margin:0;background:#fff}}
#page{{width:{W_CSS}px;height:{H_CSS}px;box-sizing:border-box;padding:{m:.2f}px;overflow:hidden;color:#000;
 font-family:'{family}';font-size:{size_pt * 96 / 72:.3f}px;line-height:{lh};text-align:{'justify' if justify else 'left'}}}
p,h1,h2{{margin:0 0 .8em 0}} h1{{font-size:1.7em;font-weight:700;line-height:1.2}} h2{{font-size:1.25em;font-weight:700;line-height:1.25}}
.li{{margin:0 0 .25em 0;padding-left:2em;text-indent:-1.4em;text-align:left}} .lg{{margin-bottom:.8em}}
</style></head><body><div id="page"><div id="c"></div></div></body></html>"""


def block_html(i: int, kind: str, lines: list[str]) -> str:
    if kind == "list":
        return f'<div class="lg" data-b="{i}">' + "".join(f'<p class="li">{html.escape(t)}</p>' for t in lines) + "</div>"
    return f'<{kind} data-b="{i}">{html.escape(lines[0])}</{kind}>'


FILL_JS = """([blocks, limit]) => {
  const c = document.getElementById('c'); let n = 0;
  for (const h of blocks) {
    const before = c.children.length; c.insertAdjacentHTML('beforeend', h);
    const el = c.children[before];
    if (el.getBoundingClientRect().bottom > limit) { c.removeChild(el); break; }
    n++;
  }
  return n;
}"""

WORDS_JS = "() => {" + FRAGS_JS + """
  const out = [];
  for (const blk of document.querySelectorAll('#c [data-b]')) {
    const b = +blk.dataset.b;
    const tw = document.createTreeWalker(blk, NodeFilter.SHOW_TEXT);
    for (let nd = tw.nextNode(); nd; nd = tw.nextNode()) {
      const re = /\\S+/g; let m;
      while ((m = re.exec(nd.data))) {
        for (const f of frags(nd, m.index, m.index + m[0].length)) out.push({...f, b});
      }
    }
  }
  return out;
}"""


def render_set(count: int, seed: int, out: Path, set_name: str = SET_NAME, post=None) -> dict:
    """`post(img, words, i, rng)` turns the rendered page into (image bytes, extension, words, params) for derived sets."""
    from playwright.sync_api import sync_playwright

    paras = load_paragraphs()
    fams = list(FAMILIES)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    entries = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": W_CSS, "height": H_CSS}, device_scale_factor=SCALE)
        for i in range(count):
            pid = f"p{i + 1:03d}"
            rng = random.Random(f"{set_name}:{seed}:{i}")
            family = fams[i % len(fams)]
            size_pt = rng.choice([10, 10.5, 11, 11.5, 12])
            lh = rng.choice(LINE_HEIGHTS)
            margin = rng.choice([18, 22, 25])
            justify = rng.random() < 0.5
            blocks = make_blocks(rng, paras)
            fill = rng.uniform(0.55, 1.0)
            limit = H_CSS - margin * 96 / 25.4 - (1 - fill) * (H_CSS - 2 * margin * 96 / 25.4)

            page = ctx.new_page()
            # file:// origin: about:blank pages may not load file:// fonts
            src = out / "page.html"
            src.write_text(page_html(family, size_pt, lh, margin, justify), encoding="utf-8")
            page.goto(src.as_uri())
            # fonts load lazily; measure only once regular and bold are in
            page.evaluate("f => Promise.all([document.fonts.load(`1em '${f}'`), document.fonts.load(`bold 1em '${f}'`)])", family)
            n = page.evaluate(FILL_JS, [[block_html(j, k, t) for j, (k, t) in enumerate(blocks)], limit])
            page.evaluate("document.fonts.ready")
            loaded = page.evaluate("[...document.fonts].filter(f=>f.status==='loaded').map(f=>f.family)")
            if family.replace("'", "") not in [x.strip("'\"") for x in loaded]:
                raise RuntimeError(f"{pid}: font {family} did not load ({loaded})")
            words = page.evaluate(WORDS_JS)
            png = page.screenshot(clip={"x": 0, "y": 0, "width": W_CSS, "height": H_CSS})
            page.close()

            img = Image.open(io.BytesIO(png)).convert("L").crop((0, 0, W_PX, H_PX))
            buf = io.BytesIO()
            img.save(buf, "PNG", compress_level=6)
            png_bytes = buf.getvalue()
            ext, params = "png", None

            tw = truth_words(words, SCALE)
            text = text_of(tw)
            add_ink(tw, img)  # from the clean render; the degradations move it with the box
            image = {"width": W_PX, "height": H_PX, "dpi": DPI}
            if post:
                png_bytes, tw, params = post(img, tw, i, rng)
                ext = "jpg"
                with Image.open(io.BytesIO(png_bytes)) as im2:
                    image = {"width": im2.size[0], "height": im2.size[1]}
            truth = {
                "id": pid, "set": set_name, "seed": seed, "page_index": i, "font": family, "size_pt": size_pt,
                "line_height": lh, "margin_mm": margin, "justify": justify, "blocks": n,
                "image": image,
                "words": tw,
                "text": text,
            }
            if params:
                truth["degradation"] = params
            truth_bytes = (json.dumps(truth, ensure_ascii=False, indent=1) + "\n").encode()
            (out / "pages" / f"{pid}.{ext}").write_bytes(png_bytes)
            (out / "truth" / f"{pid}.json").write_bytes(truth_bytes)
            (out / "truth" / f"{pid}.txt").write_text(text, encoding="utf-8")
            entries.append({"id": pid, "font": family, "words": len(words), "png_sha256" if ext == "png" else "image_sha256": sha256(png_bytes),
                            "truth_sha256": sha256(truth_bytes), "text_sha256": sha256(text.encode())})
        version = browser.version
        browser.close()
    return {"set": SET_NAME, "seed": seed, "count": count, "dpi": DPI, "chromium": version, "pages": entries}
