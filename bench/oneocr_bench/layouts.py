"""layout-v1: born-digital A4 pages with real layout structure, rendered by Chromium at 300 DPI.

Page kinds: two-column article, three-column newsletter, table page, form, and a mixed page (full-width heading,
two columns, then a table). Truth comes from DOM ranges (see render.py).

Reading-order rule, which is the document order of the generated HTML:
  * full-width title, byline and abstract first;
  * multi-column text column by column, each top to bottom (CSS columns fill in this order);
  * a table row by row, cell by cell left to right, header row first;
  * a form top to bottom by section; inside a section row by row, each label before its value,
    a checkbox's label after the (text-less) box.
"""

from __future__ import annotations

import html
import io
import random

from PIL import Image

from .assets import FAMILIES
from .gen import H_CSS, H_PX, SCALE, W_CSS, W_PX, load_paragraphs, prose
from .render import (
    WORDS_JS,
    font_css,
    png_bytes,
    text_of,
    truth_words,
    wait_fonts,
    write_page,
)
from .screens import (
    CITY,
    FIELD,
    STATUS,
    amount,
    date,
    ident,
    name,
    value_for,
)

KINDS = ["article2", "article3", "table", "form", "mixed"]
TITLES = ["Quarterly review of the regional office", "Notes on harbour traffic and the winter timetable",
          "A study of reading habits in small towns", "The cost of keeping a village library open",
          "Minutes of the finance committee", "Field report: orchard surveys and yields",
          "Newsletter of the riverside residents", "Annual summary of the cooperative"]
COLS = ["Item", "Reference", "Date", "Quantity", "Unit price", "Amount", "Status", "Owner", "City"]
SECTIONS = ["Applicant", "Contact", "Delivery", "Payment", "Declaration", "Office use"]
SKEL = ("<!doctype html><html><head><meta charset='utf-8'><style>{fonts}"
        "html,body{{margin:0;background:#fff}}#page{{width:{w}px;height:{h}px;box-sizing:border-box;padding:{m:.1f}px;overflow:hidden;"
        "color:#111;font-family:'{fam}';font-size:{fs:.2f}px;line-height:{lh};text-align:{ta}}}"
        "h1{{font-size:1.9em;margin:0 0 .3em;line-height:1.15}}h2{{font-size:1.2em;margin:.7em 0 .3em;break-after:avoid}}"
        "p{{margin:0 0 .6em}}.by{{color:#444;margin-bottom:.8em}}.abs{{font-style:normal;border-top:1px solid #333;border-bottom:1px solid #333;padding:.4em 0;margin-bottom:.9em}}"
        ".cols{{column-gap:{gap}px;column-count:{nc}}} .cols p,.cols h2{{break-inside:avoid-column}}"
        "table{{border-collapse:collapse;width:100%;margin:.4em 0 .8em}}th,td{{border:1px solid #555;padding:.28em .5em;text-align:left;vertical-align:top}}th{{background:#e4e4e4}}"
        ".sec{{background:#222;color:#fff;padding:.2em .5em;margin:.7em 0 .4em;font-weight:700}}.row{{display:flex;gap:.9em;margin-bottom:.5em}}.f{{flex:1}}"
        ".lb{{font-size:.8em;color:#333;margin-bottom:.1em}}.v{{border:1.5px solid #222;padding:.25em .45em;min-height:1.3em}}"
        ".ck{{display:flex;align-items:center;gap:.5em;margin:.25em 0}}.bx{{display:inline-block;width:.9em;height:.9em;border:1.5px solid #222}}"
        "</style></head><body><div id='page'>{body}</div></body></html>")


def e(t: str) -> str:
    return html.escape(t)


def cols_block(rng, paras, nc: int, n_paras: int) -> list[str]:
    out = []
    for i in range(n_paras):
        if i and i % 4 == 0:
            out.append(f"<h2>{e(rng.choice(['Background', 'Findings', 'Costs', 'Next steps', 'Method', 'Remarks']))}</h2>")
        out.append(f"<p>{e(prose(rng, paras, 220, 520))}</p>")
    return out


def table_html(rng, nrows: int) -> str:
    cols = rng.sample(COLS, rng.randint(4, 6))

    def cell(c):
        return {"Item": rng.choice(["Brass keys", "Field guides", "Ink cartridges", "Sample cases", "Tide tables", "Receipt books"]),
                "Reference": ident(rng), "Date": date(rng), "Quantity": str(rng.randint(1, 480)), "Unit price": amount(rng),
                "Amount": amount(rng), "Status": rng.choice(STATUS), "Owner": name(rng), "City": rng.choice(CITY)}[c]
    head = "".join(f"<th>{e(c)}</th>" for c in cols)
    rows = "".join("<tr>" + "".join(f"<td>{e(cell(c))}</td>" for c in cols) + "</tr>" for _ in range(nrows))
    return f"<table><tr>{head}</tr>{rows}</table>"


def form_blocks(rng) -> list[str]:
    out = []
    for sec in rng.sample(SECTIONS, rng.randint(3, 5)):
        out.append(f'<div class="sec">{e(sec)}</div>')
        for _ in range(rng.randint(1, 3)):
            labels = rng.sample(FIELD, rng.randint(1, 3))
            out.append('<div class="row">' + "".join(f'<div class="f"><div class="lb">{e(l)}</div><div class="v">{e(value_for(rng, l))}</div></div>' for l in labels) + "</div>")
        if rng.random() < 0.6:
            for c in rng.sample(["Same as billing address", "Deliver to a pick-up point", "I accept the terms", "Send a paper receipt", "Keep me informed"], rng.randint(1, 3)):
                out.append(f'<div class="ck"><span class="bx"></span><span>{e(c)}</span></div>')
    return out


FILL = """([blocks, limit, sel]) => {
  const c = document.querySelector(sel); let n = 0;
  for (const h of blocks) {
    const keep = c.innerHTML; c.insertAdjacentHTML('beforeend', h);
    const r = c.getBoundingClientRect();
    if (r.bottom > limit || c.scrollWidth > c.clientWidth + 1) { c.innerHTML = keep; break; }
    n++;
  }
  return n;
}"""


def render(count: int, seed: int, out, set_name: str) -> tuple[list[dict], str]:
    from playwright.sync_api import sync_playwright

    paras = load_paragraphs()
    fams = list(FAMILIES)
    entries = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": W_CSS, "height": H_CSS}, device_scale_factor=SCALE)
        for i in range(count):
            pid = f"p{i + 1:03d}"
            rng = random.Random(f"{set_name}:{seed}:{i}")
            fam = fams[i % len(fams)]
            kind = KINDS[i % len(KINDS)]
            fs = rng.choice([9.5, 10, 10.5, 11]) * 96 / 72
            m = rng.choice([16, 20, 24]) * 96 / 25.4
            nc = {"article2": 2, "article3": 3, "mixed": 2}.get(kind, 1)
            body_top, cols_blocks, tail = "", [], ""
            if kind in ("article2", "article3", "mixed"):
                body_top = (f"<h1>{e(rng.choice(TITLES))}</h1><div class='by'>{e(name(rng))} and {e(name(rng))} · {e(date(rng))}</div>"
                            f"<div class='abs'>{e(prose(rng, paras, 280, 480))}</div>")
                cols_blocks = cols_block(rng, paras, nc, 14)
            if kind == "mixed":
                tail = table_html(rng, rng.randint(4, 8))
            if kind == "table":
                body_top = f"<h1>{e(rng.choice(TITLES))}</h1><p>{e(prose(rng, paras, 200, 400))}</p>"
                cols_blocks = [table_html(rng, 40)]
            if kind == "form":
                body_top = f"<h1>{e(rng.choice(['Application form', 'Order form', 'Registration', 'Delivery request']))}</h1><p>{e(prose(rng, paras, 120, 240))}</p>"
                cols_blocks = form_blocks(rng)
            doc = SKEL.format(fonts=font_css(fam), w=W_CSS, h=H_CSS, m=m, fam=fam, fs=fs, lh=rng.choice([1.3, 1.4]),
                              ta=rng.choice(["left", "justify"]) if kind.startswith("article") else "left", gap=round(rng.uniform(18, 30)), nc=nc,
                              body=f"{body_top}<div class='cols' id='c'></div>{tail if kind != 'mixed' else ''}<div id='t'></div>")
            page = ctx.new_page()
            src = out / "page.html"
            out.mkdir(parents=True, exist_ok=True)
            src.write_text(doc, encoding="utf-8")
            page.goto(src.as_uri())
            wait_fonts(page, fam)
            limit = H_CSS - m
            if kind == "table":
                # the table is one block: add rows until the page is full
                page.evaluate("document.getElementById('c').style.columnCount='1'")
                rows_html = cols_blocks[0]
                import re
                head_row = re.search(r"<tr>.*?</tr>", rows_html).group(0)
                body_rows = re.findall(r"<tr>.*?</tr>", rows_html)[1:]
                page.evaluate("([h]) => { document.getElementById('c').innerHTML = '<table id=tt>' + h + '</table>'; }", [head_row])
                fill = rng.uniform(0.6, 1.0)
                lim = H_CSS - m - (1 - fill) * (H_CSS - 2 * m) * 0.6
                n = page.evaluate("""([rows, limit]) => { const t = document.getElementById('tt'); let n = 0;
                    for (const r of rows) { t.insertAdjacentHTML('beforeend', r); if (t.getBoundingClientRect().bottom > limit) { t.lastElementChild.remove(); break; } n++; } return n; }""", [body_rows, lim])
            elif kind == "mixed":
                n = page.evaluate(FILL, [cols_blocks, limit - 0.28 * (H_CSS - 2 * m), "#c"])
                page.evaluate("([h]) => { document.getElementById('t').innerHTML = h; }", [tail])
                if page.evaluate("document.getElementById('page').scrollHeight > document.getElementById('page').clientHeight + 1"):
                    raise RuntimeError(f"{pid}: mixed page overflows")
            else:
                fill = rng.uniform(0.7, 1.0)
                lim = limit - (1 - fill) * (H_CSS - 2 * m) * 0.5
                n = page.evaluate(FILL, [cols_blocks, lim, "#c"])
            if page.evaluate("document.getElementById('page').scrollHeight > document.getElementById('page').clientHeight + 1"):
                raise RuntimeError(f"{pid}: content overflows the page")
            words = page.evaluate(WORDS_JS, "#page")
            png = page.screenshot(clip={"x": 0, "y": 0, "width": W_CSS, "height": H_CSS})
            page.close()
            img = Image.open(io.BytesIO(png)).convert("L").crop((0, 0, W_PX, H_PX))
            tw = truth_words(words, SCALE)
            truth = {"id": pid, "set": set_name, "seed": seed, "page_index": i, "kind": kind, "font": fam, "columns": nc,
                     "font_px": round(fs, 2), "blocks_placed": n, "image": {"width": W_PX, "height": H_PX, "dpi": 300},
                     "words": tw, "text": text_of(tw)}
            ent = write_page(out, pid, png_bytes(img), "png", truth)
            ent.update(font=fam, kind=kind)
            entries.append(ent)
        version = browser.version
        browser.close()
    return entries, version
