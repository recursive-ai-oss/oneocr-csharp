"""Tables of contents and lists of figures with dot leaders, rendered by Chromium at 300 DPI with exact truth.

    uv run python scripts/gen_leader_sets.py OUT [COUNT] [SEED] [FONTDIR]

Each page is a list of entries (`Figure 3. Collision geometry ........ 12`, wrapped titles with the leader on the last line, chapter
lines with a number in front). The leader is a run of full stops whose count is chosen so that it fills the space up to the page
number, and the stops and the number are one word in the truth (`.......12`) as in a text layer that comes from a PDF.
Fonts are open training-pool families given by FONTDIR (a directory with `PTSerif-Regular.ttf`, `PTSerif-Bold.ttf`, `PTSans-Regular.ttf`,
`PTSans-Bold.ttf`, `CrimsonText-Regular.ttf`, `CrimsonText-Bold.ttf`), never the families the evaluation sets hold out.
"""

from __future__ import annotations

import html
import io
import random
import sys
from pathlib import Path

from PIL import Image

from oneocr_bench.gen import H_CSS, H_PX, SCALE, W_CSS, W_PX, load_paragraphs
from oneocr_bench.render import WORDS_JS, add_ink, png_bytes, text_of, truth_words, write_page

FAMS = {
    "PT Serif": ("PTSerif-Regular.ttf", "PTSerif-Bold.ttf"),
    "PT Sans": ("PTSans-Regular.ttf", "PTSans-Bold.ttf"),
    "Crimson Text": ("CrimsonText-Regular.ttf", "CrimsonText-Bold.ttf"),
}
KINDS = ["figures", "contents", "tables"]
SKEL = ("<!doctype html><html><head><meta charset='utf-8'><style>{fonts}"
        "html,body{{margin:0;background:#fff}}#page{{width:{w}px;height:{h}px;box-sizing:border-box;padding:{m:.1f}px;overflow:hidden;"
        "color:#111;font-family:'{fam}';font-size:{fs:.2f}px;line-height:1.25}}"
        "h1{{font-size:1.5em;margin:0 0 .9em}}"
        ".e{{display:flex;align-items:last baseline;margin:0 0 {gap:.2f}em}}.e.c2{{margin-left:{ind}em}}"
        ".t{{display:block;max-width:calc(100% - 3.2em)}}.k{{flex:none;margin-left:auto;white-space:nowrap}}"
        "</style></head><body><div id='page'>{body}</div></body></html>")
FIT = """([] ) => {
  for (const e of document.querySelectorAll('.e')) {
    const t = e.querySelector('.t'), k = e.querySelector('.k');
    const num = k.dataset.num, rr = e.getBoundingClientRect();
    const r = document.createRange(); r.selectNodeContents(t);
    const rects = r.getClientRects(); let last = rects[rects.length - 1];
    const probe = document.createElement('span'); probe.style.whiteSpace = 'nowrap'; k.textContent = '.'.repeat(50) + num;
    const wAll = k.getBoundingClientRect().width; k.textContent = num; const wNum = k.getBoundingClientRect().width;
    const dot = (wAll - wNum) / 50;
    const room = rr.right - last.right - wNum - 0.35 * parseFloat(getComputedStyle(e).fontSize);
    const n = Math.max(0, Math.floor(room / dot));
    k.textContent = '.'.repeat(Math.min(n, +k.dataset.max)) + num;
  }
}"""


def e(t: str) -> str:
    return html.escape(t)


def title(rng: random.Random, paras: list[list[str]], lo: int, hi: int) -> str:
    words = " ".join(rng.choice(rng.choice(paras)) for _ in range(3)).replace("\u201c", "").replace("\u201d", "").split()
    n = rng.randint(lo, hi)
    start = rng.randrange(0, max(1, len(words) - n))
    seg = words[start:start + n]
    s = " ".join(seg).strip(" ,;:-\u2014\u2013")
    s = s.rstrip(".").rstrip(",")
    return s[:1].upper() + s[1:]


def entry(label: str, text: str, page: int, indent: bool = False) -> str:
    return (f"<div class='e{' c2' if indent else ''}'><span class='t'>{e(label + text)}</span>"
            f"<span class='k' data-num='{page}' data-max='400'>{page}</span></div>")


def render(count: int, seed: int, out: Path, set_name: str, fontdir: Path) -> list[dict]:
    from playwright.sync_api import sync_playwright

    paras = load_paragraphs()
    entries = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": W_CSS, "height": H_CSS}, device_scale_factor=SCALE)
        for i in range(count):
            pid = f"t{i + 1:03d}"
            rng = random.Random(f"{set_name}:{seed}:{i}")
            fam = list(FAMS)[i % len(FAMS)]
            kind = KINDS[(i // len(FAMS)) % len(KINDS)]
            fs = rng.choice([10, 11, 12]) * 96 / 72
            m = rng.choice([22, 25, 30]) * 96 / 25.4
            reg, bold = FAMS[fam]
            fonts = (f"@font-face{{font-family:'{fam}';src:url('{(fontdir / reg).resolve().as_uri()}');font-weight:400}}"
                     f"@font-face{{font-family:'{fam}';src:url('{(fontdir / bold).resolve().as_uri()}');font-weight:700}}")
            page_no = rng.randint(1, 9)
            rows = []
            head = {"figures": "Figures", "contents": "Contents", "tables": "Tables"}[kind]
            for j in range(rng.randint(14, 20)):
                page_no += rng.choice([0, 1, 1, 2, 3, 5, 8, 14])
                if kind == "figures":
                    rows.append(entry(f"Figure {j + 1}. ", title(rng, paras, 3, 17), page_no))
                elif kind == "tables":
                    rows.append(entry(f"Table {j // 3 + 1}-{j % 3 + 1}. ", title(rng, paras, 3, 15), page_no))
                else:
                    sub = rng.random() < 0.4
                    lab = f"{j // 4 + 1}.{j % 4 + 1} " if sub else f"{j // 4 + 1} "
                    rows.append(entry(lab, title(rng, paras, 1, 9), page_no, indent=sub))
            doc = SKEL.format(fonts=fonts, w=W_CSS, h=H_CSS, m=m, fam=fam, fs=fs, ind=1.5, gap=rng.choice([0.5, 0.7, 0.9]),
                              body=f"<h1>{e(head)}</h1>" + "".join(rows))
            page = ctx.new_page()
            src = out / "page.html"
            out.mkdir(parents=True, exist_ok=True)
            src.write_text(doc, encoding="utf-8")
            page.goto(src.as_uri())
            page.evaluate("f => Promise.all([document.fonts.load(`1em '${f}'`), document.fonts.load(`bold 1em '${f}'`)])", fam)
            page.evaluate(FIT, [])
            # Rows below the page bottom are removed so that nothing is clipped.
            page.evaluate("""(lim) => { const es = [...document.querySelectorAll('.e')];
                for (const x of es.reverse()) { if (x.getBoundingClientRect().bottom > lim) x.remove(); } }""", H_CSS - m)
            words = page.evaluate(WORDS_JS, "#page")
            png = page.screenshot(clip={"x": 0, "y": 0, "width": W_CSS, "height": H_CSS})
            page.close()
            img = Image.open(io.BytesIO(png)).convert("L").crop((0, 0, W_PX, H_PX))
            tw = truth_words(words, SCALE)
            add_ink(tw, img)
            truth = {"id": pid, "set": set_name, "seed": seed, "page_index": i, "kind": kind, "font": fam,
                     "font_px": round(fs, 2), "image": {"width": W_PX, "height": H_PX, "dpi": 300}, "words": tw, "text": text_of(tw)}
            entries.append(write_page(out, pid, png_bytes(img), "png", truth))
        browser.close()
    return entries


if __name__ == "__main__":
    out = Path(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20290101
    fontdir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("fonts")
    name = out.name
    ents = render(count, seed, out, name, fontdir)
    print(f"{name}: {len(ents)} pages in {out}")
