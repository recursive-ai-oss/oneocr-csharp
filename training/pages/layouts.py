"""Page kinds that are not business documents: prose (1-3 columns, headings, lists, figures, footnotes, running header and
footer), tables (ruled and unruled, spans) and forms (key-value fields, checkboxes).

Every builder returns a PageSpec: the HTML with truth annotations (see htmlkit), what to do when the content does not fit, and
the discrete template choices that decide which page split a layout belongs to. A layout whose template hash falls in the
validation share is never produced for the train split and the other way round.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, replace

from . import corpus
from .fontpool import Family, pick
from .htmlkit import Look, blk, frame, head_text, inline

PAGE_SIZES = [(794, 1123, 0.55), (816, 1056, 0.33), (719, 1016, 0.07), (559, 794, 0.05)]  # A4, Letter, B5, A5 at 96 dpi
PAPERS = ["#ffffff"] * 6 + ["#fdfcf7", "#f7f3e8", "#f2f2f2", "#eef3f8", "#faf6ee"]
INKS = ["#000000"] * 4 + ["#111111", "#1c1c1c", "#222233", "#2b2b2b", "#10203a"]
ACCENTS = ["#000000", "#1f3a68", "#7a1f1f", "#1f5a3a", "#555555", "#8a5a00"]
# light-on-dark pages (web dark themes, terminals-in-documents, slides): paper, ink, accent
DARK_PAPERS = ["#000000", "#0d0d0d", "#121212", "#16181d", "#1b1f27", "#1e2a38", "#0f2a43", "#222222", "#2d2d30", "#1c1c3a"]
LIGHT_INKS = ["#ffffff", "#f2f2f2", "#e6e6e6", "#dcdcdc", "#cfd8dc", "#d8dee9", "#c9d1d9", "#b8c4d0"]
LIGHT_ACCENTS = ["#ffffff", "#8ab4f8", "#ffb4a8", "#9ad9a8", "#d0d0d0", "#f2c46d"]
BAND_PAPERS = ["#1f2937", "#112e51", "#0b3d2e", "#2b2b2b", "#4a1f1f", "#162447", "#000000"]
VAL_SHARE = 8  # one template hash in 8 is reserved for the validation split


@dataclass
class PageSpec:
    html: str
    kind: str
    lang: str
    template: dict
    fonts: list[str]
    width: int
    height: int
    fit: str = "trim"  # trim: drop trailing flow items until it fits; retry: the builder is called again smaller; none
    extra: dict = field(default_factory=dict)


def template_split(template: dict) -> str:
    h = hashlib.sha256(repr(sorted(template.items())).encode()).digest()
    return "val" if h[0] % VAL_SHARE == 0 else "train"


@dataclass
class Ctx:
    rng: random.Random
    bank: corpus.TextBank
    pool: dict[str, Family]
    split: str
    root: object
    lang: str = "en"
    cps: frozenset[int] = frozenset()
    kv: int = 0
    dark_p: float = 0.0  # share of pages drawn light-on-dark (whole page, or dark bands on a light page)
    receipt_canvas: tuple = ("tight", "page", "page")  # receipts: what the slip lies on (a page-sized surface shrinks the type)

    def clean(self, s: str) -> str:
        return "".join(c for c in s if c.isspace() or ord(c) in self.cps) if self.cps else s

    def next_kv(self) -> int:
        self.kv += 1
        return self.kv

    def para(self, lo=25, hi=100):
        return self.clean(self.bank.paragraph(self.rng, self.lang, (lo, hi)))

    def phrase(self, lo=2, hi=6):
        return self.clean(self.bank.sentence(self.rng, self.lang, (lo, hi)))


def script_of(lang: str) -> str:
    return "greek" if lang == "el" else "cyrillic" if lang in ("ru", "bg", "sr") else "latin"


def _darken(ctx: Ctx, look: Look, w: int, h: int, mx: float, my: float) -> Look:
    """With probability `ctx.dark_p` turn the page light-on-dark in small type: the whole page, or dark bands (running head and foot,
    headings) on a light page. Drawn from a generator keyed by the page's own look, so with dark_p = 0 the page generator consumes
    exactly the random numbers it always did."""
    if ctx.dark_p <= 0:
        return look
    dr = random.Random(f"dark:{look.px:.4f}:{mx:.4f}:{my:.4f}:{w}:{h}")
    if dr.random() >= ctx.dark_p:
        return look
    px = dr.uniform(9.0, 13.0)  # the small type real site chrome is set in
    if dr.random() < 0.6:
        return replace(look, px=px, ink=dr.choice(LIGHT_INKS), paper=dr.choice(DARK_PAPERS), accent=dr.choice(LIGHT_ACCENTS))
    bp, bi = dr.choice(BAND_PAPERS), dr.choice(LIGHT_INKS)
    bars = (f".hd,.ft{{background:{bp};color:{bi};border:none;padding:.5em .8em}}"
            f"h1.h,h2.h{{background:{bp};color:{bi};padding:.3em .6em;border:none}}")
    return replace(look, px=px, bars=bars)


def draw_look(ctx: Ctx, tpl: dict, need_mono: bool = False) -> Look:
    from .fontpool import SCRIPT_SAMPLES

    rng = ctx.rng
    need = set(SCRIPT_SAMPLES[script_of(ctx.lang)])
    serif = tpl["body_role"] == "serif"
    body = pick(ctx.pool, rng, ctx.split, (tpl["body_role"],), need)
    head = (
        body
        if tpl["head_same"]
        else pick(ctx.pool, rng, ctx.split, ("sans", "serif", "display") if rng.random() < 0.15 else ("sans", "serif"), need)
    )
    mono = pick(ctx.pool, rng, ctx.split, ("mono",), set(SCRIPT_SAMPLES["latin"])) if need_mono else None
    ctx.cps = body.files[0].cps
    w, h, _ = rng.choices(PAGE_SIZES, [s[2] for s in PAGE_SIZES])[0]
    px = rng.uniform(11.5, 15.5) if w > 600 else rng.uniform(10.5, 13.5)
    if tpl["cols"] == 3:
        px = min(px, 13.0)
    mx = rng.uniform(0.06, 0.13) * w
    my = rng.uniform(0.05, 0.11) * h
    look = Look(
        body=body, head=head, mono=mono, px=px, lh=rng.uniform(1.25, 1.55) if serif or rng.random() < 0.5 else rng.uniform(1.3, 1.6),
        justify=tpl["justify"], indent=tpl["indent"], ink=rng.choice(INKS), paper=rng.choice(PAPERS), accent=rng.choice(ACCENTS),
        margins=(my, mx, my, mx * rng.uniform(0.9, 1.15)), cols=tpl["cols"], gap=px * rng.uniform(1.6, 2.6), col_rule=tpl["col_rule"],
        width=w, height=h, head_scale=rng.uniform(1.15, 1.7), head_bold=rng.random() < 0.8, head_caps=tpl["head_caps"],
        head_rule=tpl["head_rule"], head_align=tpl["head_align"],
    )  # fmt: skip
    return _darken(ctx, look, w, h, mx, my)


def draw_template(ctx: Ctx, kind: str, **fixed) -> dict:
    """Discrete layout choices, redrawn until the template belongs to the page split."""
    rng = ctx.rng
    for _ in range(200):
        t = {
            "kind": kind, "cols": rng.choices([1, 2, 3], [5, 4, 2])[0], "body_role": rng.choice(["serif", "sans"]),
            "head_same": rng.random() < 0.6, "justify": rng.random() < 0.5, "indent": rng.random() < 0.4,
            "col_rule": rng.random() < 0.3, "head_caps": rng.random() < 0.2, "head_rule": rng.random() < 0.3,
            "head_align": rng.choice(["left", "left", "center"]), "title": rng.random() < 0.7,
            "running_head": rng.choice(["none", "split", "center", "rule"]), "page_no": rng.choice(["none", "center", "right", "split"]),
            "list_mark": rng.choice(["bullet", "dash", "num", "alpha", "paren"]), "footnote": rng.random() < 0.2,
        }  # fmt: skip
        t.update(fixed)
        if template_split(t) == ("train" if ctx.split == "train" else "val"):
            return t
    return t


# --- prose ------------------------------------------------------------------------------------------------------------------


def _marker(style: str, i: int) -> str:
    return {"bullet": "•", "dash": "–", "num": f"{i + 1}.", "alpha": f"{chr(97 + i % 26)})", "paren": f"({i + 1})"}[style]


def figure_svg(rng: random.Random, w: int, h: int, ink: str, accent: str) -> str:
    kind = rng.choice(["bars", "lines", "blocks", "gradient", "scatter"])
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    cols = [accent, ink, "#4a7a9b", "#b0b0b0", "#c26a2d"]
    if kind == "bars":
        n = rng.randint(4, 10)
        bw = (w - 40) / n
        s.append(f'<path d="M20 10V{h - 20}H{w - 10}" stroke="{ink}" fill="none" stroke-width="1.5"/>')
        for i in range(n):
            bh = rng.uniform(0.15, 0.95) * (h - 40)
            s.append(
                f'<rect x="{25 + i * bw:.1f}" y="{h - 20 - bh:.1f}" width="{bw * 0.7:.1f}" height="{bh:.1f}" fill="{rng.choice(cols)}"/>'
            )
    elif kind == "lines":
        s.append(f'<path d="M20 10V{h - 20}H{w - 10}" stroke="{ink}" fill="none" stroke-width="1.5"/>')
        for _ in range(rng.randint(1, 3)):
            pts = " ".join(f"{20 + i * (w - 30) / 12:.1f},{rng.uniform(15, h - 25):.1f}" for i in range(13))
            s.append(f'<polyline points="{pts}" fill="none" stroke="{rng.choice(cols)}" stroke-width="2"/>')
    elif kind == "blocks":
        for _ in range(rng.randint(5, 14)):
            x, y = rng.uniform(0, w * 0.8), rng.uniform(0, h * 0.8)
            s.append(
                f'<rect x="{x:.0f}" y="{y:.0f}" width="{rng.uniform(20, w * 0.4):.0f}" height="{rng.uniform(15, h * 0.4):.0f}" fill="{rng.choice(cols)}" opacity="{rng.uniform(0.3, 0.95):.2f}"/>'
            )
    elif kind == "gradient":
        s.append(
            f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{rng.choice(cols)}"/><stop offset="1" stop-color="#ddd"/></linearGradient></defs>'
        )
        s.append(f'<rect width="{w}" height="{h}" fill="url(#g)"/>')
        for _ in range(rng.randint(2, 6)):
            s.append(
                f'<circle cx="{rng.uniform(0, w):.0f}" cy="{rng.uniform(0, h):.0f}" r="{rng.uniform(8, h * 0.3):.0f}" fill="{rng.choice(cols)}" opacity="0.5"/>'
            )
    else:
        s.append(f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" fill="none" stroke="{ink}"/>')
        for _ in range(rng.randint(30, 120)):
            s.append(
                f'<circle cx="{rng.uniform(5, w - 5):.0f}" cy="{rng.uniform(5, h - 5):.0f}" r="{rng.uniform(1.5, 4):.1f}" fill="{rng.choice(cols)}"/>'
            )
    s.append("</svg>")
    return "".join(s)


def table_html(ctx: Ctx, ncols: int, nrows: int, st: dict, font_px: float) -> str:
    """A table with grid positions and spans on every cell. st: rules, zebra, band, pad, head_span, row_span."""
    rng = ctx.rng
    types = [rng.choice(["text", "text", "int", "money", "date", "code", "pct"]) for _ in range(ncols)]
    types[0] = rng.choice(["text", "text", "code"])
    loc = ctx.lang if ctx.lang in corpus.LOCALES else "en"
    grid: list[list[dict]] = []

    def val(t: str) -> str:
        if t == "text":
            return ctx.phrase(1, 3)
        if t == "int":
            return str(rng.randint(0, 9999))
        if t == "money":
            return corpus.money(rng.randint(100, 2_000_000), loc)
        if t == "date":
            return corpus.fmt_date(corpus.rand_date(rng), loc, 0)
        if t == "pct":
            return f"{rng.uniform(0, 100):.1f}%".replace(".", "," if loc != "en" else ".")
        return rng.choice("ABCDEFGHKLMNPRSTXZ") + str(rng.randint(100, 99999))

    r = 0
    if st["head_span"] and ncols >= 4:
        row, c = [], 0
        while c < ncols:
            cs = min(rng.randint(1, 3), ncols - c)
            row.append({"r": r, "c": c, "rs": 1, "cs": cs, "text": ctx.phrase(1, 2).capitalize(), "th": True, "al": "center"})
            c += cs
        grid.append(row)
        r += 1
    grid.append([{"r": r, "c": c, "rs": 1, "cs": 1, "text": ctx.phrase(1, 2).capitalize(), "th": True,
                  "al": "right" if types[c] in ("int", "money", "pct") else "left"} for c in range(ncols)])  # fmt: skip
    r + 1
    r += 1
    spans: dict[int, int] = {}  # rows still covered by a rowspan in column 0
    for k in range(nrows):
        row = []
        for c in range(ncols):
            if c == 0 and spans.get(0, 0) > 0:
                spans[0] -= 1
                continue
            rs = 1
            if c == 0 and st["row_span"] and nrows - k >= 2 and rng.random() < 0.25:
                rs = min(rng.randint(2, 3), nrows - k)
                spans[0] = rs - 1
            row.append({"r": r, "c": c, "rs": rs, "cs": 1, "text": val(types[c]), "th": False,
                        "al": "right" if types[c] in ("int", "money", "pct") else "left"})  # fmt: skip
        grid.append(row)
        r += 1
    # fix rowspans that run past the table
    for row in grid:
        for cell in row:
            cell["rs"] = min(cell["rs"], r - cell["r"])
    rules = st["rules"]
    ink = "currentColor"
    bd = {"grid": f"border:1px solid {ink}", "box": f"border:1px solid {ink}", "hlines": f"border-bottom:1px solid {ink}",
          "header": "", "none": ""}[rules]  # fmt: skip
    band = f"background:{ctx.look.accent};color:#fff;" if st["band"] else ""
    cls = f"t{ctx.next_kv()}"
    extra = ""
    if st["zebra"]:
        extra += f"table.{cls} tr:nth-child(even) td{{background:{ctx.look.ink}14}}"
    if rules == "hlines":
        extra += f"table.{cls} tr:last-child td{{border-bottom:1.5px solid {ink}}}"
    if rules in ("header", "hlines") and not st["band"]:
        extra += f"table.{cls} th{{border-bottom:1.5px solid {ink}}}"
    css = f"""<style>
table.{cls}{{border-collapse:collapse;font-size:{font_px:.2f}px;{('border:1.5px solid ' + ink + ';') if rules == 'box' else ''}{'width:100%;' if st['wide'] else ''}}}
table.{cls} td,table.{cls} th{{padding:{st['pad']:.1f}px {st['pad'] * 1.6:.1f}px;{bd};vertical-align:top;line-height:1.25}}
table.{cls} th{{font-weight:700;{band}}}
{extra}
</style>"""  # fmt: skip
    rows_html = []
    for row in grid:
        cells = []
        for c in row:
            tag = "th" if c["th"] else "td"
            span = (f' rowspan="{c["rs"]}"' if c["rs"] > 1 else "") + (f' colspan="{c["cs"]}"' if c["cs"] > 1 else "")
            cells.append(
                f'<{tag}{span} style="text-align:{c["al"]}" data-cell data-r="{c["r"]}" data-c="{c["c"]}" data-rs="{c["rs"]}" '
                f'data-cs="{c["cs"]}">{inline(c["text"])}</{tag}>'
            )
        rows_html.append(f"<tr data-row>{''.join(cells)}</tr>")
    ncols_total = ncols
    return (
        css + f'<table class="{cls}" data-blk="body" data-tbl data-ncols="{ncols_total}" data-nrows="{r}" data-reg="table" data-dl="Table">'
        f"{''.join(rows_html)}</table>"
    )


def draw_table_style(rng: random.Random) -> dict:
    return {
        "rules": rng.choice(["grid", "grid", "hlines", "header", "none", "box"]), "zebra": rng.random() < 0.3,
        "band": rng.random() < 0.3, "pad": rng.uniform(2, 7), "head_span": rng.random() < 0.25, "row_span": rng.random() < 0.2,
        "wide": rng.random() < 0.7,
    }  # fmt: skip


def prose_page(ctx: Ctx, kind: str = "prose") -> PageSpec:
    rng = ctx.rng
    ctx.lang = ctx.bank.pick_lang(rng)
    tpl = draw_template(ctx, kind)
    look = draw_look(ctx, tpl)
    ctx.look = look  # type: ignore[attr-defined]
    ind = " ind" if look.indent else ""
    rich = 0.04 if rng.random() < 0.5 else 0.0

    def p() -> str:
        return blk("body", "text", "Text", "para" + ind, inline(ctx.para(), rng, rich), "p")

    def heading(level: int) -> str:
        t = ctx.phrase(2, 7)
        t = t.capitalize() if rng.random() < 0.5 else t
        dl = "Title" if level == 1 else "Section-header"
        return blk("body", "title", dl, "h hr" if level < 3 else "h", inline(head_text(look, t)), f"h{level}")

    items: list[str] = []
    n_items = rng.randint(4, 40) if kind == "prose" else rng.randint(1, 4)
    if tpl["title"] and kind == "prose":
        pass  # the title is placed above the flow, spanning every column
    last_h = False
    for _ in range(n_items):
        roll = rng.random()
        if roll < 0.55 or (kind == "prose" and not items):
            items.append(p())
            last_h = False
        elif roll < 0.68 and not last_h:
            items.append(heading(2 if rng.random() < 0.7 else 3))
            last_h = True
        elif roll < 0.8:
            style = tpl["list_mark"]
            lis = []
            for i in range(rng.randint(2, 6)):
                lis.append(
                    blk(
                        "body",
                        "list",
                        "List-item",
                        "li",
                        f'<span class="mk">{_marker(style, i)}</span><span>{inline(ctx.para(6, 28), rng, 0.0)}</span>',
                    )
                )
            items.append("<div>" + "".join(lis) + "</div>")
            last_h = False
        elif roll < 0.9:
            w = int(min(look.width - look.margins[1] - look.margins[3], look.width) / look.cols * rng.uniform(0.5, 0.95))
            h = int(w * rng.uniform(0.35, 0.8))
            fig = f'<div class="fig" data-reg="figure" data-dl="Picture" data-box style="width:{w}px;height:{h}px">{figure_svg(rng, w, h, look.ink, look.accent)}</div>'
            cap = (
                blk(
                    "body",
                    "text",
                    "Caption",
                    "cap",
                    inline(f"{rng.choice(['Figure', 'Fig.'])} {rng.randint(1, 12)}. {ctx.phrase(3, 12)}"),
                    "p",
                )
                if rng.random() < 0.8
                else ""
            )
            items.append(f"<figure style='margin:0'>{fig}{cap}</figure>")
            last_h = False
        else:
            st = draw_table_style(rng)
            st["wide"] = True
            tb = table_html(ctx, rng.randint(2, 4), rng.randint(2, 6), st, look.px * 0.88)
            cap = (
                blk("body", "text", "Caption", "cap", inline(f"Table {rng.randint(1, 9)}. {ctx.phrase(2, 8)}"), "p")
                if rng.random() < 0.6
                else ""
            )
            items.append(f'<div style="margin:.4em 0 .8em">{cap}{tb}</div>')
            last_h = False
    flow = "".join(f"<div data-item>{it}</div>" if not it.startswith("<p") else it.replace("<p ", "<p data-item ", 1) for it in items)

    header = ""
    if tpl["title"] and kind == "prose":
        header += blk("body", "title", "Title", "h hr", inline(head_text(look, ctx.phrase(2, 8))), "h1")
    hd = ""
    rh = tpl["running_head"]
    if rh != "none":
        left, right = ctx.phrase(1, 4), ctx.phrase(1, 3)
        pieces = {"split": [left, right], "center": [left], "rule": [left]}[rh]
        spans = "".join(blk("header", "header", "Page-header", "", inline(t), "span") for t in pieces)
        hd = f'<div class="hd" style="{"justify-content:center;" if rh == "center" else ""}">{spans}</div>'
    ft = ""
    pn = tpl["page_no"]
    if pn != "none":
        num = str(rng.randint(1, 400))
        pieces = {"center": [num], "right": [num], "split": [ctx.phrase(1, 3), num]}[pn]
        just = {"center": "center", "right": "flex-end", "split": "space-between"}[pn]
        spans = "".join(blk("footer", "footer", "Page-footer", "", inline(t), "span") for t in pieces)
        ft = f'<div class="ft" style="justify-content:{just}">{spans}</div>'
    tail = ""
    if tpl["footnote"]:
        tail = blk("body", "text", "Footnote", "fn", inline(f"{rng.randint(1, 9)} {ctx.para(8, 30)}"))
    html = frame(look, ctx.root, "", hd + header, flow, tail, ft, ctx.lang)
    return PageSpec(html, kind, ctx.lang, tpl, sorted({look.body.name, look.head.name}), look.width, look.height, "trim")


def table_page(ctx: Ctx) -> PageSpec:
    rng = ctx.rng
    ctx.lang = ctx.bank.pick_lang(rng, set(corpus.LOCALES) | {"pt", "pl"})
    tpl = draw_template(ctx, "table", cols=1, title=rng.random() < 0.5)
    tpl.update({"cols": 1})
    look = draw_look(ctx, tpl)
    ctx.look = look  # type: ignore[attr-defined]
    items = []
    for _ in range(rng.randint(1, 3)):
        st = draw_table_style(rng)
        st["wide"] = rng.random() < 0.7
        ncols = rng.randint(2, 9 if look.width > 700 else 6)
        tb = table_html(ctx, ncols, rng.randint(3, 30), st, look.px * rng.uniform(0.8, 1.0))
        cap = blk("body", "text", "Caption", "cap", inline(f"Table {rng.randint(1, 9)}. {ctx.phrase(3, 10)}"), "p")
        top = rng.random() < 0.7
        txt = blk("body", "text", "Text", "para", inline(ctx.para(15, 45)), "p") if rng.random() < 0.5 else ""
        items.append(f'<div data-item style="margin-bottom:1.2em">{txt}{cap if top else ""}{tb}{"" if top else cap}</div>')
    header = blk("body", "title", "Title", "h hr", inline(head_text(look, ctx.phrase(2, 7))), "h1") if tpl["title"] else ""
    html = frame(look, ctx.root, "", "", "".join(items), "", "", ctx.lang).replace('<div id="flow">', f'{header}<div id="flow">', 1)
    return PageSpec(html, "table", ctx.lang, tpl, sorted({look.body.name, look.head.name}), look.width, look.height, "trim")


# --- forms ------------------------------------------------------------------------------------------------------------------


def _l(s: str) -> list[str]:
    return s.split("|")


LOCS = ("en", "de", "fr", "es", "it", "nl")
FIELDS = {  # key -> labels per locale, in LOCS order
    "first": _l("First name|Vorname|Prénom|Nombre|Nome|Voornaam"), "last": _l("Last name|Nachname|Nom|Apellidos|Cognome|Achternaam"),
    "dob": _l("Date of birth|Geburtsdatum|Date de naissance|Fecha de nacimiento|Data di nascita|Geboortedatum"),
    "street": _l("Street and number|Straße und Hausnummer|Adresse|Dirección|Indirizzo|Straat en huisnummer"),
    "zip": _l("Postcode|PLZ|Code postal|Código postal|CAP|Postcode"), "city": _l("City|Ort|Ville|Ciudad|Città|Plaats"),
    "phone": _l("Telephone|Telefon|Téléphone|Teléfono|Telefono|Telefoon"), "mail": _l("E-mail|E-Mail|E-mail|Correo electrónico|E-mail|E-mail"),
    "job": _l("Occupation|Beruf|Profession|Profesión|Professione|Beroep"), "date": _l("Date|Datum|Date|Fecha|Data|Datum"),
    "ref": _l("Reference no.|Aktenzeichen|Référence|Referencia|Riferimento|Referentie"),
    "note": _l("Comments|Bemerkungen|Remarques|Observaciones|Note|Opmerkingen"),
    "sign": _l("Signature|Unterschrift|Signature|Firma|Firma|Handtekening"),
}  # fmt: skip
QUESTIONS = [
    (_l("Do you agree to the terms?|Stimmen Sie den Bedingungen zu?|Acceptez-vous les conditions ?|¿Acepta las condiciones?|Accetta le condizioni?|Gaat u akkoord met de voorwaarden?"),
     [_l("Yes|Ja|Oui|Sí|Sì|Ja"), _l("No|Nein|Non|No|No|Nee")]),
    (_l("Title|Anrede|Civilité|Tratamiento|Titolo|Aanhef"),
     [_l("Mr|Herr|M.|Sr.|Sig.|Dhr."), _l("Ms|Frau|Mme|Sra.|Sig.ra|Mevr."), _l("Other|Divers|Autre|Otro|Altro|Anders")]),
    (_l("Payment|Zahlungsart|Paiement|Pago|Pagamento|Betaling"),
     [_l("Cash|Bar|Espèces|Efectivo|Contanti|Contant"), _l("Card|Karte|Carte|Tarjeta|Carta|Pinpas"),
      _l("Transfer|Überweisung|Virement|Transferencia|Bonifico|Overschrijving")]),
]  # fmt: skip
FORM_TITLE = _l("Application form|Antragsformular|Formulaire de demande|Formulario de solicitud|Modulo di richiesta|Aanvraagformulier")
SECTIONS = {
    "personal": _l("Personal details|Persönliche Angaben|Coordonnées|Datos personales|Dati personali|Persoonlijke gegevens"),
    "address": _l("Address|Anschrift|Adresse|Domicilio|Residenza|Adres"),
    "decl": _l("Declaration|Erklärung|Déclaration|Declaración|Dichiarazione|Verklaring"),
}  # fmt: skip

_CB_MARKS = [
    '<svg width="100%" height="100%" viewBox="0 0 10 10"><path d="M 1.5 1.5 L 8.5 8.5 M 8.5 1.5 L 1.5 8.5" stroke="currentColor" stroke-width="1.4"/></svg>',
    '<svg width="100%" height="100%" viewBox="0 0 10 10"><path d="M 1.5 5.5 L 4 8 L 8.5 1.5" stroke="currentColor" stroke-width="1.5" fill="none"/></svg>',
    '<svg width="100%" height="100%" viewBox="0 0 10 10"><rect x="2.2" y="2.2" width="5.6" height="5.6" fill="currentColor"/></svg>',
]


def form_page(ctx: Ctx) -> PageSpec:
    rng = ctx.rng
    li = rng.randrange(len(LOCS))
    loc = LOCS[li]
    ctx.lang = loc
    for _ in range(200):
        tpl = {
            "kind": "form", "cols": 1, "body_role": rng.choice(["sans", "sans", "serif"]), "head_same": rng.random() < 0.5,
            "justify": False, "indent": False, "col_rule": False, "head_caps": rng.random() < 0.3, "head_rule": rng.random() < 0.5,
            "head_align": rng.choice(["left", "center"]), "title": True, "running_head": "none", "page_no": rng.choice(["none", "center", "right"]),
            "list_mark": "bullet", "footnote": False,
            "field_style": rng.choice(["line", "box", "inline", "cell"]), "grid": rng.choice([1, 2, 2, 3]), "cb_style": rng.choice(["square", "round", "bracket"]),
        }  # fmt: skip
        if template_split(tpl) == ("train" if ctx.split == "train" else "val"):
            break
    look = draw_look(ctx, tpl)
    ctx.look = look  # type: ignore[attr-defined]
    fs = tpl["field_style"]
    ink = look.ink
    cb_radius = "50%" if tpl["cb_style"] == "round" else "0" if tpl["cb_style"] == "square" else "2px"
    pad = rng.uniform(3, 7)
    css = f"""
.sec{{margin:0 0 1.1em}}
.grid{{display:grid;grid-template-columns:repeat({tpl["grid"]},1fr);gap:{rng.uniform(0.5, 1.1):.2f}em {rng.uniform(1, 2):.2f}em}}
.f{{display:flex;flex-direction:{"column" if fs in ("line", "box", "cell") else "row"};gap:.15em;padding:{pad:.1f}px;
 {"border:1px solid " + ink + ";" if fs in ("box", "cell") else ""}{"align-items:baseline;gap:.5em;" if fs == "inline" else ""}}}
.f .lab{{font-size:{0.8 if fs != "inline" else 1}em;{"color:" + ink + "bb;" if fs != "inline" else ""}}}
.f .val{{{"border-bottom:1px solid " + ink + ";" if fs in ("line",) else ""}{"border-bottom:1px dotted " + ink + ";flex:1;" if fs == "inline" else ""}min-height:1.4em;font-weight:{rng.choice([400, 400, 500])}}}
.f.wide{{grid-column:1/-1}}
.q{{display:flex;flex-wrap:wrap;align-items:center;gap:.4em 1.2em;padding:{pad:.1f}px}}
.q .lab{{margin-right:.6em}}
.o{{display:inline-flex;align-items:center;gap:.4em}}
.cb{{display:inline-block;width:1em;height:1em;border:1.3px solid {ink};border-radius:{cb_radius};padding:1px;flex:none}}
"""
    secs = []

    def fld(key: str, wide=False, fill=0.85) -> str:
        label = FIELDS[key][li]
        empty = rng.random() > fill
        v = {
            "first": lambda: rng.choice(corpus._GIVEN),
            "last": lambda: rng.choice(corpus._FAMILY),
            "dob": lambda: corpus.fmt_date(corpus.rand_date(rng, 1940, 2006), loc, 0),
            "street": lambda: corpus.address(rng, loc)["street"],
            "zip": lambda: corpus.postcode(rng, loc),
            "city": lambda: corpus.city(rng, loc),
            "phone": lambda: corpus.phone(rng, loc),
            "mail": lambda: corpus.email(rng, corpus.person(rng)),
            "job": lambda: ctx.phrase(1, 2),
            "date": lambda: corpus.fmt_date(corpus.rand_date(rng, 2023, 2026), loc, 0),
            "ref": lambda: f"{rng.choice('ABCDEFGHKLMNPRSTXZ')}{rng.randint(1000, 999999)}",
            "note": lambda: ctx.phrase(5, 12),
            "sign": lambda: "",
        }[key]()
        if key in ("note", "sign"):
            wide = True
        if key == "sign":
            empty = True
        g = ctx.next_kv()
        valh = f'<span class="val" data-kv="{g}" data-kvr="value">{inline(v)}</span>' if v and not empty else '<span class="val"></span>'
        sty = ' style="min-height:3.2em"' if key == "sign" else ""
        return (
            f'<div class="f{" wide" if wide else ""}"{sty} data-blk="body" data-reg="formField" data-box>'
            f'<span class="lab" data-kv="{g}" data-kvr="key">{inline(label + (":" if fs == "inline" else ""))}</span>{valh}</div>'
        )

    def question(qi: int) -> str:
        qlabels, opts = QUESTIONS[qi]
        g = ctx.next_kv()
        chosen = rng.randrange(len(opts)) if rng.random() < 0.85 else -1
        parts = [f'<span class="lab" data-kv="{g}" data-kvr="key">{inline(qlabels[li])}</span>']
        mark = rng.choice(_CB_MARKS)
        for oi, o in enumerate(opts):
            on = oi == chosen
            parts.append(
                f'<span class="o" data-opt><i class="cb" data-reg="checkbox" data-box data-cb="{int(on)}">{mark if on else ""}</i>'
                f"<span{f' data-kv="{g}" data-kvr="value"' if on else ''}>{inline(o[li])}</span></span>"
            )
        return f'<div class="q" data-blk="body" data-reg="formField" data-box>{"".join(parts)}</div>'

    keys_pool = [
        ("personal", ["first", "last", "dob", "phone", "mail", "job"]),
        ("address", ["street", "zip", "city"]),
        ("decl", ["date", "ref", "note", "sign"]),
    ]
    for name, keys in keys_pool:
        if rng.random() < 0.15 and name != "personal":
            continue
        ks = [k for k in keys if rng.random() < 0.9] or keys[:2]
        body = "".join(fld(k) for k in ks)
        qs = "".join(question(rng.randrange(len(QUESTIONS))) for _ in range(rng.randint(0, 2)))
        head = blk("body", "title", "Section-header", "h hr", inline(head_text(look, SECTIONS[name][li])), "h2")
        secs.append(f'<div class="sec" data-item>{head}<div class="grid">{body}</div>{qs}</div>')
    title = blk("body", "title", "Title", "h hr", inline(head_text(look, FORM_TITLE[li])), "h1")
    pn = ""
    if tpl["page_no"] != "none":
        pn = (
            f'<div class="ft" style="justify-content:{"center" if tpl["page_no"] == "center" else "flex-end"}">'
            + blk("footer", "footer", "Page-footer", "", inline(str(rng.randint(1, 9))), "span")
            + "</div>"
        )
    html = frame(look, ctx.root, css, title, "".join(secs), "", pn, loc)
    return PageSpec(html, "form", loc, tpl, sorted({look.body.name, look.head.name}), look.width, look.height, "trim")
