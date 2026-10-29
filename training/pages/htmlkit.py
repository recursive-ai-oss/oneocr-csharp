"""HTML snippets and the page frame shared by every page kind.

Truth is carried by data attributes on the markup the generator itself writes; the browser's layout turns them into boxes
(pages/render.py reads them back). The vocabulary:

  data-blk="body|header|footer"   a text block; every word on the page must lie inside exactly one
  data-reg="<role>" data-dl="<DocLayNet class>"   a layout region (role: text title list table figure header footer formField checkbox)
  data-box                         the region box is the element's own rectangle, not the union of its words
  data-tbl, data-row, data-cell + data-r data-c data-rs data-cs   table, row and cell with its grid position and spans
  data-kv="<n>" data-kvr="key|value"   one side of key-value pair n
  data-fld="<id>" data-fname data-val  a document field (invoice / receipt) with its normalized value; data-li="<k>" on a line-item row
  data-cb="0|1"                    a checkbox (empty element), checked or not
  data-item                        a child of the flow container that may be dropped to make the page fit
"""

from __future__ import annotations

import html
import random
import re
from dataclasses import dataclass

from .fontpool import Family

_NOWRAP = re.compile(r"[-‐–—/]")


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def inline(text: str, rng: random.Random | None = None, rich: float = 0.0) -> str:
    """Escaped words; tokens that could wrap at a hyphen or slash are kept whole so a word never spans two lines.
    With `rich` > 0 a short run of words may be set in bold or italic."""
    toks = text.split()
    out = []
    i = 0
    while i < len(toks):
        t = toks[i]
        piece = f'<span class="nw">{esc(t)}</span>' if _NOWRAP.search(t) and len(t) > 1 else esc(t)
        if rng and rich and rng.random() < rich and i + 1 < len(toks):
            k = min(len(toks) - i, rng.randint(1, 3))
            tag = rng.choice(["b", "i"])
            run = " ".join(f'<span class="nw">{esc(w)}</span>' if _NOWRAP.search(w) and len(w) > 1 else esc(w) for w in toks[i : i + k])
            out.append(f"<{tag}>{run}</{tag}>")
            i += k
            continue
        out.append(piece)
        i += 1
    return " ".join(out)


@dataclass
class Look:
    body: Family
    head: Family
    mono: Family | None
    px: float
    lh: float
    justify: bool
    indent: bool
    ink: str
    paper: str
    accent: str
    margins: tuple[float, float, float, float]  # top right bottom left, css px
    cols: int
    gap: float
    col_rule: bool
    width: int
    height: int
    head_scale: float
    head_bold: bool
    head_caps: bool
    head_rule: bool
    head_align: str
    bars: str = ""  # css for light-on-dark bands (running head, foot and headings) on an otherwise light page


def font_faces(look: Look, root) -> str:
    fams = {look.body.name: look.body, look.head.name: look.head}
    if look.mono:
        fams[look.mono.name] = look.mono
    return "\n".join(f.css(root) for f in fams.values())


def page_css(look: Look, root, extra: str = "") -> str:
    m = look.margins
    hf = f"'{look.head.name}'"
    return f"""{font_faces(look, root)}
*{{box-sizing:border-box}}
html,body{{margin:0;padding:0;background:#fff}}
#page{{position:relative;width:{look.width}px;height:{look.height}px;background:{look.paper};color:{look.ink};
 padding:{m[0]}px {m[1]}px {m[2]}px {m[3]}px;display:flex;flex-direction:column;overflow:hidden;
 font-family:'{look.body.name}',serif;font-size:{look.px}px;line-height:{look.lh};font-kerning:normal}}
.nw{{white-space:nowrap}}
#flow{{flex:1 1 auto;min-height:0;overflow:hidden;column-count:{look.cols};column-gap:{look.gap}px;column-fill:auto;
 {"column-rule:1px solid " + look.ink + "66;" if look.col_rule else ""}}}
#flow>*{{break-inside:avoid}}
.para{{margin:0 0 {0 if look.indent else 0.6}em 0;text-align:{"justify" if look.justify else "left"};text-indent:0}}
.para.ind{{text-indent:1.6em}}
.h{{font-family:{hf},serif;margin:0.5em 0 0.35em;line-height:1.2;break-after:avoid;text-align:{look.head_align};
 font-weight:{700 if look.head_bold else 400};{"letter-spacing:.04em;" if look.head_caps else ""}}}
h1.h{{font-size:{look.head_scale * 1.5:.2f}em;margin:0 0 .5em}}
h2.h{{font-size:{look.head_scale:.2f}em}}
h3.h{{font-size:{max(1.0, look.head_scale * 0.82):.2f}em}}
.hr{{{"border-bottom:1.5px solid " + look.accent + ";padding-bottom:.15em;" if look.head_rule else ""}}}
.li{{display:flex;gap:.6em;margin:0 0 .25em 1.2em;text-align:left}}
.li .mk{{flex:none;min-width:1.2em;text-align:right}}
.fig{{margin:.3em 0 .2em;display:block}}
.cap{{font-size:.88em;margin:0 0 .8em;text-align:left;font-style:italic}}
.fn{{font-size:.8em;line-height:1.25;border-top:1px solid {look.ink}88;margin-top:.5em;padding-top:.3em;flex:none}}
.hd,.ft{{display:flex;justify-content:space-between;gap:1em;font-size:.8em;flex:none;color:{look.ink}cc}}
.hd{{border-bottom:1px solid {look.ink}66;padding-bottom:.35em;margin-bottom:.9em}}
.ft{{border-top:1px solid {look.ink}66;padding-top:.35em;margin-top:.7em}}
{look.bars}
{extra}"""


def frame(look: Look, root, css_extra: str, header: str, flow: str, tail: str, footer: str, lang: str) -> str:
    return (
        f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><style>{page_css(look, root, css_extra)}</style></head>'
        f'<body><div id="page">{header}<div id="flow">{flow}</div>{tail}{footer}</div></body></html>'
    )


def blk(role: str, reg: str | None, dl: str | None, cls: str, content: str, tag: str = "div", extra: str = "", box: bool = False) -> str:
    r = f' data-reg="{reg}"' if reg else ""
    d = f' data-dl="{dl}"' if dl else ""
    b = " data-box" if box else ""
    return f'<{tag} class="{cls}" data-blk="{role}"{r}{d}{b}{" " + extra if extra else ""}>{content}</{tag}>'


def head_text(look: Look, text: str) -> str:
    """Heading text as drawn; capitals are applied to the text itself because truth is the string the browser lays out."""
    return text.upper() if look.head_caps else text
