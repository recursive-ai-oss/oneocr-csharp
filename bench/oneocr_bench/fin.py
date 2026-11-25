"""fin: degraded financial documents with exact truth, rendered by Chromium and captured by the recipes of `findeg.py`.

Bank statements (running balances, carried-forward and brought-forward lines, several pages), invoices and receipts, in many
fonts and table styles. Money is written in US and European conventions with currency symbols or codes, negatives as minus,
parentheses, trailing minus or CR/DR. Truth is the browser's own layout (DOM ranges, see render.py) plus tags written when the
HTML is built:

  tags       every money value, date and account-like number (`cls`), the table quantity cells (`qty`), with their words;
  statement  for statements, the page's start balance and every row's amount, balance and sign, so a scorer can check
             previous balance +/- amount = balance on what an engine read.

Every statement the generator writes reconciles in exact decimal arithmetic (checked when the page is built). The pages are
captured as scan, fax or phone (see `findeg.py`); the truth boxes move with the pixels. Fonts are held out: the seven eval
families of `assets.py` and three more that no training pool uses (`fin_assets.py`).

    python -m oneocr_bench fin gen --set fin-scan --seeds fin-dev-seeds.json --out DIR [--count N]
    python -m oneocr_bench fin check DIR        # truth passes the pixel, tag and reconciliation checks
    python -m oneocr_bench fin draw DIR pID     # truth boxes on a page (tags in red)
"""

from __future__ import annotations

import html
import io
import json
import random
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

from . import findeg
from .fin_assets import ALL as FONTS
from .fin_assets import MONO, font_css, wait_fonts
from .gen import H_CSS, H_PX, SCALE, W_CSS, W_PX
from .render import FRAGS_JS, add_ink, sha256, text_of

CENT = Decimal("0.01")
FAMILIES = list(FONTS)
SET_KINDS = {
    "fin-scan": ("scan",), "fin-fax": ("fax-std", "fax-fine"), "fin-phone": ("phone",),
    "fin-tables": ("scan", "fax-fine", "phone", "fax-std"),
}
# document mix per 40 pages: statements, invoices, receipts (the tables cut has no plain receipts, only long ones)
MIX = {"fin-scan": (16, 12, 12), "fin-fax": (16, 12, 12), "fin-phone": (16, 12, 12), "fin-tables": (18, 14, 8)}

BANKS = ["Harbor Savings Bank", "Northfield Credit Union", "Maple Ridge Bank", "Summit Trust", "Lakeshore Federal Savings",
         "Cedar Valley Bank", "Prairie Community Bank", "Granite State Credit Union", "Bluewater Bank", "Oakline Financial",
         "Rheinblick Sparkasse", "Banque du Lac", "Costa Verde Banco"]
MERCH = ["GROCERY MART", "FUEL STOP", "CITY POWER & LIGHT", "WATER UTILITY", "PHARMACY PLUS", "HARDWARE DEPOT", "COFFEE CORNER",
         "BOOK NOOK", "AIR TRAVEL CO", "INSURANCE PREMIUM", "RENT PAYMENT", "ONLINE MARKETPLACE", "DENTAL CARE", "TRANSIT PASS",
         "STREAMING SERVICE", "GYM MEMBERSHIP", "TELECOM MOBILE", "HOME INTERNET", "PET SUPPLIES", "GARDEN CENTRE"]
CREDITS = ["PAYROLL DEPOSIT", "ACH CREDIT REFUND", "TRANSFER FROM SAVINGS", "INTEREST PAID", "MOBILE CHECK DEPOSIT", "WIRE IN",
           "TAX REFUND", "CASH DEPOSIT"]
DEBITS = ["CHECK", "ATM WITHDRAWAL", "TRANSFER TO SAVINGS", "SERVICE FEE", "ONLINE PAYMENT", "CARD PURCHASE", "DIRECT DEBIT"]
VENDORS = ["Alder & Finch Supplies Ltd", "Brightwave Logistics", "Copperline Engineering GmbH", "Dunmore Office Products",
           "Evergreen Print Works", "Fairhaven Catering", "Greystone Consulting", "Hollis Hardware Wholesale", "Ironbridge Tools",
           "Juniper Packaging Co"]
ITEMS = ["Consulting services", "Cable assembly 2m", "Office chair, ergonomic", "Toner cartridge black", "Courier delivery",
         "Annual licence renewal", "Onsite training day", "Packaging materials", "Maintenance contract Q3", "Safety gloves (box)",
         "Steel shelving unit", "Printer paper A4", "Network switch 24-port", "Inspection fee", "Laminated signs", "Hand cart",
         "Calibration service", "Replacement filters", "Freight surcharge", "Software support"]
STORES = ["QUICKMART", "GREEN GROCER", "CORNER CAFE", "HOME & GARDEN", "TECH HUB", "FUEL EXPRESS", "BAKE HOUSE", "PET PLANET",
          "RIVERSIDE PHARMACY", "TOOL BARN"]
GOODS = ["MILK 2%", "BREAD WHOLE", "APPLES GALA", "COFFEE BEANS 12OZ", "CHICKEN BREAST", "EGGS LARGE 12", "PASTA PENNE",
         "TOMATOES VINE", "CEREAL OATS", "PAPER TOWELS", "BATTERIES AA 8PK", "DISH SOAP", "ORANGE JUICE", "BUTTER SALTED",
         "RICE LONG GRAIN", "YOGURT PLAIN", "BANANAS", "CHEESE CHEDDAR", "NOTEBOOK A5", "LIGHT BULB LED"]
FIRST = ["Maria", "Jonas", "Aisha", "Tomas", "Chen", "Olga", "Priya", "Lucas", "Fatima", "Dmitri", "Elena", "Samuel", "Ingrid"]
LAST = ["Alvarez", "Berg", "Okafor", "Novak", "Wei", "Petrova", "Nair", "Silva", "Haddad", "Ivanov", "Costa", "Osei", "Lund"]
STREETS = ["Elm Street", "Harbour Road", "Mill Lane", "Station Avenue", "Orchard Way", "Market Square", "Birch Close"]
CITIES = ["Lisbon", "Oslo", "Denver", "Valencia", "Rotterdam", "Toronto", "Leeds", "Austin", "Lyon", "Graz"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ------------------------------------------------------------------------------------------------ formatting

@dataclass
class Fmt:
    locale: str  # us | eu | ch
    cur: str | None  # currency symbol or code printed with amounts, or None
    cur_at: str  # pre | post | word-pre | word-post (a separate word)
    neg: str  # minus | paren | trail | crdr
    dates: str  # mdy | dmy | iso | mon
    year: bool


def make_fmt(rng: random.Random, locale: str | None = None) -> Fmt:
    locale = locale or rng.choices(["us", "eu", "ch"], [6, 3, 1])[0]
    if locale == "us":
        cur = rng.choice(["$", "$", "USD", None])
        dates = rng.choice(["mdy", "mdy", "iso", "mon"])
    elif locale == "eu":
        cur = rng.choice(["€", "EUR", None, "€"])
        dates = rng.choice(["dmy", "dmy", "iso"])
    else:
        cur = rng.choice(["CHF", None])
        dates = rng.choice(["dmy", "iso"])
    at = rng.choice(["pre", "word-pre"] if cur in ("$", "€") and locale == "us" else ["post", "word-post", "word-pre"]) if cur else "pre"
    if cur in ("USD", "EUR", "CHF") and at == "pre":
        at = "word-pre"
    return Fmt(locale, cur, at, rng.choice(["minus", "paren", "trail", "crdr"]), dates, rng.random() < 0.5)


def money(v: Decimal, f: Fmt, cur: bool = True, sign: bool = True) -> str:
    """The amount as printed. `sign` False prints the magnitude (a debit column); a negative value follows `f.neg`."""
    s = f"{abs(v):,.2f}"
    if f.locale == "eu":
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    elif f.locale == "ch":
        s = s.replace(",", "'")
    if cur and f.cur:
        s = {"pre": f.cur + s, "post": s + f.cur, "word-pre": f"{f.cur} {s}", "word-post": f"{s} {f.cur}"}[f.cur_at]
    if not sign:
        return s
    if v < 0:
        return {"minus": f"-{s}", "paren": f"({s})", "trail": f"{s}-", "crdr": f"{s} DR"}[f.neg]
    return f"{s} CR" if f.neg == "crdr" else s


def fdate(day: tuple[int, int, int], f: Fmt, short: bool = False) -> str:
    y, m, d = day
    keep_year = f.year and not short
    if f.dates == "mdy":
        return f"{m:02d}/{d:02d}/{y}" if keep_year else f"{m:02d}/{d:02d}"
    if f.dates == "dmy":
        return f"{d:02d}.{m:02d}.{y}" if keep_year else f"{d:02d}.{m:02d}."
    if f.dates == "iso":
        return f"{y}-{m:02d}-{d:02d}"
    return f"{MONTHS[m - 1]} {d:02d}, {y}" if keep_year and f.locale == "us" else f"{d:02d} {MONTHS[m - 1]}" if not keep_year else f"{d:02d} {MONTHS[m - 1]} {y}"


def cents(rng: random.Random, lo: float = 2.5, hi: float = 4800.0) -> Decimal:
    v = Decimal(str(round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 2)))
    if rng.random() < 0.08:
        v = Decimal(rng.choice([20, 40, 50, 100, 150, 200, 250, 500, 1000]))
    return v.quantize(CENT)


class Reg:
    """Tags of one page: `tag()` wraps the text in a span the DOM walk will find, and remembers what it means."""

    def __init__(self) -> None:
        self.n = 0
        self.meta: dict[str, dict] = {}

    def tag(self, text: str, cls: str, role: str, value: str | None = None) -> str:
        self.n += 1
        g = f"g{self.n}"
        self.meta[g] = {"cls": cls, "role": role, "value": value}
        return f'<span data-g="{g}">{html.escape(text)}</span>'


def e(t: str) -> str:
    return html.escape(t)


def person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def address(rng: random.Random) -> list[str]:
    return [f"{rng.randint(1, 240)} {rng.choice(STREETS)}", f"{rng.choice(CITIES)} {rng.randint(10000, 99999)}"]


def rand_date(rng: random.Random, y: int | None = None, m: int | None = None) -> tuple[int, int, int]:
    return (y or rng.randint(2022, 2026), m or rng.randint(1, 12), rng.randint(1, 28))


def account_no(rng: random.Random, f: Fmt) -> tuple[str, str]:
    """(printed text, kind): an account number, or an IBAN grouped in fours for European documents."""
    if f.locale in ("eu", "ch") and rng.random() < 0.7:
        cc = "DE" if f.locale == "eu" and rng.random() < 0.5 else ("CH" if f.locale == "ch" else rng.choice(["FR", "NL", "ES", "AT"]))
        digits = "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([18, 20])))
        raw = f"{cc}{rng.randint(10, 98)}{digits}"
        return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4)), "iban"
    return rng.choice(["", "", "", "0"]) + "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([8, 10, 12]))), "acct"


# --------------------------------------------------------------------------------------------- page skeletons

def css(family: str, fs_pt: float, style: str, accent: str, line_h: float) -> str:
    fs = fs_pt * 96 / 72
    border = "1px solid #666" if style == "ruled" else "0"
    row = {"rows": "border-bottom:1px solid #888;", "leaders": "border-bottom:1.5px dotted #666;"}.get(style, "")
    th = "background:#e6e6e6;" if style == "ruled" else "border-bottom:2px solid #222;"
    if style == "zebra":
        th = f"background:{accent};color:#fff;"
    zebra = "tr.z:nth-child(even) td{background:#eef0f3}" if style == "zebra" else ""
    return (f"{font_css(family)}html,body{{margin:0;background:#fff}}"
            f"#page{{width:{W_CSS}px;min-height:{H_CSS}px;box-sizing:border-box;padding:46px 52px 44px;color:#111;font-family:'{family}';"
            f"font-size:{fs:.2f}px;line-height:{line_h}}}"
            f"table{{border-collapse:collapse;width:100%}}td,th{{padding:{.14 + (line_h - 1.15) * .2:.2f}em .45em;border:{border};vertical-align:top;{row}}}"
            f"th{{font-weight:700;text-align:left;{th}}}td.n,th.n{{text-align:right;white-space:nowrap}}{zebra}"
            f".bank{{font-size:1.9em;font-weight:700;color:{accent};line-height:1.1}}.band{{background:{accent};color:#fff;padding:.5em .7em}}"
            f".kv{{display:flex;justify-content:space-between;gap:1em}}.sm{{font-size:.85em;color:#333}}.b{{font-weight:700}}"
            f".box{{border:1.5px solid #333;padding:.4em .6em}}.tot td{{font-weight:700;border-top:2px solid #222}}")


def doc_html(css_text: str, body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css_text}</style></head><body><div id='page'><div id='content'>{body}</div></div></body></html>"


FILL_JS = """([html, sel, limit]) => {
  const c = document.querySelector(sel); c.insertAdjacentHTML('beforeend', html);
  const b = document.getElementById('content').getBoundingClientRect().bottom;
  if (b > limit) { c.lastElementChild.remove(); return false; }
  return true;
}"""

WORDS_FIN_JS = "(rootSel) => {" + FRAGS_JS + """
  const root = document.querySelector(rootSel);
  const out = []; const ids = new Map();
  const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let nd = tw.nextNode(); nd; nd = tw.nextNode()) {
    const el = nd.parentElement;
    if (getComputedStyle(el).visibility === 'hidden') continue;
    if (!ids.has(el)) ids.set(el, ids.size);
    const tg = el.closest('[data-g]');
    const re = /\\S+/g; let m;
    while ((m = re.exec(nd.data))) {
      for (const f of frags(nd, m.index, m.index + m[0].length)) out.push({...f, b: ids.get(el), g: tg ? tg.dataset.g : ''});
    }
  }
  return out;
}"""


@dataclass
class Page:
    png: bytes
    raw: list
    reg: Reg
    width: int  # CSS px of the page element
    statement: dict | None
    info: dict


# ------------------------------------------------------------------------------------------------ statements

class Statement:
    """One statement document: rows come from `next_row` in order, balances run on, pages are filled by the renderer."""

    def __init__(self, rng: random.Random, dense: bool, pages: int) -> None:
        self.rng = rng
        self.span = rng.randint(28, 46) * pages * (2 if dense else 1)  # rows the statement is expected to hold, to spread its dates over the month
        self.count = 0
        self.f = make_fmt(rng)
        self.dense = dense
        self.bank = rng.choice(BANKS if self.f.locale == "us" else BANKS[7:] + BANKS[:3])
        self.holder, self.addr = person(rng), address(rng)
        self.acct, self.acct_kind = account_no(rng, self.f)
        self.mode = rng.choice(["split", "signed", "inout", "crdr"] if not dense else ["split", "inout", "split"])
        if self.mode == "crdr":
            self.f.neg = "crdr"
        self.show_cur = rng.random() < 0.35
        self.y, self.m = rng.randint(2023, 2026), rng.randint(1, 12)
        self.day = 1
        self.balance = cents(rng, 120, 18000)
        self.opening = self.balance
        self.band = rng.random() < 0.5
        self.refs = rng.random() < 0.6 and self.mode in ("split", "crdr")
        self.cased = rng.random() < 0.5
        self.n_checks = rng.randint(1000, 4000)
        self.totals = {"credit": Decimal(0), "debit": Decimal(0)}  # of the rows placed so far

    def next_row(self) -> dict:
        rng = self.rng
        self.count += 1
        self.day = max(self.day, min(28, 1 + self.count * 27 // self.span + rng.choice([0, 0, 1])))
        credit = rng.random() < 0.3
        amt = cents(rng, 30, 6500) if credit else cents(rng, 2.5, 1900)
        if not credit and rng.random() < 0.05:  # overdraft: the balance goes negative
            amt = max(self.balance, Decimal(0)) + cents(rng, 5, 300)
        if credit:
            desc = rng.choice(CREDITS)
        else:
            kind = rng.choice(DEBITS + MERCH * 3)
            desc = f"CHECK {self.n_checks}" if kind == "CHECK" else kind
            if kind == "CHECK":
                self.n_checks += rng.randint(1, 3)
            elif kind in MERCH or kind.startswith(("CARD", "DIRECT")):
                desc = f"{kind} #{rng.randint(100, 9999)}" + (f" {rng.choice(CITIES).upper()}" if rng.random() < 0.4 else "")
        if not self.cased:
            desc = desc.title()
        ref = "".join(rng.choice("0123456789") for _ in range(rng.choice([6, 8, 9])))
        if rng.random() < 0.3:
            ref = rng.choice(["TXN", "REF", "ACH"]) + ref
        signed = amt if credit else -amt
        self.balance += signed
        return {"date": (self.y, self.m, self.day), "desc": desc, "ref": ref, "amount": signed, "balance": self.balance}

    def head(self, first: bool, page_no: int, pages: int, reg: Reg) -> str:
        f = self.f
        title = f"<div class='bank'>{e(self.bank)}</div>"
        if self.band:
            title = f"<div class='band'><div class='bank' style='color:#fff'>{e(self.bank)}</div><div class='sm' style='color:#fff'>Statement of account</div></div>"
        pg = f"<div class='sm'>Page {page_no} of {pages}</div>"
        lab = "IBAN" if self.acct_kind == "iban" else "Account number"
        acct = f"{e(lab)} {reg.tag(self.acct, 'account', 'account')}"
        if not first:
            return f"<div class='kv' style='margin-bottom:.8em'><div>{title}</div><div style='text-align:right'>{pg}<div>{acct}</div></div></div>"
        d0, d1 = (self.y, self.m, 1), (self.y, self.m, 28)
        period = f"{reg.tag(fdate(d0, f), 'date', 'period')} - {reg.tag(fdate(d1, f), 'date', 'period')}"
        return (f"{title}<div class='kv' style='margin:.9em 0 .7em'><div><div class='b'>{e(self.holder)}</div>"
                + "".join(f"<div>{e(a)}</div>" for a in self.addr)
                + f"</div><div style='text-align:right'><div>{acct}</div><div>Statement period {period}</div>{pg}</div></div>")

    def columns(self) -> list[tuple[str, str]]:
        ref = [("Reference", "d")] if self.refs else []
        return {"split": [("Date", "d"), ("Description", "d"), *ref, ("Debit", "n"), ("Credit", "n"), ("Balance", "n")],
                "signed": [("Date", "d"), ("Description", "d"), ("Amount", "n"), ("Balance", "n")],
                "inout": [("Date", "d"), ("Details", "d"), ("Withdrawals", "n"), ("Deposits", "n"), ("Balance", "n")],
                "crdr": [("Date", "d"), ("Description", "d"), ("Reference", "d"), ("Amount", "n"), ("Balance", "n")]}[self.mode]

    def thead(self) -> str:
        return "<thead><tr>" + "".join(f"<th class='{c}'>{h}</th>" for h, c in self.columns()) + "</tr></thead>"

    def row_html(self, row: dict, reg: Reg, z: bool) -> tuple[str, dict]:
        """(<tr> html, {amount, balance}: the tag spans of the row), for the statement structure."""
        f, cs = self.f, self.show_cur
        cells, info = [], {}
        for h, c in self.columns():
            if h == "Date":
                cells.append(f"<td class='d'>{reg.tag(fdate(row['date'], f, short=True), 'date', 'txn')}</td>")
            elif h in ("Description", "Details"):
                cells.append(f"<td class='d'>{e(row['desc'])}</td>")
            elif h == "Reference":
                cells.append(f"<td class='d'>{reg.tag(row['ref'], 'account', 'ref')}</td>")
            elif h == "Balance":
                t = reg.tag(money(row["balance"], f, cs), "money", "balance", str(row["balance"]))
                info["balance"] = t
                cells.append(f"<td class='n'>{t}</td>")
            else:
                v = row["amount"]
                here = h == "Amount" or (h in ("Debit", "Withdrawals") and v < 0) or (h in ("Credit", "Deposits") and v > 0)
                if here:
                    t = reg.tag(money(v, f, cs, sign=h == "Amount"), "money", "amount", str(v))
                    info["amount"] = t
                    cells.append(f"<td class='n'>{t}</td>")
                else:
                    cells.append("<td class='n'></td>")
        return f"<tr class='z'>" + "".join(cells) + "</tr>", info

    def special_row(self, label: str, bal: Decimal, reg: Reg, role: str) -> tuple[str, str]:
        cols = self.columns()
        nnum = sum(1 for _, c in cols if c == "n")
        t = reg.tag(money(bal, self.f, self.show_cur), "money", role, str(bal))
        return (f"<tr class='z'><td class='d'></td><td class='d b' colspan='{len(cols) - 1 - nnum}'>{e(label)}</td>"
                + "<td class='n'></td>" * (nnum - 1) + f"<td class='n b'>{t}</td></tr>"), t

    def summary_html(self, reg: Reg | None, closing: Decimal) -> str:
        f = self.f
        vals = [("Total debits", self.totals["debit"], "total"), ("Total credits", self.totals["credit"], "total"), ("Closing balance", closing, "closing")]

        def cell(v, role):
            s = money(v, f, True, sign=role == "closing")
            return reg.tag(s, "money", role, str(v)) if reg else e(s)
        return ("<table class='tot' style='width:56%;margin:1em 0 0 auto'><tbody>"
                + "".join(f"<tr><td class='d'>{lab}</td><td class='n'>{cell(v, role)}</td></tr>" for lab, v, role in vals) + "</tbody></table>"
                + "<div class='sm' style='margin-top:1em'>Please examine this statement and report any difference within 30 days.</div>")


_SHOW = Path(tempfile.mkdtemp(prefix="fin-")) / "page.html"


def show(page, doc: str) -> None:
    """Load a document from a file: a page made with set_content has no file origin and cannot load the fonts from disk."""
    _SHOW.write_text(doc, encoding="utf-8")
    page.goto(_SHOW.as_uri())


def snap(page, width_css: int) -> tuple[bytes, list[dict]]:
    """Screenshot of the page element (an A4 sheet, or a receipt cut to its content) and the words found in the DOM."""
    h = H_CSS
    if width_css != W_CSS:
        h = min(H_CSS, int(page.evaluate("document.getElementById('page').getBoundingClientRect().height")) + 1)
    png = page.screenshot(clip={"x": 0, "y": 0, "width": width_css, "height": h})
    return png, page.evaluate(WORDS_FIN_JS, "#page")


def render_statement(page, doc: Statement, pages: int, rng: random.Random, family: str, fs_pt: float, style: str, accent: str) -> list[Page]:
    out: list[Page] = []
    queue: list[dict] = []  # a row that did not fit starts the next page
    bf: Decimal | None = None
    limit = H_CSS - 44
    for k in range(pages):
        reg = Reg()
        last = k == pages - 1
        f = doc.f
        body = doc.head(k == 0, k + 1, pages, reg) + f"<table id='t'>{doc.thead()}<tbody id='tb'></tbody></table><div id='after'></div>"
        show(page, doc_html(css(family, fs_pt, style, accent, 1.18 if doc.dense else 1.3), body))
        wait_fonts(page, family)
        label, role = ("Opening balance", "opening") if k == 0 else ("Balance brought forward", "bf")
        start_bal = doc.opening if k == 0 else bf
        sp, start_t = doc.special_row(label, start_bal, reg, role)
        assert page.evaluate(FILL_JS, [sp, "#tb", limit])
        if last:  # the closing summary goes in first with dummy numbers, so the rows leave room for it
            page.evaluate("([h]) => { document.getElementById('after').innerHTML = h; }", [doc.summary_html(None, Decimal("9999999.99"))])
        placed: list[tuple[dict, dict]] = []
        target = rng.uniform(0.45, 1.0) if last else None
        reserve = 0 if last else 3.0 * fs_pt * 96 / 72 * 1.4  # room for the carried-forward line
        while True:
            row = queue.pop(0) if queue else doc.next_row()
            hh, info = doc.row_html(row, reg, False)
            if not page.evaluate(FILL_JS, [hh, "#tb", limit - reserve]):
                queue.insert(0, row)
                break
            placed.append((row, info))
            if target is not None and len(placed) >= 3 and page.evaluate("document.getElementById('content').getBoundingClientRect().bottom") / limit >= target:
                break
        for row, _ in placed:
            doc.totals["credit" if row["amount"] > 0 else "debit"] += abs(row["amount"])
        if not last:
            bf = placed[-1][0]["balance"] if placed else start_bal
            sp2, _ = doc.special_row("Carried forward", bf, reg, "cf")
            assert page.evaluate(FILL_JS, [sp2, "#tb", limit])
        else:
            closing = placed[-1][0]["balance"] if placed else start_bal
            page.evaluate("([h]) => { document.getElementById('after').innerHTML = h; }", [doc.summary_html(reg, closing)])
            if page.evaluate("document.getElementById('content').getBoundingClientRect().bottom") > limit + 1:
                raise RuntimeError("the closing summary overflows the page")
        prev = start_bal
        for row, _ in placed:  # the statement's arithmetic, exactly
            assert prev + row["amount"] == row["balance"]
            prev = row["balance"]
        gs = lambda t: t.split('data-g="')[1].split('"')[0]  # noqa: E731
        stmt = {"mode": doc.mode, "start": {"g": gs(start_t), "value": str(start_bal)},
                "rows": [{"amount": {"g": gs(i["amount"]), "value": str(r["amount"])}, "balance": {"g": gs(i["balance"]), "value": str(r["balance"])},
                          "sign": 1 if r["amount"] > 0 else -1} for r, i in placed]}
        png, raw = snap(page, W_CSS)
        out.append(Page(png, raw, reg, W_CSS, stmt, {"doc": "statement", "page": k + 1, "pages": pages, "font": family, "font_pt": fs_pt, "style": style,
                                                      "mode": doc.mode, "locale": f.locale, "neg": f.neg, "currency": f.cur}))
    return out


# -------------------------------------------------------------------------------------------------- invoices

def invoice_body(rng: random.Random, nitems: int, dense: bool, reg: Reg, f: Fmt, cur: bool) -> str:
    vendor, buyer = rng.choice(VENDORS), person(rng)
    inv_no = f"{rng.choice(['INV', 'RE', 'FA'])}-{rng.randint(2022, 2026)}-{rng.randint(100, 99999):05d}"
    issue = rand_date(rng)
    due = (issue[0], min(12, issue[1] + 1), issue[2])
    acct, kind = account_no(rng, f)
    tax_rate = Decimal(rng.choice(["0", "5", "7", "7.7", "8.25", "19", "20", "21"]))
    qty_pool = [1, 1, 2, 3, 4, 5, 6, 8, 9, 12, 24, 50] if not dense else [1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 1, 12]
    rows, sub = [], Decimal(0)
    for _ in range(nitems):
        q = rng.choice(qty_pool)
        unit = cents(rng, 1.2, 950)
        amt = (unit * q).quantize(CENT)
        sub += amt
        rows.append((rng.choice(ITEMS), q, unit, amt))
    tax = (sub * tax_rate / 100).quantize(CENT, ROUND_HALF_EVEN)
    total = sub + tax
    show_tax_col = rng.random() < 0.4
    cols = ["Pos", "Description", "Qty", "Unit price"] + (["Tax %"] if show_tax_col else []) + ["Amount"]
    head = "".join(f"<th class='{'d' if c == 'Description' else 'n'}'>{c}</th>" for c in cols)
    trs = []
    for i, (desc, q, unit, amt) in enumerate(rows):
        tds = [f"<td class='n'>{i + 1}</td>", f"<td class='d'>{e(desc)}</td>", f"<td class='n'>{reg.tag(str(q), 'qty', 'qty', str(q))}</td>",
               f"<td class='n'>{reg.tag(money(unit, f, cur, sign=False), 'money', 'unit', str(unit))}</td>"]
        if show_tax_col:
            tds.append(f"<td class='n'>{tax_rate}%</td>")
        tds.append(f"<td class='n'>{reg.tag(money(amt, f, cur, sign=False), 'money', 'amount', str(amt))}</td>")
        trs.append(f"<tr class='z'>{''.join(tds)}</tr>")
    span = len(cols) - 1
    foot = [("Subtotal", sub, "subtotal"), (f"Tax {tax_rate}%", tax, "tax"), ("Total due" if rng.random() < .5 else "Amount due", total, "total")]
    foot_html = "".join(f"<tr class='{'tot' if r == 'total' else ''}'><td class='n b' colspan='{span}'>{e(lab)}</td>"
                        f"<td class='n'>{reg.tag(money(v, f, True, sign=False), 'money', r, str(v))}</td></tr>" for lab, v, r in foot)
    lab = "IBAN" if kind == "iban" else "Account no."
    bic = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(8))
    routing = reg.tag("".join(str(rng.randint(0, 9)) for _ in range(9)), "account", "routing")
    pay = (f"<div class='box' style='margin-top:1em'><div class='b'>Payment details</div><div>Bank {e(vendor.split()[0])} Bank</div>"
           f"<div>{lab} {reg.tag(acct, 'account', 'account')}</div>" + (f"<div>BIC {e(bic)}</div>" if kind == "iban" else f"<div>Routing {routing}</div>") + "</div>")
    a1, a2 = address(rng), address(rng)
    head_block = (f"<div class='kv'><div><div class='bank'>{e(vendor)}</div><div>{e(a1[0])}</div><div>{e(a1[1])}</div></div>"
                  f"<div style='text-align:right'><div class='b' style='font-size:1.6em'>INVOICE</div><div>Invoice no. {reg.tag(inv_no, 'account', 'invoice')}</div>"
                  f"<div>Date {reg.tag(fdate(issue, f), 'date', 'issue')}</div><div>Due {reg.tag(fdate(due, f), 'date', 'due')}</div>"
                  f"<div>Order {reg.tag(str(rng.randint(10000, 999999)), 'account', 'po')}</div></div></div>"
                  f"<div style='margin:1em 0'><div class='b'>Bill to</div><div>{e(buyer)}</div><div>{e(a2[0])}</div></div>")
    return head_block + f"<table id='t'><thead><tr>{head}</tr></thead><tbody id='tb'>{''.join(trs)}</tbody><tfoot>{foot_html}</tfoot></table>" + pay


def receipt_body(rng: random.Random, nitems: int, reg: Reg, f: Fmt) -> str:
    store = rng.choice(STORES)
    lines, sub = [], Decimal(0)
    for _ in range(nitems):
        q = rng.choice([1, 1, 1, 2, 3, 4])
        unit = cents(rng, 0.6, 38)
        amt = (unit * q).quantize(CENT)
        sub += amt
        name = rng.choice(GOODS)
        if q > 1:
            lines.append(f"<div class='r'><span>{e(name)}</span><span></span></div><div class='r sm'><span>&nbsp;&nbsp;{reg.tag(str(q), 'qty', 'qty', str(q))} @ "
                         f"{reg.tag(money(unit, f, False, False), 'money', 'unit', str(unit))}</span>"
                         f"<span>{reg.tag(money(amt, f, False, False), 'money', 'amount', str(amt))}</span></div>")
        else:
            lines.append(f"<div class='r'><span>{e(name)}</span><span>{reg.tag(money(amt, f, False, False), 'money', 'amount', str(amt))}</span></div>")
    rate = Decimal(rng.choice(["0", "6", "7", "8.25", "19"]))
    tax = (sub * rate / 100).quantize(CENT, ROUND_HALF_EVEN)
    total = sub + tax
    card_pay = rng.random() < 0.6
    paid = total if card_pay else (total // 10 * 10 + 10 if total < 190 else total + Decimal(20))
    d = rand_date(rng)
    stamp = f"{rng.randint(7, 21):02d}:{rng.randint(0, 59):02d}"
    if card_pay:
        tender = (f"<div class='r'><span>VISA ****{reg.tag(''.join(str(rng.randint(0, 9)) for _ in range(4)), 'account', 'card')}</span>"
                  f"<span>{reg.tag(money(paid, f, False, False), 'money', 'tender', str(paid))}</span></div>"
                  f"<div class='r'><span>AUTH {reg.tag(str(rng.randint(100000, 999999)), 'account', 'auth')}</span><span></span></div>")
    else:
        tender = (f"<div class='r'><span>CASH</span><span>{reg.tag(money(paid, f, False, False), 'money', 'tender', str(paid))}</span></div>"
                  f"<div class='r'><span>CHANGE</span><span>{reg.tag(money(paid - total, f, False, False), 'money', 'change', str(paid - total))}</span></div>")
    a = address(rng)
    return (f"<div class='c b' style='font-size:1.25em'>{e(store)}</div><div class='c'>{e(a[0])}</div><div class='c'>{e(a[1])}</div>"
            f"<div class='r' style='margin-top:.5em'><span>{reg.tag(fdate(d, f), 'date', 'sale')}</span><span>{stamp}</span></div><div class='hr'></div>"
            + "".join(lines) + "<div class='hr'></div>"
            f"<div class='r'><span>SUBTOTAL</span><span>{reg.tag(money(sub, f, False, False), 'money', 'subtotal', str(sub))}</span></div>"
            f"<div class='r'><span>TAX {rate}%</span><span>{reg.tag(money(tax, f, False, False), 'money', 'tax', str(tax))}</span></div>"
            f"<div class='r b' style='font-size:1.15em'><span>TOTAL</span><span>{reg.tag(money(total, f, True, False), 'money', 'total', str(total))}</span></div>"
            "<div class='hr'></div>" + tender + "<div class='c' style='margin-top:.8em'>THANK YOU</div>")


# ------------------------------------------------------------------------------------------------- the set

def plan(set_name: str, count: int, seed: int) -> list[dict]:
    """Page specs: document types by the set's mix, statements of 1 to 3 pages, a capture per document (a batch shares its capture)."""
    ns, ni, nr = (round(count * x / 40) for x in MIX[set_name])
    ni += count - ns - ni - nr
    kinds = SET_KINDS[set_name]
    rng = random.Random(f"plan:{set_name}:{seed}")
    specs: list[dict] = []
    left, d = ns, 0
    while left > 0:
        n = min(left, rng.choice([1, 1, 2, 3]))
        specs += [{"doc": "statement", "doc_index": d, "pages": n, "page": k + 1} for k in range(n)]
        left -= n
        d += 1
    specs += [{"doc": "invoice", "doc_index": d + i, "pages": 1, "page": 1} for i in range(ni)]
    specs += [{"doc": "receipt", "doc_index": d + ni + i, "pages": 1, "page": 1} for i in range(nr)]
    for s in specs:
        s["capture"] = kinds[s["doc_index"] % len(kinds)]
    return specs


def tag_truth(words: list[dict], page: Page) -> tuple[list[dict], dict | None]:
    """Resolve group ids to tag entries (and the statement structure); tags the DOM walk did not find were on rows that moved to the next page."""
    by_g: dict[str, list[int]] = {}
    for i, w in enumerate(words):
        if w["g"]:
            by_g.setdefault(w["g"], []).append(i)
    tags, index = [], {}
    for g in sorted(by_g, key=lambda x: int(x[1:])):
        m = page.reg.meta[g]
        index[g] = len(tags)
        tags.append({"cls": m["cls"], "role": m["role"], "value": m["value"], "words": by_g[g]})
    st = None
    if page.statement:
        s = page.statement

        def v(x):
            return {"words": by_g[x["g"]], "value": x["value"]}
        st = {"mode": s["mode"], "start": v(s["start"]), "rows": [{"amount": v(r["amount"]), "balance": v(r["balance"]), "sign": r["sign"]} for r in s["rows"]]}
    for w in words:
        g = w.pop("g")
        if g:
            w["tag"] = index[g]
    return tags, st


def clean_words(raw: list[dict]) -> list[dict]:
    out = []
    for w in raw:
        d = {"text": w["t"], "box": [round(w[k] * SCALE, 1) for k in ("x0", "y0", "x1", "y1")], "block": w["b"], "g": w["g"]}
        if w.get("split"):
            d["split"], d["part"] = True, w["part"]
        out.append(d)
    return out


def build_doc(page, spec: dict, set_name: str, seed: int) -> list[Page]:
    rng = random.Random(f"fin:{set_name}:{seed}:{spec['doc_index']}")
    dense = set_name == "fin-tables"
    capture = spec["capture"]
    if spec["doc"] == "receipt":
        family = rng.choice(list(MONO)) if rng.random() < 0.6 else rng.choice(FAMILIES)
    else:
        family = rng.choice([x for x in FAMILIES if x not in MONO])
    style = rng.choice(["ruled", "rows", "zebra", "plain", "leaders"])
    accent = rng.choice(["#1f3b5c", "#7a1f2b", "#1d5a45", "#3a3a3a", "#4b2e83"])
    lo, hi = (7.5, 9.0) if dense else (9.0, 11.0)
    if capture.startswith("fax"):  # at 100 lines per inch smaller type is not text any more
        lo, hi = 10.0, 11.5
    fs_pt = rng.choice([x / 2 for x in range(int(lo * 2), int(hi * 2) + 1)])
    if spec["doc"] == "statement":
        return render_statement(page, Statement(rng, dense, spec["pages"]), spec["pages"], rng, family, fs_pt, style, accent)
    f = make_fmt(rng) if spec["doc"] == "invoice" else make_fmt(rng, "us" if rng.random() < 0.7 else None)
    n = (rng.randint(22, 34) if dense else rng.randint(4, 12)) if spec["doc"] == "invoice" else (rng.randint(24, 34) if dense else rng.randint(3, 11))
    use_cur = rng.random() < 0.4
    receipt_px = rng.choice([11.5, 12, 12.5, 13, 14])
    while True:
        reg = Reg()
        if spec["doc"] == "invoice":
            html_doc = doc_html(css(family, fs_pt, style, accent, 1.15 if dense else 1.3), invoice_body(random.Random(f"{set_name}:{seed}:{spec['doc_index']}:{n}"), n, dense, reg, f, use_cur))
            width, limit = W_CSS, H_CSS - 44
        else:
            rcss = (font_css(family) + "html,body{margin:0;background:#fff}#page{width:302px;box-sizing:border-box;padding:18px 16px 22px;color:#111;"
                    f"font-family:'{family}';font-size:{receipt_px}px;line-height:1.28}}.r{{display:flex;justify-content:space-between;gap:.6em}}"
                    ".c{text-align:center}.b{font-weight:700}.sm{font-size:.9em}.hr{border-top:1.5px dashed #444;margin:.45em 0}")
            html_doc = doc_html(rcss, receipt_body(random.Random(f"{set_name}:{seed}:{spec['doc_index']}:{n}"), n, reg, f))
            width, limit = 302, H_CSS - 20
        show(page, html_doc)
        wait_fonts(page, family)
        if page.evaluate("document.getElementById('content').getBoundingClientRect().bottom") <= limit:
            break
        n -= 2  # too long for the sheet: fewer rows, drawn again
        assert n > 1
    png, raw = snap(page, width)
    info = {"doc": spec["doc"], "page": 1, "pages": 1, "font": family, "font_pt": fs_pt if spec["doc"] == "invoice" else receipt_px, "style": style,
            "locale": f.locale, "currency": f.cur}
    return [Page(png, raw, reg, width, None, info)]


def render_set(set_name: str, count: int, seed: int, out: Path) -> dict:
    from playwright.sync_api import sync_playwright

    specs = plan(set_name, count, seed)
    entries: list[dict] = []
    for d in ("pages", "truth"):
        (out / d).mkdir(parents=True, exist_ok=True)
    versions = {"pillow": PIL.__version__, "numpy": np.__version__}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": W_CSS, "height": H_CSS}, device_scale_factor=SCALE)
        page = ctx.new_page()
        i = 0
        while i < len(specs):
            pages_out = build_doc(page, specs[i], set_name, seed)
            for k, pg in enumerate(pages_out):
                entries.append(capture_page(pg, specs[i + k], set_name, seed, i + k, out))
            i += len(pages_out)
        version = browser.version
        browser.close()
    return {"set": set_name, "seed": seed, "count": len(entries), "chromium": version, **versions, "pages": entries}


def capture_page(pg: Page, spec: dict, set_name: str, seed: int, idx: int, out: Path) -> dict:
    img = Image.open(io.BytesIO(pg.png)).convert("L")
    if pg.width == W_CSS:
        img = img.crop((0, 0, W_PX, H_PX))  # the screenshot of 794 x 1123 CSS px is a pixel or two larger than A4 at 300 DPI
    words = clean_words(pg.raw)
    add_ink(words, img)
    tags, statement = tag_truth(words, pg)
    capture = spec["capture"]
    sheet = False
    if pg.width != W_CSS and (capture.startswith("fax") or (capture == "scan" and random.Random(f"{set_name}:{seed}:{idx}:sheet").random() < 0.5)):
        # a receipt copied onto an A4 sheet, as it is faxed or scanned
        a4 = Image.new("L", (W_PX, H_PX), 255)
        r0 = random.Random(f"{set_name}:{seed}:{idx}:place")
        ox, oy = r0.randint(120, W_PX - img.size[0] - 120), r0.randint(120, 600)
        a4.paste(img, (ox, oy))
        for w in words:
            w["box"] = [w["box"][0] + ox, w["box"][1] + oy, w["box"][2] + ox, w["box"][3] + oy]
            w["ink"] = [w["ink"][0] + ox, w["ink"][1] + oy, w["ink"][2] + ox, w["ink"][3] + oy]
        img, sheet = a4, True
    canvas = (2016, 2688) if pg.width == W_CSS else (1500, 2688)
    return deliver(img, words, tags, statement, {**pg.info, "on_a4_sheet": sheet}, capture, set_name, seed, idx, out, canvas)


def deliver(img: Image.Image, words: list[dict], tags: list[dict], statement: dict | None, info: dict, capture: str, set_name: str, seed: int,
            idx: int, out: Path, canvas: tuple[int, int] = (2016, 2688), extra: dict | None = None) -> dict:
    """Capture a clean 300 DPI page (scan, fax or phone), keeping the capture whose words all show ink, and write page, truth and text."""
    pid = f"p{idx + 1:03d}"
    best = None
    for attempt in range(8):
        r = random.Random(f"fin:{set_name}:{seed}:{idx}:{capture}:{attempt}")
        n = np.random.default_rng([seed, idx, attempt, 7])
        if capture == "scan":
            data, w2, params = findeg.scan(img, words, r, n)
            ext = "jpg"
        elif capture.startswith("fax"):
            data, w2, params = findeg.fax(img, words, r, n, capture.split("-")[1])
            ext = "tif"
        else:
            data, w2, params = findeg.phone(img, words, lambda a, at=attempt: random.Random(f"fin:{set_name}:{seed}:{idx}:{a}:{at}"),
                                            lambda a, at=attempt: np.random.default_rng([seed, idx, a, at]), canvas)
            ext = "jpg"
        share = ink_share(data, w2)
        params["attempt"] = attempt
        if best is None or share > best[0]:
            best = (share, data, w2, params, ext)
        if share >= 0.995:
            break
    share, data, w2, params, ext = best
    with Image.open(io.BytesIO(data)) as im2:
        size = im2.size
    truth = {"id": pid, "set": set_name, "seed": seed, "page_index": idx, **info, **(extra or {}), "capture": capture,
             "image": {"width": size[0], "height": size[1]}, "degradation": params, "ink_share": round(share, 4),
             "words": w2, "tags": tags, "text": text_of(w2)}
    if statement:
        truth["statement"] = statement
    truth_bytes = (json.dumps(truth, ensure_ascii=False, indent=1) + "\n").encode()
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    (out / "pages" / f"{pid}.{ext}").write_bytes(data)
    (out / "truth" / f"{pid}.json").write_bytes(truth_bytes)
    (out / "truth" / f"{pid}.txt").write_text(truth["text"], encoding="utf-8")
    return {"id": pid, "doc": info.get("doc"), "capture": capture, "words": len(w2), "tags": len(tags), "image_sha256": sha256(data),
            "truth_sha256": sha256(truth_bytes), "text_sha256": sha256(truth["text"].encode())}


def ink_share(data: bytes, words: list[dict]) -> float:
    """Share of truth words that show ink in the delivered image: a contrast of 40 grey levels (any black dot, on a 1-bit page)."""
    with Image.open(io.BytesIO(data)) as im:
        g = np.asarray(im.convert("L"), np.uint8)
    h, w = g.shape
    ok = 0
    for t in words:
        x0, y0, x1, y1 = t.get("ink", t["box"])
        reg = g[max(0, int(y0)): min(h, int(y1) + 1), max(0, int(x0)): min(w, int(x1) + 1)]
        if reg.size and int(reg.max()) - int(reg.min()) >= 40:
            ok += 1
    return ok / max(1, len(words))


# ------------------------------------------------------------------------------------------------ commands

def check(root: Path) -> int:
    """Truth checks on every page under `root/<set>/` (or `root` itself): returns the number of failed checks and prints each."""
    from decimal import Decimal as D

    from .amounts import parse_number

    bad = 0
    dirs = [root] if (root / "truth").is_dir() else sorted(d for d in root.iterdir() if (d / "truth").is_dir())
    for d in dirs:
        n = tags_n = rows_n = 0
        for f in sorted((d / "truth").glob("*.json")):
            t = json.loads(f.read_text())
            n += 1
            problems = []
            w, h = t["image"]["width"], t["image"]["height"]
            words = t["words"]
            for k in words:
                x0, y0, x1, y1 = k["box"]
                ix0, iy0, ix1, iy1 = k["ink"]
                if not (k["text"] and -3 <= x0 < x1 <= w + 3 and -3 <= y0 < y1 <= h + 3 and ix0 < ix1 and iy0 < iy1):
                    problems.append(f"bad word {k['text']!r} {k['box']}")
                    break
            if t["text"].split() != " ".join(k["text"] if k.get("part", 0) == 0 else "\0" + k["text"] for k in words).replace(" \0", "").split():
                problems.append("text does not match the words")
            img = next((d / "pages").glob(f"{f.stem}.*"))
            share = ink_share(img.read_bytes(), words)
            if share < 0.95:  # dense small print on a 100 dpi fax loses a few words outright; the share is in the truth file
                problems.append(f"only {share:.4f} of the words show ink")
            tg = t.get("tags") or []
            tags_n += len(tg)
            if not any(x["cls"] == "money" for x in tg):
                problems.append("no money tag")
            for x in tg:
                text = " ".join(words[i]["text"] for i in x["words"])
                if any(not 0 <= i < len(words) for i in x["words"]):
                    problems.append(f"tag words out of range {x}")
                elif x.get("role") == "real":
                    if text != x["value"]:
                        problems.append(f"tag text {text!r} differs from {x['value']!r}")
                elif x["cls"] == "money" and x["value"] is not None:
                    p = parse_number(text)
                    if p is None or abs(p) != abs(D(x["value"])) or (x["role"] in ("balance", "opening", "bf", "cf", "closing") and p != D(x["value"]) and t.get("neg") != "crdr"):
                        problems.append(f"tag text {text!r} does not parse to {x['value']}")
            st = t.get("statement")
            if st:
                def val(v):
                    return D(v["value"])
                prev = val(st["start"])
                for r in st["rows"]:
                    rows_n += 1
                    if prev + r["sign"] * abs(val(r["amount"])) != val(r["balance"]):
                        problems.append("statement row does not reconcile")
                        break
                    prev = val(r["balance"])
            if img.suffix == ".tif":
                with Image.open(img) as im:
                    if im.format != "TIFF" or im.mode != "1":
                        problems.append(f"fax page is {im.format} {im.mode}")
            for p in problems:
                print(f"{d.name}/{f.stem}: {p}")
            bad += len(problems)
        print(f"{d.name}: {n} pages, {tags_n} tags, {rows_n} statement rows checked")
    return bad


def draw(root: Path, pid: str) -> None:
    from PIL import ImageDraw

    t = json.loads((root / "truth" / f"{pid}.json").read_text())
    img = Image.open(next((root / "pages").glob(f"{pid}.*"))).convert("RGB")
    dr = ImageDraw.Draw(img)
    for w in t["words"]:
        q = w["quad"]
        dr.polygon([(q[0], q[1]), (q[2], q[3]), (q[4], q[5]), (q[6], q[7])], outline=(220, 0, 0) if "tag" in w else (0, 140, 255), width=2)
    dest = root / "samples" / f"{pid}-boxes.png"
    dest.parent.mkdir(exist_ok=True)
    if max(img.size) > 1800:
        f = 1800 / max(img.size)
        img = img.resize((round(img.size[0] * f), round(img.size[1] * f)))
    img.save(dest)
    print(dest)


def main(a) -> None:
    from .fin_assets import fetch
    from .paths import MANIFESTS

    if a.action == "fetch":
        fetch(a.update)
    elif a.action == "gen":
        seeds = json.loads(Path(a.seeds).expanduser().read_text())
        out = Path(a.out).expanduser()
        names = [a.set] if a.set else list(SET_KINDS)
        manifest = {"sets": {}}
        for name in names:
            m = render_set(name, a.count or 40, int(seeds[name]), out / name)
            manifest["sets"][name] = m
            print(f"{name}: {len(m['pages'])} pages", flush=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        if a.check:
            pinned = json.loads((MANIFESTS / "fin-dev-v1.json").read_text())["synthetic"]
            bad = [n for n in names if manifest["sets"][n]["pages"] != pinned[n]["pages"]]
            if bad:
                raise SystemExit(f"hash mismatch in {bad}")
            print("pages reproduce the committed hashes")
    elif a.action == "freeze":
        root = Path(a.root).expanduser()
        man = {"synthetic": json.loads((root / "manifest.json").read_text())["sets"],
               "real": json.loads((root / "manifest-real.json").read_text())["sets"]}
        if a.manifest:
            Path(a.manifest).expanduser().write_text(json.dumps(man, indent=1) + "\n")
        lines = [f"{sha256(p.read_bytes())}  {p.relative_to(root).as_posix()}" for p in sorted(root.rglob("*"))
                 if p.is_file() and p.name != "SHA256SUMS" and "samples" not in p.parts]
        (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        print(f"{len(lines)} files hashed; digest {sha256(chr(10).join(lines).encode())}")
    elif a.action == "check":
        raise SystemExit(1 if check(Path(a.root).expanduser()) else 0)
    elif a.action == "draw":
        draw(Path(a.root).expanduser(), a.page)
