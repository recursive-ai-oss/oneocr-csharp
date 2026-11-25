"""Amount exact match and running-balance reconciliation (scorer v3.2).

Both metrics read the word boxes an engine reports, so they need a truth sidecar with `tags` (and, for statements, `statement`)
and a hypothesis sidecar `<id>.json` with `words[{text, box|quad}]`. Pages whose truth has no tags are skipped, and a page with
no hypothesis sidecar scores every value as wrong.

Truth `tags`: one entry per tagged value `{cls, role, value, words: [indices into the truth words]}`. `cls` is `money`, `date`
or `account` (account-like numbers: account, IBAN, routing, card, reference) for the exact-match score, and `qty` (table
quantity cells) which is reported on its own. A value spans one or more truth words (`1.234,56 EUR` is two).

Exact match. A truth word is read when a hypothesis word has the same text (after the scorer's normalisation) and its centre
lies inside the truth word's quad grown by `GROW` about its centre. Every hypothesis word is used for at most one truth word, nearest
centre first. A value is read when all of its words are read. So a value read correctly somewhere else on the page earns nothing
(position is part of the match), a value split by the engine into pieces or fused with its neighbour is wrong, and nothing
depends on the order in which the engine lists its words. The test is identical for every engine; an engine that reports no word
boxes would score zero, which is why the runner refuses such outputs.

Reconciliation. Truth `statement` = `{start: V, rows: [{amount: V, balance: V, sign: +1|-1}]}` with V = `{words, value}`. A row's
amount and balance, and the page's opening or brought-forward balance, are read from the hypothesis as the text of the hypothesis words
whose centres fall inside the truth words' grown quads, left to right, parsed as a number (US and European separators, currency
symbols and codes, `(1,234.56)`, a leading or trailing minus, `CR` and `DR`). A row reconciles when
`previous balance + sign * |amount| == balance` as read, where the previous balance is the row above's balance as read (the
page's start for the first row) and the sign is the truth column's (credit +, debit -). A value that is not read, or does not parse, fails
the row. The rate is reconciled rows / truth rows.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

from .score import _aabb, normalize

GROW = 1.3
CLASSES = ("money", "date", "account")
CURRENCY = {"$", "€", "£", "¥", "USD", "EUR", "GBP", "CHF", "US$"}


def _quad(w: dict) -> list[tuple[float, float]]:
    q = w.get("quad") or w.get("ink_quad")
    if q:
        return [(float(q[i]), float(q[i + 1])) for i in range(0, 8, 2)]
    x0, y0, x1, y1 = _aabb(w, "box")
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _region(w: dict) -> list[tuple[float, float]]:
    q = _quad(w)
    cx = sum(p[0] for p in q) / 4
    cy = sum(p[1] for p in q) / 4
    return [(cx + (x - cx) * GROW, cy + (y - cy) * GROW) for x, y in q]


def _centre(w: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = _aabb(w)
    return (x0 + x1) / 2, (y0 + y1) / 2


def _inside(p: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = p
    sign = 0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        c = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        if c:
            if sign and (c > 0) != (sign > 0):
                return False
            sign = 1 if c > 0 else -1
    return True


def _key(text: str) -> str:
    return " ".join(normalize(text, True))


def _match(truth_words: list[dict], hyp_words: list[dict], wanted: list[int]) -> dict[int, int]:
    """Truth word index -> hypothesis word index, one hypothesis word per truth word, nearest centre first."""
    hc = [_centre(h) for h in hyp_words]
    hk = [_key(h["text"]) for h in hyp_words]
    pairs = []
    for i in wanted:
        tw = truth_words[i]
        reg = _region(tw)
        tc = _centre(tw)
        tk = _key(tw["text"])
        for j, c in enumerate(hc):
            if hk[j] == tk and _inside(c, reg):
                pairs.append((math.hypot(c[0] - tc[0], c[1] - tc[1]), i, j))
    pairs.sort()
    used_t: set[int] = set()
    used_h: set[int] = set()
    out: dict[int, int] = {}
    for _, i, j in pairs:
        if i in used_t or j in used_h:
            continue
        used_t.add(i)
        used_h.add(j)
        out[i] = j
    return out


def parse_number(text: str) -> Decimal | None:
    """A money amount as written on a statement, or None. `CR` is positive and `DR` negative; parentheses and minus signs negate."""
    neg = False
    parts = []
    for tok in text.split():
        u = tok.upper()
        if u == "DR":
            neg = True
        elif u == "CR" or u in CURRENCY:
            continue
        else:
            parts.append(tok)
    s = "".join(parts).replace("\u2212", "-").replace("\u2013", "-")
    for c in sorted(CURRENCY, key=len, reverse=True):
        s = s.replace(c, "")
    if s.upper().endswith("DR"):
        neg, s = True, s[:-2]
    elif s.upper().endswith("CR"):
        s = s[:-2]
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.startswith("-"):
        neg, s = not neg, s[1:]
    if s.endswith("-"):
        neg, s = not neg, s[:-1]
    if not re.fullmatch(r"\d[\d.,']*", s):
        return None
    last = max(s.rfind("."), s.rfind(","))
    if last >= 0 and 1 <= len(s) - last - 1 <= 2:
        whole, frac = s[:last], s[last + 1:]
    else:
        whole, frac = s, ""
    whole = re.sub(r"[.,']", "", whole)
    if not whole.isdigit():
        return None
    try:
        v = Decimal(f"{whole}.{frac}" if frac else whole)
    except InvalidOperation:
        return None
    return -v if neg else v


def _read(truth_words: list[dict], hyp_words: list[dict], words: list[int]) -> str:
    regs = [_region(truth_words[i]) for i in words]
    hit = [h for h in hyp_words if any(_inside(_centre(h), r) for r in regs)]
    hit.sort(key=lambda h: _centre(h)[0])
    return " ".join(h["text"] for h in hit)


def score_page(truth: dict, hyp_words: list[dict] | None) -> dict | None:
    """Per-class counts of tagged values read exactly, and the reconciliation counts, for one page; None when the truth has no tags."""
    tags = truth.get("tags")
    if not tags:
        return None
    tw = truth["words"]
    hw = hyp_words or []
    wanted = sorted({i for t in tags for i in t["words"]})
    got = _match(tw, hw, wanted)
    res: dict = {"values": {}, "read": {}, "decisions": []}
    for t in tags:
        ok = all(i in got for i in t["words"])
        res["values"][t["cls"]] = res["values"].get(t["cls"], 0) + 1
        res["read"][t["cls"]] = res["read"].get(t["cls"], 0) + int(ok)
        res["decisions"].append({"cls": t["cls"], "role": t.get("role"), "words": t["words"], "read": ok})
    st = truth.get("statement")
    res["rows"] = res["reconciled"] = 0
    if st:
        def value(v: dict) -> Decimal | None:
            return parse_number(_read(tw, hw, v["words"])) if hyp_words else None

        prev = value(st["start"])
        for row in st["rows"]:
            amt, bal = value(row["amount"]), value(row["balance"])
            res["rows"] += 1
            if prev is not None and amt is not None and bal is not None and prev + row["sign"] * abs(amt) == bal:
                res["reconciled"] += 1
            prev = bal
    return res


def add_page(total: dict, page: dict) -> None:
    for k in ("values", "read"):
        for c, n in page[k].items():
            total.setdefault(k, {})
            total[k][c] = total[k].get(c, 0) + n
    total["rows"] = total.get("rows", 0) + page["rows"]
    total["reconciled"] = total.get("reconciled", 0) + page["reconciled"]


def summarize(total: dict) -> dict:
    """Rates from summed counts: exact match overall (money + date + account) and per class, qty on its own, reconciliation."""
    vals, read = total.get("values", {}), total.get("read", {})
    n = sum(vals.get(c, 0) for c in CLASSES)
    r = sum(read.get(c, 0) for c in CLASSES)
    out = {"amount_values": n, "amount_exact": r / n if n else None,
           "classes": {c: {"values": vals[c], "exact": read[c] / vals[c]} for c in vals},
           "rows": total.get("rows", 0), "reconciled": total.get("reconciled", 0),
           "reconciliation": total["reconciled"] / total["rows"] if total.get("rows") else None}
    return out
