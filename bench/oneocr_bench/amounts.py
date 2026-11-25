"""Amount exact match and running-balance reconciliation (scorer v3.2).

Both metrics read the word boxes an engine reports, so they need a truth sidecar with `tags` (and, for statements, `statement`)
and a hypothesis sidecar `<id>.json` with `words[{text, box|quad}]`. Pages whose truth has no tags are skipped. A page with no
hypothesis sidecar (an engine that reports no boxes) is scored by the text-only rule below instead, so missing boxes cost nothing.

Truth `tags`: one entry per tagged value `{cls, role, value, words: [indices into the truth words]}`. `cls` is `money`, `date`
or `account` (account-like numbers: account, IBAN, routing, card, reference) for the exact-match score, and `qty` (table
quantity cells) which is reported on its own. A value spans one or more truth words (`1.234,56 EUR` is two).

Exact match in place (the rule the gate uses). The value's region is the box around its words grown by `GROW` about its centre. A
value is read when some run of consecutive hypothesis words among those whose centres lie in its region (left to right, written
without spaces) spells the value's words written without spaces; neighbours' words at either end of the run are ignored. A hypothesis word that lies in several regions belongs to the nearest value. So a value
read correctly somewhere else on the page earns nothing (position is part of the match); a value fused with a neighbour or
with extra characters is wrong; and the engine's tokenisation does not matter: `250,00 €` read as one word `250,00€`, or
`1 234.56` as two words, is the same reading. Nothing depends on the order in which the engine lists its words.

Text-only match (information, and the fallback without boxes). Each value's words written without spaces must equal the text of
consecutive hypothesis tokens (one to four of them), each token used for one value at most. No position is used, so it is an
upper bound: a value read right in the wrong place counts. It is what the in-place rule is compared with to see whether box
granularity costs an engine anything.

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

import numpy as np

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


def value_region(truth_words: list[dict], idxs: list[int]) -> list[tuple[float, float]]:
    """Where the words of one tagged value may be found: a word's own quad grown by GROW, and for a value of several words (`1.234,56 EUR`)
    the box around all of them grown the same way. Engines place the words of one value with different accuracy, and a value is one
    thing on the page; the unit that has to be in place is the value."""
    if len(idxs) == 1:
        return _region(truth_words[idxs[0]])
    pts = [p for i in idxs for p in _quad(truth_words[i])]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    hw, hh = (max(xs) - min(xs)) / 2 * GROW, (max(ys) - min(ys)) / 2 * GROW
    return [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]


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


def _nospace(text: str) -> str:
    return "".join(normalize(text, True))


def read_values(truth_words: list[dict], hyp_words: list[dict], tags: list[dict]) -> list[bool]:
    """For each tag, whether its value is read exactly in place (module docstring)."""
    # one printed value can carry two tags (a reference that is also a heading): they are read together, not competing for the words
    uniq: dict[tuple, int] = {}
    for t in tags:
        uniq.setdefault(tuple(t["words"]), len(uniq))
    keys = list(uniq)
    tags_u = [{"words": list(k)} for k in keys]
    got = _read_unique(truth_words, hyp_words, tags_u)
    return [got[uniq[tuple(t["words"])]] for t in tags]


def _read_unique(truth_words: list[dict], hyp_words: list[dict], tags: list[dict]) -> list[bool]:
    regions = [value_region(truth_words, t["words"]) for t in tags]
    centres = []
    for t in tags:
        pts = [p for i in t["words"] for p in _quad(truth_words[i])]
        centres.append((sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)))
    owned: list[list[tuple[float, str]]] = [[] for _ in tags]
    box = np.array([[min(p[0] for p in r), min(p[1] for p in r), max(p[0] for p in r), max(p[1] for p in r)] for r in regions]).reshape(-1, 4)
    for h in hyp_words:
        c = _centre(h)
        near = np.nonzero((box[:, 0] <= c[0]) & (c[0] <= box[:, 2]) & (box[:, 1] <= c[1]) & (c[1] <= box[:, 3]))[0]
        best = None
        for k in near:
            if _inside(c, regions[k]):
                d = math.hypot(c[0] - centres[k][0], c[1] - centres[k][1])
                if best is None or d < best[0]:
                    best = (d, int(k))
        if best:
            owned[best[1]].append((c[0], _nospace(h["text"])))
    return [_spells([t for _, t in sorted(o)], "".join(_nospace(truth_words[i]["text"]) for i in tags[k]["words"]))
            for k, o in enumerate(owned)]


def _spells(tokens: list[str], want: str) -> bool:
    """Whether consecutive tokens, written without spaces, spell `want`. Words of the neighbours that fall in the grown region before or
    after the value are ignored; a token that holds the value together with other characters does not spell it."""
    for i in range(len(tokens)):
        acc = ""
        for t in tokens[i:]:
            acc += t
            if acc == want:
                return True
            if not want.startswith(acc):
                break
    return False


def read_values_text(truth_words: list[dict], tokens: list[str], tags: list[dict]) -> list[bool]:
    """For each tag, whether its value is read exactly by the text-only rule (module docstring)."""
    toks = [_nospace(t) for t in tokens]
    index: dict[str, list[tuple[int, int]]] = {}
    for i in range(len(toks)):
        acc = ""
        for n in range(1, 5):
            if i + n > len(toks):
                break
            acc += toks[i + n - 1]
            index.setdefault(acc, []).append((i, n))
    used = [False] * len(toks)
    out = []
    for t in tags:
        want = "".join(_nospace(truth_words[i]["text"]) for i in t["words"])
        hit = next(((i, n) for i, n in index.get(want, []) if not any(used[i:i + n])), None)
        if hit:
            used[hit[0]:hit[0] + hit[1]] = [True] * hit[1]
        out.append(hit is not None)
    return out


def _page(tags: list[dict], reads: list[bool]) -> dict:
    res: dict = {"values": {}, "read": {}, "decisions": [], "rows": 0, "reconciled": 0}
    for t, ok in zip(tags, reads):
        res["values"][t["cls"]] = res["values"].get(t["cls"], 0) + 1
        res["read"][t["cls"]] = res["read"].get(t["cls"], 0) + int(ok)
        res["decisions"].append({"cls": t["cls"], "role": t.get("role"), "words": t["words"], "read": ok})
    return res


def score_page_text(truth: dict, hyp_text: str) -> dict | None:
    """The text-only twin of `score_page`; None when the truth has no tags."""
    tags = truth.get("tags")
    if not tags:
        return None
    return _page(tags, read_values_text(truth["words"], normalize(hyp_text, True), tags))


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
    res = _page(tags, read_values(tw, hw, tags))
    st = truth.get("statement")
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
           "reconciliation": total["reconciled"] / total["rows"] if total.get("rows") else None, "counts": total}
    return out
