"""Strings of financial documents for the line sampler: amounts in the number conventions of statements, invoices and reports, with
their signs and currency marks, dates, account and reference numbers, quantities, and the short label words and descriptions around
them.

Everything is drawn from a seeded generator and the rendered string is the label (exact truth). The text recipes are written from the
printing conventions of such documents, not from any evaluation page: amounts are random digit strings, descriptions are windows of the
training books or a short list of generic statement words. The goal is a recognizer that does not drop or swap the characters whose
loss changes a figure: the sign, the decimal separator, a thousands separator, a lone digit.
"""

from __future__ import annotations

import numpy as np

_CODES = ["USD", "EUR", "GBP", "CHF", "CAD", "AUD", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF"]
_LABELS = [
    "Opening balance", "Closing balance", "Balance forward", "Balance brought forward", "Carried forward", "Total", "Subtotal",
    "Sub-total", "Total due", "Amount due", "Balance due", "Payment received", "Interest", "Fee", "Fees", "Service charge", "Withdrawal",
    "Deposit", "Debit", "Credit", "Date", "Description", "Amount", "Balance", "Qty", "Unit price", "Tax", "VAT", "Net", "Gross", "Total net",
    "Total gross", "Discount", "Shipping", "Previous balance", "New balance", "Minimum payment", "Payment due", "Total deposits",
    "Total withdrawals", "Net change", "Ending balance", "Beginning balance", "Receipts", "Outlays", "Surplus", "Deficit",
    "Saldo", "Kontostand", "Betrag", "Datum", "Rechnung", "Gesamt", "Summe", "MwSt.", "Solde", "Montant", "Total TTC", "Importe", "Fecha",
    "Totale", "Saldo iniziale", "Saldo finale", "Zahlung", "Soll", "Haben", "Débit", "Crédit", "Intérêts",
]  # fmt: skip
_DESC = [
    "CHECK", "ATM WITHDRAWAL", "TRANSFER", "SERVICE FEE", "ONLINE PAYMENT", "CARD PURCHASE", "DIRECT DEBIT", "PAYROLL", "DEPOSIT", "INTEREST",
    "REFUND", "WIRE", "MONTHLY FEE", "STANDING ORDER", "POS PURCHASE", "CASH DEPOSIT", "ACH CREDIT", "BILL PAYMENT", "TRANSFER FROM SAVINGS",
    "Consulting services", "Office supplies", "Software licence", "Freight", "Rent", "Utilities", "Maintenance", "Subscription",
]  # fmt: skip
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _digits(rng: np.random.Generator, n: int) -> str:
    return "".join(str(int(x)) for x in rng.integers(0, 10, size=n))


def _lead(rng: np.random.Generator) -> str:
    """First digit: about half follow the 1-heavy law of ledger figures, the rest uniform, so no digit is rare."""
    if rng.random() < 0.5:
        return str(int(rng.choice(9, p=np.log10(1 + 1 / np.arange(1, 10))) + 1))
    return str(int(rng.integers(1, 10)))


def _int_part(rng: np.random.Generator) -> str:
    n = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.1, 0.16, 0.2, 0.2, 0.13, 0.08, 0.05, 0.03]))
    body = _lead(rng) + _digits(rng, n - 1)
    if n >= 4 and rng.random() < 0.18:  # round figures: runs of zeros are where digits get dropped
        k = int(rng.integers(1, min(4, n)))
        body = body[:-k] + "0" * k
    return body


def _group(body: str, sep: str) -> str:
    out = []
    while len(body) > 3:
        out.append(body[-3:])
        body = body[:-3]
    out.append(body)
    return sep.join(reversed(out))


def number(rng: np.random.Generator) -> str:
    """An unsigned amount: grouping and decimal marks of one convention, mostly two decimals."""
    conv = rng.choice(["us", "eu", "ch", "sp_dot", "sp_comma", "plain_dot", "plain_comma"], p=[0.38, 0.22, 0.04, 0.04, 0.06, 0.15, 0.11])
    body = _int_part(rng)
    thou, dec = {"us": (",", "."), "eu": (".", ","), "ch": ("'", "."), "sp_dot": (" ", "."), "sp_comma": (" ", ","),
                 "plain_dot": ("", "."), "plain_comma": ("", ",")}[str(conv)]  # fmt: skip
    r = rng.random()
    nd = 2 if r < 0.88 else 0 if r < 0.95 else int(rng.choice([1, 3, 4]))
    s = _group(body, thou)
    if nd:
        frac = _digits(rng, nd)
        if nd == 2 and rng.random() < 0.12:
            frac = str(rng.choice(["00", "50", "99", "95", "25", "75"]))
        s += dec + frac
    return s


def money(rng: np.random.Generator) -> str:
    """A printed amount with its sign convention and currency mark."""
    s = number(rng)
    r = rng.random()
    neg = r < 0.38
    cur = str(rng.choice(["", "$", "€", "£", "code", "Fr.", "¥"], p=[0.5, 0.22, 0.08, 0.05, 0.1, 0.01, 0.04]))
    if cur == "code":
        code = str(rng.choice(_CODES))
        pre = rng.random() < 0.4
        cur_s = code + (" " if rng.random() < 0.7 else "")
        body = cur_s + s if pre else s + (" " if rng.random() < 0.8 else "") + code
        cur = ""
    else:
        gap = " " if (cur and rng.random() < 0.12) else ""
        body = (cur + gap + s) if cur != "" and (cur not in ("Fr.",) or rng.random() < 0.5) else s + (" " + cur if cur else "")
    if not neg:
        if rng.random() < 0.03:
            body = "+" + body
        return body
    style = rng.choice(["minus", "paren", "trail", "crdr", "uminus", "dash", "front_after_cur"], p=[0.40, 0.22, 0.12, 0.12, 0.05, 0.05, 0.04])
    if style == "minus":
        return "-" + body
    if style == "uminus":
        return "\u2212" + body
    if style == "dash":
        return "\u2013" + body
    if style == "paren":
        return "(" + body + ")"
    if style == "trail":
        return body + str(rng.choice(["-", "-", "\u2212"]))
    if style == "crdr":
        return body + str(rng.choice([" DR", " CR", "DR", "CR", " Dr", " Cr"]))
    # sign between the currency mark and the digits: $-12.00 / -$12.00
    return f"-{body}" if rng.random() < 0.5 else body.replace(s, "-" + s, 1)


def date(rng: np.random.Generator) -> str:
    y = int(rng.integers(1995, 2031))
    m = int(rng.integers(1, 13))
    d = int(rng.integers(1, 29))
    k = int(rng.integers(0, 12))
    yy = f"{y % 100:02d}"
    forms = [f"{y}-{m:02d}-{d:02d}", f"{m:02d}/{d:02d}/{y}", f"{d:02d}.{m:02d}.{y}", f"{d} {_MONTHS[m - 1]} {y}", f"{_MONTHS[m - 1]} {d}, {y}",
             f"{_MONTHS_LONG[m - 1]} {d}, {y}", f"{m:02d}/{d:02d}/{yy}", f"{d:02d}-{_MONTHS[m - 1]}-{y}", f"{m}/{d}/{yy}", f"{d:02d}/{m:02d}/{y}",
             f"{m:02d}/{d:02d}", f"{d:02d}.{m:02d}.{yy}"]  # fmt: skip
    return forms[k]


def account(rng: np.random.Generator) -> str:
    k = int(rng.integers(0, 10))
    d = lambda n: _digits(rng, n)  # noqa: E731
    up = lambda n: "".join(str(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789"))) for _ in range(n))  # noqa: E731
    if k == 0:
        return f"{rng.choice(['DE', 'FR', 'GB', 'NL', 'ES', 'IT', 'CH', 'AT', 'BE'])}{d(2)} " + " ".join(d(4) for _ in range(int(rng.integers(4, 7))))
    if k == 1:
        return f"****{d(4)}" if rng.random() < 0.5 else f"xxxx{d(4)}"
    if k == 2:
        return " ".join(d(4) for _ in range(int(rng.integers(2, 5))))
    if k == 3:
        return d(int(rng.integers(8, 13)))
    if k == 4:
        return f"{d(2)}-{d(2)}-{d(2)}"
    if k == 5:
        return f"{rng.choice(['INV', 'IN', 'RE', 'PO', 'REF', 'No.', '#', 'Inv.'])}{rng.choice(['-', ' ', ''])}{d(int(rng.integers(3, 9)))}"
    if k == 6:
        return f"{rng.choice(['INV', 'INV', 'FA', 'RG'])}-{int(rng.integers(2015, 2031))}-{d(int(rng.integers(3, 6)))}"
    if k == 7:
        return f"{d(3)}-{d(int(rng.integers(5, 10)))}-{d(int(rng.integers(1, 3)))}"
    if k == 8:
        return f"{rng.choice(['Ref:', 'Ref.', 'Acct', 'Account', 'Check', 'Chk', 'Order'])} {up(int(rng.integers(5, 11)))}"
    return f"{rng.choice(['BIC', 'SWIFT', 'IBAN', 'Acct'])} {up(8)}"


def quantity(rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.45:
        return str(int(rng.choice([0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9])))
    if r < 0.7:
        return str(int(rng.integers(10, 100)))
    if r < 0.82:
        return f"{int(rng.integers(1, 20))} {rng.choice(['pcs', 'PCS', 'ea', 'hrs', 'x', 'pc', 'kg'])}"
    if r < 0.92:
        return f"{int(rng.integers(0, 100))}{rng.choice(['.', ','])}{int(rng.integers(0, 100)):02d}%"
    return f"{int(rng.integers(1, 30))}%"


def _words(rng: np.random.Generator, books, n: int) -> str:
    w, _t = books.window(rng, short=True)
    w = " ".join(w.split()[:n])
    q = rng.random()
    return w.upper() if q < 0.5 else w.title() if q < 0.8 else w


def fin_line(rng: np.random.Generator, books) -> tuple[str, str]:
    """One line and its kind (for the manifest's source counts)."""
    r = rng.random()
    if r < 0.30:
        return money(rng), "amount"
    if r < 0.42:
        return " ".join(money(rng) for _ in range(int(rng.integers(2, 4)))), "amounts"
    if r < 0.50:
        return date(rng), "date"
    if r < 0.60:
        return account(rng), "account"
    if r < 0.66:
        return quantity(rng), "qty"
    if r < 0.82:  # a statement row: date, description, amount, balance
        parts = [date(rng)] if rng.random() < 0.75 else []
        parts.append(str(rng.choice(_DESC)) if rng.random() < 0.5 else _words(rng, books, int(rng.integers(1, 4))))
        if rng.random() < 0.3:
            parts.append(account(rng))
        parts.append(money(rng))
        if rng.random() < 0.6:
            parts.append(money(rng))
        return " ".join(parts), "row"
    if r < 0.92:  # label and figure: totals, balances, tax lines
        lab = str(rng.choice(_LABELS))
        if rng.random() < 0.25:
            lab += str(rng.choice([":", "", " :"]))
        if rng.random() < 0.12:
            lab = lab.upper()
        parts = [lab, money(rng)]
        if rng.random() < 0.2:
            parts.insert(1, quantity(rng))
        return " ".join(parts), "label"
    if r < 0.96:  # invoice item row: description, quantity, unit price, amount
        return " ".join([_words(rng, books, int(rng.integers(1, 4))), quantity(rng), money(rng), money(rng)]), "item"
    return " ".join([str(rng.choice(_LABELS)), date(rng)]), "label-date"
