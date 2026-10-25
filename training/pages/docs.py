"""Invoices and receipts with field truth.

Invoice fields carry the EN 16931 business term (BT) ids; the value of every field is stored normalized (ISO dates, plain
decimals, country codes) next to the words it was printed in. All content is invented (see corpus.py). Layouts vary in the
discrete choices recorded in the template (logo and title placement, address arrangement, meta block, item table style and
columns, totals style, bank block, footer) and in fonts, sizes, colours and wording per locale (en de fr es it nl).

Receipt fields: merchant, date, time, items (name, quantity, unit price, amount), subtotal, tax, tip, total, payment method.
"""

from __future__ import annotations

import random
from datetime import timedelta

from . import corpus
from .corpus import LOCALES, money
from .fontpool import SCRIPT_SAMPLES, pick
from .htmlkit import Look, esc, frame, inline
from .layouts import ACCENTS, INKS, PAPERS, Ctx, PageSpec, template_split


def _l(s: str) -> list[str]:
    return s.split("|")


LAB = {  # en|de|fr|es|it|nl
    "invoice": _l("Invoice|Rechnung|Facture|Factura|Fattura|Factuur"), "inv_no": _l("Invoice no.|Rechnungsnr.|N° de facture|N.º de factura|N. fattura|Factuurnr."),
    "date": _l("Date|Datum|Date|Fecha|Data|Datum"), "due": _l("Due date|Fälligkeitsdatum|Échéance|Vencimiento|Scadenza|Vervaldatum"),
    "delivery": _l("Delivery date|Lieferdatum|Date de livraison|Fecha de entrega|Data di consegna|Leverdatum"),
    "po": _l("Order no.|Bestellnr.|N° de commande|N.º de pedido|N. ordine|Ordernr."),
    "currency": _l("Currency|Währung|Devise|Moneda|Valuta|Valuta"),
    "pos": _l("Pos.|Pos.|Pos.|Pos.|Pos.|Pos."), "desc": _l("Description|Beschreibung|Désignation|Descripción|Descrizione|Omschrijving"),
    "qty": _l("Qty|Menge|Qté|Cant.|Qtà|Aantal"), "unit": _l("Unit|Einheit|Unité|Unidad|Unità|Eenheid"),
    "price": _l("Unit price|Einzelpreis|Prix unitaire|Precio|Prezzo|Prijs"), "vat": _l("VAT|MwSt.|TVA|IVA|IVA|BTW"),
    "amount": _l("Amount|Betrag|Montant|Importe|Importo|Bedrag"), "subtotal": _l("Subtotal|Zwischensumme|Sous-total|Subtotal|Subtotale|Subtotaal"),
    "tax": _l("VAT total|MwSt. gesamt|Total TVA|Total IVA|Totale IVA|Totaal BTW"), "total": _l("Total|Gesamtbetrag|Total TTC|Total|Totale|Totaal"),
    "due_amt": _l("Amount due|Zahlbetrag|Net à payer|Importe a pagar|Importo dovuto|Te betalen"),
    "terms": _l("Payment terms|Zahlungsbedingungen|Conditions de paiement|Condiciones de pago|Condizioni di pagamento|Betalingsvoorwaarden"),
    "bank": _l("Bank details|Bankverbindung|Coordonnées bancaires|Datos bancarios|Coordinate bancarie|Bankgegevens"),
    "iban": _l("IBAN|IBAN|IBAN|IBAN|IBAN|IBAN"), "bic": _l("BIC|BIC|BIC|BIC|BIC|BIC"),
    "vat_id": _l("VAT ID|USt-IdNr.|N° TVA|NIF|P. IVA|BTW-nr."), "bill_to": _l("Bill to|Rechnung an|Facturé à|Facturar a|Fatturare a|Factuur aan"),
    "taxable": _l("Net|Netto|Base|Base|Imponibile|Netto"), "thanks": _l("Thank you for your business.|Vielen Dank für Ihren Auftrag.|Merci de votre confiance.|Gracias por su confianza.|Grazie per la fiducia.|Bedankt voor uw opdracht."),
    "tel": _l("Tel.|Tel.|Tél.|Tel.|Tel.|Tel."),
}  # fmt: skip
CONTRACT = _l("Contract no.|Vertragsnr.|N° de contrat|N.º de contrato|N. contratto|Contractnr.")
TERMS = _l(
    "Payable within {n} days|Zahlbar innerhalb von {n} Tagen|Payable sous {n} jours|Pagadero en {n} días|Pagabile entro {n} giorni|Betaalbaar binnen {n} dagen"
)
RATES = {"en": [0, 5, 20], "de": [7, 19], "fr": [5.5, 10, 20], "es": [4, 10, 21], "it": [4, 10, 22], "nl": [9, 21]}
UNITS = {"en": ("pcs", "h"), "de": ("Stk.", "Std."), "fr": ("pce", "h"), "es": ("ud.", "h"), "it": ("pz.", "h"), "nl": ("st.", "uur")}
CUR = {"en": ("GBP", "£"), "de": ("EUR", "€"), "fr": ("EUR", "€"), "es": ("EUR", "€"), "it": ("EUR", "€"), "nl": ("EUR", "€")}
IDS = {
    "invoiceNumber": "BT-1", "issueDate": "BT-2", "currency": "BT-5", "dueDate": "BT-9", "poNumber": "BT-13", "paymentTerms": "BT-20",
    "sellerName": "BT-27", "sellerVatId": "BT-31", "sellerStreet": "BT-35", "sellerCity": "BT-37", "sellerPostcode": "BT-38",
    "sellerCountry": "BT-40", "buyerName": "BT-44", "buyerVatId": "BT-48", "buyerStreet": "BT-50", "buyerCity": "BT-52",
    "buyerPostcode": "BT-53", "buyerCountry": "BT-55", "deliveryDate": "BT-72", "iban": "BT-84", "bic": "BT-86", "subtotal": "BT-106",
    "taxTotal": "BT-110", "total": "BT-112", "amountDue": "BT-115", "lineId": "BT-126", "lineNet": "BT-131", "quantity": "BT-129",
    "unitCode": "BT-130", "unitPrice": "BT-146", "lineVatRate": "BT-152", "itemName": "BT-153", "taxableAmount": "BT-116",
    "taxAmount": "BT-117", "taxRate": "BT-119",
    "contractNumber": "custom.contractNumber",  # only on the contract variant of the invoice page (invoice_page(contract=True))
}  # fmt: skip


def fld(name: str, val: str, text: str | None = None, extra: str = "") -> str:
    return (
        f'<span data-fld="{IDS[name]}" data-fname="{name}" data-val="{esc(val)}"{extra}>{inline(text if text is not None else val)}</span>'
    )


def kvpair(ctx: Ctx, label: str, value_html: str, cls: str = "kv") -> str:
    g = ctx.next_kv()
    return f'<div class="{cls}"><span class="k" data-kv="{g}" data-kvr="key">{inline(label)}</span> <span class="v" data-kv="{g}" data-kvr="value">{value_html}</span></div>'


def tblk(role: str, reg: str, dl: str, cls: str, content: str, extra: str = "") -> str:
    return f'<div class="{cls}" data-blk="{role}" data-reg="{reg}" data-dl="{dl}" {extra}>{content}</div>'


def _make_invoice(rng: random.Random, ctx: Ctx, loc: str, n_items: int) -> dict:
    cur, sym = CUR[loc]
    seller = {"name": corpus.company(rng, loc), "addr": corpus.address(rng, loc)}
    buyer = {"name": corpus.company(rng, loc) if rng.random() < 0.7 else corpus.person(rng), "addr": corpus.address(rng, loc)}
    seller["vat"] = corpus.vat_id(rng, seller["addr"]["cc"])
    buyer["vat"] = corpus.vat_id(rng, buyer["addr"]["cc"]) if rng.random() < 0.5 else ""
    issue = corpus.rand_date(rng, 2020, 2026)
    terms_n = rng.choice([7, 14, 30, 30, 45, 60])
    inv = {
        "seller": seller, "buyer": buyer, "issue": issue, "due": issue + timedelta(days=terms_n), "terms_n": terms_n,
        "no": rng.choice(["", "INV-", "RE", "F", "2026/", "R-"]) + str(rng.randint(1, 99999)).zfill(rng.choice([3, 4, 5])),
        "po": rng.choice(["PO", "B", "", "ORD-"]) + str(rng.randint(1000, 999999)) if rng.random() < 0.5 else "",
        "delivery": issue - timedelta(days=rng.randint(0, 20)) if rng.random() < 0.4 else None, "cur": cur, "sym": sym,
        "iban": corpus.iban(rng, seller["addr"]["cc"]), "bic": corpus.bic(rng, seller["addr"]["cc"][:2]),
    }  # fmt: skip
    rates = RATES[loc]
    items = []
    for k in range(n_items):
        service = rng.random() < 0.35
        unit = UNITS[loc][1] if service else UNITS[loc][0]
        qty = round(rng.choice([1, 1, 2, 3, 5, 10, 12, 25, 100]) if not service else rng.choice([1, 2, 4, 7.5, 8, 12.25, 16]), 2)
        price = rng.randint(150, 90000) if not service else rng.randint(4000, 15000)
        items.append({
            "id": str(k + 1), "name": corpus.item_name(rng, loc, "service" if service else "goods"), "qty": qty, "unit": unit,
            "code": "HUR" if service else "C62", "price": price, "rate": rng.choice(rates) if rng.random() < 0.6 else rates[-1],
            "net": round(qty * price),
        })  # fmt: skip
    inv["items"] = items
    sub = sum(i["net"] for i in items)
    br = {}
    for i in items:
        br[i["rate"]] = br.get(i["rate"], 0) + i["net"]
    inv["breakdown"] = [(r, t, round(t * r / 100)) for r, t in sorted(br.items())]
    inv["sub"] = sub
    inv["tax"] = sum(b[2] for b in inv["breakdown"])
    inv["total"] = sub + inv["tax"]
    return inv


def _qty_txt(q: float, loc: str) -> str:
    s = f"{q:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", "," if loc != "en" else ".")


def invoice_page(ctx: Ctx, seed: int, n_items: int, contract: bool = False) -> PageSpec:
    """An invoice page. With `contract` the meta block also carries a contract number, a field the shipped extraction model has no
    class for; it is drawn from its own random stream, so pages without it are unchanged."""
    rng = random.Random(seed)
    ctx.rng = rng
    li = rng.randrange(len(LOCALES))
    loc = LOCALES[li]
    ctx.lang = loc
    for _ in range(300):
        tpl = {
            "kind": "invoice", "head": rng.choice(["left", "right", "split", "band"]), "parties": rng.choice(["window", "stack", "side"]),
            "meta": rng.choice(["grid", "strip", "stack"]), "rules": rng.choice(["grid", "hlines", "none", "box", "header"]),
            "band": rng.random() < 0.35, "zebra": rng.random() < 0.25, "totals": rng.choice(["right", "box", "full"]),
            "bank": rng.choice(["footer", "body"]), "footer": rng.choice(["cols", "line", "none"]), "role": rng.choice(["sans", "sans", "serif"]),
            "pos": rng.random() < 0.5, "unit": rng.random() < 0.5, "vat": rng.random() < 0.6, "breakdown": rng.random() < 0.5,
            "title_case": rng.choice(["upper", "plain"]),
        }  # fmt: skip
        if template_split(tpl) == ("train" if ctx.split == "train" else "val"):
            break
    inv = _make_invoice(rng, ctx, loc, n_items)
    ctx.look = None  # type: ignore[attr-defined]
    body = pick(ctx.pool, rng, ctx.split, (tpl["role"],), set(SCRIPT_SAMPLES["latin"]) | set("äöüßéèàçñ£€"))
    head = body if rng.random() < 0.5 else pick(ctx.pool, rng, ctx.split, ("sans", "serif"), set(SCRIPT_SAMPLES["latin"]))
    w, h, _ = rng.choices([(794, 1123), (816, 1056)], [0.65, 0.35])[0] + (0,)
    px = rng.uniform(11.0, 13.8) if w > 700 else 12
    mx = rng.uniform(0.07, 0.11) * w
    ink, paper, acc = rng.choice(INKS), rng.choice(PAPERS), rng.choice(ACCENTS)
    look = Look(body, head, None, px, rng.uniform(1.25, 1.45), False, False, ink, paper, acc, (mx * 0.9, mx, mx * 0.7, mx), 1, 0, False, w, h, 1.0,
                True, False, False, "left")  # fmt: skip
    ctx.look = look  # type: ignore[attr-defined]
    s, b = inv["seller"], inv["buyer"]

    def addr(party: dict, pre: str, with_name=True, with_vat=True) -> str:
        a = party["addr"]
        lines = []
        if with_name:
            lines.append(fld(pre + "Name", party["name"]))
        lines.append(fld(pre + "Street", a["street"]))
        lines.append(fld(pre + "Postcode", a["postcode"]) + " " + fld(pre + "City", a["city"]))
        lines.append(fld(pre + "Country", a["cc"], a["country"]))
        if with_vat and party["vat"]:
            lines.append(kvpair(ctx, LAB["vat_id"][li] + ":", fld(pre + "VatId", party["vat"]), "kvi"))
        return "<br>".join(x if not x.startswith("<div") else x for x in lines)

    fmt = rng.randrange(4)
    D = lambda d: corpus.fmt_date(d, loc, fmt)  # noqa: E731
    meta = [
        (LAB["inv_no"][li], fld("invoiceNumber", inv["no"])),
        (LAB["date"][li], fld("issueDate", inv["issue"].isoformat(), D(inv["issue"]))),
        (LAB["due"][li], fld("dueDate", inv["due"].isoformat(), D(inv["due"]))),
    ]
    if inv["delivery"]:
        meta.append((LAB["delivery"][li], fld("deliveryDate", inv["delivery"].isoformat(), D(inv["delivery"]))))
    if inv["po"]:
        meta.append((LAB["po"][li], fld("poNumber", inv["po"])))
    if contract:
        crng = random.Random(seed * 7919 + 13)
        number = crng.choice(["CT-", "K", "AG/", "C", ""]) + str(crng.randint(100, 99999999)).zfill(crng.choice([4, 6]))
        pos = crng.randrange(len(meta) + 1)
        meta.insert(pos, (CONTRACT[li], fld("contractNumber", number)))
    cur_in_meta = rng.random() < 0.3
    if cur_in_meta:
        meta.append((LAB["currency"][li], fld("currency", inv["cur"])))
    sym_html = lambda: fld("currency", inv["cur"], inv["sym"]) if not cur_in_meta else esc(inv["sym"])  # noqa: E731
    msty = tpl["meta"]
    meta_html = "".join(kvpair(ctx, k + ":", v) for k, v in meta)
    meta_blk = tblk("body", "text", "Text", f"meta {msty}", meta_html)

    title_txt = LAB["invoice"][li].upper() if tpl["title_case"] == "upper" else LAB["invoice"][li]
    title = f'<div class="title" data-blk="body" data-reg="title" data-dl="Title">{inline(title_txt)}</div>'
    logo = '<div class="logo" data-reg="figure" data-dl="Picture" data-box></div>' if rng.random() < 0.6 else ""
    sname = tblk("body", "text", "Text", "sname", fld("sellerName", s["name"]))
    sender = (
        tblk("body", "text", "Text", "sender", inline(f"{s['name']} · {s['addr']['street']} · {s['addr']['postcode']} {s['addr']['city']}"))
        if tpl["parties"] == "window" and rng.random() < 0.6
        else ""
    )
    s_addr = tblk("body", "text", "Text", "party", addr(s, "seller", with_name=False))
    b_addr = tblk("body", "text", "Text", "party buyer", addr(b, "buyer"))
    bill_lbl = (
        f'<div class="small" data-blk="body" data-reg="text" data-dl="Text">{inline(LAB["bill_to"][li])}</div>'
        if rng.random() < 0.5
        else ""
    )

    hd = tpl["head"]
    if hd == "band":
        top = f'<div class="band"><div class="hl">{logo}{sname}</div>{title}</div>'
    elif hd == "split":
        top = f'<div class="top"><div class="hl">{logo}{sname}</div>{title}</div>'
    elif hd == "right":
        top = f'<div class="top r">{title}<div class="hl">{sname}{logo}</div></div>'
    else:
        top = f'<div class="top">{logo}<div>{sname}</div></div>' + title
    if tpl["parties"] == "window":
        mid = f'<div class="mid"><div>{sender}{bill_lbl}{b_addr}</div><div>{s_addr}{meta_blk}</div></div>'
    elif tpl["parties"] == "side":
        mid = f'<div class="mid"><div>{s_addr}</div><div>{bill_lbl}{b_addr}</div></div>{meta_blk}'
    else:
        mid = f"{s_addr}{bill_lbl}{b_addr}{meta_blk}"

    # item table
    cols = (
        (["pos"] if tpl["pos"] else [])
        + ["desc", "qty"]
        + (["unit"] if tpl["unit"] else [])
        + ["price"]
        + (["vat"] if tpl["vat"] else [])
        + ["amount"]
    )
    if rng.random() < 0.3 and "unit" not in cols:
        cols[cols.index("qty")], cols[cols.index("price")] = "price", "qty"
    rows = []
    hdr = "".join(
        f'<th data-cell data-r="0" data-c="{c}" data-rs="1" data-cs="1" style="text-align:{"left" if k in ("pos", "desc", "unit") else "right"}">{inline(LAB[k][li])}</th>'
        for c, k in enumerate(cols)
    )
    rows.append(f"<tr data-row>{hdr}</tr>")
    for r, it in enumerate(inv["items"]):
        tds = []
        for c, k in enumerate(cols):
            val = {
                "pos": fld("lineId", it["id"]),
                "desc": fld("itemName", it["name"]),
                "qty": fld("quantity", f"{it['qty']:g}", _qty_txt(it["qty"], loc)),
                "unit": fld("unitCode", it["code"], it["unit"]),
                "price": fld("unitPrice", f"{it['price'] / 100:.2f}", money(it["price"], loc)),
                "vat": fld("lineVatRate", f"{it['rate']:g}", f"{_qty_txt(it['rate'], loc)} %"),
                "amount": fld("lineNet", f"{it['net'] / 100:.2f}", money(it["net"], loc)),
            }[k]
            al = "left" if k in ("pos", "desc", "unit") else "right"
            tds.append(f'<td data-cell data-r="{r + 1}" data-c="{c}" data-rs="1" data-cs="1" style="text-align:{al}">{val}</td>')
        rows.append(f'<tr data-row data-li="{r}">{"".join(tds)}</tr>')
    table = f'<table class="items" data-blk="body" data-tbl data-reg="table" data-dl="Table" data-ncols="{len(cols)}" data-nrows="{len(rows)}">{"".join(rows)}</table>'

    # totals
    tot = [kvpair(ctx, LAB["subtotal"][li], fld("subtotal", f"{inv['sub'] / 100:.2f}", money(inv["sub"], loc)))]
    if tpl["breakdown"]:
        for rate, taxable, tax in inv["breakdown"]:
            g = ctx.next_kv()
            tot.append(
                f'<div class="kv"><span class="k" data-kv="{g}" data-kvr="key">{inline(LAB["vat"][li])} {fld("taxRate", f"{rate:g}", _qty_txt(rate, loc) + " %")}</span>'
                f'<span class="v"><span class="sub">{inline(LAB["taxable"][li])} {fld("taxableAmount", f"{taxable / 100:.2f}", money(taxable, loc))}</span> '
                f'<span data-kv="{g}" data-kvr="value">{fld("taxAmount", f"{tax / 100:.2f}", money(tax, loc))}</span></span></div>'
            )
    tot.append(kvpair(ctx, LAB["tax"][li], fld("taxTotal", f"{inv['tax'] / 100:.2f}", money(inv["tax"], loc))))
    pre = sym_html() + " " if loc == "en" else ""
    post = "" if loc == "en" else " " + sym_html()
    tot.append(kvpair(ctx, LAB["total"][li], pre + fld("total", f"{inv['total'] / 100:.2f}", money(inv["total"], loc)) + post, "kv grand"))
    if rng.random() < 0.6:
        tot.append(kvpair(ctx, LAB["due_amt"][li], fld("amountDue", f"{inv['total'] / 100:.2f}", money(inv["total"], loc)), "kv grand"))
    totals = tblk("body", "text", "Text", f"totals {tpl['totals']}", "".join(tot))

    terms_txt = TERMS[li].format(n=inv["terms_n"])
    terms = tblk("body", "text", "Text", "terms", kvpair(ctx, LAB["terms"][li] + ":", fld("paymentTerms", terms_txt)))
    bank = tblk(
        "body",
        "text",
        "Text",
        "bank",
        f"<b>{inline(LAB['bank'][li])}</b><br>"
        + kvpair(ctx, LAB["iban"][li] + ":", fld("iban", inv["iban"], corpus.group4(inv["iban"])))
        + kvpair(ctx, LAB["bic"][li] + ":", fld("bic", inv["bic"])),
    )
    thanks = tblk("body", "text", "Text", "thanks", inline(LAB["thanks"][li])) if rng.random() < 0.5 else ""
    foot = ""
    if tpl["footer"] == "cols":
        foot = "".join(
            f'<span data-blk="footer" data-reg="footer" data-dl="Page-footer">{inline(t)}</span>'
            for t in (s["name"], f"{LAB['tel'][li]} {corpus.phone(rng, loc)}", corpus.email(rng, s["name"].split()[0]))
        )
        foot = f'<div class="ft cols">{foot}</div>'
    elif tpl["footer"] == "line":
        foot = f'<div class="ft"><span data-blk="footer" data-reg="footer" data-dl="Page-footer">{inline(s["name"] + " · " + s["addr"]["street"] + " · " + s["addr"]["postcode"] + " " + s["addr"]["city"])}</span></div>'
    bank_body = bank if tpl["bank"] == "body" else ""
    foot_bank = ""
    if tpl["bank"] == "footer":
        foot_bank = bank.replace('class="bank"', 'class="bank fb"')
    # the footer bank block precedes the footer strip in reading order
    content = f"{top}{mid}{table}{totals}{terms}{bank_body}{thanks}"
    t_rule = {"grid": "border:1px solid", "box": "border:1px solid", "hlines": "border-bottom:1px solid", "header": "", "none": ""}[
        tpl["rules"]
    ]
    css = f"""
#page{{font-size:{px:.2f}px}}
.top,.band{{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.4em;gap:1em}}
.top.r{{flex-direction:row}}
.band{{background:{acc};color:#fff;padding:.8em 1em;margin:-{mx * 0.9:.0f}px -{mx:.0f}px 1.6em}}
.hl{{display:flex;align-items:center;gap:.7em}}
.logo{{width:{rng.randint(36, 70)}px;height:{rng.randint(30, 60)}px;background:linear-gradient(135deg,{acc},{rng.choice(ACCENTS)});border-radius:{rng.choice([0, 6, 50])}%}}
.sname{{font-family:'{head.name}';font-weight:700;font-size:{rng.uniform(1.2, 1.9):.2f}em}}
.title{{font-family:'{head.name}';font-weight:700;font-size:{rng.uniform(1.4, 2.4):.2f}em;color:{acc if hd != 'band' else '#fff'};margin-bottom:.5em;letter-spacing:{rng.choice([0, 0.05])}em}}
.top .title,.band .title{{margin:0}}
.mid{{display:grid;grid-template-columns:1fr 1fr;gap:1.2em;margin-bottom:1.2em}}
.party{{line-height:1.4;margin-bottom:.8em}} .small{{font-size:.8em;opacity:.8}} .sender{{font-size:.7em;text-decoration:underline;margin-bottom:.4em}}
.kv{{display:flex;gap:.6em}} .kvi{{display:block}} .kvi .k{{margin-right:.4em}}
.meta.grid .kv,.meta.stack .kv{{justify-content:space-between}} .meta.grid{{max-width:19em;margin-left:auto}}
.meta.strip{{display:flex;flex-wrap:wrap;gap:.5em 2em;border-top:1px solid {ink}88;border-bottom:1px solid {ink}88;padding:.5em 0;margin-bottom:1em}}
.meta .k{{opacity:.75}}
table.items{{width:100%;border-collapse:collapse;margin:.6em 0 1em}}
table.items td,table.items th{{padding:.4em .5em;{t_rule};vertical-align:top}}
table.items th{{{("background:" + acc + ";color:#fff;") if tpl["band"] else ("border-bottom:1.5px solid " + ink)};font-weight:700;{"border:none;" if tpl["band"] else ""}}}
{"table.items tr:nth-child(even) td{background:" + ink + "12}" if tpl["zebra"] else ""}
.totals{{margin-bottom:1em}} .totals .kv{{justify-content:space-between;padding:.15em 0}}
.totals.right{{margin-left:auto;width:{rng.randint(38, 55)}%}} .totals.box{{margin-left:auto;width:48%;border:1px solid {ink};padding:.5em .7em}}
.totals.full{{border-top:1px solid {ink};padding-top:.4em}}
.grand{{font-weight:700;font-size:1.1em}} .sub{{opacity:.75;margin-right:1em}}
.terms,.bank,.thanks{{margin-bottom:.8em}} .bank .kv{{gap:.4em}}
.ft{{margin-top:auto;border-top:1px solid {ink}88;padding-top:.5em;font-size:.78em;display:flex;gap:1.5em;justify-content:{rng.choice(['space-between', 'center'])}}}
.fb{{margin-top:auto;font-size:.85em}}
#flow{{overflow:visible !important}}
"""  # fmt: skip
    html = frame(look, ctx.root, css, "", content, foot_bank, foot, loc)
    return PageSpec(html, "invoice", loc, tpl, sorted({body.name, head.name}), w, h, "retry", {"n_items": n_items, "currency": inv["cur"]})


# --- receipts ---------------------------------------------------------------------------------------------------------------

R_LAB = {
    "subtotal": _l("Subtotal|Zwischensumme|Sous-total|Subtotal|Subtotale|Subtotaal"), "tax": _l("Tax|MwSt.|TVA|IVA|IVA|BTW"),
    "tip": _l("Tip|Trinkgeld|Pourboire|Propina|Mancia|Fooi"), "total": _l("TOTAL|SUMME|TOTAL|TOTAL|TOTALE|TOTAAL"),
    "paid": _l("Paid by|Bezahlt mit|Payé par|Pagado con|Pagato con|Betaald met"),
    "cash": _l("Cash|Bar|Espèces|Efectivo|Contanti|Contant"), "card": _l("Card|Karte|Carte|Tarjeta|Carta|Pinpas"),
    "thanks": _l("Thank you!|Danke für Ihren Einkauf!|Merci de votre visite !|¡Gracias por su visita!|Grazie e arrivederci!|Bedankt voor uw bezoek!"),
    "receipt": _l("Receipt|Beleg|Ticket|Recibo|Scontrino|Bon"), "date": _l("Date|Datum|Date|Fecha|Data|Datum"), "time": _l("Time|Uhrzeit|Heure|Hora|Ora|Tijd"),
    "cashier": _l("Cashier|Kassierer|Caissier|Cajero|Cassiere|Kassier"),
}  # fmt: skip


def rfld(name: str, val: str, text: str | None = None, extra: str = "") -> str:
    return f'<span data-fld="receipt.{name}" data-fname="{name}" data-val="{esc(val)}"{extra}>{inline(text if text is not None else val)}</span>'


def receipt_page(ctx: Ctx, seed: int) -> PageSpec:
    rng = random.Random(seed)
    ctx.rng = rng
    li = rng.randrange(len(LOCALES))
    loc = LOCALES[li]
    ctx.lang = loc
    for _ in range(300):
        tpl = {
            "kind": "receipt", "align": rng.choice(["center", "left"]), "item": rng.choice(["dots", "plain", "grid"]), "sep": rng.choice(["dash", "star", "rule", "none"]),
            "role": rng.choice(["mono", "mono", "sans"]), "qtyfmt": rng.choice(["x", "pre", "none"]), "canvas": rng.choice(["tight", "page", "page"]),
            "barcode": rng.random() < 0.5, "taxcode": rng.random() < 0.3,
        }  # fmt: skip
        if template_split(tpl) == ("train" if ctx.split == "train" else "val"):
            break
    cur, sym = CUR[loc]
    pool_roles = ("mono",) if tpl["role"] == "mono" else ("sans",)
    body = pick(ctx.pool, rng, ctx.split, pool_roles, set(SCRIPT_SAMPLES["latin"]) | set("äöüß€£éèàç"))
    px = rng.uniform(11.5, 15.0)
    ink = rng.choice(["#000000", "#111111", "#222222", "#2a2a2a", "#333333"])
    paper = rng.choice(["#ffffff", "#fcfcfa", "#f6f4ee", "#efefef"])
    cw = int(rng.uniform(19, 34) * px * 0.62 + 20)  # receipt width in css px: 19-34 characters
    merchant = (
        corpus.company(rng, loc)
        if rng.random() < 0.6
        else corpus.city(rng, loc) + " " + rng.choice(["Café", "Market", "Bistro", "Kiosk", "Bakery"])
    )
    addr = corpus.address(rng, loc)
    kind = rng.choice(["food", "food", "goods"])
    n = rng.randint(2, 22)
    items = []
    for _ in range(n):
        q = rng.choice([1, 1, 1, 2, 3])
        p = rng.randint(80, 4500)
        items.append({"name": corpus.item_name(rng, loc, kind), "qty": q, "price": p, "amount": q * p, "rate": rng.choice(RATES[loc])})
    sub = sum(i["amount"] for i in items)
    rate = items[0]["rate"]
    tax = round(sub * rate / (100 + rate))  # prices include tax, as on a till receipt
    tip = rng.choice([0, 0, 0, round(sub * 0.1), 100, 200]) if kind == "food" else 0
    total = sub + tip
    pay = rng.choice(["cash", "card", "card"])
    d = corpus.rand_date(rng, 2022, 2026)
    tm = f"{rng.randint(7, 22):02d}:{rng.randint(0, 59):02d}"
    sepch = {"dash": "dashed", "star": "dotted", "rule": "solid", "none": None}[tpl["sep"]]
    # a run of dashes or stars would be a word of its own, so separators are drawn as borders
    sep = f'<div class="sep" style="border-top:1px {sepch} {ink}"></div>' if sepch else ""

    def kv2(label: str, v: str, cls="kv") -> str:
        return kvpair(ctx, label, v, cls)

    head = tblk("body", "text", "Text", "m", f"<b>{fld_merchant(merchant)}</b>", "") + tblk(
        "body",
        "text",
        "Text",
        "a",
        inline(addr["street"]) + "<br>" + inline(f"{addr['postcode']} {addr['city']}") + "<br>" + inline(corpus.phone(rng, loc)),
    )
    dt = tblk(
        "body",
        "text",
        "Text",
        "dt",
        kv2(LAB["date"][li] + ":", rfld("date", d.isoformat(), corpus.fmt_date(d, loc, 0)))
        + kv2(R_LAB["time"][li] + ":", rfld("time", tm)),
    )
    rows = []
    for k, it in enumerate(items):
        qtxt = "" if tpl["qtyfmt"] == "none" or it["qty"] == 1 and rng.random() < 0.5 else f"{it['qty']}x"
        qpre = f"{rfld('quantity', str(it['qty']), qtxt)} " if qtxt else ""
        name = rfld("itemName", it["name"])
        code = f' <span class="tc">{rng.choice("AB")}</span>' if tpl["taxcode"] else ""
        unit = rfld("unitPrice", f"{it['price'] / 100:.2f}", money(it["price"], loc)) if it["qty"] > 1 else ""
        unit_html = f'<span class="up">{unit}</span>' if unit else ""
        rows.append(
            f'<div class="it {tpl["item"]}" data-blk="body" data-reg="list" data-dl="List-item" data-li="{k}"><span class="n">{qpre}{name}</span>{unit_html}'
            f'<span class="p">{rfld("amount", f"{it['amount'] / 100:.2f}", money(it["amount"], loc))}{code}</span></div>'
        )
    items_blk = "".join(rows)
    pre = f"{sym} " if loc == "en" else ""
    post = "" if loc == "en" else f" {sym}"
    tots = kv2(R_LAB["subtotal"][li], rfld("subtotal", f"{sub / 100:.2f}", money(sub, loc)))
    tots += kv2(f"{R_LAB['tax'][li]} {_qty_txt(rate, loc)} %", rfld("tax", f"{tax / 100:.2f}", money(tax, loc)))
    if tip:
        tots += kv2(R_LAB["tip"][li], rfld("tip", f"{tip / 100:.2f}", money(tip, loc)))
    tots += kv2(R_LAB["total"][li], pre + rfld("total", f"{total / 100:.2f}", money(total, loc)) + post, "kv big")
    paid_txt = R_LAB[pay][li] + (f" **** {rng.randint(1000, 9999)}" if pay == "card" else "")
    tots += kv2(R_LAB["paid"][li], rfld("paymentMethod", pay, paid_txt))
    totals = tblk("body", "text", "Text", "tot", tots)
    thanks = tblk("body", "text", "Text", "th", inline(R_LAB["thanks"][li]))
    bars = ""
    if tpl["barcode"]:
        stripes = ",".join(f"#000 {x}px,#000 {x + w}px,transparent {x + w}px,transparent {x + w + g}px" for x, w, g in _bar(rng, cw - 60))
        bars = f'<div class="bc" data-reg="figure" data-dl="Picture" data-box style="background:linear-gradient(90deg,{stripes})"></div>'
    pad = rng.randint(8, 22)
    css = f"""
html,body{{background:#fff}}
#page{{font-family:'{body.name}',monospace;font-size:{px:.2f}px;line-height:{rng.uniform(1.2, 1.4):.2f};display:block;width:fit-content}}
.canvas{{display:flex;justify-content:{rng.choice(['center', 'flex-start', 'flex-end'])};align-items:flex-start;padding:{pad if tpl["canvas"] == "tight" else rng.randint(30, 120)}px;
 background:{rng.choice(['#ffffff', '#e8e8e8', '#d9d4c9', '#cfd8dc', '#3a3a3a']) if tpl["canvas"] == "page" else paper};
 {"min-height:" + str(rng.choice([700, 1000, 1123])) + "px;width:" + str(rng.choice([794, 816, 900])) + "px;" if tpl["canvas"] == "page" else ""}}}
.rc{{width:{cw}px;background:{paper};color:{ink};padding:{rng.randint(10, 22)}px {rng.randint(8, 16)}px {rng.randint(14, 30)}px;text-align:{tpl["align"] if tpl["align"] == "center" else "left"}}}
.rc .m{{font-size:1.25em;margin-bottom:.3em;text-align:center}} .rc .a,.rc .dt{{text-align:center;margin-bottom:.5em}} .rc .dt .kv{{justify-content:center}}
.kv{{display:flex;gap:.6em}} .rc .tot .kv,.rc .it{{justify-content:space-between}} .it{{display:flex;gap:.5em;text-align:left;margin:.1em 0}}
.it.dots .n{{flex:1;border-bottom:1px dotted {ink}88;padding-bottom:1px}} .it.grid{{border-bottom:1px solid {ink}33}} .it .n{{flex:1}}
.it .up{{opacity:.8;font-size:.9em}} .tc{{margin-left:.5em}} .big{{font-size:1.3em;font-weight:700;margin:.3em 0}}
.sep{{margin:.5em 0;min-height:1px}} .th{{text-align:center;margin-top:.8em}} .tot{{margin-top:.4em}} .bc{{height:{rng.randint(32, 56)}px;margin:.8em auto 0;width:{cw - 60}px}}
.k{{text-align:left}}
"""  # fmt: skip
    ff = body.css(ctx.root)
    html = (
        f'<!doctype html><html lang="{loc}"><head><meta charset="utf-8"><style>{ff}{css}</style></head><body><div id="page"><div class="canvas">'
        f'<div class="rc">{head}{sep}{dt}{sep}{items_blk}{sep}{totals}{sep}{thanks}{bars}</div></div></div></body></html>'
    )
    return PageSpec(html, "receipt", loc, tpl, [body.name], 0, 0, "none", {"currency": cur})


def fld_merchant(name: str) -> str:
    return rfld("merchant", name)


def _bar(rng: random.Random, width: int):
    x = 0
    while x < width:
        w, g = rng.choice([1, 1, 2, 3]), rng.choice([1, 2, 2, 3])
        yield x, w, g
        x += w + g
