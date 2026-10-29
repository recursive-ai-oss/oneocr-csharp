# ruff: noqa: C408, UP031
"""screenshot-v1: UI-like and web-like screens rendered by Chromium at 1x and 2x device scale.

Each screen is a random composition of an app bar, an optional sidebar and one content template (form, table,
settings list, dialog, dashboard, web article) in a light, dark or tinted theme, set in one of the held-out
families. Truth comes from DOM ranges (see render.py). Reading order is document order, which the templates
author as: app bar left to right, sidebar top to bottom, then the content top to bottom, left to right
(a form reads label, value, label, value row by row; a table reads row by row). A screen whose text does not
fit the viewport is rejected and redrawn, so no printed word is missing from the truth.
"""

from __future__ import annotations

import html
import random

from .assets import FAMILIES
from .render import WORDS_JS, add_ink, font_css, text_of, truth_words, wait_fonts

THEMES = {
    "light": dict(bg="#ffffff", fg="#1b1b1f", mut="#5b606b", card="#f4f5f7", line="#cfd3da", acc="#1a5fd0", accfg="#ffffff", bar="#f0f2f5"),
    "dark": dict(bg="#1c1d22", fg="#e6e8ec", mut="#9aa0aa", card="#26282f", line="#41444d", acc="#7aa7ff", accfg="#0c1630", bar="#15161a"),
    "sepia": dict(bg="#fbf5e6", fg="#3a352b", mut="#6f6756", card="#f2e9d2", line="#d8ccae", acc="#8a4b12", accfg="#ffffff", bar="#efe4c8"),
    "blue": dict(bg="#eef3fb", fg="#14213d", mut="#4a5a78", card="#ffffff", line="#c3d0e6", acc="#0b3d91", accfg="#ffffff", bar="#0b3d91"),
}
VIEWPORTS = [(1280, 800), (1024, 768), (1440, 900), (900, 700), (1366, 768)]

NAV = ["Home", "Overview", "Projects", "Reports", "Messages", "Calendar", "Billing", "Help", "Settings", "Team",
       "Orders", "Customers", "Inventory", "Analytics", "Support", "Documents"]
SIDE = ["Dashboard", "Activity", "Invoices", "Payments", "Subscriptions", "Integrations", "Security", "Audit log",
        "Notifications", "Profile", "Shared with me", "Archive", "Trash", "API keys", "Webhooks"]
FIRST = ["Maria", "Jonas", "Aisha", "Tomas", "Chen", "Olga", "Priya", "Lucas", "Fatima", "Dmitri", "Elena", "Samuel",
         "Ingrid", "Mateo", "Noor", "Kofi"]
LAST = ["Alvarez", "Berg", "Okafor", "Novak", "Wei", "Petrova", "Nair", "Silva", "Haddad", "Ivanov", "Costa", "Osei",
        "Lund", "Reyes", "Khan", "Mensah"]
CITY = ["Lisbon", "Oslo", "Kraków", "Lagos", "Toronto", "Osaka", "Denver", "Valencia", "Rotterdam", "Nairobi"]
STATUS = ["Active", "Pending", "Paid", "Overdue", "Draft", "Shipped", "Failed", "Refunded", "Archived", "In review"]
FIELD = ["Full name", "Email address", "Phone number", "Company", "Street address", "City", "Postal code", "Country",
         "Job title", "Account number", "Tax ID", "Reference", "Department", "Start date", "Billing contact"]
CHECK = ["Email me product updates", "Remember this device", "Send a copy to my address", "Enable two-factor sign-in",
         "I agree to the terms of service", "Show archived items", "Notify me about new comments"]
SETTING = [("Dark appearance", "Use a darker colour scheme in all windows."), ("Auto-save drafts", "Keep a copy every 30 seconds while typing."),
           ("Weekly summary", "A short report is sent on Monday mornings."), ("Sync over mobile data", "Large files wait for Wi-Fi."),
           ("Session timeout", "Sign out after a period of inactivity."), ("Language", "Choose the language of menus and messages."),
           ("Download location", "Files are saved in this folder by default."), ("Spell check", "Underline words that are not in the dictionary."),
           ("Compact view", "Show more rows on each screen."), ("Backup schedule", "Nightly copies are kept for thirty days.")]
BUTTON = ["Save changes", "Cancel", "Continue", "Apply", "Delete", "Send", "Export", "Create account", "Back", "Next step"]


def name(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def email(rng):
    return f"{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower()}@{rng.choice(['example.com', 'mail.example.org', 'corp.example.net'])}"


def amount(rng):
    return f"{rng.choice(['$', 'EUR ', '£'])}{rng.uniform(3, 9999):,.2f}"


def date(rng):
    return rng.choice([f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2019, 2026)}",
                       f"{rng.randint(2019, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                       f"{rng.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])} {rng.randint(1, 28)}, {rng.randint(2019, 2026)}"])


def ident(rng):
    return f"{rng.choice(['INV', 'ORD', 'TKT', 'ACC'])}-{rng.randint(1000, 999999)}"


def value_for(rng, label):
    k = label.lower()
    if "email" in k or "contact" in k:
        return email(rng)
    if "name" in k:
        return name(rng)
    if "phone" in k:
        return f"+{rng.randint(1, 99)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"
    if "date" in k:
        return date(rng)
    if "city" in k:
        return rng.choice(CITY)
    if "postal" in k:
        return f"{rng.randint(10000, 99999)}"
    if "street" in k:
        return f"{rng.randint(1, 240)} {rng.choice(LAST)} Street"
    return ident(rng)


def e(t):
    return html.escape(t)


def t_form(rng, p):
    n = rng.randint(4, 8)
    labels = rng.sample(FIELD, n)
    cells = "".join(f'<div class="fld"><div class="lb">{e(l)}</div><div class="in">{e(value_for(rng, l))}</div></div>' for l in labels)
    checks = "".join(f'<div class="chk"><span class="box"></span><span>{e(c)}</span></div>' for c in rng.sample(CHECK, rng.randint(1, 3)))
    b = rng.sample(BUTTON, 2)
    return (f'<h1>{e(rng.choice(["Account details", "Edit contact", "Billing information", "New customer", "Shipping address"]))}</h1>'
            f'<p class="mut">{e("Fields marked in the form are required before you can continue.")}</p>'
            f'<div class="card"><div class="grid">{cells}</div>{checks}'
            f'<div class="btns"><span class="btn pri">{e(b[0])}</span><span class="btn">{e(b[1])}</span></div></div>')


def t_table(rng, p):
    cols = rng.sample(["Reference", "Customer", "Date", "Amount", "Status", "Owner", "City"], rng.randint(4, 6))
    rows = rng.randint(6, 11)

    def cell(c):
        return {"Reference": ident(rng), "Customer": name(rng), "Date": date(rng), "Amount": amount(rng),
                "Status": rng.choice(STATUS), "Owner": name(rng), "City": rng.choice(CITY)}[c]
    head = "".join(f"<th>{e(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{e(cell(c))}</td>" for c in cols) + "</tr>" for _ in range(rows))
    return (f'<div class="bar2"><h1>{e(rng.choice(["Invoices", "Orders", "Customers", "Transactions", "Tickets"]))}</h1>'
            f'<span class="in sm">Search by reference or name</span><span class="btn pri">New entry</span></div>'
            f'<table><tr>{head}</tr>{body}</table>'
            f'<p class="mut">Showing 1 to {rows} of {rng.randint(rows, 900)} results</p>')


def t_settings(rng, p):
    items = rng.sample(SETTING, rng.randint(5, 8))
    rows = "".join(f'<div class="set"><div><div class="st">{e(a)}</div><div class="mut">{e(b)}</div></div>'
                   f'<div class="val">{e(rng.choice(["On", "Off", "Every day", "English", "Default", "30 minutes", "Enabled"]))}</div></div>'
                   for a, b in items)
    return f'<h1>{e(rng.choice(["Preferences", "Settings", "Notifications", "Privacy"]))}</h1><div class="card">{rows}</div>'


def t_dialog(rng, p):
    msg = rng.choice(["This action cannot be undone. All files in the folder will be removed after thirty days.",
                      "Your session will expire in five minutes. Save your work to avoid losing recent changes.",
                      "A newer version of this document exists. Do you want to replace your copy with it?",
                      "The upload finished with warnings. Review the list below before you publish the report."])
    b = rng.sample(BUTTON, 2)
    return (f'<div class="dlgwrap"><div class="dlg"><h2>{e(rng.choice(["Delete folder?", "Session expiring", "Replace file?", "Upload complete"]))}</h2>'
            f'<p>{e(msg)}</p><div class="chk"><span class="box"></span><span>{e(rng.choice(CHECK))}</span></div>'
            f'<div class="btns r"><span class="btn">{e(b[0])}</span><span class="btn pri">{e(b[1])}</span></div></div></div>')


def t_dash(rng, p):
    stats = "".join(f'<div class="card stat"><div class="mut">{e(l)}</div><div class="big">{e(v)}</div><div class="mut">{e(d)}</div></div>'
                    for l, v, d in [("Revenue", amount(rng), "up 4.2% on last month"), ("Orders", f"{rng.randint(100, 9999):,}", "since Monday"),
                                    ("New users", f"{rng.randint(10, 999)}", "last 7 days"), ("Refunds", amount(rng), "3 open requests")][: rng.randint(3, 4)])
    acts = "".join(f'<div class="set"><div>{e(name(rng))} {e(rng.choice(["updated", "created", "approved", "commented on"]))} {e(ident(rng))}</div>'
                   f'<div class="mut">{rng.randint(1, 23)}:{rng.randint(10, 59)}</div></div>' for _ in range(rng.randint(4, 7)))
    return f'<h1>{e(rng.choice(["Dashboard", "Overview", "Team activity"]))}</h1><div class="stats">{stats}</div><h2>Recent activity</h2><div class="card">{acts}</div>'


def t_article(rng, p):
    paras = "".join(f"<p>{e(' '.join(rng.choice(p['paras'])[:2]))}</p>" for _ in range(rng.randint(2, 3)))
    rel = "".join(f'<div class="rel">{e(rng.choice(["How to", "Why", "Ten ways to", "A guide to"]))} {e(rng.choice(["plan a budget", "read a timetable", "write clear notes", "store receipts", "pack for a trip"]))}</div>' for _ in range(4))
    return (f'<div class="art"><div><h1>{e(rng.choice(["The quiet end of the season", "A short history of the harbour", "Notes from the road"]))}</h1>'
            f'<p class="mut">By {e(name(rng))} · {e(date(rng))} · {rng.randint(3, 12)} min read</p>{paras}</div>'
            f'<div class="side2"><h2>Related</h2>{rel}</div></div>')


TEMPLATES = [t_form, t_table, t_settings, t_dialog, t_dash, t_article]

CSS = """
*{box-sizing:border-box} html,body{margin:0;overflow:hidden}
body{background:var(--bg);color:var(--fg);font-family:'%(fam)s';font-size:%(fs)spx;line-height:1.4}
#app{width:%(w)spx;height:%(h)spx;overflow:hidden;position:relative}
.top{display:flex;align-items:center;gap:22px;padding:12px 20px;background:var(--bar);border-bottom:1px solid var(--line)}
.top .logo{font-weight:700;font-size:1.2em;margin-right:12px;color:var(--acc)} .top .usr{margin-left:auto}
.blue .top .logo,.blue .top{color:#fff} .blue .top .logo{color:#fff}
.main{display:flex}.side{width:190px;padding:16px 14px;border-right:1px solid var(--line);%(sidemin)s}
.side div{padding:7px 10px;border-radius:6px}.side div.on{background:var(--card);font-weight:700}
.content{flex:1;padding:22px 28px}
h1{font-size:1.7em;margin:0 0 .4em}h2{font-size:1.25em;margin:.8em 0 .4em}p{margin:0 0 .8em}
.mut{color:var(--mut);font-size:.92em}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px 18px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px 24px;margin-bottom:14px}.lb{color:var(--mut);font-size:.88em;margin-bottom:3px}
.in{border:1px solid var(--line);background:var(--bg);border-radius:5px;padding:7px 10px}.in.sm{display:inline-block;color:var(--mut);margin:0 14px}
.chk{display:flex;align-items:center;gap:9px;margin:8px 0}.box{display:inline-block;width:15px;height:15px;border:2px solid var(--mut);border-radius:3px}
.btns{display:flex;gap:10px;margin-top:16px}.btns.r{justify-content:flex-end}
.btn{display:inline-block;padding:8px 16px;border-radius:6px;border:1px solid var(--line);font-weight:600}
.btn.pri{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
table{border-collapse:collapse;width:100%%;margin:10px 0}th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}
th{background:var(--card);font-size:.9em}.bar2{display:flex;align-items:center}.bar2 h1{margin:0}.bar2 .btn{margin-left:auto}
.set{display:flex;justify-content:space-between;align-items:center;padding:11px 2px;border-bottom:1px solid var(--line)}
.st{font-weight:600}.val{color:var(--acc);font-weight:600}
.dlgwrap{position:absolute;left:0;top:0;width:100%%;height:100%%;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.45)}
.dlg{width:430px;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:22px}
.stats{display:flex;gap:16px;margin-bottom:12px}.stat{flex:1}.big{font-size:1.9em;font-weight:700}
.art{display:flex;gap:36px}.art>div:first-child{flex:1}.side2{width:230px}.rel{padding:8px 0;border-bottom:1px solid var(--line)}
"""


def screen_html(rng: random.Random, fam: str, w: int, h: int, paras, fit_sidebar: bool = False) -> tuple[str, dict]:
    """`fit_sidebar` makes the sidebar column as tall as the space under the top bar. The frozen sets were generated
    with the sidebar as tall as the whole viewport, which always overflowed and was rejected, so they hold no sidebar
    pages; the default keeps their generation unchanged."""
    theme = rng.choice(list(THEMES))
    th = THEMES[theme]
    fs = rng.choice([12, 13, 14, 15, 16])
    tmpl = rng.choice(TEMPLATES)
    p = {"paras": paras}
    nav = rng.sample(NAV, rng.randint(4, 7))
    top = (f'<div class="top"><span class="logo">{e(rng.choice(["Northwind", "Brightdesk", "Fieldnote", "Lumen", "Harbor"]))}</span>'
           + "".join(f"<span>{e(n)}</span>" for n in nav) + f'<span class="usr">{e(name(rng))}</span></div>')
    body = tmpl(rng, p)
    side = ""
    if tmpl is not t_dialog and rng.random() < 0.55:
        items = rng.sample(SIDE, rng.randint(5, 8))
        side = '<div class="side">' + "".join(f'<div class="{"on" if i == 0 else ""}">{e(s)}</div>' for i, s in enumerate(items)) + "</div>"
    vars_ = ";".join(f"--{k}:{v}" for k, v in th.items())
    doc = (f'<!doctype html><html><head><meta charset="utf-8"><style>{font_css(fam)}'
           f'{CSS % dict(fam=fam, fs=fs, w=w, h=h, sidemin=(f'min-height:{h - 72}px' if fit_sidebar else f'min-height:{h}px'))}</style></head>'
           f'<body class="{theme}" style="{vars_}"><div id="app">{top}<div class="main">{side}<div class="content">{body}</div></div></div></body></html>')
    return doc, {"theme": theme, "template": tmpl.__name__[2:], "font_px": fs, "sidebar": bool(side)}


def render(count: int, seed: int, out, set_name: str, paras, fit_sidebar: bool = False) -> list[dict]:
    from playwright.sync_api import sync_playwright

    from .render import write_page

    fams = list(FAMILIES)
    entries = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for i in range(count):
            pid = f"p{i + 1:03d}"
            fam = fams[i % len(fams)]
            scale = 1 if i % 2 == 0 else 2
            for attempt in range(40):
                rng = random.Random(f"{set_name}:{seed}:{i}:{attempt}")
                w, h = rng.choice(VIEWPORTS)
                doc, info = screen_html(rng, fam, w, h, paras, fit_sidebar)
                ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=scale)
                page = ctx.new_page()
                src = out / "page.html"
                out.mkdir(parents=True, exist_ok=True)
                src.write_text(doc, encoding="utf-8")
                page.goto(src.as_uri())
                wait_fonts(page, fam)
                words = page.evaluate(WORDS_JS, "#app")
                ok = bool(words) and all(0 <= q["x0"] and q["x1"] <= w and 0 <= q["y0"] and q["y1"] <= h for q in words)
                if ok:
                    # content taller than the viewport is clipped by overflow:hidden even if inside it
                    ok = page.evaluate("document.querySelector('#app').scrollHeight <= %d && document.querySelector('#app').scrollWidth <= %d" % (h, w))
                if not ok:
                    ctx.close()
                    continue
                png = page.screenshot(clip={"x": 0, "y": 0, "width": w, "height": h})
                ctx.close()
                break
            else:
                raise RuntimeError(f"{pid}: no screen fit after 40 attempts")
            import io

            from PIL import Image
            img = Image.open(io.BytesIO(png)).convert("RGB")
            tw = truth_words(words, scale)
            add_ink(tw, img)
            truth = {"id": pid, "set": set_name, "seed": seed, "page_index": i, "font": fam, "device_scale": scale,
                     "viewport": [w, h], "attempt": attempt, **info,
                     "image": {"width": img.size[0], "height": img.size[1]}, "words": tw, "text": text_of(tw)}
            from .render import png_bytes
            e_ = write_page(out, pid, png_bytes(img), "png", truth)
            e_.update(font=fam, scale=scale, template=info["template"], theme=info["theme"])
            entries.append(e_)
        version = browser.version
        browser.close()
    return entries, version
