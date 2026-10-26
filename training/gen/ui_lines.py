# ruff: noqa: E501
"""Interface and web text lines rendered by headless Chromium, with the label taken from the DOM.

Why: the line generator draws with OneImage.Text, but the text the library meets on screens and web pages is drawn by a browser:
its own anti-aliasing, sub-pixel placement at fractional device-pixel ratios, CSS colours and weights. This generator
renders rows of interface text in a page, screenshots the page, and cuts one crop per row the way the page reader's line finder
does (ink box plus margins), scaled to the 32 px recognizer height.

Truth: every label is the string the generator wrote into a row's cells; the DOM is read back (cell texts, element boxes) and a row
is kept only when the browser laid it out as one unwrapped line that reads back as the label. Nothing is read from an image by a
model. The crop box uses the row's own pixels (ink against the known background), the same thing the library's finder does.

    uv run python -m gen.ui_lines --config configs/ui-lines-v1.yaml --split train --n 20000 --seed 85 --out OUT --workers 8
    uv run python -m gen.ui_lines ... --sheet SHEET.png      # a contact sheet of the first crops, label under each

Sample `i` depends only on (seed, split, shard of i): the same seed gives the same labels, parameters and pixels whatever the number
of workers. Needs ONEOCR_FONT_ROOT (data/fonts.py fetch for fonts.yaml and fonts_ui.yaml) and ONEOCR_TEXT_ROOT (data/texts.py), and a
Chromium for Playwright (PLAYWRIGHT_BROWSERS_PATH, or ONEOCR_CHROMIUM naming a Chromium binary). Output: OUT/<split>/images/<id>.png (8-bit gray, 32 px high), labels.jsonl (the
LineSet format plus the parameters of each row), run.json.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import multiprocessing as mp
import os
import platform
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from data import charset as cs
from gen.plan import Books, Sampler, figures_line, gap_cells, numtext_line, sep_cells, tight_cells, ui_cells, uiword_cells
from pages.fontpool import Family, FontFile, _describe

BASE = Path(__file__).resolve().parents[1]
OUT_H = 32
SPLIT_ID = {"train": 0, "val": 1}
# display, script and decorative faces: pages and screens do not set running interface text in them
NOT_UI = {"pacifico", "lobster", "russoone", "anton", "specialelite", "amaticsc", "ruslandisplay", "bangers", "abrilfatface",
          "yanonekaffeesatz", "comfortaa", "tenorsans", "didactgothic", "cormorantgaramond", "oswald"}  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------- fonts


def font_root() -> Path:
    return Path(os.environ["ONEOCR_FONT_ROOT"])


def load_pool(split: str) -> dict[str, Family]:
    """Families of fonts.yaml and fonts_ui.yaml for one split, with their files described by fontTools (cached per process)."""
    root = font_root()
    fams: dict[str, Family] = {}
    for name in ("fonts.yaml", "fonts_ui.yaml"):
        for e in yaml.safe_load((BASE / "data" / name).read_text(encoding="utf-8"))["fonts"]:
            if e["split"] != split or e["family"] in NOT_UI:
                continue
            d = _describe(root, e["file"])
            fam = fams.setdefault(e["family"], Family(e["family"], "sans", split, []))
            fam.files.append(FontFile(d["path"], d["weight"], d["italic"], frozenset(d["cps"])))
    for fam in fams.values():
        fam.files.sort(key=lambda f: (f.italic, f.weight, f.path))
    return fams


def weights_of(fam: Family) -> list[int]:
    out: set[int] = set()
    for f in fam.files:
        if " " in f.weight:
            lo, hi = (int(x) for x in f.weight.split())
            out |= {w for w in (300, 400, 500, 600, 700) if lo <= w <= hi}
        else:
            out.add(int(f.weight))
    return sorted(out)


def family_chars(fam: Family) -> frozenset[int]:
    """Code points every file of the family can draw, so any weight or style of it can set a row."""
    cps = None
    for f in fam.files:
        cps = f.cps if cps is None else cps & f.cps
    return cps or frozenset()


# --------------------------------------------------------------------------------------------------------------- colours


def _lin(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb) -> float:
    r, g, b = (_lin(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def hsl(h: float, s: float, light: float) -> tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hls_to_rgb(h % 1.0, light, s)
    return round(r * 255), round(g * 255), round(b * 255)


def blend(a, b, t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def pick_colors(rng: np.random.Generator, cfg: dict) -> tuple[tuple, tuple, str]:
    """Background and ink for a row: light, dark, mid-tone or brand-colour surface; ink contrast log-uniform over the configured range."""
    theme = str(rng.choice(list(cfg["themes"]), p=np.array(list(cfg["themes"].values()), dtype=float) / sum(cfg["themes"].values())))
    if theme == "light":
        bg = hsl(rng.random(), rng.uniform(0, 0.25) * (rng.random() < 0.4), rng.uniform(0.9, 1.0))
    elif theme == "dark":
        bg = hsl(rng.random(), rng.uniform(0, 0.35) * (rng.random() < 0.5), rng.uniform(0.04, 0.2))
    elif theme == "mid":
        bg = hsl(rng.random(), rng.uniform(0.1, 0.6), rng.uniform(0.3, 0.7))
    else:  # brand: saturated button or banner
        bg = hsl(rng.random(), rng.uniform(0.5, 0.9), rng.uniform(0.3, 0.55))
    lo, hi = cfg["contrast"]
    target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    away = (255, 255, 255) if luminance(bg) < 0.18 else (0, 0, 0)
    if theme == "brand" and rng.random() < 0.7:
        away = (255, 255, 255) if luminance(bg) < 0.4 else (0, 0, 0)
    if rng.random() < cfg["tint_ink"]:  # coloured ink: links, errors, muted labels
        away = blend(away, hsl(rng.random(), 0.8, 0.5), float(rng.uniform(0.3, 0.8)))
    best = away
    t_lo, t_hi = 0.0, 1.0
    for _ in range(14):  # bisection on the blend from bg towards the far colour until the contrast matches
        t = (t_lo + t_hi) / 2
        best = blend(bg, away, t)
        if contrast(best, bg) < target:
            t_lo = t
        else:
            t_hi = t
    return bg, best, theme


# ------------------------------------------------------------------------------------------------------------------ text


def make_row_text(
    rng: np.random.Generator, kind: str, charset: list[str], allowed: frozenset[str], books: Books
) -> tuple[list[str], list[float], str]:
    s = Sampler(charset, allowed, rng)
    if kind in ("ui", "gap", "tight", "sep", "uiword"):
        fn = {"ui": ui_cells, "gap": gap_cells, "tight": tight_cells, "sep": sep_cells, "uiword": uiword_cells}[kind]
        cells, gaps, layout = fn(rng, s, books)
        return cells, gaps, f"synthetic:{kind}-{layout}"
    if kind == "book":
        text, src = books.window(rng, short=bool(rng.random() < 0.5))
        return [text], [], src
    if kind == "numtext":
        return [numtext_line(rng, s, books)], [], "synthetic:numtext"
    if kind == "figures":
        return [figures_line(rng, s)], [], "synthetic:figures"
    text = s.numbers_line() if kind == "numbers" else s.codes_line()
    return [text], [], f"synthetic:{kind}"


# ----------------------------------------------------------------------------------------------------------------- page

JS_READ = r"""() => [...document.querySelectorAll('.row')].map(r => {
  const l = r.querySelector('.line'), rr = r.getBoundingClientRect(), lr = l.getBoundingClientRect();
  const cells = [...l.querySelectorAll('.c')].map(c => c.textContent);
  const cs = getComputedStyle(l);
  return {cells, row: [rr.left, rr.top, rr.right, rr.bottom], line: [lr.left, lr.top, lr.right, lr.bottom],
          fits: l.scrollWidth <= r.clientWidth, fontSize: parseFloat(cs.fontSize), fontFamily: cs.fontFamily};
})"""


def shard_rng(seed: int, split: str, shard: int) -> np.random.Generator:
    return np.random.default_rng([seed, SPLIT_ID[split], shard])


class Renderer:
    """One Chromium per anti-aliasing mode, started lazily in the worker. (`--font-render-hinting` changed no pixel of the headless build
    tested, so hinting is not a dimension here.)"""

    def __init__(self):
        from playwright.sync_api import sync_playwright

        self.pw = sync_playwright().start()
        self.browsers: dict[str, object] = {}

    def browser(self, aa: str):
        key = aa
        if key not in self.browsers:
            args = ["--enable-lcd-text" if aa == "subpixel" else "--disable-lcd-text"]
            self.browsers[key] = self.pw.chromium.launch(args=args, executable_path=os.environ.get("ONEOCR_CHROMIUM") or None)
        return self.browsers[key]

    def close(self):
        for b in self.browsers.values():
            b.close()
        self.pw.stop()


_W: dict = {}


def _init(cfg: dict, split: str, seed: int, out: str):
    charset = cs.load(BASE / cfg["charset"])
    pool = load_pool("train" if split == "train" else "heldout")
    _W.update(cfg=cfg, split=split, seed=seed, out=Path(out), charset=charset, pool=pool, books=Books("train" if split == "train" else "heldout", charset),
              chars={n: frozenset(chr(c) for c in family_chars(f)) for n, f in pool.items()}, weights={n: weights_of(f) for n, f in pool.items()},
              names=sorted(pool), renderer=Renderer())  # fmt: skip


def pick(rng: np.random.Generator, table: dict) -> str:
    keys = list(table)
    p = np.array([table[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def render_shard(shard: int) -> tuple[list[dict], dict]:
    """Renders one page of rows; returns the label rows and counters of what was dropped."""
    cfg, split, seed, out = _W["cfg"], _W["split"], _W["seed"], _W["out"]
    rng = shard_rng(seed, split, shard)
    charset, pool, books = _W["charset"], _W["pool"], _W["books"]
    dpr = float(cfg["dpr"][int(rng.integers(0, len(cfg["dpr"])))])
    aa = pick(rng, cfg["aa"])
    drops = {"text": 0, "layout": 0, "read": 0, "ink": 0, "width": 0}
    rows: list[dict] = []
    faces: set[str] = set()
    n_rows = cfg["rows_per_page"]
    # plan the rows first; the page is built from the plan
    plan = []
    guard = 0
    while len(plan) < n_rows and guard < n_rows * 40:
        guard += 1
        fam_name = _W["names"][int(rng.integers(0, len(_W["names"])))]
        fam = pool[fam_name]
        kind = pick(rng, cfg["mix"])
        allowed = frozenset(c for c in _W["chars"][fam_name] if c in set(charset[1:]))
        cells, gaps, src = make_row_text(rng, kind, charset, allowed, books)
        text = unicodedata.normalize("NFC", " ".join(" ".join(c.split()) for c in cells))
        cells = [unicodedata.normalize("NFC", " ".join(c.split())) for c in cells]
        if not text or any(not c for c in cells) or len(text) > cfg["max_chars"] or sum(gaps) > 70:
            drops["text"] += 1
            continue
        if not set(text) - {" "} <= allowed or len({cs.script_of(c) for c in text if c.isalpha()}) > 1:
            drops["text"] += 1
            continue
        if rng.random() < cfg["p_upper"] and len(cells) == 1:
            cells = [cells[0].upper()]
            text = cells[0]
            if not set(text) - {" "} <= allowed:
                drops["text"] += 1
                continue
        px = float(cfg["css_px"][int(rng.integers(0, len(cfg["css_px"])))])
        px = round(px * float(rng.uniform(0.96, 1.04)), 2)
        bg, fg, theme = pick_colors(rng, cfg)
        w = int(rng.choice(_W["weights"][fam_name]))
        if rng.random() > cfg["p_weight_variety"]:
            w = min(_W["weights"][fam_name], key=lambda x: abs(x - 400))
        italic = fam.has_italic and bool(rng.random() < cfg["p_italic"])
        ls = round(float(rng.uniform(*cfg["letter_spacing"])), 3) if rng.random() < cfg["p_letter_spacing"] else 0.0
        plan.append({"fam": fam_name, "kind": kind, "src": src, "cells": cells, "gaps": gaps, "text": text, "px": px, "bg": bg, "fg": fg,
                     "theme": theme, "weight": w, "italic": italic, "ls": ls, "underline": bool(rng.random() < cfg["p_underline"]),
                     "lig": bool(rng.random() < cfg["p_ligatures"]) and not fam_name.startswith("firacode"),
                     "pad": [round(float(rng.uniform(*cfg["pad_y"])), 1), round(float(rng.uniform(*cfg["pad_x"])), 1)],
                     "sep": bool(rng.random() < cfg["p_rule"]), "seed": int(rng.integers(0, 2**31 - 1))})  # fmt: skip
        faces.add(fam_name)
    css = "".join(pool[f].css(font_root()) for f in sorted(faces))
    body = []
    for i, r in enumerate(plan):
        cells_html = []
        for k, c in enumerate(r["cells"]):
            g = r["gaps"][k - 1] if k and k - 1 < len(r["gaps"]) else 0.0
            cells_html.append(f'<span class="c" style="margin-left:{g:.2f}em">{html.escape(c)}</span>')
        style = (f"background:rgb{tuple(r['bg'])};color:rgb{tuple(r['fg'])};padding:{r['pad'][0]}px {r['pad'][1]}px;"
                 f"font:{'italic ' if r['italic'] else ''}{r['weight']} {r['px']}px/normal '{r['fam']}';letter-spacing:{r['ls']}em;"
                 f"font-variant-ligatures:{'normal' if r['lig'] else 'none'};{'text-decoration:underline;' if r['underline'] else ''}")  # fmt: skip
        body.append(f'<div class="row" id="r{i}" style="{style}"><div class="line">{"".join(cells_html)}</div></div>')
        if r["sep"]:
            body.append('<div style="height:1px;background:#8884"></div>')
        body.append(f'<div style="height:{round(float(rng.uniform(*cfg["gap_y"])), 1)}px;background:#{"".join(f"{int(v):02x}" for v in rng.integers(0, 256, 3))}"></div>')  # fmt: skip
    doc = (f"<!doctype html><meta charset=utf-8><style>{css}html,body{{margin:0;background:#fff}}"
           f".row{{width:{cfg['page_width']}px;box-sizing:border-box;overflow:hidden}}.line{{display:inline-block;white-space:pre}}</style>"
           f"<body><div style='height:20px'></div>{''.join(body)}<div style='height:20px'></div></body>")  # fmt: skip
    ctx = _W["renderer"].browser(aa).new_context(device_scale_factor=dpr, viewport={"width": cfg["page_width"], "height": 800})
    page = ctx.new_page()
    page.set_content(doc)
    page.evaluate("document.fonts.ready.then(() => true)")
    read = page.evaluate(JS_READ)
    png = page.screenshot(full_page=True)
    ctx.close()
    img = Image.open(io.BytesIO(png)).convert("RGB")
    arr = np.asarray(img)
    gray = np.asarray(img.convert("L"))
    (out / "images").mkdir(parents=True, exist_ok=True)
    for r, d in zip(plan, read, strict=True):
        if not d["fits"] or " ".join(d["cells"]) != r["text"]:
            drops["read"] += 1
            continue
        lh = d["line"][3] - d["line"][1]
        if lh > 2.2 * r["px"]:  # wrapped or stacked: not one line
            drops["layout"] += 1
            continue
        # the row's pixels, inset so that anti-aliased row edges and the neighbouring strips at fractional offsets are not counted as ink
        inset = int(np.ceil(dpr)) + 1
        x0, y0, x1, y1 = (int(round(v * dpr)) for v in d["row"])
        x0, y0, x1, y1 = x0 + inset, y0 + inset, min(x1, arr.shape[1]) - inset, min(y1, arr.shape[0]) - inset
        reg = gray[y0:y1, x0:x1].astype(np.float32)
        if reg.size == 0 or min(reg.shape) < 4:
            drops["ink"] += 1
            continue
        # background level from the row's frame (padding), ink where the pixel departs from it by a share of the ink/paper step
        border = np.concatenate([reg[0], reg[-1], reg[:, 0], reg[:, -1]])
        bgl = float(np.median(border))
        step = abs(255.0 * (luminance(r["fg"]) ** 0.4545 - luminance(r["bg"]) ** 0.4545))
        mask = np.abs(reg - bgl) > max(10.0, 0.25 * step)
        ys, xs = np.nonzero(mask)
        if len(ys) < 6:
            drops["ink"] += 1
            continue
        ix0, ix1, iy0, iy1 = xs.min() + x0, xs.max() + 1 + x0, ys.min() + y0, ys.max() + 1 + y0
        h = iy1 - iy0
        if h < cfg["min_ink_h"]:
            drops["ink"] += 1
            continue
        crng = np.random.default_rng([r["seed"]])
        broad = crng.random() < cfg["p_broad"]
        vm = [crng.uniform(0.0, 0.45) if broad else crng.uniform(0.18, 0.32) for _ in range(2)]
        hm = [crng.uniform(0.05, 1.0) if broad else crng.uniform(0.35, 0.65) for _ in range(2)]
        if crng.random() < cfg["p_tight_ends"]:
            hm = [crng.uniform(0.0, 0.2) for _ in range(2)]
        cx0, cx1 = max(0, int(round(ix0 - hm[0] * h))), min(arr.shape[1], int(round(ix1 + hm[1] * h)))
        cy0, cy1 = max(0, int(round(iy0 - vm[0] * h))), min(arr.shape[0], int(round(iy1 + vm[1] * h)))
        crop = img.crop((cx0, cy0, cx1, cy1)).convert("L")
        ow = max(8, round(crop.width * OUT_H / crop.height))
        if ow > cfg["max_width"]:
            drops["width"] += 1
            continue
        small = crop.resize((ow, OUT_H), Image.BICUBIC)  # PIL's bicubic reduces with a widened kernel: the antialiased scale of the library
        buf = io.BytesIO()
        small.save(buf, format="PNG")
        rid = f"{_W['split']}-{shard:05d}{len(rows):03d}"
        (out / "images" / f"{rid}.png").write_bytes(buf.getvalue())
        rows.append({"id": rid, "text": r["text"], "w": ow, "h": OUT_H, "font": f"{r['fam']}", "sha256": hashlib.sha256(buf.getvalue()).hexdigest(),
                     "src": r["src"], "dpr": dpr, "aa": aa, "css_px": r["px"], "em_px": round(r["px"] * dpr, 2),
                     "ink_h": int(h), "weight": r["weight"], "italic": r["italic"], "letter_spacing": r["ls"], "underline": r["underline"],
                     "ligatures": r["lig"], "theme": r["theme"], "bg": list(r["bg"]), "fg": list(r["fg"]),
                     "contrast": round(contrast(r["fg"], r["bg"]), 2), "crop": [cx0, cy0, cx1, cy1], "scale": round(OUT_H / crop.height, 4)})  # fmt: skip
    del arr
    return rows, drops


def sheet(root: Path, rows: list[dict], path: Path, n: int = 48) -> None:
    """Contact sheet: each crop at 3x with its label and rendering parameters under it."""
    from PIL import ImageDraw

    pad, lh = 4, 14
    rows = rows[:n]
    imgs = [Image.open(root / "images" / f"{r['id']}.png").convert("L") for r in rows]
    cols = 2
    cw = min(1000, max(im.width for im in imgs) * 2)
    canvas = Image.new("RGB", (cols * (cw + pad), ((len(imgs) + 1) // cols) * (OUT_H * 2 + lh + pad)), "white")
    d = ImageDraw.Draw(canvas)
    for k, (r, im) in enumerate(zip(rows, imgs, strict=True)):
        x, y = (k % cols) * (cw + pad), (k // cols) * (OUT_H * 2 + lh + pad)
        canvas.paste(im.resize((min(cw, im.width * 2), OUT_H * 2)).convert("RGB"), (x, y))
        d.text(
            (x, y + OUT_H * 2),
            f"{r['text'][:90]}  [{r['font']} {r['em_px']}px dpr{r['dpr']} {r['aa']} {r['theme']}]",
            fill="red",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=list(SPLIT_ID), required=True)
    ap.add_argument("--n", type=int, required=True, help="target number of kept crops (rounded up to whole pages)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sheet")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    out = Path(a.out) / a.split
    out.mkdir(parents=True, exist_ok=True)
    pages = -(-a.n // max(1, int(cfg["rows_per_page"] * cfg["expected_keep"])))
    t0 = time.time()
    rows: list[dict] = []
    drops: dict[str, int] = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(cfg, a.split, a.seed, str(out))) as pool:
        for r, d in pool.imap(render_shard, range(pages)):
            rows += r
            for k, v in d.items():
                drops[k] = drops.get(k, 0) + v
    rows = rows[: a.n] if len(rows) > a.n else rows
    keep = {r["id"] for r in rows}
    for p in (out / "images").glob("*.png"):  # pages are whole; crops past --n are not part of the set
        if p.stem not in keep:
            p.unlink()
    (out / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    import playwright

    manifest = {
        "config": cfg, "seed": a.seed, "split": a.split, "pages": pages, "kept": len(rows), "dropped": drops,
        "git_commit": os.environ.get("ONEOCR_GIT_COMMIT", "unknown"),
        "labels_sha256": hashlib.sha256((out / "labels.jsonl").read_bytes()).hexdigest(),
        "label_provenance": "exact-truth synthetic (gen/ui_lines.py, text written by the generator, read back from the DOM)",
        "versions": {"python": platform.python_version(), "playwright": getattr(playwright, "__version__", "?"), "numpy": np.__version__},
        "fonts": sorted({r["font"] for r in rows}), "seconds": round(time.time() - t0, 1), "workers": a.workers,
    }  # fmt: skip
    (out / "run.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("kept", "dropped", "seconds", "labels_sha256")}))
    if a.sheet:
        sheet(out, rows, Path(a.sheet))


if __name__ == "__main__":
    sys.exit(main())
