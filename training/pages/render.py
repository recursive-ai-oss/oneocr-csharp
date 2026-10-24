"""Chromium rendering of a PageSpec and assembly of the page truth from the browser's own layout.

Word boxes are DOM Range client rectangles of the words (runs of non-space characters in text nodes); nothing is read from the
image. Everything else (blocks, regions, tables, cells, key-value pairs, fields, checkboxes) comes from the data attributes the
generator wrote (htmlkit.py) resolved against the DOM geometry. The truth is the `oneocr.page` JSON of docs/spec/oneocr-page-v1.md
(same member names, order, rounding and joining rules) plus members a reader ignores: `docLayNet` on regions, `fields`,
`checkboxes`, `documentType` on the page and `synthetic` on the document.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .layouts import PageSpec

DSFS = (1.0, 1.5625, 2.0833333, 3.125)  # 96, 150, 200 and 300 dpi

JS_FIT = r"""() => {
  const c = document.getElementById('flow'); let n = 0;
  // text that sticks out of the flow box (a table wider than its column, a block cut at a column break) is clipped by the page
  const clipped = () => {
    const fr = c.getBoundingClientRect(), tw = document.createTreeWalker(c, NodeFilter.SHOW_TEXT);
    for (let nd = tw.nextNode(); nd; nd = tw.nextNode()) {
      if (!/\S/.test(nd.data)) continue;
      const r = document.createRange(); r.selectNodeContents(nd);
      for (const q of r.getClientRects()) {
        const tol = 0.3 * q.height;
        if (q.left < fr.left - tol || q.right > fr.right + tol || q.bottom > fr.bottom + tol) return true;
      }
    }
    return false;
  };
  const over = () => c.scrollHeight > c.clientHeight + 1 || c.scrollWidth > c.clientWidth + 1 || clipped();
  const items = () => [...c.querySelectorAll(':scope > [data-item]')];
  while (over() && items().length > 1) { items().pop().remove(); n++; }
  return n;
}"""

JS_EXTRACT = r"""() => {
  const page = document.getElementById('page');
  const pr = page.getBoundingClientRect();
  const R = e => { const r = e.getBoundingClientRect(); return [r.left - pr.left, r.top - pr.top, r.right - pr.left, r.bottom - pr.top]; };
  const idx = sel => { const a = [...page.querySelectorAll(sel)]; return [a, new Map(a.map((e, i) => [e, i]))]; };
  const [blks, bm] = idx('[data-blk]'), [rows, rm] = idx('[data-row]'), [cells, cm] = idx('[data-cell]');
  const [flds, fm] = idx('[data-fld]'), [kvs, km] = idx('[data-kvr]'), [regs] = idx('[data-reg]');
  const [tbls, tm] = idx('[data-tbl]'), [opts, om] = idx('[data-opt]');
  const words = [], orphans = [];
  const tw = document.createTreeWalker(page, NodeFilter.SHOW_TEXT);
  for (let nd = tw.nextNode(); nd; nd = tw.nextNode()) {
    const el = nd.parentElement;
    if (el.closest('style,script') || !/\S/.test(nd.data)) continue;
    if (getComputedStyle(el).visibility === 'hidden') continue;
    const b = el.closest('[data-blk]');
    if (!b) { orphans.push(nd.data.trim().slice(0, 30)); continue; }
    const row = el.closest('[data-row]'), cell = el.closest('[data-cell]'), kv = el.closest('[data-kvr]'), op = el.closest('[data-opt]');
    const chain = [];
    for (let e = el.closest('[data-fld]'); e; e = e.parentElement && e.parentElement.closest('[data-fld]')) chain.push(fm.get(e));
    const re = /\S+/g; let m;
    while ((m = re.exec(nd.data))) {
      const r = document.createRange(); r.setStart(nd, m.index); r.setEnd(nd, m.index + m[0].length);
      const rs = [...r.getClientRects()].filter(q => q.width > 0 && q.height > 0);
      if (!rs.length) continue;
      const f = rs[0], fc = (f.top + f.bottom) / 2;
      const line = rs.filter(q => Math.abs((q.top + q.bottom) / 2 - fc) < f.height * 0.5);
      words.push({t: m[0], r: [Math.min(...line.map(q => q.left)) - pr.left, Math.min(...line.map(q => q.top)) - pr.top,
                               Math.max(...line.map(q => q.right)) - pr.left, Math.max(...line.map(q => q.bottom)) - pr.top],
                  b: bm.get(b), row: row ? rm.get(row) : -1, cell: cell ? cm.get(cell) : -1, fl: chain,
                  kv: kv ? km.get(kv) : -1, opt: op ? om.get(op) : -1, split: line.length < rs.length});
    }
  }
  const flow = document.getElementById('flow');
  // words that lie (even partly) outside the clipped area of the flow are not fully visible; the page is redrawn instead
  let clipped = 0;
  if (flow) {
    const fr = flow.getBoundingClientRect();
    for (const w of words) {
      const r = [w.r[0] + pr.left, w.r[1] + pr.top, w.r[2] + pr.left, w.r[3] + pr.top];
      const tol = 0.3 * (r[3] - r[1]);
      if (r[0] < fr.left - tol || r[2] > fr.right + tol || r[3] > fr.bottom + tol) clipped++;
    }
  }
  return { clipped,
    size: [pr.width, pr.height], orphans, words,
    overflow: !!(flow && (flow.scrollHeight > flow.clientHeight + 1 || flow.scrollWidth > flow.clientWidth + 1)) || page.scrollHeight > page.clientHeight + 1 || page.scrollWidth > page.clientWidth + 1,
    blocks: blks.map(e => ({role: e.dataset.blk, r: R(e), tbl: tm.has(e) ? tm.get(e) : -1})),
    tables: tbls.map(e => ({r: R(e), nrows: +e.dataset.nrows, ncols: +e.dataset.ncols})),
    cells: cells.map(e => ({r: +e.dataset.r, c: +e.dataset.c, rs: +e.dataset.rs, cs: +e.dataset.cs, box: R(e), tbl: tm.get(e.closest('[data-tbl]'))})),
    regions: regs.map(e => ({role: e.dataset.reg, dl: e.dataset.dl || null, box: e.hasAttribute('data-box'), r: R(e),
                             blks: blks.map((b, i) => e.contains(b) ? i : -1).filter(i => i >= 0)})),
    fields: flds.map(e => { const li = e.closest('[data-li]'); return {id: e.dataset.fld, name: e.dataset.fname, val: e.dataset.val, li: li ? +li.dataset.li : null}; }),
    kvs: kvs.map(e => ({g: +e.dataset.kv, role: e.dataset.kvr})),
    checks: [...page.querySelectorAll('[data-cb]')].map(e => { const o = e.closest('[data-opt]'); return {r: R(e), on: e.dataset.cb === '1', opt: o ? om.get(o) : -1}; }),
    text: page.innerText,
  };
}"""


class Overflow(Exception):
    pass


class Renderer:
    def __init__(self, tmp: Path):
        from playwright.sync_api import sync_playwright

        self.tmp = tmp
        tmp.mkdir(parents=True, exist_ok=True)
        self.pw = sync_playwright().start()
        self.browser = self.pw.chromium.launch(
            args=["--font-render-hinting=none", "--disable-lcd-text", "--force-color-profile=srgb", "--disable-gpu"]
        )
        self.ctx: dict[float, object] = {}
        self.pages: dict[float, object] = {}

    def close(self) -> None:
        self.browser.close()
        self.pw.stop()

    def _page(self, dsf: float):
        if dsf not in self.pages:
            ctx = self.browser.new_context(viewport={"width": 1000, "height": 1400}, device_scale_factor=dsf)
            self.ctx[dsf] = ctx
            self.pages[dsf] = ctx.new_page()
        return self.pages[dsf]

    def render(self, spec: PageSpec, dsf: float) -> tuple[bytes, dict]:
        page = self._page(dsf)
        src = self.tmp / "page.html"
        src.write_text(spec.html, encoding="utf-8")
        page.set_viewport_size(
            {"width": max(spec.width, 400) + 40 if spec.width else 1000, "height": max(spec.height, 600) + 40 if spec.height else 1400}
        )
        page.goto(src.as_uri())
        page.evaluate("fams => Promise.all(fams.map(f => document.fonts.load(`1em '${f}'`)))", list(spec.fonts))
        page.evaluate("document.fonts.ready.then(() => true)")
        dropped = page.evaluate(JS_FIT) if spec.fit == "trim" else 0
        raw = page.evaluate(JS_EXTRACT)
        raw["dropped"] = dropped
        if raw["orphans"]:
            raise AssertionError(f"text outside any block: {raw['orphans'][:5]}")
        if spec.fit != "none" and raw["overflow"]:
            raise Overflow
        loaded = page.evaluate("[...document.fonts].filter(f => f.status === 'loaded').map(f => f.family)")
        raw["fonts_loaded"] = sorted({x.strip("'\"") for x in loaded})
        raw["font_status"] = page.evaluate("[...document.fonts].map(f => f.family + ':' + f.status + ':' + f.weight + ':' + f.style)")
        png = page.locator("#page").screenshot(type="png")
        return png, raw


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _r3(v):
    return round(float(v) + 0.0, 3)


def build_truth(raw: dict, spec: PageSpec, img_size: tuple[int, int], xf, meta: dict) -> tuple[dict, list[dict]]:
    """Assemble the oneocr.page document. `xf(box)` maps a clean-image-pixel box to (bounds, quad or None).
    Returns (document, flat word list in reading order with their clean boxes, for the checker)."""
    W, H = img_size
    cw, ch = raw["size"]
    sx, sy = W / cw, H / ch
    px = lambda b: [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]  # noqa: E731

    words = raw["words"]
    order = sorted(range(len(words)), key=lambda i: (words[i]["b"], i))  # DOM order already; stable by block
    assert order == list(range(len(words))), "words are not contiguous per block"
    n_blocks = len(raw["blocks"])
    by_block: dict[int, list[int]] = {}
    for i, w in enumerate(words):
        by_block.setdefault(w["b"], []).append(i)

    wbox = [px(w["r"]) for w in words]
    lines_of: dict[int, list[list[int]]] = {}
    for bi, idxs in by_block.items():
        blk = raw["blocks"][bi]
        lines: list[list[int]] = []
        if blk["tbl"] >= 0:
            cur = None
            for i in idxs:
                if words[i]["row"] != cur:
                    lines.append([])
                    cur = words[i]["row"]
                lines[-1].append(i)
        else:
            first = None
            for i in idxs:
                b = wbox[i]
                if first is None or abs((b[1] + b[3]) / 2 - (first[1] + first[3]) / 2) > 0.5 * (first[3] - first[1]):
                    lines.append([])
                    first = b
                lines[-1].append(i)
        lines_of[bi] = lines

    def cell_starts(line: list[int], tbl: bool) -> list[int]:
        starts = [0]
        for k in range(1, len(line)):
            a, b = line[k - 1], line[k]
            if tbl:
                if words[a]["cell"] != words[b]["cell"]:
                    starts.append(k)
            else:
                h = wbox[a][3] - wbox[a][1]
                if wbox[b][0] - wbox[a][2] > 1.4 * h:
                    starts.append(k)
        return starts

    def box_out(b):
        bounds, quad = xf(b)
        d = {"bounds": [_r3(v) for v in bounds]}
        return d, quad

    def with_quad(d: dict, quad):
        if quad is not None:
            d["quad"] = [_r3(v) for v in quad]
        return d

    _out_words, out_blocks = [], []
    for bi in range(n_blocks):
        blk = raw["blocks"][bi]
        lines = lines_of.get(bi, [])
        if not lines:
            out_blocks.append(None)  # a block without words (decor); skipped
            continue
        olines = []
        for line in lines:
            starts = cell_starts(line, blk["tbl"] >= 0)
            ows = []
            for i in line:
                d, q = box_out(wbox[i])
                wd = {
                    "text": words[i]["t"],
                    "bounds": d["bounds"],
                    "visible": True,
                    "source": "textLayer",
                    "confidence": 1,
                    "rawConfidence": 1,
                }
                ows.append(with_quad(wd, q))
            lb, lq = box_out(_union([wbox[i] for i in line]))
            texts = []
            for k, s in enumerate(starts):
                e = starts[k + 1] if k + 1 < len(starts) else len(line)
                texts.append(" ".join(words[i]["t"] for i in line[s:e]))
            ld = {"direction": "ltr", "vertical": False, "bounds": lb["bounds"], "cells": starts, "text": "\t".join(texts), "words": ows,
                  "source": "textLayer", "confidence": 1}  # fmt: skip
            olines.append(with_quad(ld, lq))
        bb, bq = box_out(_union([wbox[i] for ln in lines for i in ln]) if blk["tbl"] < 0 else px(raw["tables"][blk["tbl"]]["r"]))
        bd = {
            "role": blk["role"],
            "table": blk["tbl"] >= 0,
            "bounds": bb["bounds"],
            "lines": olines,
            "source": "textLayer",
            "confidence": 1,
        }
        out_blocks.append(with_quad(bd, bq))

    # blocks without words are dropped; remap indices
    remap, kept = {}, []
    for bi, ob in enumerate(out_blocks):
        if ob is not None:
            remap[bi] = len(kept)
            kept.append(ob)
    # word index in reading order = position in raw words (already block-ordered, and only blocks with words exist)
    regions = []
    for rg in raw["regions"]:
        blks = [remap[b] for b in rg["blks"] if b in remap]
        ws = [i for b in rg["blks"] for i in by_block.get(b, [])]
        if rg["box"] or not ws:
            box = px(rg["r"])
        else:
            box = _union([wbox[i] for i in ws])
        if rg["role"] not in ("checkbox", "figure", "formField") and not blks:
            continue
        bd, q = box_out(box)
        regions.append(with_quad({"role": rg["role"], "bounds": bd["bounds"], "confidence": 1, "source": "textLayer", "blocks": blks,
                                  "docLayNet": rg["dl"]}, q))  # fmt: skip

    tables = []
    for ti, tb in enumerate(raw["tables"]):
        bi = next((i for i, b in enumerate(raw["blocks"]) if b["tbl"] == ti), None)
        if bi is None or bi not in remap:
            continue
        cells = []
        for ci, c in enumerate(raw["cells"]):
            if c["tbl"] != ti:
                continue
            ws = [i for i, w in enumerate(words) if w["cell"] == ci]
            bd, _ = box_out(px(c["box"]))
            cells.append({"row": c["r"], "column": c["c"], "rowSpan": c["rs"], "columnSpan": c["cs"], "bounds": bd["bounds"], "words": ws})
        bd, q = box_out(px(tb["r"]))
        tables.append(with_quad({"bounds": bd["bounds"], "rows": tb["nrows"], "columns": tb["ncols"], "cells": cells, "confidence": 1,
                                 "source": "textLayer", "block": remap[bi]}, q))  # fmt: skip

    kvs = []
    groups: dict[int, dict] = {}
    for i, w in enumerate(words):
        if w["kv"] >= 0:
            k = raw["kvs"][w["kv"]]
            groups.setdefault(k["g"], {"key": [], "value": []})[k["role"]].append(i)
    for g in sorted(groups):
        gr = groups[g]
        if gr["key"] and gr["value"]:
            bd, q = box_out(_union([wbox[i] for i in gr["key"] + gr["value"]]))
            kvs.append(
                with_quad({"key": gr["key"], "value": gr["value"], "bounds": bd["bounds"], "confidence": 1, "source": "textLayer"}, q)
            )

    fields = []
    for fi, f in enumerate(raw["fields"]):
        ws = [i for i, w in enumerate(words) if fi in w["fl"]]
        if ws:
            fields.append({"id": f["id"], "name": f["name"], "value": f["val"], "lineItem": f["li"], "words": ws})
    checks = []
    for c in raw["checks"]:
        bd, q = box_out(px(c["r"]))
        label = [i for i, w in enumerate(words) if c["opt"] >= 0 and w["opt"] == c["opt"]]
        checks.append(with_quad({"bounds": bd["bounds"], "checked": c["on"], "label": label}, q))

    text = "\n\n".join("\n".join(ln["text"] for ln in b["lines"]) for b in kept)
    page = {"index": 0, "width": W, "height": H, "unitsPerInch": _r3(meta["units_per_inch"]), "diagnostics": [], "text": text,
            "blocks": kept, "regions": regions, "tables": tables, "keyValues": kvs, "documentType": spec.kind}  # fmt: skip
    if fields:
        page["fields"] = fields
    if checks:
        page["checkboxes"] = checks
    doc = {"format": "oneocr.page", "version": 1, "pageCount": 1, "pages": [page], "synthetic": meta["synthetic"]}
    flat = [{"text": w["t"], "box": wbox[i], "split": w["split"]} for i, w in enumerate(words)]
    return doc, flat


def dumps(doc: dict) -> bytes:
    return (json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def identity_xf(box):
    return list(box), None
