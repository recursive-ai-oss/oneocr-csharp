"""Shared Chromium helpers for the generated sets: fonts, word geometry from DOM ranges, truth files.

Truth is always the browser's own layout. For every word (a run of non-space characters in a text node) a DOM
Range gives the client rectangles. A word that wraps at a hyphen or em dash is printed on two lines; each printed
fragment gets its own entry and box (`split`, `part`), so a box never spans two lines or columns, while the text
file still holds the word whole. Reading order is document order; the word's block is the index of its parent
element (first appearance), and the text file starts a new line whenever the block changes.

Every word also gets `ink`: the extent of its ink on the rendered page, found from the pixels (see `add_ink`). The
layout box is the browser's rectangle (font ascent to descent, advance width); an engine that reports tight boxes
is compared against `ink`.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .assets import FAMILIES, font_path

# One entry per printed fragment of a word. Almost always that is the whole word (one rectangle per line from a DOM Range);
# a word that wraps at a hyphen or em dash is cut into one fragment per line by measuring its characters, so every
# printed piece gets its own box. Fragments carry `part` (0, 1, ...) and the text file is unaffected: fragments of one word
# are joined without a space.
FRAGS_JS = """function frags(nd, s, e) {
  const r = document.createRange(); r.setStart(nd, s); r.setEnd(nd, e);
  const rs = [...r.getClientRects()].filter(q => q.width > 0 && q.height > 0);
  if (!rs.length) return [];
  const f = rs[0], fc = (f.top + f.bottom) / 2;
  if (rs.every(q => Math.abs((q.top + q.bottom) / 2 - fc) < f.height * 0.5)) {
    return [{t: nd.data.slice(s, e), x0: Math.min(...rs.map(q => q.left)), y0: Math.min(...rs.map(q => q.top)),
             x1: Math.max(...rs.map(q => q.right)), y1: Math.max(...rs.map(q => q.bottom))}];
  }
  const out = []; let cur = null;
  for (let k = s; k < e; k++) {
    const c = document.createRange(); c.setStart(nd, k); c.setEnd(nd, k + 1);
    const q = [...c.getClientRects()].find(q => q.width > 0 && q.height > 0);
    if (!q) { if (cur) cur.t += nd.data[k]; continue; }
    if (cur && Math.abs((q.top + q.bottom) / 2 - cur.cy) < q.height * 0.5) {
      cur.t += nd.data[k]; cur.x0 = Math.min(cur.x0, q.left); cur.y0 = Math.min(cur.y0, q.top);
      cur.x1 = Math.max(cur.x1, q.right); cur.y1 = Math.max(cur.y1, q.bottom);
    } else {
      cur = {t: nd.data[k], x0: q.left, y0: q.top, x1: q.right, y1: q.bottom, cy: (q.top + q.bottom) / 2};
      out.push(cur);
    }
  }
  return out.map((o, i) => ({t: o.t, x0: o.x0, y0: o.y0, x1: o.x1, y1: o.y1, split: out.length > 1, part: i}));
}
"""

WORDS_JS = "(rootSel) => {" + FRAGS_JS + """
  const root = document.querySelector(rootSel);
  const out = []; const ids = new Map();
  const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let nd = tw.nextNode(); nd; nd = tw.nextNode()) {
    const el = nd.parentElement;
    if (getComputedStyle(el).visibility === 'hidden') continue;
    if (!ids.has(el)) ids.set(el, ids.size);
    const re = /\\S+/g; let m;
    while ((m = re.exec(nd.data))) {
      for (const f of frags(nd, m.index, m.index + m[0].length)) out.push({...f, b: ids.get(el)});
    }
  }
  return out;
}"""


def font_css(family: str) -> str:
    rules = []
    for name, weight, style in FAMILIES[family][1]:
        rules.append(
            f"@font-face{{font-family:'{family}';src:url('{font_path(family, name).as_uri()}');"
            f"font-weight:{weight};font-style:{style};}}"
        )
    return "\n".join(rules)


def wait_fonts(page, family: str) -> None:
    page.evaluate(
        "f => Promise.all([document.fonts.load(`1em '${f}'`), document.fonts.load(`bold 1em '${f}'`)])", family
    )
    page.evaluate("document.fonts.ready")
    loaded = page.evaluate("[...document.fonts].filter(f=>f.status==='loaded').map(f=>f.family)")
    if family not in [x.strip("'\"") for x in loaded]:
        raise RuntimeError(f"font {family} did not load ({loaded})")


def text_of(words: list[dict]) -> str:
    """Words in order; a new line wherever the block changes."""
    lines: list[list[str]] = []
    cur = None
    for w in words:
        if w.get("part", 0) > 0:  # a later fragment of the word before it
            lines[-1][-1] += w["text"]
            continue
        if w["block"] != cur:
            lines.append([])
            cur = w["block"]
        lines[-1].append(w["text"])
    return "\n".join(" ".join(ws) for ws in lines) + "\n"


def truth_words(raw: list[dict], scale: float) -> list[dict]:
    out = []
    for w in raw:
        d = {"text": w["t"], "box": [round(w[k] * scale, 1) for k in ("x0", "y0", "x1", "y1")], "block": w["b"]}
        if w.get("split"):
            d["split"] = True
            d["part"] = w["part"]
        out.append(d)
    return out


INK_REL = 0.3  # a pixel is ink when it differs from the background by this share of the word's peak contrast
INK_MIN = 12  # grey levels; below this peak contrast the word has no visible ink and keeps its layout box


def add_ink(words: list[dict], img: Image.Image) -> int:
    """Add `ink` = [x0, y0, x1, y1] to each truth word: the extent of its ink in the rendered, undegraded page.

    Deterministic from the pixels inside the word's layout box (one pixel of margin sideways, none vertically, so the
    neighbouring words and lines are not picked up): the background is the most common grey level (4-level bins), ink is
    every pixel that differs from it by at least INK_REL of the strongest difference. Geometric degradations move
    `ink` with the same homography as `box`. Returns the number of words with no visible ink (they keep `box`).
    """
    a = np.asarray(img.convert("L"), np.int16)
    h, w = a.shape
    blank = 0
    for t in words:
        x0, y0, x1, y1 = t["box"]
        px0, py0 = max(0, math.floor(x0) - 1), max(0, math.floor(y0))
        px1, py1 = min(w, math.ceil(x1) + 1), min(h, math.ceil(y1))
        reg = a[py0:py1, px0:px1]
        peak = 0
        if reg.size:
            bg = np.bincount((reg // 4).ravel()).argmax() * 4 + 2
            d = np.abs(reg - bg)
            peak = int(d.max())
        if peak < INK_MIN:
            t["ink"] = list(t["box"])
            blank += 1
            continue
        m = d >= max(8.0, INK_REL * peak)
        cols = np.flatnonzero(m.any(axis=0))
        rows = np.flatnonzero(m.any(axis=1))
        t["ink"] = [float(px0 + cols[0]), float(py0 + rows[0]), float(px0 + cols[-1] + 1), float(py0 + rows[-1] + 1)]
    return blank


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=6)
    return buf.getvalue()


def write_page(out: Path, pid: str, image: bytes, ext: str, truth: dict) -> dict:
    """Write pages/<id>.<ext>, truth/<id>.json and truth/<id>.txt; return the manifest entry."""
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    truth_bytes = (json.dumps(truth, ensure_ascii=False, indent=1) + "\n").encode()
    (out / "pages" / f"{pid}.{ext}").write_bytes(image)
    (out / "truth" / f"{pid}.json").write_bytes(truth_bytes)
    (out / "truth" / f"{pid}.txt").write_text(truth["text"], encoding="utf-8")
    return {"id": pid, "words": len(truth["words"]), "image_sha256": sha256(image), "truth_sha256": sha256(truth_bytes),
            "text_sha256": sha256(truth["text"].encode())}
