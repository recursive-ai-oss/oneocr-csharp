"""Shared Chromium helpers for the generated sets: fonts, word geometry from DOM ranges, truth files.

Truth is always the browser's own layout. For every word (a run of non-space characters in a text node) a DOM
Range gives the client rectangles. A word that wraps at a hyphen or em dash has rectangles on two lines; its box
is the union of the rectangles on the first line only, and the word is flagged `split` in the truth file, so a box
never spans two columns. Reading order is document order; the word's block is the index of its parent element
(first appearance), and the text file starts a new line whenever the block changes.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from .assets import FAMILIES, font_path

WORDS_JS = """(rootSel) => {
  const root = document.querySelector(rootSel);
  const out = []; const ids = new Map();
  const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let nd = tw.nextNode(); nd; nd = tw.nextNode()) {
    const el = nd.parentElement;
    if (getComputedStyle(el).visibility === 'hidden') continue;
    if (!ids.has(el)) ids.set(el, ids.size);
    const re = /\\S+/g; let m;
    while ((m = re.exec(nd.data))) {
      const r = document.createRange(); r.setStart(nd, m.index); r.setEnd(nd, m.index + m[0].length);
      const rs = [...r.getClientRects()].filter(q => q.width > 0 && q.height > 0);
      if (!rs.length) continue;
      const f = rs[0], fc = (f.top + f.bottom) / 2;
      const line = rs.filter(q => Math.abs((q.top + q.bottom) / 2 - fc) < f.height * 0.5);
      out.push({t: m[0], x0: Math.min(...line.map(q => q.left)), y0: Math.min(...line.map(q => q.top)),
                x1: Math.max(...line.map(q => q.right)), y1: Math.max(...line.map(q => q.bottom)),
                b: ids.get(el), split: line.length < rs.length});
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
        out.append(d)
    return out


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
    return {"id": pid, "words": len(truth["words"]), "image_sha256": sha256(image), "truth_sha256": sha256(truth_bytes)}
