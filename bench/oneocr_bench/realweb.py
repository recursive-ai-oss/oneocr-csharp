"""Real web pages as screenshot evaluation pages, with exact truth from the DOM.

    python -m oneocr_bench realweb capture --list PAGES.json --out DIR [--target N]

PAGES.json is a private list (never committed): [{"url": ..., "viewport": [w, h], "scale": 1|2, "scrolls": [0, 700],
"source": "label"}, {"wikipedia": "Title", "mobile": false, ...}]. A Wikipedia entry is pinned to the article's current
revision id, so the capture is of an immutable page version.

Each page is loaded in headless Chromium, frozen (no animation or transition, fonts loaded, lazy content triggered) and
then made *text-exact* before the screenshot, so that every visible glyph is a DOM text node:

- images, video, canvas, SVG and iframes are made invisible (no image of text can appear; layout is unchanged);
- background images, list markers and generated (::before / ::after) content are removed;
- form-control values and placeholders are made transparent (their text is not DOM text) and left out of the truth;
- hyphenation, ellipsis, capitalize and small caps are switched off, so that the printed text is the DOM text;
- elements made only of private-use (icon font) characters are hidden.

Truth is then read the way the generated screenshot set reads it (one DOM Range rectangle per word, document order), and
only words a hit test confirms as visible are kept: hidden, clipped and covered text is dropped, and a capture is thrown
away when any word is cut by the viewport edge, partly covered or clipped, or invisible in the pixels (more than a few).
The screenshot is cut at row gaps so that no word is sliced. Reading order is the DOM order of visible text; that is not
always the visual order, so the set's order metric is information, not a gate.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from .render import add_ink, png_bytes, sha256, text_of, truth_words, write_page

UA = "oneocr-bench-fetch/1.0 (https://github.com/recursive-ai-oss/oneocr-csharp; evaluation pages, a few requests per run)"

FREEZE_CSS = """
*,*::before,*::after{animation:none !important;transition:none !important;caret-color:transparent !important;
  scroll-behavior:auto !important;background-image:none !important;hyphens:manual !important;text-overflow:clip !important;
  font-variant-caps:normal !important;font-variant:normal !important}
*::before,*::after{content:none !important}
img,picture,video,audio,canvas,svg,iframe,object,embed,map{visibility:hidden !important}
ul,ol,li,summary{list-style:none !important}
summary::-webkit-details-marker{display:none !important}
input,select,textarea,option{color:transparent !important;-webkit-text-fill-color:transparent !important;text-shadow:none !important}
::placeholder{color:transparent !important;opacity:0 !important}
sup.reference,.mw-editsection,.mw-cite-backlink{visibility:hidden !important}
"""

# Words as the page shows them: visible ones only, with the hit test's verdict on each rectangle.
EXTRACT_JS = r"""([W, H, css]) => {
  const out = [], ids = new Map(), fonts = {};
  const partials = [];
  const stats = {hidden: 0, partial: 0, outside: 0, pua: 0, controls: 0, shadow: 0, copyright: 0};
  const roots = [document.body];
  const collect = (node) => node.querySelectorAll('*').forEach(e => {
    if (e.shadowRoot) { roots.push(e.shadowRoot); stats.shadow++; collect(e.shadowRoot); }
  });
  collect(document);
  for (const r of roots.slice(1)) { const st = document.createElement('style'); st.textContent = css; r.appendChild(st); }
  const deepHit = (x, y) => {
    let h = document.elementFromPoint(x, y);
    while (h && h.shadowRoot) { const i = h.shadowRoot.elementFromPoint(x, y); if (!i || i === h) break; h = i; }
    return h;
  };
  for (const r of roots) for (const el of r.querySelectorAll('*')) {
    if (getComputedStyle(el).textTransform === 'capitalize') el.style.setProperty('text-transform', 'none', 'important');
  }
  const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'TITLE', 'HEAD']);
  const clipped = (el, r) => {  // 0 visible, 1 hidden, 2 partial: the word rectangle against every clipping ancestor
    let verdict = 0;
    for (let a = el; a && a !== document.documentElement; a = a.parentElement) {
      const cs = getComputedStyle(a);
      if (cs.overflowX === 'visible' && cs.overflowY === 'visible' && cs.clipPath === 'none') continue;
      if (cs.clipPath !== 'none' || cs.clip !== 'auto') return 2;
      const c = a.getBoundingClientRect();
      const x0 = cs.overflowX === 'visible' ? -1e9 : c.left, x1 = cs.overflowX === 'visible' ? 1e9 : c.right;
      const y0 = cs.overflowY === 'visible' ? -1e9 : c.top, y1 = cs.overflowY === 'visible' ? 1e9 : c.bottom;
      if (r.right <= x0 + 0.5 || r.left >= x1 - 0.5 || r.bottom <= y0 + 0.5 || r.top >= y1 - 0.5) return 1;
      if (r.left < x0 - 0.5 || r.right > x1 + 0.5 || r.top < y0 - 0.5 || r.bottom > y1 + 0.5) verdict = 2;
    }
    return verdict;
  };
  const verdictOf = (el, r) => {
    // the centre, and for wider words a quarter in from each end, on the middle of the line: points near the edges
    // of a rectangle often fall on a neighbouring inline element or the next line's box
    const cy = (r.top + r.bottom) / 2, w = r.right - r.left;
    const pts = w >= 24 ? [[(r.left + r.right) / 2, cy], [r.left + w * 0.25, cy], [r.left + w * 0.75, cy]] : [[(r.left + r.right) / 2, cy]];
    let inside = 0, hit = 0;
    for (const [x, y] of pts) {
      if (x < 0 || y < 0 || x >= W || y >= H) continue;
      inside++;
      const h = deepHit(x, y);
      if (h && (el.contains(h) || h.contains(el))) hit++;
    }
    if (inside === 0) return 'outside';
    if (inside < pts.length) return 'partial';
    return hit === pts.length ? 'visible' : hit === 0 ? 'hidden' : 'partial';
  };
  const puaEls = [];
  const texts = [];
  for (const r of roots) { const tw = document.createTreeWalker(r, NodeFilter.SHOW_TEXT); for (let n = tw.nextNode(); n; n = tw.nextNode()) texts.push(n); }
  for (const nd of texts) {
    const el = nd.parentElement;
    if (!el || SKIP.has(el.tagName) || el.closest('input,textarea,select,option')) continue;
    const cs = getComputedStyle(el);
    if (!el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true, opacityProperty: true, visibilityProperty: true})) continue;
    if (/rgba?\(\s*\d+,\s*\d+,\s*\d+,\s*0\s*\)/.test(cs.color) || cs.color === 'transparent') continue;
    const tf = cs.textTransform;
    const re = /\S+/g; let m;
    while ((m = re.exec(nd.data))) {
      let t = m[0].replace(/[\u00ad\u200b\u200c\u200d\u2060\ufeff]/g, '');
      if (!t) continue;
      if (/[\ue000-\uf8ff]/.test(t)) { stats.pua++; puaEls.push(el); continue; }
      if (/[\u0000-\u001f\u007f-\u009f]/.test(t)) { stats.controls++; continue; }
      if (tf === 'uppercase') t = t.toUpperCase(); else if (tf === 'lowercase') t = t.toLowerCase();
      const r = document.createRange(); r.setStart(nd, m.index); r.setEnd(nd, m.index + m[0].length);
      const rs = [...r.getClientRects()].filter(q => q.width > 0 && q.height > 0);
      if (!rs.length) continue;
      const f = rs[0], fc = (f.top + f.bottom) / 2;
      let parts;
      if (rs.every(q => Math.abs((q.top + q.bottom) / 2 - fc) < f.height * 0.5)) {
        parts = [{x0: Math.min(...rs.map(q => q.left)), y0: Math.min(...rs.map(q => q.top)), x1: Math.max(...rs.map(q => q.right)),
                  y1: Math.max(...rs.map(q => q.bottom)), s: t, split: false}];
      } else {  // a word wrapped across lines: one fragment per printed line, cut by character
        parts = []; let cur = null; const src = nd.data.slice(m.index, m.index + m[0].length);
        for (let k = 0; k < src.length; k++) {
          const c = document.createRange(); c.setStart(nd, m.index + k); c.setEnd(nd, m.index + k + 1);
          const q = [...c.getClientRects()].find(q => q.width > 0 && q.height > 0);
          if (!q) { if (cur) cur.s += src[k]; continue; }
          if (cur && Math.abs((q.top + q.bottom) / 2 - cur.cy) < q.height * 0.5) {
            cur.s += src[k]; cur.x0 = Math.min(cur.x0, q.left); cur.y0 = Math.min(cur.y0, q.top);
            cur.x1 = Math.max(cur.x1, q.right); cur.y1 = Math.max(cur.y1, q.bottom);
          } else { cur = {s: src[k], x0: q.left, y0: q.top, x1: q.right, y1: q.bottom, cy: (q.top + q.bottom) / 2}; parts.push(cur); }
        }
        parts.forEach((p, i) => { p.split = parts.length > 1; p.part = i; });
        if (tf === 'uppercase' || tf === 'lowercase') parts.forEach(p => { p.s = tf === 'uppercase' ? p.s.toUpperCase() : p.s.toLowerCase(); });
        parts = parts.map(p => ({...p, s: p.s.replace(/[\u00ad\u200b\u200c\u200d\u2060\ufeff]/g, '')})).filter(p => p.s);
      }
      for (const p of parts) {
        const rect = {left: p.x0, right: p.x1, top: p.y0, bottom: p.y1};
        let v = verdictOf(el, rect);
        if (v === 'visible') { const c = clipped(el, rect); v = c === 0 ? 'visible' : c === 1 ? 'hidden' : 'partial'; }
        if (v !== 'visible') { stats[v]++; if (v === 'partial') partials.push([p.y0, p.y1]); continue; }
        if (!ids.has(el)) ids.set(el, ids.size);
        const fam = cs.fontFamily.split(',')[0].trim().replace(/^["']|["']$/g, '');
        fonts[fam] = (fonts[fam] || 0) + 1;
        const o = {t: p.s, x0: p.x0, y0: p.y0, x1: p.x1, y1: p.y1, b: ids.get(el)};
        if (p.split) { o.split = true; o.part = p.part; }
        out.push(o);
      }
    }
  }
  puaEls.forEach(e => e.style.setProperty('visibility', 'hidden', 'important'));
  const all = out.map(o => o.t).join(' ');
  if (/©|\(c\)\s*\d{4}|copyright|used with permission|all rights reserved/i.test(all)) stats.copyright = 1;
  return {words: out, fonts, stats, partials};
}"""


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def wikipedia_revision(title: str) -> int:
    q = urllib.parse.urlencode({"action": "query", "prop": "revisions", "rvprop": "ids", "titles": title, "format": "json",
                                "redirects": 1})
    data = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{q}"))
    page = next(iter(data["query"]["pages"].values()))
    return int(page["revisions"][0]["revid"])


def resolve(entry: dict) -> dict:
    """Entry with a concrete `url`; Wikipedia titles become permanent revision links."""
    e = dict(entry)
    if "wikipedia" in e:
        rev = wikipedia_revision(e["wikipedia"])
        host = "en.m.wikipedia.org" if e.get("mobile") else "en.wikipedia.org"
        e["url"] = f"https://{host}/w/index.php?title={urllib.parse.quote(e['wikipedia'].replace(' ', '_'))}&oldid={rev}"
        e["revision"] = rev
        e.setdefault("source", "wikipedia")
    if e.get("mobile"):
        e.setdefault("viewport", [390, 844])
        e.setdefault("scale", 2)
    e.setdefault("viewport", [1280, 800])
    e.setdefault("scale", 1)
    e.setdefault("scrolls", [0, 1])
    return e


def row_cut(words: list[dict], lo: float, hi: float, want_low: bool) -> float | None:
    """A y position inside [lo, hi] that no word box crosses (scanning from the top when want_low, else from the bottom)."""
    ys = np.arange(lo, hi + 0.5, 1.0)
    if not want_low:
        ys = ys[::-1]
    for y in ys:
        if not any(w["y0"] < y < w["y1"] for w in words):
            return float(y)
    return None


def capture_one(ctx, entry: dict, k: int, out: Path, pid: str, set_name: str) -> tuple[dict | None, str]:
    w, h = entry["viewport"]
    scale = entry["scale"]
    page = ctx.new_page()
    try:
        resp = page.goto(entry["url"], wait_until="load", timeout=60000)
        if resp is None or resp.status >= 400:
            return None, f"http {resp.status if resp else 'none'}"
        body = resp.body()
        with contextlib.suppress(PlaywrightTimeout):  # some pages never go idle; the freeze below is what matters
            page.wait_for_load_state("networkidle", timeout=10000)
        page.add_style_tag(content=FREEZE_CSS)
        page.evaluate("document.fonts.ready")
        total = page.evaluate("document.documentElement.scrollHeight")
        for y in range(0, int(total), int(h * 0.8)):  # trigger lazy content, then come back
            page.evaluate("y => window.scrollTo(0, y)", y)
            page.wait_for_timeout(150)
        pos = 0 if k == 0 else int(h * 0.85 * k)
        if pos + h > total + 2 and k > 0:
            return None, "page too short for this scroll"
        page.evaluate("y => window.scrollTo(0, y)", pos)
        page.wait_for_timeout(600)
        page.evaluate("document.fonts.ready")
        res = page.evaluate(EXTRACT_JS, [w, h, FREEZE_CSS])
        words, stats = res["words"], res["stats"]
        if stats["hidden"] > 0.15 * (len(res["words"]) + stats["hidden"]):
            return None, f"{stats['hidden']} covered or clipped words"
        if stats["copyright"]:
            return None, "copyright or permission notice in the text"
        if stats["pua"] > 30:
            return None, f"icon text {stats['pua']}"
        if len(words) < 60:
            return None, f"only {len(words)} words"
        if any(x["x0"] < -0.5 or x["x1"] > w + 0.5 for x in words):
            return None, "words cross the left or right edge"
        top = row_cut(words, 0, h * 0.12, True)
        bot = row_cut(words, h * 0.85, h - 1, False)
        for lo, hi in sorted(res["partials"]):  # move the cut past a partly covered word that touches the top or bottom band
            if top is not None and lo < top + 1 and hi > 0 and hi <= h * 0.2:
                top = row_cut(words, hi + 1, h * 0.3, True)
            if bot is not None and hi > bot - 1 and lo >= h * 0.8:
                bot = row_cut(words, h * 0.7, lo - 1, False)
        if top is None or bot is None:
            return None, "no clean row to cut at"
        # a partly covered word (a sticky bar, a popover) is no truth: the cut must leave every such word outside the crop
        inside = [p for p in res["partials"] if p[1] > top and p[0] < bot]
        if inside:
            return None, f"{len(inside)} partly covered words inside the cut"
        words = [x for x in words if x["y0"] >= top and x["y1"] <= bot]
        if len(words) < 60:
            return None, f"only {len(words)} words in the cut"
        png = page.screenshot(animations="disabled")  # the whole viewport; cut below, so clip coordinates are never in doubt
        title = page.title()
    finally:
        page.close()

    img = Image.open(io.BytesIO(png)).convert("L")
    img = img.crop((0, round(top * scale), img.size[0], round(bot * scale)))
    for x in words:
        x["y0"] -= top
        x["y1"] -= top
    tw = truth_words(words, scale)
    g = np.asarray(img, np.float32)
    dead = partial = 0
    keep = []
    for t in tw:  # does the word's box show ink across its width? white text on a blanked background fails this
        x0, y0, x1, y1 = (round(v) for v in t["box"])
        reg = g[max(0, y0): max(0, y1), max(0, x0): max(0, x1)]
        cov = float(((reg.max(axis=0) - reg.min(axis=0)) >= 12).mean()) if reg.size else 0.0
        if cov < 0.1:
            dead += 1
        else:
            if cov < 0.25 and len(t["text"]) > 2:
                partial += 1
            keep.append(t)
    if partial > 2 or dead > max(2, 0.02 * len(tw)):
        return None, f"{dead} invisible and {partial} part-visible words of {len(tw)} in the pixels"
    if dead:  # drop them from the truth; there is nothing on the page to read
        tw = keep
    add_ink(tw, img)
    fonts = res["fonts"]
    truth = {"id": pid, "set": set_name, "source": entry.get("source", "web"), "url_kind": "private list",
             "image": {"width": img.size[0], "height": img.size[1]}, "viewport": [w, h], "device_scale": scale,
             "scroll_step": k, "title_sha256": sha256(title.encode()), "fonts": fonts, "words": tw, "text": text_of(tw),
             "order": "DOM order of visible text", "dropped": {"hidden": stats["hidden"], "outside": stats["outside"],
                                                                "invisible_in_pixels": dead}}
    e = write_page(out, pid, png_bytes(img), "png", truth)
    e.update(url=entry["url"], revision=entry.get("revision"), source=entry.get("source", "web"), response_sha256=sha256(body),
             retrieved=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), fonts=sorted(fonts), viewport=[w, h], scale=scale)
    return e, "ok"


def capture(a) -> None:
    from playwright.sync_api import sync_playwright

    entries = []
    for e in json.loads(Path(a.list).expanduser().read_text()):
        try:
            entries.append(resolve(e))
        except Exception as exc:  # noqa: BLE001 - an entry that cannot be pinned is skipped and reported
            print(f"skipped {e.get('wikipedia') or e.get('url')}: {type(exc).__name__} {exc}", flush=True)
    out = Path(a.out).expanduser()
    (out / "pages").mkdir(parents=True, exist_ok=True)
    pages: list[dict] = []
    log: list[dict] = []
    set_name = "screenshot-real"
    started = time.time()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for entry in entries:
            if len(pages) >= a.target or time.time() - started > a.seconds:
                break
            ctx = browser.new_context(viewport={"width": entry["viewport"][0], "height": entry["viewport"][1]},
                                      device_scale_factor=entry["scale"], reduced_motion="reduce", locale="en-US",
                                      is_mobile=bool(entry.get("mobile")), has_touch=bool(entry.get("mobile")))
            for k in range(len(entry["scrolls"])):
                if len(pages) >= a.target:
                    break
                pid = f"s{len(pages) + 1:03d}"
                try:
                    e, why = capture_one(ctx, entry, k, out, pid, set_name)
                except Exception as exc:  # noqa: BLE001 - one bad page must not stop the run
                    e, why = None, f"{type(exc).__name__}: {str(exc)[:120]}"
                log.append({"url": entry["url"], "k": k, "result": why})
                print(f"{pid} {entry.get('source', 'web')} k={k}: {why}", flush=True)
                if e:
                    pages.append(e)
            ctx.close()
        version = browser.version
        browser.close()
    manifest = {"set": set_name, "count": len(pages), "chromium": version, "source": "real web pages (private list)",
                "pages": pages, "log": log}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(pages)} pages written to {out}")

