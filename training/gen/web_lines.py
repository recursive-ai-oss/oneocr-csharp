# ruff: noqa: E501
"""Text lines from real public-domain web pages, rendered in headless Chromium, with the label taken from the DOM.

Why: the recognizer meets screenshots of real sites, whose fonts, CSS, hinting, anti-aliasing and widgets differ from anything we
draw ourselves. Pages of US federal agencies whose own notices place their content in the public domain (data/web_sources.yaml)
give that variety with exact truth: every word is a DOM text node, its rectangle comes from a DOM Range, and a line is kept only
when the pixels agree with the DOM.

    uv run python -m gen.web_lines discover --config configs/web-lines-v1.yaml            # writes the pinned page list
    uv run python -m gen.web_lines render   --config configs/web-lines-v1.yaml --out OUT --workers 5 [--limit-pages N]

Discover fetches seed pages politely (robots.txt, an honest User-Agent, a pause between requests) and writes data/web_urls.txt
(publisher, URL); the list is committed, pages are not. Render loads each URL, makes the page text-exact (images, background
images and generated content are not drawn; form values are hidden), reads the visible words with their boxes, groups them into
visual lines, checks each line against the screenshot, cuts the crop the way the page reader does (ink box, margins of a quarter
of the line height above and below and half of it left and right, a minimum line height of 0.85 of the page's typical one) and
scales it to 32 px. Output: OUT/<split>/images/<id>.png, labels.jsonl (the LineSet format plus parameters), OUT/run.json and
OUT/pages.jsonl (one record per page, with the response hash, so a later run can tell whether a page changed).

Truth: the label of a line is the DOM text of its words, joined by the gaps the browser laid out; nothing is read from an image.
A line is dropped, and counted, when anything about it is uncertain: a word the hit test finds covered or clipped near it, a
character outside the charset, an ambiguous gap, a held-out font, a superscript, ink that does not match the DOM box.
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
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from data import charset as cs

BASE = Path(__file__).resolve().parents[1]
OUT_H = 32
FILE_OK = re.compile(r"^(?:[^.]*|.*\.(?:html?|php|aspx?|shtml|jsp))$", re.I)  # no extension, or a page extension
COPYRIGHT = re.compile(r"©|\(c\)\s*\d{4}|copyright\s*(?:\d{4}|by)|used with permission|all rights reserved", re.I)

# ------------------------------------------------------------------------------------------------------------------ crawler


class Politeness:
    """One request at a time per host, at least `interval` seconds apart (a robots.txt Crawl-delay raises it), robots.txt honoured."""

    def __init__(self, token: str, info_url: str, interval: float):
        self.ua = f"{token}/1.0 (+{info_url}; text-line training data from public-domain pages)"
        self.token, self.interval = token, interval
        self.robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self.last: dict[str, float] = {}
        self.delay: dict[str, float] = {}

    def _robots(self, host: str):
        if host not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                req = urllib.request.Request(f"https://{host}/robots.txt", headers={"User-Agent": self.ua})
                with urllib.request.urlopen(req, timeout=30) as r:
                    rp.parse(r.read().decode("utf-8", "replace").splitlines())
            except urllib.error.HTTPError as e:
                if e.code >= 500:
                    rp = None  # robots.txt cannot be read: do not crawl
                else:
                    rp.parse([])  # none published
            except Exception:  # noqa: BLE001 - unreachable host
                rp = None
            self.robots[host] = rp
            d = rp.crawl_delay(self.token) if rp else None
            self.delay[host] = max(self.interval, float(d or 0))
        return self.robots[host]

    def allowed(self, url: str) -> bool:
        host = urllib.parse.urlparse(url).netloc
        rp = self._robots(host)
        return bool(rp and rp.can_fetch(self.token, url))

    def wait(self, host: str) -> None:
        self._robots(host)
        gap = self.delay.get(host, self.interval) - (time.time() - self.last.get(host, 0))
        if gap > 0:
            time.sleep(gap)
        self.last[host] = time.time()

    def get(self, url: str) -> str | None:
        host = urllib.parse.urlparse(url).netloc
        if not self.allowed(url):
            return None
        self.wait(host)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept": "text/html"})
            with urllib.request.urlopen(req, timeout=40) as r:
                if "html" not in (r.headers.get("Content-Type") or ""):
                    return None
                return r.read(2_000_000).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return None


def path_key(path: str) -> str:
    """One spelling per page: no empty path, no trailing slash except the root."""
    return "/" if path in ("", "/") else path.rstrip("/")


def clean_url(base: str, href: str, pub: dict) -> str | None:
    u = urllib.parse.urljoin(base, html.unescape(href.strip()))
    p = urllib.parse.urlparse(u)
    if p.scheme != "https" or p.netloc != pub["host"] or p.query:
        return None
    path = urllib.parse.unquote(p.path) or "/"
    if not FILE_OK.match(path.rsplit("/", 1)[-1] or "x"):
        return None
    if any(x in path for x in pub.get("exclude", [])):
        return None
    if path != "/" and not any(path.startswith(x) for x in pub["prefixes"]):
        return None
    return f"https://{p.netloc}{path_key(p.path)}"


def discover(cfg: dict, only: str | None = None) -> None:
    src = yaml.safe_load((BASE / cfg["sources"]).read_text(encoding="utf-8"))
    crawler = src["crawler"]
    pol = Politeness(crawler["token"], crawler["info_url"], crawler["min_interval_s"])
    d = cfg["discover"]
    lines: list[str] = []
    for pub in src["publishers"]:
        if only and pub["id"] != only:
            continue
        rng = random.Random(f"{cfg['seed']}:{pub['id']}")
        seen: set[str] = set()
        frontier: list[str] = []
        for s in pub["seeds"]:
            u = f"https://{pub['host']}{path_key(s)}"
            if u not in seen:
                seen.add(u)
                frontier.append(u)
        pool = list(frontier)
        fetched = 0
        queue = list(frontier)
        while queue and fetched < d["fetch_limit"]:
            u = queue.pop(0)
            body = pol.get(u)
            fetched += 1
            if body is None:
                continue
            new = []
            for m in re.finditer(r'(?is)<a\s[^>]*?href="([^"#]+)"', body):
                c = clean_url(u, m.group(1), pub)
                if c and c not in seen:
                    seen.add(c)
                    new.append(c)
            pool += new
            rng.shuffle(new)
            if fetched <= len(pub["seeds"]):  # after the seeds, follow a seeded sample of what they linked to
                queue += new[:3]
        pool = sorted(set(pool))
        rng.shuffle(pool)
        pick = pool[: d["pages_per_publisher"]]
        print(f"{pub['id']}: fetched {fetched}, {len(pool)} candidate pages, {len(pick)} pinned", flush=True)
        lines += [f"{pub['id']}\t{u}" for u in sorted(pick)]
    path = BASE / cfg["urls"]
    if only and path.exists():  # replace one publisher's lines, keep the others
        keep = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith(only + "\t")]
        lines = sorted(keep + lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} pages in {path}")


# ----------------------------------------------------------------------------------------------------------------- browser

FREEZE_CSS = """
*,*::before,*::after{animation:none !important;transition:none !important;caret-color:transparent !important;scroll-behavior:auto !important;
  background-image:none !important;hyphens:none !important;text-overflow:clip !important;font-variant-caps:normal !important;
  text-shadow:none !important}
*::before,*::after{content:none !important}
img,picture,video,audio,canvas,svg,iframe,object,embed,map{visibility:hidden !important}
ul,ol,li,summary{list-style:none !important}
summary::-webkit-details-marker{display:none !important}
input,select,textarea,option{color:transparent !important;-webkit-text-fill-color:transparent !important}
::placeholder{color:transparent !important;opacity:0 !important}
"""

KEYS_JS = r"""() => {
  const keys = new Map();
  document.querySelectorAll('[data-font-key]').forEach(e => e.removeAttribute('data-font-key'));
  for (const el of document.querySelectorAll('*')) {
    if (![...el.childNodes].some(n => n.nodeType === 3 && n.data.trim())) continue;
    const cs = getComputedStyle(el);
    const key = cs.fontFamily + '|' + cs.fontWeight + '|' + cs.fontStyle;
    if (!keys.has(key)) keys.set(key, {i: keys.size, family: cs.fontFamily, weight: cs.fontWeight, style: cs.fontStyle});
    el.setAttribute('data-font-key', String(keys.get(key).i));
  }
  return [...keys.values()];
}"""
SUBSTITUTE_JS = r"""(bad) => {
  const set = new Set(bad); let n = 0;
  for (const el of document.querySelectorAll('*')) {
    if (![...el.childNodes].some(nd => nd.nodeType === 3 && nd.data.trim())) continue;
    const cs = getComputedStyle(el);
    if (set.has(cs.fontFamily + '|' + cs.fontWeight + '|' + cs.fontStyle)) { el.style.setProperty('font-family', "'HeldOutStandIn', sans-serif", 'important'); n++; }
  }
  return n;
}"""


# Visible words with their rectangles and font keys. A word is visible when a hit test at its centre (and a quarter in from each end)
# lands on its own element and no clipping ancestor cuts it; words that fail are returned as blockers, so a line next to one is dropped.
EXTRACT_JS = r"""([W, H, css]) => {
  const out = [], blockers = [], keys = new Map();
  document.querySelectorAll('[data-font-key]').forEach(e => e.removeAttribute('data-font-key'));
  const stats = {hidden: 0, partial: 0, outside: 0, pua: 0, controls: 0, nodes: 0};
  const roots = [document.body];
  const collect = (node) => node.querySelectorAll('*').forEach(e => { if (e.shadowRoot) { roots.push(e.shadowRoot); collect(e.shadowRoot); } });
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
  const clipped = (el, r) => {
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
  const strip = (s) => s.replace(/[\u00ad\u200b\u200c\u200d\u2060\ufeff]/g, '');
  const puaEls = [], texts = [];
  for (const r of roots) { const tw = document.createTreeWalker(r, NodeFilter.SHOW_TEXT); for (let n = tw.nextNode(); n; n = tw.nextNode()) texts.push(n); }
  for (const nd of texts) {
    const el = nd.parentElement;
    if (!el || SKIP.has(el.tagName) || el.closest('input,textarea,select,option')) continue;
    const cs = getComputedStyle(el);
    if (!el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true, opacityProperty: true, visibilityProperty: true})) continue;
    if (/rgba?\(\s*\d+,\s*\d+,\s*\d+,\s*0\s*\)/.test(cs.color) || cs.color === 'transparent') continue;
    const tf = cs.textTransform, fs = parseFloat(cs.fontSize);
    const key = cs.fontFamily + '|' + cs.fontWeight + '|' + cs.fontStyle;
    if (!keys.has(key)) keys.set(key, {i: keys.size, family: cs.fontFamily, weight: cs.fontWeight, style: cs.fontStyle});
    el.setAttribute('data-font-key', String(keys.get(key).i));
    const fk = keys.get(key).i;
    const va = cs.verticalAlign;
    const sup = !!el.closest('sup,sub') || /^(super|sub|text-top|text-bottom)$/.test(va) || (/px$|%$/.test(va) && parseFloat(va) !== 0);
    const re = /\S+/g; let m;
    while ((m = re.exec(nd.data))) {
      let t = strip(m[0]);
      if (!t) continue;
      stats.nodes++;
      if (/[\ue000-\uf8ff]/.test(t)) { stats.pua++; puaEls.push(el); continue; }
      if (/[\u0000-\u001f\u007f-\u009f]/.test(t)) { stats.controls++; continue; }
      if (tf === 'uppercase') t = t.toUpperCase(); else if (tf === 'lowercase') t = t.toLowerCase();
      const rg = document.createRange(); rg.setStart(nd, m.index); rg.setEnd(nd, m.index + m[0].length);
      const rs = [...rg.getClientRects()].filter(q => q.width > 0 && q.height > 0);
      if (!rs.length) continue;
      const f = rs[0], fc = (f.top + f.bottom) / 2;
      let parts;
      if (rs.every(q => Math.abs((q.top + q.bottom) / 2 - fc) < f.height * 0.5)) {
        parts = [{x0: Math.min(...rs.map(q => q.left)), y0: Math.min(...rs.map(q => q.top)), x1: Math.max(...rs.map(q => q.right)), y1: Math.max(...rs.map(q => q.bottom)), s: t}];
      } else {  // a word wrapped across lines: one fragment per printed line, cut by character
        parts = []; let cur = null; const src = nd.data.slice(m.index, m.index + m[0].length);
        for (let k = 0; k < src.length; k++) {
          const c = document.createRange(); c.setStart(nd, m.index + k); c.setEnd(nd, m.index + k + 1);
          const q = [...c.getClientRects()].find(q => q.width > 0 && q.height > 0);
          if (!q) { if (cur) cur.s += src[k]; continue; }
          if (cur && Math.abs((q.top + q.bottom) / 2 - cur.cy) < q.height * 0.5) {
            cur.s += src[k]; cur.x0 = Math.min(cur.x0, q.left); cur.y0 = Math.min(cur.y0, q.top); cur.x1 = Math.max(cur.x1, q.right); cur.y1 = Math.max(cur.y1, q.bottom);
          } else { cur = {s: src[k], x0: q.left, y0: q.top, x1: q.right, y1: q.bottom, cy: (q.top + q.bottom) / 2}; parts.push(cur); }
        }
        parts = parts.map(p => ({...p, s: strip(p.s)})).filter(p => p.s);
        if (tf === 'uppercase') parts.forEach(p => { p.s = p.s.toUpperCase(); }); else if (tf === 'lowercase') parts.forEach(p => { p.s = p.s.toLowerCase(); });
      }
      for (const p of parts) {
        const rect = {left: p.x0, right: p.x1, top: p.y0, bottom: p.y1};
        let v = verdictOf(el, rect);
        if (v === 'visible') { const c = clipped(el, rect); v = c === 0 ? 'visible' : c === 1 ? 'hidden' : 'partial'; }
        if (v !== 'visible') { stats[v]++; blockers.push([p.x0, p.y0, p.x1, p.y1]); continue; }
        out.push({t: p.s, x0: p.x0, y0: p.y0, x1: p.x1, y1: p.y1, fs, fk, sup});
      }
    }
  }
  puaEls.forEach(e => e.style.setProperty('visibility', 'hidden', 'important'));
  const all = out.map(o => o.t).join(' ');
  return {words: out, blockers, keys: [...keys.values()].map(v => ({i: v.i, family: v.family, weight: v.weight, style: v.style})), stats, copyright: all.search(COPYRIGHT_JS) >= 0};
}""".replace("COPYRIGHT_JS", r"/©|\(c\)\s*\d{4}|copyright\s*(?:\d{4}|by)|used with permission|all rights reserved/i")


def norm_family(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def heldout_patterns(cfg: dict) -> list[str]:
    """Normalised family names that must not train: the configured patterns plus every non-train family in the font manifests."""
    pats = set(cfg["heldout_font_patterns"])
    for name in ("fonts.yaml", "fonts_ui.yaml"):
        p = BASE / "data" / name
        if p.exists():
            for e in yaml.safe_load(p.read_text(encoding="utf-8"))["fonts"]:
                if e.get("split") != "train":
                    pats.add(norm_family(e["family"]))
    return sorted(pats)


def font_is_held_out(css_family: str, platform_names: list[str], held: list[str]) -> bool:
    """True when the browser used a held-out family (platform names) or the CSS names one first (the stack's primary entry)."""
    names = [norm_family(f) for f in platform_names]
    first = norm_family(css_family.split(",")[0].strip("\"' "))
    return any(p in n for p in held for n in names) or any(p in first for p in held)


def substitute_font_css(cfg: dict) -> str:
    """@font-face for the font that stands in for held-out families on a page: one pinned file of the training pool, SHA-256 checked."""
    import base64

    from data.fonts import font_root

    want = cfg["substitute_font"]
    entry = next(e for e in yaml.safe_load((BASE / "data" / "fonts.yaml").read_text(encoding="utf-8"))["fonts"] if e["file"] == want)
    path = font_root() / want
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(entry["url"], timeout=120).read()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], "font file does not match the manifest"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    b64 = base64.b64encode(path.read_bytes()).decode()
    return ("@font-face{font-family:'HeldOutStandIn';font-weight:100 900;font-stretch:75% 100%;"
            f"src:url(data:font/ttf;base64,{b64}) format('truetype')}}")  # fmt: skip


def page_draw(seed: int, url: str, cfg: dict) -> tuple[list, float, str]:
    """Viewport, device-pixel ratio and anti-aliasing for one page: a function of (seed, url) only."""
    rng = random.Random(f"{seed}:{url}")
    r = cfg["render"]
    vp = rng.choices(r["viewports"], weights=r["viewport_weights"])[0]
    dpr = rng.choice(r["dpr"])
    if vp[0] < 500:
        dpr = 2.0
    aa = rng.choices(list(r["aa"]), weights=list(r["aa"].values()))[0]
    return vp, float(dpr), aa


def platform_fonts(cdp, keys: list[dict], per_key: int = 40) -> dict[int, list[str]]:
    """Font families the browser actually used, as the union over up to `per_key` elements of each (family, weight, style) key. An
    element whose text is drawn by another element's font reports nothing, so one element is not enough. Empty = not established."""
    out: dict[int, list[str]] = {}
    cdp.send("DOM.enable")
    cdp.send("CSS.enable")
    root = cdp.send("DOM.getDocument", {"depth": 0})["root"]["nodeId"]
    for k in keys:
        names: list[str] = []
        try:
            ids = cdp.send("DOM.querySelectorAll", {"nodeId": root, "selector": f'[data-font-key="{k["i"]}"]'})["nodeIds"]
            for nid in ids[:per_key]:
                for f in cdp.send("CSS.getPlatformFontsForNode", {"nodeId": nid})["fonts"]:
                    if f["familyName"] not in names:
                        names.append(f["familyName"])
        except Exception:  # noqa: BLE001 - an element inside a shadow root
            pass
        out[k["i"]] = names
    return out


def load_allowlist(cfg: dict) -> list[str]:
    """Normalised family-name prefixes of fonts whose licence was read from the family's own licence file (data/font_allowlist.yaml)."""
    rows = yaml.safe_load((BASE / cfg["font_allowlist"]).read_text(encoding="utf-8"))["families"]
    return sorted(norm_family(r["family"]) for r in rows)


def font_allowed(platform_names: list[str], allow: list[str]) -> bool:
    """True only when the browser reported at least one font and every reported font (minus the "(Fontations)" backend suffix and icon
    fonts) starts with an allowlisted family. Unreported or unknown fonts are not allowed."""
    names = [norm_family(n.replace("(Fontations)", "")) for n in platform_names]
    names = [n for n in names if not n.startswith("fontawesome")]
    return bool(names) and all(any(n.startswith(a) for a in allow) for n in names)


# -------------------------------------------------------------------------------------------------------------------- lines


def group_lines(words: list[dict], cfg: dict) -> list[dict]:
    """Visual lines from word rectangles: same row (centres within a fifth of the smaller height), split at wide gaps."""
    lc = cfg["lines"]
    words = sorted(words, key=lambda w: ((w["y0"] + w["y1"]) / 2, w["x0"]))
    rows: list[dict] = []
    for w in words:
        cy, h = (w["y0"] + w["y1"]) / 2, w["y1"] - w["y0"]
        best = None
        for r in reversed(rows[-12:]):
            if abs(cy - r["cy"]) <= 0.2 * min(h, r["h"]):
                best = r
                break
        if best is None:
            rows.append({"cy": cy, "h": h, "w": [w]})
        else:
            best["w"].append(w)
            best["cy"] = sum((x["y0"] + x["y1"]) / 2 for x in best["w"]) / len(best["w"])
            best["h"] = min(best["h"], h)
    out = []
    for r in rows:
        ws = sorted(r["w"], key=lambda w: w["x0"])
        seg = [ws[0]]
        segs = [seg]
        for a, b in zip(ws, ws[1:], strict=False):
            gap = b["x0"] - a["x1"]
            if gap > lc["split_gap_em"] * max(a["fs"], b["fs"]):
                seg = [b]
                segs.append(seg)
            else:
                seg.append(b)
        for seg in segs:
            text, flag = seg[0]["t"], None
            for a, b in zip(seg, seg[1:], strict=False):
                gap = (b["x0"] - a["x1"]) / a["fs"]
                if gap < -0.3:
                    flag = "overlap"
                if gap < lc["join_gap_em"]:
                    text += b["t"]
                elif gap >= lc["space_gap_em"]:
                    text += " " + b["t"]
                else:
                    flag = flag or "gap"
                    text += " " + b["t"]
            out.append({"words": seg, "text": text, "flag": flag, "x0": min(w["x0"] for w in seg), "x1": max(w["x1"] for w in seg),
                        "y0": min(w["y0"] for w in seg), "y1": max(w["y1"] for w in seg), "em": float(np.median([w["fs"] for w in seg]))})  # fmt: skip
    return out


def pixel_check(G: np.ndarray, ln: dict, s: float, lc: dict) -> tuple[dict | None, str]:
    """Does the screenshot agree with the DOM box of the line? Returns the ink box (device px) and measurements, or a reason."""
    Hs, Ws = G.shape
    em = ln["em"] * s
    X0, X1 = ln["x0"] * s, ln["x1"] * s
    Y0, Y1 = ln["y0"] * s, ln["y1"] * s
    hd = Y1 - Y0
    tol = lc["edge_tol_em"] * em
    # background: the commonest gray level around the box
    zx0, zx1 = int(max(0, X0 - 0.5 * em)), int(min(Ws, X1 + 0.5 * em))
    zy0, zy1 = int(max(0, Y0 - 1)), int(min(Hs, Y1 + 1))
    if zx1 - zx0 < 4 or zy1 - zy0 < 4:
        return None, "edge"
    zone = G[zy0:zy1, zx0:zx1]
    bg = float(np.bincount(zone.ravel(), minlength=256).argmax())
    bx0, bx1, by0, by1 = int(max(0, X0)), int(min(Ws, X1)), int(max(0, Y0)), int(min(Hs, Y1))
    if bx1 - bx0 < 3 or by1 - by0 < 3:
        return None, "edge"
    diff_box = np.abs(G[by0:by1, bx0:bx1].astype(np.float32) - bg)
    contrast = float(np.percentile(diff_box, 99.5))
    if contrast < lc["min_contrast"]:
        ln["detail"] = f"contrast {contrast:.0f} bg {bg:.0f}"
        return None, "faint"
    thr = max(10.0, lc["ink_threshold"] * contrast)
    ex0, ex1 = int(max(0, X0 - tol)), int(min(Ws, X1 + tol))
    ey0, ey1 = int(max(0, Y0 - lc["vert_tol"] * hd)), int(min(Hs, Y1 + lc["vert_tol"] * hd))
    sub = np.abs(G[ey0:ey1, ex0:ex1].astype(np.float32) - bg) > thr
    ys, xs = np.nonzero(sub)
    if len(ys) < 6:
        return None, "noink"
    ix0, ix1, iy0, iy1 = xs.min() + ex0, xs.max() + 1 + ex0, ys.min() + ey0, ys.max() + 1 + ey0
    ih = iy1 - iy0
    if ix0 > X0 + lc["end_slack_em"] * em or ix1 < X1 - lc["end_slack_em"] * em:
        ln["detail"] = f"ink {ix0}-{ix1} dom {X0:.0f}-{X1:.0f} em {em:.1f}"
        return None, "ends"
    if not (lc["ink_h_range"][0] * hd <= ih <= lc["ink_h_range"][1] * hd):
        ln["detail"] = f"ink_h {ih} dom_h {hd:.1f} em {em:.1f}"
        return None, "inkheight"
    if ih < lc["min_ink_h"]:
        return None, "tiny"
    if em > lc["max_em_px"]:
        return None, "huge"
    for w in ln["words"]:  # every word shows ink across its own width
        wx0, wx1 = int(max(0, w["x0"] * s)), int(min(Ws, w["x1"] * s))
        if wx1 - wx0 < 2:
            continue
        cols = sub[:, max(0, wx0 - ex0): max(0, wx1 - ex0)].any(axis=0) if wx1 > ex0 else np.zeros(1, bool)
        if cols.size == 0 or cols.mean() < lc["word_cover_min"] * (0.6 if len(w["t"]) == 1 else 1.0):
            ln["detail"] = f"word {w['t']!r} cover {cols.mean():.2f}"
            return None, "wordblank"
    return {"ix0": int(ix0), "ix1": int(ix1), "iy0": int(iy0), "iy1": int(iy1), "bg": int(bg), "contrast": round(contrast, 1), "thr": thr}, "ok"


def engine_crop(ink: dict, typical: float, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """The page reader's crop (LinePreprocessor.CropBox): effective height max(ink, 0.85 x typical); half of it left and right,
    a quarter above and below the ink box, clamped to the page."""
    ih = ink["iy1"] - ink["iy0"]
    eff = max(ih, 0.85 * typical)
    cy = ink["iy0"] + ih / 2.0
    half, margin = eff * 0.75, eff * 0.5
    Hs, Ws = shape
    return (max(0, int(np.floor(ink["ix0"] - margin))), max(0, int(np.floor(cy - half))),
            min(Ws, int(np.ceil(ink["ix1"] + margin))), min(Hs, int(np.ceil(cy + half))))  # fmt: skip


def margin_check(G: np.ndarray, crop: tuple[int, int, int, int], ink: dict, lc: dict) -> tuple[bool, int]:
    """Ink in the crop's side margins that no label word explains. A drawn rule (a column inked over most of the line) is allowed."""
    cx0, cy0, cx1, cy1 = crop
    ih = ink["iy1"] - ink["iy0"]
    r0, r1 = int(ink["iy0"] + 0.2 * ih), int(ink["iy1"] - 0.2 * ih)
    if r1 <= r0:
        return True, 0
    bad = 0
    for a, b in ((cx0, ink["ix0"]), (ink["ix1"], cx1)):
        if b <= a:
            continue
        m = np.abs(G[r0:r1, a:b].astype(np.float32) - ink["bg"]) > ink["thr"]
        col = m.mean(axis=0)
        bad += int(((col > 0) & (col < 0.8)).sum())
    return bad <= lc["margin_cols_max"], bad


def light_on_dark(G: np.ndarray, box: tuple[int, int, int, int], bg: float) -> bool:
    """True when the text in the box is lighter than its background (the ink end that lies further from the background is the bright one)."""
    x0, y0, x1, y1 = box
    c = G[max(0, y0) : y1, max(0, x0) : x1]
    if c.size == 0:
        return False
    hi, lo = float(np.percentile(c, 99.5)), float(np.percentile(c, 0.5))
    return hi - bg > bg - lo


def cut_windows(job: dict, rec: dict, G: np.ndarray, s: float, vp: list, lines: list[dict], cands: list[dict]) -> list[dict]:
    """Viewport-sized windows of the page with detector labels, for training the text detector.

    The detector sees a whole screenshot scaled to 1024 px on its long side, so a window is one viewport (what a screenshot of the
    site shows), not the 6000 px page. Lines that passed every check are positives with their DOM line box; every other line the DOM
    holds (dropped for a covered neighbour, a charset or pixel check, a held-out font or one outside the font allowlist) is an ignore region, because the pixels there
    are text we could not verify, not background. A line cut by a window edge is ignored in that window too.
    """
    cfg = job["cfg"]["windows"]
    Hs = G.shape[0]
    wh = max(64, int(round(vp[1] * s)))
    offs = list(range(0, Hs, wh))
    offs = [o for o in offs if Hs - o >= 0.3 * wh or o == 0]
    rng = random.Random(f"win:{job['cfg']['seed']}:{job['url']}")
    keep = [offs[0]] + rng.sample(offs[1:], min(len(offs) - 1, cfg["max_per_page"] - 1))
    good = {id(c) for c in cands}
    out = []
    for oy in sorted(keep):
        y1w = min(Hs, oy + wh)
        pos, ign, flags, ems = [], [], [], []
        for ln in lines:
            b = [ln["x0"] * s, ln["y0"] * s - oy, ln["x1"] * s, ln["y1"] * s - oy]
            if b[3] <= 0 or b[1] >= y1w - oy:
                continue
            inside = b[1] >= 0 and b[3] <= y1w - oy
            if id(ln) in good and inside:
                pos.append([round(v, 1) for v in b])
                ink = ln["ink"]
                flags.append(int(light_on_dark(G, (int(b[0]), int(b[1] + oy), int(b[2]), int(b[3] + oy)), float(ink["bg"]))))
                ems.append(round(ln["em"] * s, 1))
            else:
                ign.append([round(v, 1) for v in b])
        if len(pos) < cfg["min_lines"]:
            continue
        wid = f"{rec['publisher']}-{hashlib.sha1(job['url'].encode()).hexdigest()[:8]}-w{oy // wh}"
        d = Path(job["windows_dir"]) / job["split"] / "images"
        d.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        Image.fromarray(G[oy:y1w]).save(buf, format="PNG")
        (d / f"{wid}.png").write_bytes(buf.getvalue())
        out.append({"id": wid, "publisher": rec["publisher"], "url": job["url"], "split": job["split"], "w": int(G.shape[1]), "h": int(y1w - oy),
                    "dpr": rec["dpr"], "aa": rec["aa"], "viewport": vp, "lines": pos, "ignore": ign, "dark": flags, "em_px": ems,
                    "sha256": hashlib.sha256(buf.getvalue()).hexdigest()})  # fmt: skip
    return out


def process_page(job: dict) -> dict:
    """Renders one page; returns its record, the kept line rows (with PNG bytes) and what was dropped."""
    from playwright.sync_api import TimeoutError as PwTimeout

    cfg, url, pub = job["cfg"], job["url"], job["pub"]
    lc = cfg["lines"]
    drops: Counter = Counter()
    rec = {"publisher": pub, "url": url, "result": "ok"}
    vp, dpr, aa = page_draw(cfg["seed"], url, cfg)
    rec.update(viewport=vp, dpr=dpr, aa=aa)
    br = job["browser"](aa)
    ua = f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{br.version} Safari/537.36 {job['ua_suffix']}"
    mobile = vp[0] < 500
    ctx = br.new_context(viewport={"width": vp[0], "height": vp[1]}, device_scale_factor=dpr, user_agent=ua, locale="en-US",
                         reduced_motion="reduce", is_mobile=mobile, has_touch=mobile,
                         bypass_csp=True)  # the substitute font is injected as a data URI  # fmt: skip
    page = ctx.new_page()
    page.route("**/*", lambda route: route.abort() if route.request.resource_type in ("image", "media") else route.continue_())
    try:
        t0 = time.time()
        resp = page.goto(url, wait_until="load", timeout=cfg["render"]["page_timeout_ms"])
        if resp is None or resp.status >= 400:
            rec["result"] = f"http {resp.status if resp else 'none'}"
            return {"rec": rec, "rows": [], "drops": drops}
        rec["status"] = resp.status
        try:
            rec["response_sha256"] = hashlib.sha256(resp.body()).hexdigest()
        except Exception:  # noqa: BLE001
            rec["response_sha256"] = None
        try:
            page.wait_for_load_state("networkidle", timeout=6000)
        except PwTimeout:
            pass
        page.add_style_tag(content=FREEZE_CSS)
        page.evaluate("document.fonts.ready")
        total = int(page.evaluate("document.documentElement.scrollHeight"))
        step = max(400, int(vp[1] * 0.8))
        for y in range(0, min(total, 12000), step):  # trigger lazy content, then come back
            page.evaluate("y => window.scrollTo(0, y)", y)
            page.wait_for_timeout(80)
        H = int(min(max(vp[1], total), cfg["render"]["max_page_css_px"], cfg["render"]["max_shot_px"] / dpr))
        page.set_viewport_size({"width": vp[0], "height": H})
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        page.evaluate("document.fonts.ready")
        cdp = ctx.new_cdp_session(page)
        keys0 = page.evaluate(KEYS_JS)
        pf0 = platform_fonts(cdp, keys0)
        bad0 = [f"{k['family']}|{k['weight']}|{k['style']}" for k in keys0 if font_is_held_out(k["family"], pf0.get(k["i"], []), job["heldout"])]
        rec["substituted_keys"] = len(bad0)
        if bad0 and job["sub_font"]:
            page.add_style_tag(content=job["sub_font"])
            page.evaluate(SUBSTITUTE_JS, bad0)
            page.evaluate("document.fonts.load(\"16px HeldOutStandIn\").then(() => document.fonts.ready)")
            page.wait_for_timeout(300)
        res = page.evaluate(EXTRACT_JS, [vp[0], H, FREEZE_CSS])
        if res["copyright"]:
            rec["result"] = "copyright or permission notice"
            return {"rec": rec, "rows": [], "drops": drops}
        fonts = platform_fonts(cdp, res["keys"])
        png = page.screenshot(animations="disabled")
        rec["seconds"] = round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001 - one bad page must not stop the run
        rec["result"] = f"{type(e).__name__}: {str(e)[:100]}"
        return {"rec": rec, "rows": [], "drops": drops}
    finally:
        ctx.close()

    img = Image.open(io.BytesIO(png)).convert("L")
    G = np.asarray(img, dtype=np.uint8)
    s = G.shape[1] / vp[0]
    keyinfo = {k["i"]: k for k in res["keys"]}
    held = job["heldout"]

    allow = job["allow"]

    def bad_font(k: int) -> bool:
        return font_is_held_out(keyinfo[k]["family"], fonts.get(k, []), held)

    lines = group_lines(res["words"], cfg)
    rec.update(words=len(res["words"]), blockers=len(res["blockers"]), candidates=len(lines), **{f"js_{k}": v for k, v in res["stats"].items()})
    cands = []
    all_text_lines = []
    charset = job["charset_set"]
    for ln in lines:
        ln["text"] = unicodedata.normalize("NFC", ln["text"])
        t = ln["text"]
        if ln["flag"]:
            drops[ln["flag"]] += 1
        elif any(c not in charset for c in t):
            drops["charset"] += 1
        elif len(t) < lc["min_chars"] or len(t) > lc["max_chars"] or not any(c.isalnum() for c in t):
            drops["length_or_symbols"] += 1
        elif any(bad_font(w["fk"]) for w in ln["words"]):
            drops["heldout_font"] += 1
        elif not all(font_allowed(fonts.get(w["fk"], []), allow) for w in ln["words"]):
            drops["font_not_allowlisted"] += 1
        elif any(w["sup"] for w in ln["words"]):
            drops["superscript"] += 1
        elif max(w["y1"] - w["y0"] for w in ln["words"]) > lc["max_height_ratio"] * min(w["y1"] - w["y0"] for w in ln["words"]):
            drops["mixed_sizes"] += 1
        else:
            all_text_lines.append(ln)
            pad = 0.5 * ln["em"]
            if any(b[2] > ln["x0"] - pad and b[0] < ln["x1"] + pad and b[3] > ln["y0"] and b[1] < ln["y1"] for b in res["blockers"]):
                drops["covered_neighbour"] += 1
                continue
            ink, why = pixel_check(G, ln, s, lc)
            if ink is None:
                drops[f"pixels_{why}"] += 1
                if job.get("debug_dir") and len(rec.setdefault("examples", [])) < 40:
                    rec["examples"].append([why, t[:40], ln.get("detail", "")])
                continue
            ln["ink"] = ink
            cands.append(ln)
    wins = cut_windows(job, rec, G, s, vp, lines, cands) if job.get("windows_dir") else []
    if not cands:
        rec["kept"] = 0
        rec["drops"] = dict(drops)
        return {"rec": rec, "rows": [], "drops": drops, "windows": wins}
    ratio = float(np.median([(c["ink"]["iy1"] - c["ink"]["iy0"]) / ((c["y1"] - c["y0"]) * s) for c in cands]))
    typical = float(np.median([(c["y1"] - c["y0"]) * s * ratio for c in all_text_lines]))  # the page reader's median over every found line
    rows = []
    for k, ln in enumerate(cands):
        crop = engine_crop(ln["ink"], typical, G.shape)
        ok, nbad = margin_check(G, crop, ln["ink"], lc)
        if not ok:
            drops["pixels_margin_ink"] += 1
            continue
        c = img.crop(crop)
        ow = max(8, round(c.width * OUT_H / c.height))
        if ow > lc["max_width"]:
            drops["too_wide"] += 1
            continue
        small = c.resize((ow, OUT_H), Image.BICUBIC)  # PIL's bicubic reduces with a widened kernel: the antialiased scale of the library
        buf = io.BytesIO()
        small.save(buf, format="PNG")
        names = sorted({f for w in ln["words"] for f in fonts.get(w["fk"], [])})
        css = sorted({keyinfo[w["fk"]]["family"].split(",")[0].strip("\"' ") for w in ln["words"]})
        rows.append({
            "text": ln["text"], "w": ow, "h": OUT_H, "publisher": pub, "url": url, "k": k, "dpr": dpr, "aa": aa, "viewport": vp,
            "css_px": round(ln["em"], 2), "em_px": round(ln["em"] * s, 2), "ink_h": ln["ink"]["iy1"] - ln["ink"]["iy0"], "typical_h": typical,
            "font": names or css, "css_family": css, "substituted": any("HeldOutStandIn" in keyinfo[w["fk"]]["family"] for w in ln["words"]), "bg": ln["ink"]["bg"], "contrast": ln["ink"]["contrast"], "margin_cols": nbad,
            "crop": list(crop), "scale": round(OUT_H / c.height, 4), "dom_box": [round(ln[a] * s, 1) for a in ("x0", "y0", "x1", "y1")],
            "png": buf.getvalue(),
        })  # fmt: skip
        if len(rows) >= lc["max_lines_per_page"]:
            break
    rec["kept"] = len(rows)
    rec["drops"] = dict(drops)
    dbg = job.get("debug_dir")
    if dbg and rows:
        draw_debug(img, cands, rows, res["blockers"], s, Path(dbg) / f"{pub}-{hashlib.sha1(url.encode()).hexdigest()[:8]}.png")
    return {"rec": rec, "rows": rows, "drops": drops, "windows": wins}


def draw_debug(img: Image.Image, cands: list[dict], rows: list[dict], blockers: list, s: float, path: Path) -> None:
    """The screenshot with DOM boxes (blue), ink boxes (green) and crops (red) of the kept lines, for looking at before running at scale."""
    from PIL import ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    d = ImageDraw.Draw(rgb)
    for r in rows:
        d.rectangle(r["dom_box"], outline=(0, 90, 255))
        d.rectangle(r["crop"], outline=(255, 0, 0))
    for b in blockers:
        d.rectangle([v * s for v in b], outline=(255, 0, 255))
    for c in cands:
        i = c["ink"]
        d.rectangle([i["ix0"], i["iy0"], i["ix1"], i["iy1"]], outline=(0, 170, 0))
    rgb.crop((0, 0, rgb.width, min(rgb.height, 1800))).save(path)


# -------------------------------------------------------------------------------------------------------------------- driver

_W: dict = {}


def _init(cfg: dict, args: dict):
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browsers: dict = {}

    def browser(aa: str):
        if aa not in browsers:
            flag = "--enable-lcd-text" if aa == "subpixel" else "--disable-lcd-text"
            browsers[aa] = pw.chromium.launch(args=[flag], executable_path=os.environ.get("ONEOCR_CHROMIUM") or None)
        return browsers[aa]

    charset = cs.load(BASE / cfg["charset"])
    _W.update(cfg=cfg, browser=browser, pw=pw, charset_set=frozenset(charset[1:]), heldout=heldout_patterns(cfg),
              allow=load_allowlist(cfg), sub_font=substitute_font_css(cfg) if cfg.get("substitute_font") else "", **args)  # fmt: skip


def render_group(group: dict) -> list[dict]:
    """All pages of one publisher in order, throttled: a worker owns whole publishers, so a host sees one request stream."""
    out = []
    pol = _W.setdefault("pol", Politeness(_W["token"], _W["info_url"], _W["interval"]))
    for url in group["urls"]:
        if not pol.allowed(url):
            out.append({"rec": {"publisher": group["pub"], "url": url, "result": "robots.txt"}, "rows": [], "drops": Counter()})
            continue
        pol.wait(group["host"])
        job = {"cfg": _W["cfg"], "url": url, "pub": group["pub"], "browser": _W["browser"], "ua_suffix": _W["ua_suffix"],
               "heldout": _W["heldout"], "sub_font": _W["sub_font"], "allow": _W["allow"], "charset_set": _W["charset_set"],
               "debug_dir": _W.get("debug_dir"), "windows_dir": _W.get("windows_dir"), "split": group["split"]}  # fmt: skip
        out.append(process_page(job))
    return out


def render(cfg: dict, a) -> None:
    src = yaml.safe_load((BASE / cfg["sources"]).read_text(encoding="utf-8"))
    crawler = src["crawler"]
    pubs = {p["id"]: p for p in src["publishers"]}
    by: dict[str, list[str]] = {}
    for ln in (BASE / cfg["urls"]).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            pid, url = ln.split("\t")
            by.setdefault(pid, []).append(url)
    if a.limit_pages:
        by = {k: v[: a.limit_pages] for k, v in by.items()}
    if a.only:
        by = {k: v for k, v in by.items() if k in a.only.split(",")}
    groups = [{"pub": k, "host": pubs[k]["host"], "split": pubs[k]["split"], "urls": v} for k, v in sorted(by.items())]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = {"token": crawler["token"], "info_url": crawler["info_url"], "interval": crawler["min_interval_s"],
            "ua_suffix": f"{crawler['token']}/1.0 (+{crawler['info_url']})", "debug_dir": a.debug_dir, "windows_dir": a.windows_dir}  # fmt: skip
    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(cfg, args), maxtasksperchild=None) as pool:
        for r in pool.imap_unordered(render_group, groups):
            results += r
            print(f"{r[0]['rec']['publisher'] if r else '?'}: {len(r)} pages, {sum(len(x['rows']) for x in r)} lines  [{time.time() - t0:.0f}s]", flush=True)
    write_set(cfg, a, out, pubs, results, time.time() - t0)


def write_set(cfg: dict, a, out: Path, pubs: dict, results: list[dict], seconds: float) -> None:
    """Merge in a fixed order (publisher, URL, line index) so the set does not depend on the worker count; cap repeated texts per publisher."""
    lc = cfg["lines"]
    results.sort(key=lambda r: (r["rec"]["publisher"], r["rec"]["url"]))
    drops: Counter = Counter()
    seen: Counter = Counter()
    labels = {"train": [], "heldout": []}
    pages = []
    for r in results:
        drops.update(r["drops"])
        rec = r["rec"]
        pages.append(rec)
        split = pubs[rec["publisher"]]["split"]
        n = 0
        for row in r["rows"]:
            key = (rec["publisher"], row["text"])
            if seen[key] >= lc["max_copies_per_text"]:
                drops["repeated_text"] += 1
                continue
            seen[key] += 1
            png = row.pop("png")
            rid = f"{rec['publisher']}-{hashlib.sha1(rec['url'].encode()).hexdigest()[:8]}-{row['k']:03d}"
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "images" / f"{rid}.png").write_bytes(png)
            row.update(id=rid, sha256=hashlib.sha256(png).hexdigest(), split=split)
            row.pop("k")
            labels[split].append(row)
            n += 1
        rec["kept_after_dedupe"] = n
    for split, rows in labels.items():
        (out / split).mkdir(parents=True, exist_ok=True)
        (out / split / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    if getattr(a, "windows_dir", None):
        wd = Path(a.windows_dir)
        for split in ("train", "heldout"):
            ws = [w for r in results for w in r.get("windows", []) if w["split"] == split]
            (wd / split).mkdir(parents=True, exist_ok=True)
            (wd / split / "windows.jsonl").write_text("".join(json.dumps(w) + "\n" for w in ws), encoding="utf-8")
            print(f"windows {split}: {len(ws)} ({sum(len(w['lines']) for w in ws)} lines, {sum(sum(w['dark']) > 0 for w in ws)} with light-on-dark text)")
    (out / "pages.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pages), encoding="utf-8")
    import playwright

    manifest = {
        "config": cfg, "git_commit": os.environ.get("ONEOCR_GIT_COMMIT", "unknown"), "seconds": round(seconds, 1), "workers": a.workers,
        "label_provenance": "exact truth: DOM text of public-domain web pages rendered by Chromium (gen/web_lines.py); pixel-checked against the DOM boxes",
        "urls_sha256": hashlib.sha256((BASE / cfg["urls"]).read_bytes()).hexdigest(),
        "labels_sha256": {k: hashlib.sha256((out / k / "labels.jsonl").read_bytes()).hexdigest() for k in labels},
        "lines": {k: len(v) for k, v in labels.items()}, "dropped_lines": dict(drops),
        "page_results": dict(Counter(re.sub(r"\d+", "N", p["result"]) for p in pages)),
        "versions": {"python": platform.python_version(), "playwright": getattr(playwright, "__version__", "?"), "numpy": np.__version__},
    }  # fmt: skip
    (out / "run.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("lines", "page_results", "dropped_lines")}, indent=1))


def sheet(root: Path, path: Path, n: int, seed: int) -> None:
    """A contact sheet of n random crops (seeded), the label under each, for looking at before a set is used."""
    from PIL import ImageDraw

    rows = [json.loads(x) for x in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines() if x]
    rows = random.Random(seed).sample(rows, min(n, len(rows)))
    cw, lh = 760, 16
    canvas = Image.new("RGB", (cw * 2, ((len(rows) + 1) // 2) * (OUT_H + lh + 4)), "white")
    d = ImageDraw.Draw(canvas)
    for k, r in enumerate(rows):
        im = Image.open(root / "images" / f"{r['id']}.png").convert("RGB")
        x, y = (k % 2) * cw, (k // 2) * (OUT_H + lh + 4)
        canvas.paste(im.crop((0, 0, min(im.width, cw - 4), OUT_H)), (x, y))
        d.text((x, y + OUT_H), f"{r['text'][:80]}  [{r['publisher']} {r['em_px']}px dpr{r['dpr']} {r['aa']}]", fill="red")
    canvas.save(path)


def merge(base: Path, extra: Path, only: str) -> None:
    """Replaces one publisher's pages and lines in a finished set with those of a re-render of just that publisher (texts are capped
    per publisher, so publishers do not interact)."""
    import shutil

    for split in ("train", "heldout"):
        lab = base / split / "labels.jsonl"
        keep = [x for x in lab.read_text(encoding="utf-8").splitlines() if x and json.loads(x)["publisher"] != only] if lab.exists() else []
        for img in (base / split / "images").glob(f"{only}-*.png") if (base / split / "images").exists() else []:
            img.unlink()
        new = []
        if (extra / split / "labels.jsonl").exists():
            new = [x for x in (extra / split / "labels.jsonl").read_text(encoding="utf-8").splitlines() if x]
            (base / split / "images").mkdir(parents=True, exist_ok=True)
            for img in (extra / split / "images").glob("*.png"):
                shutil.copy2(img, base / split / "images" / img.name)
        lab.write_text("".join(x + "\n" for x in keep + new), encoding="utf-8")
    pages = [x for x in (base / "pages.jsonl").read_text(encoding="utf-8").splitlines() if x and json.loads(x)["publisher"] != only]
    pages += [x for x in (extra / "pages.jsonl").read_text(encoding="utf-8").splitlines() if x]
    (base / "pages.jsonl").write_text("".join(x + "\n" for x in pages), encoding="utf-8")
    run = json.loads((base / "run.json").read_text(encoding="utf-8"))
    run["merged"] = run.get("merged", []) + [{"publisher": only, "run": json.loads((extra / "run.json").read_text(encoding="utf-8"))}]
    for split in ("train", "heldout"):
        run["lines"][split] = sum(1 for _ in (base / split / "labels.jsonl").open(encoding="utf-8"))
        run["labels_sha256"][split] = hashlib.sha256((base / split / "labels.jsonl").read_bytes()).hexdigest()
    (base / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run["lines"]))


def resplit(base: Path, publisher: str, to: str, seed: int) -> None:
    """Moves one publisher's lines between splits of a finished set (no re-render). The destination's labels are shuffled with `seed`, so
    a validation run that reads the first N lines sees every site."""
    import shutil

    frm = "train" if to == "heldout" else "heldout"
    rows = {s: [json.loads(x) for x in (base / s / "labels.jsonl").read_text(encoding="utf-8").splitlines() if x] for s in (frm, to)}
    moved = [r for r in rows[frm] if r["publisher"] == publisher]
    (base / to / "images").mkdir(parents=True, exist_ok=True)
    for r in moved:
        shutil.move(str(base / frm / "images" / f"{r['id']}.png"), str(base / to / "images" / f"{r['id']}.png"))
        r["split"] = to
    rows[frm] = [r for r in rows[frm] if r["publisher"] != publisher]
    rows[to] = rows[to] + moved
    random.Random(seed).shuffle(rows[to])
    for s in (frm, to):
        (base / s / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[s]), encoding="utf-8")
    run = json.loads((base / "run.json").read_text(encoding="utf-8"))
    run.setdefault("resplit", []).append({"publisher": publisher, "to": to, "lines": len(moved), "seed": seed})
    for s in (frm, to):
        run["lines"][s] = len(rows[s])
        run["labels_sha256"][s] = hashlib.sha256((base / s / "labels.jsonl").read_bytes()).hexdigest()
    (base / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"moved": len(moved), "lines": run["lines"]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--config", required=True)
    d.add_argument("--only")
    r = sub.add_parser("render")
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=5)
    r.add_argument("--limit-pages", type=int)
    r.add_argument("--only", help="comma-separated publisher ids")
    r.add_argument("--debug-dir", help="write annotated screenshots of the pages here")
    r.add_argument("--windows-dir", help="also write viewport windows with line boxes for detector training here (cfg `windows`)")
    rs = sub.add_parser("resplit")
    rs.add_argument("--base", required=True)
    rs.add_argument("--publisher", required=True)
    rs.add_argument("--to", choices=["train", "heldout"], required=True)
    rs.add_argument("--seed", type=int, default=95)
    mg = sub.add_parser("merge")
    mg.add_argument("--base", required=True)
    mg.add_argument("--extra", required=True)
    mg.add_argument("--publisher", required=True)
    sh = sub.add_parser("sheet")
    sh.add_argument("--data", required=True)
    sh.add_argument("--out", required=True)
    sh.add_argument("--n", type=int, default=60)
    sh.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "resplit":
        resplit(Path(a.base), a.publisher, a.to, a.seed)
        return
    if a.cmd == "merge":
        merge(Path(a.base), Path(a.extra), a.publisher)
        return
    if a.cmd == "sheet":
        sheet(Path(a.data), Path(a.out), a.n, a.seed)
        return
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if a.cmd == "discover":
        discover(cfg, a.only)
    else:
        render(cfg, a)


if __name__ == "__main__":
    sys.exit(main())
