"""Real born-digital PDFs as evaluation pages, with truth from the PDF text layer.

    python -m oneocr_bench realpdf fetch --list PDFS.json --out DIR
    python -m oneocr_bench realpdf cut   --pdfs DIR --out STAGING --seeds SEEDS.json --pdftruth PdfTruth.dll [--exclude HASHES]

PDFS.json is a private list: [{"url": ..., "source": "label"}]. `fetch` records size and SHA-256 per file.

A page becomes an evaluation page only when its truth is verifiably exact. Everything is checked from two independent
readers of the text layer: poppler (`pdftotext -raw -bbox`, content-stream order, also the renderer at 300 DPI) and OnePdf
(`PdfTruth`, content order and reading-order analysis). A page is kept when all of these hold:

- the character stream of both readers is the same (ratio >= CHAR_RATIO) and their word sequences agree (>= WORD_RATIO);
- at least BOX_SHARE of the words have a OnePdf word with the same text whose centre lies within 3 pt;
- the order is credible: the content-stream order agrees with poppler's own reading-order analysis and with OnePdf's by at
  least ORDER_RATIO (clean) or ORDER_RATIO_LAYOUT; otherwise the page is dropped;
- no word is invisible, no font is Type 3 or a CID font without a ToUnicode map, no word has U+FFFD, a private-use or a
  control character, and the page has no embedded image;
- every word box shows ink in the 300 DPI render (>= INK_SHARE), and the ink outside the words, after removing long rules, is
  at most OUTSIDE_INK (so text drawn as paths or figures is not hiding behind the truth).

Page classes by a stated rule: *layout* when the words fall into two or more columns (a gutter at least GUTTER_PT wide that
at most 4% of the lines cross) or the page has at least 4 long horizontal and 2 long vertical rules (a table or form
grid); *clean* when it is one column with at most two long rules and 250 to 900 words; anything else is dropped. Degraded
pages are other accepted pages of the same files, run through the bench's scan-degradation simulator.

Truth words are the text layer's words as poppler splits them, in content-stream order; a word wrapped with a hyphen stays
two words, as printed. Ligatures are written as letters.
"""

from __future__ import annotations

import difflib
import html
import io
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import urllib.request
from itertools import zip_longest
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

from . import degrade
from .render import add_ink, sha256, text_of, write_page

DPI = 300
PT = DPI / 72.0
CHAR_RATIO = 0.998
WORD_RATIO = 0.90  # the readers cut words differently; the character stream and the box match carry the check
BOX_SHARE = 0.985
ORDER_RATIO = 0.99
ORDER_RATIO_LAYOUT = 0.97
INK_SHARE = 0.99
OUTSIDE_INK = 0.02
GUTTER_PT = 10.0
SCAN_PER_DOC = 25
MAX_PER_DOC = {"clean-real": 4, "layout-real": 5, "degraded-real": 5}
UA = "oneocr-bench-fetch/1.0 (https://github.com/recursive-ai-oss/oneocr-csharp; evaluation documents, a few requests per run)"
LIG = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
# A page that carries a third party's notice is not ours to redistribute: drop it, whatever else is true of it.
COPYRIGHT_NOTICE = re.compile(r"©|\(c\)\s*(?:19|20)\d\d|used with permission|all rights reserved", re.IGNORECASE)
MAX_FETCH_BYTES = 80_000_000
BAD_CHARS = re.compile("[\ufffd\ue000-\uf8ff\u0000-\u001f\u007f-\u009f]")
WORD_RE = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', re.DOTALL)
PAGE_RE = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">')


def norm(s: str) -> str:
    s = "".join(LIG.get(c, c) for c in s)
    return unicodedata.normalize("NFC", s)


# ----------------------------------------------------------------------------------------------------------- fetch

def fetch(a) -> None:
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, e in enumerate(json.loads(Path(a.list).expanduser().read_text())):
        dest = out / f"d{i + 1:03d}.pdf"
        row = {"file": dest.name, "url": e["url"], "source": e.get("source", "")}
        try:
            if not dest.exists():
                req = urllib.request.Request(e["url"], headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read(a.max_bytes + 1)
                if len(data) > a.max_bytes:
                    raise ValueError(f"larger than the {a.max_bytes} byte cap")
                if not data.startswith(b"%PDF"):
                    raise ValueError("not a PDF")
                dest.write_bytes(data)
                time.sleep(1.0)
            data = dest.read_bytes()
            row.update(ok=True, bytes=len(data), sha256=sha256(data), pages=pdf_pages(dest))
            if e.get("sha256") and e["sha256"] != row["sha256"]:
                row.update(ok=False, why="hash differs from the pinned value")
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the fetch
            row.update(ok=False, why=f"{type(exc).__name__}: {str(exc)[:100]}")
        rows.append(row)
        print(row["file"], row.get("ok"), row.get("pages"), row.get("why", ""), flush=True)
    (out / "FETCHED.json").write_text(json.dumps(rows, indent=1) + "\n")


def run(cmd: list[str], timeout: int = 120) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout


def pdf_pages(path: Path) -> int:
    m = re.search(r"^Pages:\s+(\d+)", run(["pdfinfo", str(path)]), re.MULTILINE)
    return int(m.group(1)) if m else 0


# ------------------------------------------------------------------------------------------------------------- cut

def poppler_words(path: Path, page: int, raw: bool) -> tuple[tuple[float, float], list[dict]]:
    xml = run(["pdftotext", *(["-raw"] if raw else []), "-bbox", "-enc", "UTF-8", "-f", str(page), "-l", str(page), str(path), "-"])
    m = PAGE_RE.search(xml)
    size = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
    words = [{"t": norm(html.unescape(t)), "b": [float(x0), float(y0), float(x1), float(y1)]}
             for x0, y0, x1, y1, t in WORD_RE.findall(xml)]
    return size, [w for w in words if w["t"]]


def font_problem(path: Path, page: int) -> str | None:
    # pdffonts columns: name type encoding emb sub uni object-number generation
    for line in run(["pdffonts", "-f", str(page), "-l", str(page), str(path)]).splitlines()[2:]:
        if "Type 3" in line:
            return "Type 3 font"
        cells = line.split()
        if "CID" in line and len(cells) >= 5 and cells[-3] == "no":
            return "CID font without ToUnicode"
    return None


def has_images(path: Path, page: int) -> bool:
    """True when the page holds an image that could carry text: more than 2 KB of data (flat rules and colour bars are a few
    hundred bytes). The ink-outside-the-words check backs this up on the rendered page."""
    for line in run(["pdfimages", "-list", "-f", str(page), "-l", str(page), str(path)]).splitlines()[2:]:
        cells = line.split()
        if len(cells) < 15 or cells[2] == "smask":
            continue
        size = cells[-2]
        mult = {"B": 1, "K": 1024, "M": 1 << 20, "G": 1 << 30}.get(size[-1], 1)
        if float(size[:-1] or 0) * mult > 2048:
            return True
    return False


def render_page(path: Path, page: int) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "p"
        subprocess.run(["pdftoppm", "-r", str(DPI), "-gray", "-png", "-f", str(page), "-l", str(page), "-singlefile",
                        str(path), str(root)], capture_output=True, timeout=180, check=True)
        with Image.open(root.with_suffix(".png")) as im:
            return im.convert("L")


def ratio(a: list, b: list) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a or b else 1.0


def long_runs(dark: np.ndarray, minlen: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the longest dark run, and a mask of the pixels in runs of at least `minlen`."""
    h, _ = dark.shape
    longest = np.zeros(h, np.int32)
    mask = np.zeros_like(dark)
    pad = np.zeros((h, 1), np.int8)
    d = np.diff(np.concatenate([pad, dark.astype(np.int8), pad], axis=1), axis=1)
    for i in range(h):
        s = np.flatnonzero(d[i] == 1)
        if s.size:
            e = np.flatnonzero(d[i] == -1)
            ln = e - s
            longest[i] = ln.max()
            for a, b, n in zip(s, e, ln, strict=True):
                if n >= minlen:
                    mask[i, a:b] = True
    return longest, mask


def groups(flags: np.ndarray) -> int:
    f = np.concatenate([[False], flags, [False]]).astype(np.int8)
    return int((np.diff(f) == 1).sum())


def columns(words: list[dict], width: float) -> int:
    """Text columns from the words' horizontal coverage: a gutter is a stretch at least GUTTER_PT wide that few lines cross."""
    xs = np.zeros(int(width) + 2, np.int32)
    for w in words:
        xs[int(w["b"][0]): int(w["b"][2]) + 1] += 1
    lines = max(1, len({round((w["b"][1] + w["b"][3]) / 2 / 4) for w in words}))
    thr = max(2, 0.04 * lines)
    cover = np.flatnonzero(xs > 0)
    if not cover.size:
        return 1
    lo, hi = cover[0] + 0.15 * (cover[-1] - cover[0]), cover[-1] - 0.15 * (cover[-1] - cover[0])
    gutters, run = 0, 0
    for x in range(int(lo), int(hi) + 1):
        if xs[x] <= thr:
            run += 1
        else:
            gutters += run >= GUTTER_PT
            run = 0
    gutters += run >= GUTTER_PT
    return 1 + gutters


def judge(path: Path, page: int, truth_page: dict) -> tuple[dict | None, str, dict]:
    """Apply the acceptance rules to one page; returns (candidate or None, reason, measurements)."""
    info: dict = {}
    size, raw = poppler_words(path, page, True)
    _, phys = poppler_words(path, page, False)
    if not raw or size[0] <= 0:
        return None, "no text", info
    if len(raw) < 150:
        return None, "too few words", info
    if any(w["b"][2] - w["b"][0] < 1.0 or w["b"][3] - w["b"][1] < 2.0 for w in raw):
        return None, "degenerate word box", info
    if COPYRIGHT_NOTICE.search(" ".join(w["t"] for w in raw)):
        return None, "copyright or permission notice in the text", info
    if any(BAD_CHARS.search(w["t"]) for w in raw):
        return None, "replacement, private-use or control characters", info
    if truth_page.get("error"):
        return None, "OnePdf error", info
    if abs(truth_page["width"] - size[0]) > 1 or abs(truth_page["height"] - size[1]) > 1:
        return None, "page size differs between readers", info
    if any(not w["v"] for w in truth_page["words"]):
        return None, "invisible text", info
    why = font_problem(path, page)
    if why:
        return None, why, info
    if has_images(path, page):
        return None, "embedded image", info

    a_tok = [w["t"] for w in raw]
    c_tok = [norm(t) for t in truth_page["content"].split()]
    r_tok = [norm(w["t"]) for w in truth_page["words"]]
    p_tok = [w["t"] for w in phys]
    info["char_ratio"] = round(ratio(list("".join(a_tok)), list("".join(c_tok))), 5)
    info["word_ratio"] = round(ratio(a_tok, c_tok), 4)
    if info["char_ratio"] < CHAR_RATIO or info["word_ratio"] < WORD_RATIO:
        return None, f"readers differ (chars {info['char_ratio']}, words {info['word_ratio']})", info

    # Boxes: each poppler word must be matched by the other reader's words at the same place. The readers cut words differently
    # at hyphens and wide gaps, so a match is either the concatenation of the OnePdf words centred inside the poppler box, or a
    # single OnePdf word that contains the poppler word's text and whose box holds the poppler word's centre.
    ones = []
    for w in truth_page["words"]:
        l, b, r, t = w["b"]
        ones.append((norm(w["t"]), l, size[1] - t, r, size[1] - b))
    hit = 0
    for w in raw:
        x0, y0, x1, y1 = w["b"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        inside = sorted((o for o in ones if x0 - 2 <= (o[1] + o[3]) / 2 <= x1 + 2 and y0 - 2 <= (o[2] + o[4]) / 2 <= y1 + 2),
                        key=lambda o: o[1])
        if "".join(o[0] for o in inside) == w["t"] or any(
                w["t"] in o[0] and o[1] - 1 <= cx <= o[3] + 1 and o[2] - 1 <= cy <= o[4] + 1 for o in ones if abs((o[2] + o[4]) / 2 - cy) < 6):
            hit += 1
    info["box_share"] = round(hit / len(raw), 4)
    if info["box_share"] < BOX_SHARE:
        return None, f"boxes differ ({info['box_share']})", info

    # Order is compared on the character stream (spaces removed), so that a different word cut does not read as a different order.
    a_chars = list("".join(a_tok))
    info["order_poppler_layout"] = round(ratio(a_chars, list("".join(p_tok))), 4)
    info["order_onepdf_reading"] = round(ratio(a_chars, list("".join(r_tok))), 4)

    img = render_page(path, page)
    scale = img.size[0] / size[0]
    if abs(scale - PT) > 0.02:
        return None, "render scale differs", info
    g = np.asarray(img, np.uint8)
    dark = g < 140
    h, w = dark.shape
    row_longest, hmask = long_runs(dark, 60)
    col_longest, vmask_t = long_runs(np.ascontiguousarray(dark.T), 60)
    h_rules = groups(row_longest >= 0.25 * w)
    v_rules = groups(col_longest >= 0.08 * h)
    cols = columns(raw, size[0])
    info.update(columns=cols, h_rules=h_rules, v_rules=v_rules, words=len(raw))

    words_px = []
    for i, wd in enumerate(raw):
        x0, y0, x1, y1 = (v * scale for v in wd["b"])
        words_px.append({"text": wd["t"], "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)], "block": 0, "_i": i})
    add_ink(words_px, img)
    weak = 0
    cover = np.zeros_like(dark)
    for t in words_px:
        x0, y0, x1, y1 = (round(v) for v in t["ink"])
        reg = g[max(0, y0): max(0, y1), max(0, x0): max(0, x1)]
        if reg.size == 0 or float(reg.std()) < 5.0:
            weak += 1
        bx0, by0, bx1, by1 = (round(v) for v in t["box"])
        cover[max(0, by0 - 3): by1 + 3, max(0, bx0 - 3): bx1 + 3] = True
    info["ink_share"] = round(1 - weak / len(words_px), 4)
    if info["ink_share"] < INK_SHARE:
        return None, f"words without ink ({info['ink_share']})", info
    free = dark & ~cover & ~hmask & ~vmask_t.T
    info["outside_ink"] = round(float(free.sum()) / max(1, int(dark.sum())), 4)
    if info["outside_ink"] > OUTSIDE_INK:
        return None, f"ink outside the words ({info['outside_ink']})", info

    if cols >= 2 or (h_rules >= 4 and v_rules >= 2):
        klass = "layout"
    elif cols == 1 and h_rules + v_rules <= 2 and 250 <= len(raw) <= 900:
        klass = "clean"
    else:
        return None, "neither clean nor layout", info
    need = ORDER_RATIO if klass == "clean" else ORDER_RATIO_LAYOUT
    if info["order_poppler_layout"] < need or info["order_onepdf_reading"] < need:
        return None, f"order not credible ({info['order_poppler_layout']}, {info['order_onepdf_reading']})", info
    info["class"] = klass
    # lines for the text file: a new block whenever the next word starts a new line
    block, prev = 0, None
    for t in words_px:
        cy = (t["box"][1] + t["box"][3]) / 2
        hgt = t["box"][3] - t["box"][1]
        if prev is not None and abs(cy - prev) > 0.6 * hgt:
            block += 1
        t["block"] = block
        prev = cy
    return {"words": words_px, "img": img, "klass": klass, "size": size}, "ok", info


def dotnet_run(dll: str, pdf: Path, pages: list[int]) -> dict:
    spec = ",".join(str(p) for p in pages)
    out = subprocess.run(["dotnet", dll, str(pdf), spec], capture_output=True, timeout=600, check=False)
    data = json.loads(out.stdout) if out.stdout else {"pages": []}
    return {p["page"]: p for p in data["pages"]}


def cut(a) -> None:
    seeds = json.loads(Path(a.seeds).expanduser().read_text())
    base = int(seeds["real-pdf"])
    pdf_dir = Path(a.pdfs).expanduser()
    out = Path(a.out).expanduser()
    fetched = [r for r in json.loads((pdf_dir / "FETCHED.json").read_text()) if r.get("ok")]
    exclude = set(Path(a.exclude).read_text().split()) if a.exclude else set()
    cands: dict[str, list[dict]] = {"clean": [], "layout": []}
    log: list[dict] = []
    for r in fetched:
        path = pdf_dir / r["file"]
        if r["sha256"] in exclude:
            log.append({"file": r["file"], "result": "in the excluded set (training data hash list)"})
            continue
        n = r["pages"]
        rng = random.Random(f"{base}:{r['file']}")
        pages = sorted(rng.sample(range(1, n + 1), min(n, a.scan_per_doc)))
        tp = dotnet_run(a.pdftruth, path, pages)
        for p in pages:
            if p not in tp:
                log.append({"file": r["file"], "page": p, "result": "PdfTruth returned no page"})
                continue
            try:
                cand, why, info = judge(path, p, tp[p])
            except Exception as exc:  # noqa: BLE001 - one bad page must not stop the cut
                cand, why, info = None, f"{type(exc).__name__}: {str(exc)[:80]}", {}
            log.append({"file": r["file"], "page": p, "result": why, **info})
            if cand:
                cand.update(file=r["file"], page=p, info=info, url=r["url"], sha=r["sha256"], source=r["source"])
                cands[cand["klass"]].append(cand)
        why: dict[str, int] = {}
        for x in log:
            if x["file"] == r["file"] and "page" in x:
                k = re.sub(r"\(.*?\)", "", x["result"]).strip()
                why[k] = why.get(k, 0) + 1
        print(r["file"], {k: len(v) for k, v in cands.items()}, why, flush=True)

    targets = {"clean-real": a.clean, "layout-real": a.layout, "degraded-real": a.degraded}
    rng = random.Random(base)
    chosen: dict[str, list[dict]] = {k: [] for k in targets}
    used: set[tuple[str, int]] = set()

    def take(set_name: str, pool: list[dict]) -> None:
        pool = [c for c in pool if (c["file"], c["page"]) not in used]
        rng.shuffle(pool)
        # round-robin over publishers, so that one publisher's many easy pages do not fill the set
        by_src: dict[str, list[dict]] = {}
        for c in pool:
            by_src.setdefault(c["source"], []).append(c)
        pool = [c for row in zip_longest(*by_src.values()) for c in row if c]
        per: dict[str, int] = {}
        for c in pool:
            if len(chosen[set_name]) >= targets[set_name]:
                break
            if per.get(c["file"], 0) >= MAX_PER_DOC[set_name]:
                continue
            per[c["file"]] = per.get(c["file"], 0) + 1
            chosen[set_name].append(c)
            used.add((c["file"], c["page"]))

    take("clean-real", cands["clean"])
    take("layout-real", cands["layout"])
    take("degraded-real", cands["clean"] + cands["layout"])  # other pages of the same files

    versions = {"pillow": PIL.__version__, "numpy": np.__version__}
    for set_name, picks in chosen.items():
        d = out / set_name
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        pages_out = []
        for i, c in enumerate(picks):
            pid = f"{set_name[0]}{i + 1:03d}"
            words, img = c["words"], c["img"]
            for t in words:
                t.pop("_i", None)
            ext, params = "png", None
            data = None
            if set_name == "degraded-real":
                r = random.Random(f"{base}:{pid}:degrade")
                n = np.random.default_rng([base, i])
                data, words, params = degrade.degrade_scan(img, words, r, n)
                ext = "jpg"
                with Image.open(io.BytesIO(data)) as im2:
                    size = im2.size
            else:
                buf = io.BytesIO()
                img.save(buf, "PNG", compress_level=6)
                data, size = buf.getvalue(), img.size
            truth = {"id": pid, "set": set_name, "source": "born-digital PDF text layer", "pdf_page": c["page"],
                     "image": {"width": size[0], "height": size[1], "dpi": DPI}, "words": words, "text": text_of(words),
                     "order": "content-stream order of the text layer, checked against two reading-order analyses",
                     "checks": c["info"]}
            if params:
                truth["degradation"] = params
            e = write_page(d, pid, data, ext, truth)
            e.update(source=c["source"], url=c["url"], pdf_sha256=c["sha"], pdf_file=c["file"], pdf_page=c["page"],
                     klass=c["klass"], checks=c["info"])
            pages_out.append(e)
        (d / "MANIFEST.json").write_text(json.dumps({"set": set_name, "count": len(pages_out), **versions, "pages": pages_out},
                                                    indent=1) + "\n")
        print(f"{set_name}: {len(pages_out)} pages (target {targets[set_name]})")
    (out / "CUT-LOG.json").write_text(json.dumps(log, indent=1) + "\n")
    reasons: dict[str, int] = {}
    for x in log:
        k = re.sub(r"\(.*?\)", "", x["result"]).strip()
        reasons[k] = reasons.get(k, 0) + 1
    print(json.dumps(reasons, indent=1))


def hashdir(a) -> None:
    """SHA-256 of every PDF under a directory, one per line: the exclusion list for `cut --exclude`."""
    for p in sorted(Path(a.dir).expanduser().rglob("*.pdf")):
        print(sha256(p.read_bytes()))
