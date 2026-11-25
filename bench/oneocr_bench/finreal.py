"""Real federal financial reports as evaluation pages: scan, fax, phone and table cuts with tagged amounts.

    python -m oneocr_bench finreal fetch --list fin-dev-real-urls.json --out DIR      # same as `realpdf fetch`
    python -m oneocr_bench finreal cut --pdfs DIR --out ROOT --seeds SEEDS.json --pdftruth PdfTruth.dll [--exclude HASHES]

The page truth is the realpdf method (poppler and OnePdf must agree on characters, words, boxes and order, every word shows
ink, no embedded image, no notice of a third party's rights; see `realpdf.py`). Only the page-class rule differs: a page is
kept when at least MIN_VALUES of its words are money amounts, dates or account-like numbers, whatever else it looks like.

Tags are found by pattern on the text layer's words (a value is one word): money (grouped digits with two decimals, or
grouped whole numbers as in reports stated in millions, optional currency sign, minus or parentheses), dates (`m/d/yyyy`,
`yyyy-mm-dd`) and account-like numbers (eight or more digits in a row). The pages are captured like the synthetic sets
(`fin.deliver`): the sets are `fin-scan-real`, `fin-fax-real`, `fin-phone-real` and `fin-tables-real` (pages with at least
TABLE_VALUES values, in all four captures); a page belongs to one set only.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from itertools import zip_longest
from pathlib import Path

from PIL import Image

from . import fin, realpdf
from .render import sha256

MIN_VALUES = 30
TABLE_VALUES = 80
MONEY = re.compile(r"^[(\-\u2212]?[$\u20ac\u00a3]?[(\-\u2212]?\d{1,3}(?:,\d{3})*\.\d{2}[)\-]?$|^[(\-\u2212]?[$\u20ac\u00a3]?\d{1,3}(?:,\d{3})+[)]?$")
DATE = re.compile(r"^(?:\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})|\d{4}-\d{2}-\d{2})$")
ACCOUNT = re.compile(r"^\d{8,}$")
SETS = ("fin-scan-real", "fin-fax-real", "fin-phone-real", "fin-tables-real")
CAPTURES = {"fin-scan-real": ["scan"], "fin-fax-real": ["fax-fine", "fax-std"], "fin-phone-real": ["phone"],
            "fin-tables-real": ["scan", "fax-fine", "phone", "fax-std"]}
PER_DOC = 12


def tag_of(text: str) -> str | None:
    if MONEY.match(text):
        return "money"
    if DATE.match(text):
        return "date"
    if ACCOUNT.match(text):
        return "account"
    return None


def classify(raw: list[dict], info: dict) -> str | None:
    n = sum(1 for w in raw if tag_of(w["t"]))
    info["values"] = n
    # table pages have no single reading order; the content-stream order is kept as truth but is not checked against the analyses
    info["order_exempt"] = True
    return "layout" if n >= MIN_VALUES else None


def cut(a) -> None:
    seeds = json.loads(Path(a.seeds).expanduser().read_text())
    base = int(seeds["fin-real"])
    pdf_dir = Path(a.pdfs).expanduser()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    fetched = [r for r in json.loads((pdf_dir / "FETCHED.json").read_text()) if r.get("ok")]
    exclude = set(Path(a.exclude).read_text().split()) if a.exclude else set()
    cands: list[dict] = []
    log: list[dict] = []
    for r in fetched:
        path = pdf_dir / r["file"]
        if r["sha256"] in exclude:
            log.append({"file": r["file"], "result": "in the excluded set"})
            continue
        n = r["pages"]
        rng = random.Random(f"{base}:{r['file']}")
        pages = sorted(rng.sample(range(1, n + 1), min(n, a.scan_per_doc)))
        tp = realpdf.dotnet_run(a.pdftruth, path, pages)
        kept = 0
        for p in pages:
            if p not in tp:
                log.append({"file": r["file"], "page": p, "result": "PdfTruth returned no page"})
                continue
            try:
                cand, why, info = realpdf.judge(path, p, tp[p], classify=classify)
            except Exception as exc:  # noqa: BLE001 - one bad page must not stop the cut
                cand, why, info = None, f"{type(exc).__name__}: {str(exc)[:80]}", {}
            log.append({"file": r["file"], "page": p, "result": why, **info})
            if cand:
                cand.update(file=r["file"], page=p, info=info, url=r["url"], sha=r["sha256"], source=r["source"])
                cands.append(cand)
                kept += 1
        print(r["file"], "kept", kept, "of", len(pages), flush=True)

    rng = random.Random(base)
    rng.shuffle(cands)
    by_src: dict[str, list[dict]] = {}
    for c in cands:
        by_src.setdefault(c["source"], []).append(c)
    pool = [c for row in zip_longest(*by_src.values()) for c in row if c]
    chosen: dict[str, list[dict]] = {s: [] for s in SETS}
    per_doc: dict[str, int] = {}
    targets = {"fin-tables-real": a.tables, "fin-scan-real": a.each, "fin-fax-real": a.each, "fin-phone-real": a.each}
    # densest pages first for the tables cut, then the other sets in turn
    dense = sorted(pool, key=lambda c: -c["info"]["values"])
    for c in [x for x in dense if x["info"]["values"] >= TABLE_VALUES]:
        if len(chosen["fin-tables-real"]) >= targets["fin-tables-real"]:
            break
        if per_doc.get(c["file"], 0) < PER_DOC:
            chosen["fin-tables-real"].append(c)
            per_doc[c["file"]] = per_doc.get(c["file"], 0) + 1
    taken = {(c["file"], c["page"]) for v in chosen.values() for c in v}
    rest = [c for c in pool if (c["file"], c["page"]) not in taken]
    for c in rest:
        if per_doc.get(c["file"], 0) >= PER_DOC:
            continue
        for s in ("fin-scan-real", "fin-fax-real", "fin-phone-real"):
            if len(chosen[s]) < targets[s]:
                chosen[s].append(c)
                per_doc[c["file"]] = per_doc.get(c["file"], 0) + 1
                break
        # the sets fill in turn: the first one that is not full takes the page

    manifest = {}
    for s in SETS:
        d = out / s
        shutil.rmtree(d, ignore_errors=True)
        pages_out = []
        for i, c in enumerate(chosen[s]):
            words = c["words"]
            tags: list[dict] = []
            for j, w in enumerate(words):
                w.pop("_i", None)
                cls = tag_of(w["text"])
                if cls:
                    w["tag"] = len(tags)
                    tags.append({"cls": cls, "role": "real", "value": w["text"], "words": [j]})
            img = c["img"].convert("L") if isinstance(c["img"], Image.Image) else c["img"]
            capture = CAPTURES[s][i % len(CAPTURES[s])]
            info = {"doc": "real", "order": "content-stream order of the text layer (not verified: tables)", "source": c["source"], "url": c["url"], "pdf_sha256": c["sha"], "pdf_page": c["page"],
                    "checks": c["info"]}
            e = fin.deliver(img, words, tags, None, info, capture, s, base, i, d)
            e.update(pdf_sha256=c["sha"], pdf_page=c["page"], url=c["url"])
            pages_out.append(e)
        manifest[s] = {"count": len(pages_out), "pages": pages_out}
        print(s, len(pages_out), "pages", sum(p["tags"] for p in pages_out), "tags")
    (out / "manifest-real.json").write_text(json.dumps({"sets": manifest}, indent=1) + "\n")
    if a.check:
        from .paths import MANIFESTS
        pinned = json.loads((MANIFESTS / "fin-dev-v1.json").read_text())["real"]
        bad = [s for s in SETS if manifest[s]["pages"] != pinned[s]["pages"]]
        if bad:
            raise SystemExit(f"hash mismatch in {bad}")
        print("real pages reproduce the committed hashes")
    (out / "CUT-LOG.json").write_text(json.dumps(log, indent=1) + "\n")
    reasons: dict[str, int] = {}
    for x in log:
        k = re.sub(r"\(.*?\)", "", x["result"]).strip()
        reasons[k] = reasons.get(k, 0) + 1
    print(json.dumps(reasons, indent=1))
    _ = sha256
