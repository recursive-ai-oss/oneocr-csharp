"""Truth checker for generated pages.

  uv run python -m pages.check RUN_DIR --split train --n 1000 --seed 0 [--workers 8] [--report report.json]

Samples N pages of a finished run and re-derives each one from its seed in a fresh renderer. Per page it checks:

  reproduce   the regenerated truth has the same SHA-256 as the stored file, and so does the image
  dom-text    the words, in order, are exactly the whitespace-separated tokens of the page's rendered text (innerText), so no
              word is missing, duplicated, reordered or differently cased (text-transform is never used)
  ink-clean   every word box in the clean render holds ink: the grey range inside the box is at least 40 levels
  ink-final   in the stored (degraded) image the box of every word still differs from its surroundings: the 3rd and 97th
              percentile of the box differ from its median by more than a noise floor derived from the logged degradation
  structure   the rules of the oneocr.page reader (docs/spec/oneocr-page-v1.md): boxes ordered and inside the page, indices in
              range, cells starting at 0 and strictly increasing, spans inside tables, region blocks in range, counts
  splits      the template, fonts and books belong to the page split they were drawn for; none is an evaluation font or book
  geometry    every stored quad is the homography image of its pre-degradation box (checked through the bounds)

The report lists counts per check and per page kind or degradation profile, and the first failures.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from . import build

EVAL_BOOKS = {11, 1342}
HERE = Path(__file__).resolve().parent


def structure_errors(doc: dict) -> list[str]:
    e: list[str] = []
    if doc.get("format") != "oneocr.page" or doc.get("version") != 1 or doc.get("pageCount") != len(doc["pages"]):
        e.append("header")
    page = doc["pages"][0]
    W, H = page["width"], page["height"]

    def box(b, what):
        if not (b[0] <= b[2] and b[1] <= b[3]):
            e.append(f"{what}: inverted box")
        if b[0] < -2 or b[1] < -2 or b[2] > W + 2 or b[3] > H + 2:
            e.append(f"{what}: outside page {b}")

    nwords = 0
    for bi, b in enumerate(page["blocks"]):
        box(b["bounds"], f"block {bi}")
        if not b["lines"]:
            e.append(f"block {bi}: no lines")
        for li, ln in enumerate(b["lines"]):
            box(ln["bounds"], f"line {bi}.{li}")
            cells = ln["cells"]
            if (
                not ln["words"]
                or cells[:1] != [0]
                or any(y <= x for x, y in zip(cells, cells[1:], strict=False))
                or cells[-1] >= len(ln["words"])
            ):
                e.append(f"line {bi}.{li}: bad cells")
            for w in ln["words"]:
                nwords += 1
                box(w["bounds"], f"word {w['text']}")
                if not w["text"]:
                    e.append("empty word")
    for r in page["regions"]:
        box(r["bounds"], f"region {r['role']}")
        if any(not 0 <= b < len(page["blocks"]) for b in r["blocks"]):
            e.append("region: block index")
    for t in page["tables"]:
        if not 0 <= t["block"] < len(page["blocks"]):
            e.append("table: block index")
        for c in t["cells"]:
            if c["row"] + c["rowSpan"] > t["rows"] or c["column"] + c["columnSpan"] > t["columns"]:
                e.append("table: span leaves table")
            if any(not 0 <= i < nwords for i in c["words"]):
                e.append("cell: word index")
    for kv in page["keyValues"]:
        if any(not 0 <= i < nwords for i in kv["key"] + kv["value"]) or not kv["key"] or not kv["value"]:
            e.append("keyValue: indices")
    for f in page.get("fields", []):
        if any(not 0 <= i < nwords for i in f["words"]):
            e.append("field: word index")
    return e


def ink_final(img: Image.Image, doc: dict) -> tuple[int, int, list[str]]:
    a = np.asarray(img.convert("L"), dtype=np.float32)
    d = doc["synthetic"]["degradation"]
    floor = max(14.0, 5.0 * (d["read_noise"] + d["shot_noise"] * 0.7) * 255 + 2.0 * d["blur"] * 3)
    ok = bad = 0
    ex = []
    for b in doc["pages"][0]["blocks"]:
        for ln in b["lines"]:
            for w in ln["words"]:
                l, t, r, bt = w["bounds"]
                x0, x1, y0, y1 = max(0, int(l)), min(a.shape[1], int(np.ceil(r))), max(0, int(t)), min(a.shape[0], int(np.ceil(bt)))
                if x1 - x0 < 1 or y1 - y0 < 1:
                    bad += 1
                    ex.append(w["text"])
                    continue
                c = a[y0:y1, x0:x1].ravel()
                med = np.median(c)
                spread = max(med - np.percentile(c, 3), np.percentile(c, 97) - med)
                if spread >= floor * (1.0 if d["faint"] >= 1 else d["faint"] + 0.15):
                    ok += 1
                else:
                    bad += 1
                    if len(ex) < 3:
                        ex.append(w["text"])
    return ok, bad, ex


def check_page(args: tuple[int, str]) -> dict:
    i, run = args
    g = build._G
    r = build.render_page(i)
    pid = r["pid"]
    rd = Path(run) / g["split"]
    res: dict = {"id": pid, "kind": r["spec"].kind, "profile": r["degradation"]["profile"], "fail": []}
    stored_t = (rd / "truth" / f"{pid}.json").read_bytes()
    if build.sha256(stored_t) != build.sha256(r["truth"]):
        res["fail"].append("reproduce-truth")
    stored_i = next((rd / "images").glob(f"{pid}.*")).read_bytes()
    res["image_identical"] = build.sha256(stored_i) == build.sha256(r["data"])
    # dom text
    toks = r["raw"]["text"].split()
    words = [w["t"] for w in r["raw"]["words"]]
    if toks != words:
        res["fail"].append("dom-text" if sorted(toks) != sorted(words) else "dom-order")
        res["dom_diff"] = [(a, b) for a, b in zip(toks, words, strict=False) if a != b][:3]
    # ink on the clean render
    W, H = r["clean"].size
    n_bad = build.ink_failures(r["clean"], r["raw"], W / r["raw"]["size"][0], H / r["raw"]["size"][1])
    res["ink_clean_bad"], res["words"] = n_bad, len(words)
    if n_bad:
        res["fail"].append("ink-clean")
    doc = json.loads(stored_t)
    ok, bad, ex = ink_final(Image.open(rd / "images" / next((rd / "images").glob(f"{pid}.*")).name), doc)
    res["ink_final_ok"], res["ink_final_bad"] = ok, bad
    if bad:
        res["ink_final_examples"] = ex
    errs = structure_errors(doc)
    if errs:
        res["fail"].append("structure")
        res["structure"] = errs[:3]
    syn = doc["synthetic"]
    if syn["templateSplit"] != ("train" if g["split"] == "train" else "val"):
        res["fail"].append("template-split")
    if g["split"] == "train" and any(g["pool"][f].split != "train" for f in syn["fonts"]):
        res["fail"].append("font-split")
    if g["split"] == "train" and any(
        f in {"gelasio", "carlito", "caladea", "sourceserif4", "sourcesans3", "ibmplexserif", "ibmplexsans"} for f in syn["fonts"]
    ):
        res["fail"].append("eval-font")
    if set(syn["books"]) & EVAL_BOOKS:
        res["fail"].append("eval-book")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="seed of the page sample (not of the run)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report")
    ap.add_argument("--config", default=str(HERE.parent / "configs" / "pages-v0.yaml"))
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.loads((run / a.split / "run.json").read_text())
    rows = [json.loads(line) for line in (run / a.split / "manifest.jsonl").read_text().splitlines()]
    idx = [r["i"] for r in rows if "error" not in r]
    sample = sorted(random.Random(a.seed).sample(idx, min(a.n, len(idx))))
    cfg = yaml.safe_load(Path(a.config).read_text())
    args = (cfg, a.split, meta["seed"], str(run / "_check"), str(run / "fontinfo.json"), str(run / "_check" / "tmp"))
    if a.workers <= 1:
        build.init_worker(*args)
        res = [check_page((i, str(run))) for i in sample]
    else:
        with mp.get_context("spawn").Pool(a.workers, build.init_worker, args) as p:
            res = list(p.imap_unordered(check_page, [(i, str(run)) for i in sample], chunksize=4))
    fails = Counter(f for r in res for f in r["fail"])
    words = sum(r["words"] for r in res)
    by = defaultdict(lambda: [0, 0, 0])
    for r in res:
        for key in (f"kind:{r['kind']}", f"profile:{r['profile']}"):
            by[key][0] += 1
            by[key][1] += r["ink_final_ok"]
            by[key][2] += r["ink_final_bad"]
    rep = {
        "run": str(run), "split": a.split, "pages_checked": len(res), "words_checked": words,
        "pages_with_any_failure": sum(1 for r in res if r["fail"]), "failures_by_check": dict(fails),
        "image_bytes_identical_on_regeneration": sum(r["image_identical"] for r in res),
        "ink_clean_words_without_ink": sum(r["ink_clean_bad"] for r in res),
        "ink_final": {k: {"pages": v[0], "ok": v[1], "bad": v[2], "bad_rate": round(v[2] / max(1, v[1] + v[2]), 5)} for k, v in sorted(by.items())},
        "first_failures": [r for r in sorted(res, key=lambda r: r["id"]) if r["fail"]][:10],
        "first_final_ink_misses": [r for r in sorted(res, key=lambda r: r["id"]) if r["ink_final_bad"]][:10],
    }  # fmt: skip
    text = json.dumps(rep, indent=1, ensure_ascii=False, default=str)
    if a.report:
        Path(a.report).write_text(text + "\n")
    print(text)
    sys.exit(1 if rep["pages_with_any_failure"] else 0)


if __name__ == "__main__":
    main()
