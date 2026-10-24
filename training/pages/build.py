"""Synthetic page generator: pages, exact truth and degraded images.

  uv run python -m pages --n 1000 --seed 7 --split train --out $OUT [--workers 8] [--config configs/pages-v0.yaml]

Needs ONEOCR_FONT_ROOT and ONEOCR_TEXT_ROOT (data/fonts.py and data/texts.py fetch) and a Playwright Chromium
(PLAYWRIGHT_BROWSERS_PATH, `playwright install chromium`).

Page i of a run depends only on (seed, split, i): the same seed gives the same truth bytes whatever the number of workers.
Output under OUT/<split>/: images/<id>.png|jpg, truth/<id>.json (oneocr.page, see pages/README.md), manifest.jsonl, run.json.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from . import corpus, degrade, docs, layouts
from .fontpool import assert_disjoint, load_pool
from .render import DSFS, Overflow, Renderer, build_truth, dumps, sha256

HERE = Path(__file__).resolve().parent
_G: dict = {}
CHUNK = 600  # pages per process pool
PAGE_TIMEOUT = 240  # seconds to wait for the next finished page before the chunk is abandoned


def page_seed(seed: int, split: str, i: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{split}:{i}".encode()).digest()[:8], "big")


def init_worker(*args) -> None:
    try:
        _init_worker(*args)
    except BaseException:
        import traceback

        traceback.print_exc()
        raise


def _init_worker(cfg: dict, split: str, seed: int, out: str, font_cache: str, tmp: str) -> None:
    import torch

    torch.set_num_threads(1)
    pool = load_pool(Path(font_cache))
    assert_disjoint(pool)
    _G.update(cfg=cfg, split=split, seed=seed, out=Path(out), pool=pool, bank=corpus.TextBank(split),
              renderer=Renderer(Path(tmp) / f"w{os.getpid()}"), root=Path(os.environ["ONEOCR_FONT_ROOT"]))  # fmt: skip


def choose_dsf(dpi: float) -> float:
    need = dpi / 96
    return next((d for d in DSFS if d >= need - 1e-6), DSFS[-1])


def build_spec(kind: str, ps: int, attempt: int, n_items: int):
    g = _G
    ctx = layouts.Ctx(random.Random(ps ^ (0x9E37 * (attempt + 1))), g["bank"], g["pool"], g["split"], g["root"])
    if kind == "prose":
        return layouts.prose_page(ctx)
    if kind == "table":
        return layouts.table_page(ctx)
    if kind == "form":
        return layouts.form_page(ctx)
    if kind == "invoice":
        return docs.invoice_page(ctx, ps ^ attempt, n_items)
    return docs.receipt_page(ctx, ps ^ attempt)


def render_page(i: int) -> dict:
    """Everything about page i, without writing it: spec, DOM data, clean and final images, truth document and bytes."""
    g = _G
    cfg, split = g["cfg"], g["split"]
    ps = page_seed(g["seed"], split, i)
    rng = random.Random(ps)
    kinds = cfg["mix"]
    kind = rng.choices(list(kinds), list(kinds.values()))[0]
    dcfg = cfg["degrade"]["train" if split == "train" else "heldout"]
    d = degrade.sample(random.Random(ps ^ 0xD3), dcfg)
    dsf = choose_dsf(d["dpi"])
    n_items = rng.randint(1, 26)
    spec = png = raw = None
    for attempt in range(8):
        g["bank"].used.clear()
        spec = build_spec(kind, ps, attempt, n_items)
        try:
            png, raw = g["renderer"].render(spec, dsf)
            # a word box with no ink in the clean render means something hid or recoloured it: draw the page again
            if ink_failures(Image.open(io.BytesIO(png)).convert("RGB"), raw, png_scale(png, raw), png_scale(png, raw, 1)):
                raise Overflow
            break
        except Overflow:
            n_items = max(1, n_items - 3)
    else:
        raise RuntimeError(f"page {i}: no fit after retries")
    missing = [f for f in spec.fonts if f not in raw["fonts_loaded"]]
    if missing:
        raise RuntimeError(f"page {i}: fonts did not load: {missing} {raw['font_status']}")
    clean = Image.open(io.BytesIO(png)).convert("RGB")
    W, H = clean.size
    final, Hm, jq = degrade.apply(clean, d, 96 * dsf)
    data, ext = degrade.encode(final, jq)
    pid = f"{split}-{i:07d}"
    fw, fh = final.size
    synthetic = {
        "id": pid, "generator": "oneocr-pages-v0", "seed": g["seed"], "split": split, "index": i, "kind": spec.kind, "language": spec.lang,
        "template": spec.template, "templateSplit": layouts.template_split(spec.template), "fonts": spec.fonts,
        "books": sorted(g["bank"].used), "renderDpi": round(96 * dsf, 1), "css": {"width": raw["size"][0], "height": raw["size"][1]},
        "degradation": d, "image": {"file": f"images/{pid}.{ext}", "width": fw, "height": fh}, "droppedItems": raw["dropped"],
    }  # fmt: skip
    # pixels per inch of the final image as laid out: the dpi drawn, but never more than was rendered
    ppi = min(d["dpi"], 96 * dsf)
    doc, flat = build_truth(raw, spec, (W, H), lambda b: degrade.transform_box(Hm, b), {"units_per_inch": ppi, "synthetic": synthetic})
    doc["pages"][0]["width"], doc["pages"][0]["height"] = fw, fh
    return {"pid": pid, "spec": spec, "raw": raw, "clean": clean, "final": final, "data": data, "ext": ext, "doc": doc, "flat": flat,
            "truth": dumps(doc), "degradation": d, "H": Hm}  # fmt: skip


def make_page(i: int) -> dict:
    r = render_page(i)
    g = _G
    raw, clean, spec, d = r["raw"], r["clean"], r["spec"], r["degradation"]
    W, H = clean.size
    ink_fail = ink_failures(clean, raw, W / raw["size"][0], H / raw["size"][1])
    out = g["out"] / g["split"]
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    (out / "images" / f"{r['pid']}.{r['ext']}").write_bytes(r["data"])
    (out / "truth" / f"{r['pid']}.json").write_bytes(r["truth"])
    return {"id": r["pid"], "i": i, "kind": spec.kind, "words": len(r["flat"]), "profile": d["profile"],
            "template_split": r["doc"]["synthetic"]["templateSplit"], "image_sha256": sha256(r["data"]), "truth_sha256": sha256(r["truth"]),
            "ink_fail": ink_fail, "fonts": spec.fonts, "split_words": sum(1 for w in r["flat"] if w["split"])}  # fmt: skip


def png_scale(png: bytes, raw: dict, axis: int = 0) -> float:
    w, h = Image.open(io.BytesIO(png)).size
    return (w, h)[axis] / raw["size"][axis]


def ink_failures(clean: Image.Image, raw: dict, sx: float, sy: float) -> int:
    """Words whose box in the clean render holds no ink: fewer than 2 distinct tone levels (spread under 40 grey levels)."""
    a = np.asarray(clean.convert("L"), dtype=np.int16)
    bad = 0
    for w in raw["words"]:
        l, t, r, b = w["r"]
        x0, x1 = max(0, int(l * sx)), min(a.shape[1], int(np.ceil(r * sx)))
        y0, y1 = max(0, int(t * sy)), min(a.shape[0], int(np.ceil(b * sy)))
        if x1 <= x0 or y1 <= y0:
            bad += 1
            continue
        c = a[y0:y1, x0:x1]
        if int(c.max()) - int(c.min()) < 40:
            bad += 1
    return bad


def make_page_safe(i: int) -> dict:
    try:
        return make_page(i)
    except Exception as e:  # keep the run going; failures are listed in the manifest
        return {"i": i, "error": f"{type(e).__name__}: {e}"[:300]}


def git_commit() -> str:
    """The commit of the generator: ONEOCR_GIT_COMMIT when the checkout has no git (a synced copy), else `git rev-parse HEAD`."""
    if os.environ.get("ONEOCR_GIT_COMMIT"):
        return os.environ["ONEOCR_GIT_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=str(HERE.parent / "configs" / "pages-v0.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    font_cache = str(out / "fontinfo.json")
    load_pool(Path(font_cache))  # describe the fonts once before the workers start
    tmp = str(out / "tmp")
    t0 = time.time()
    idx = list(range(a.start, a.start + a.n))
    sdir = out / a.split
    sdir.mkdir(parents=True, exist_ok=True)
    mpath = sdir / "manifest.jsonl"
    if not a.start:
        mpath.write_text("")
    rows: list[dict] = []

    def flush(part: list[dict]) -> None:
        part.sort(key=lambda r: r["i"])
        with mpath.open("a", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows.extend(part)
        print(f"{len(rows)}/{a.n} {len(rows) / (time.time() - t0):.2f} pages/s", file=sys.stderr, flush=True)

    if a.workers <= 1:
        init_worker(cfg, a.split, a.seed, str(out), font_cache, tmp)
        for k in range(0, len(idx), 100):
            flush([make_page_safe(i) for i in idx[k : k + 100]])
    else:
        # Chunks with a fresh process pool each: a hung Chromium costs one chunk's stragglers (recorded as timeouts), not the run.
        for k in range(0, len(idx), CHUNK):
            part_idx = idx[k : k + CHUNK]
            got: list[dict] = []
            pool = mp.get_context("spawn").Pool(a.workers, init_worker, (cfg, a.split, a.seed, str(out), font_cache, tmp))
            try:
                it = pool.imap_unordered(make_page_safe, part_idx, chunksize=1)
                for _ in part_idx:
                    got.append(it.next(timeout=PAGE_TIMEOUT))
            except mp.TimeoutError:
                done = {r["i"] for r in got}
                got += [{"i": i, "error": "timeout"} for i in part_idx if i not in done]
            finally:
                pool.terminate()
                pool.join()
            flush(got)
    dt = time.time() - t0
    rows.sort(key=lambda r: r["i"])
    sdir = out / a.split
    ok = [r for r in rows if "error" not in r]
    import torch

    run = {
        "generator": "oneocr-pages-v0", "seed": a.seed, "split": a.split, "start": a.start, "n": a.n, "ok": len(ok), "errors": len(rows) - len(ok),
        "seconds": round(dt, 1), "pages_per_second": round(len(rows) / dt, 3), "workers": a.workers, "git_commit": git_commit(),
        "config": cfg, "config_sha256": sha256(Path(a.config).read_bytes()),
        "libraries": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pillow": Image.__version__},
        "manifest_sha256": sha256((sdir / "manifest.jsonl").read_bytes()), "cpu_count": os.cpu_count(),
        "fonts_pinned_in": "data/fonts.yaml", "texts_pinned_in": "data/texts.yaml",
    }  # fmt: skip
    (sdir / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("n", "ok", "errors", "seconds", "pages_per_second")}))
    for r in rows:
        if "error" in r:
            print(r, file=sys.stderr)
