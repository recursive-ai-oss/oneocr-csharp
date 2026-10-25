"""Cut and verify a sealed page set: the six categories rendered from seeds that are not in this repository.

    python -m oneocr_bench seal build  ROOT --seeds SEEDS.json --avoid-text <eval-root>...
    python -m oneocr_bench seal verify ROOT --avoid-text <eval-root>...

`build` renders each category with the recipes of the pinned sets and a seed taken from SEEDS.json ({"<set>": int}),
keeps passages out that an earlier eval root already used, writes `SHA256SUMS` and `SEALED.json` (what each category is
made of, no seeds), and makes the tree read-only. Receipts come from the receipt generator: no unseen CORD images
exist. `verify` is the only check anyone should run on a sealed tree: hashes, truth structure, image sizes against
truth, ink boxes against pixels, and overlap with earlier sets, printed as counts only.

Pages are never scored here and no engine is run on them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import gen, sets
from .assets import FAMILIES

KINDS = {
    "clean300-v1": "generated page; public-domain prose; exact DOM-range truth",
    "degraded-v1": "generated page with simulated scan degradation; truth moved with the image",
    "screenshot-v1": "generated UI and web screens; exact DOM-range truth",
    "layout-v1": "generated columns, tables and forms; exact DOM-range truth",
    "phone-v1": "generated page with simulated phone capture; truth moved with the image",
    "receipts-v1": "generated receipts (synthetic text) with simulated phone capture; truth moved with the image",
}
PROSE_SETS = {"clean300-v1", "degraded-v1", "phone-v1"}
INK_STD = 5.0  # a word's ink box must show some contrast against the page
INK_SHARE = 0.97  # share of words that must pass


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS")


def write_sums(root: Path) -> str:
    lines = [f"{_sha(p)}  {p.relative_to(root).as_posix()}" for p in _files(root)]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return _sha(root / "SHA256SUMS")


def lock(root: Path) -> None:
    for p in sorted(root.rglob("*"), reverse=True):
        mode = p.stat().st_mode
        p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    root.chmod(root.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def build(a) -> None:
    root = Path(a.root).expanduser()
    seeds = json.loads(Path(a.seeds).expanduser().read_text())
    names = a.sets or list(sets.SETS)
    avoid = [Path(p).expanduser() for p in a.avoid_text]
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} is not empty; a sealed tree is cut once")
    root.mkdir(parents=True, exist_ok=True)
    gen.AVOID_TEXT[:] = avoid
    summary = {}
    for n in names:
        if n not in seeds:
            raise SystemExit(f"no seed for {n}")
        out = root / n
        manifest = sets.render(n, sets.SETS[n].count, out, int(seeds[n]), synthetic_receipts=(n == "receipts-v1"))
        sets.sanity(n, out)
        (out / "page.html").unlink(missing_ok=True)
        (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
        summary[n] = {"pages": len(manifest["pages"]), "made_of": KINDS[n], "real_images": False}
        print(f"{n}: {summary[n]['pages']} pages", flush=True)
    (root / "SEALED.json").write_text(json.dumps({"name": "bench-v3-sealed", "sets": summary}, indent=1) + "\n")
    digest = write_sums(root)
    lock(root)
    print(f"SHA256SUMS {digest}")


def _windows(root: Path) -> set[str]:
    seen: set[str] = set()
    for f in sorted(root.glob("*/truth/*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            w = line.split()
            for i in range(len(w) - 7):
                seen.add(" ".join(w[i : i + 8]))
    return seen


def verify(a) -> None:
    root = Path(a.root).expanduser()
    meta = json.loads((root / "SEALED.json").read_text())["sets"]
    names = a.sets or list(meta)
    bad: list[str] = []

    listed = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        h, rel = line.split("  ", 1)
        listed[rel] = h
    on_disk = {p.relative_to(root).as_posix() for p in _files(root)}
    if set(listed) != on_disk:
        bad.append(f"SHA256SUMS lists {len(listed)} files, tree has {len(on_disk)}")
    wrong = sum(1 for rel, h in listed.items() if rel in on_disk and _sha(root / rel) != h)
    if wrong:
        bad.append(f"{wrong} files differ from SHA256SUMS")
    print(f"hashes: {len(listed)} files, {wrong} differ")

    old_hashes: set[str] = set()
    for p in map(Path, a.avoid_text):
        s = p.expanduser() / "SHA256SUMS"
        if s.exists():
            old_hashes |= {ln.split("  ")[0] for ln in s.read_text().splitlines()}
    old_windows: set[str] = set()
    for p in a.avoid_text:
        old_windows |= _windows(Path(p).expanduser())

    for n in names:
        d = root / n
        sets.sanity(n, d)
        expected = meta[n]["pages"]
        truths = sorted((d / "truth").glob("*.json"))
        images = sorted((d / "pages").iterdir())
        if len(truths) != expected or len(images) != len(truths):
            bad.append(f"{n}: {len(truths)} truths, {len(images)} images, expected {expected}")
        words = weak = size_bad = font_bad = 0
        for t in truths:
            tr = json.loads(t.read_text())
            img = next(p for p in images if p.stem == t.stem)
            with Image.open(img) as im:
                g = np.asarray(im.convert("L"), np.float32)
            if (g.shape[1], g.shape[0]) != (tr["image"]["width"], tr["image"]["height"]):
                size_bad += 1
            if tr.get("font") and tr["font"] not in FAMILIES:
                font_bad += 1
            for w in tr["words"]:
                x0, y0, x1, y1 = (round(v) for v in w["ink"])
                reg = g[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
                words += 1
                if reg.size == 0 or float(reg.std()) < INK_STD:
                    weak += 1
        share = 1 - weak / max(1, words)
        if size_bad or font_bad or share < INK_SHARE:
            bad.append(f"{n}: {size_bad} size mismatches, {font_bad} unknown fonts, ink share {share:.4f}")
        new_windows: set[str] = set()
        for f in sorted((d / "truth").glob("*.txt")):
            for line in f.read_text(encoding="utf-8").splitlines():
                w = line.split()
                for i in range(len(w) - 7):
                    new_windows.add(" ".join(w[i : i + 8]))
        shared = new_windows & old_windows
        # a run with digits is a generated figure line (dates, amounts, item counts) that two seeds can draw alike;
        # a digit-free run in a prose set would be a repeated passage
        reused = sum(1 for k in shared if not any(c.isdigit() for c in k)) if n in PROSE_SETS else len(shared)
        same_pages = sum(1 for rel, h in listed.items() if rel.startswith(f"{n}/pages/") and h in old_hashes)
        # screens and layouts reuse fixed interface and form wording by design; only prose sets must not repeat passages
        if (reused and n in PROSE_SETS) or same_pages:
            bad.append(f"{n}: {reused} eight-word runs and {same_pages} page images also in earlier sets")
        print(f"{n}: {len(truths)} pages, {words} words, ink share {share:.4f}, size mismatches {size_bad}, "
              f"runs shared with earlier sets {reused}, identical images {same_pages}")
    if bad:
        print("FAILED:\n  " + "\n  ".join(bad), file=sys.stderr)
        raise SystemExit(1)
    print("sealed set verifies")


REAL_KINDS = {
    "screenshot-real": {
        "made_of": "real public web pages captured in headless Chromium; images and generated content blanked so every glyph is DOM text; truth from DOM ranges",
        "order_metric": "information",
        "order_note": "DOM order of visible text is not always the visual reading order",
    },
    "clean-real": {
        "made_of": "real born-digital PDF pages rendered at 300 DPI; truth from the text layer, cross-checked by two independent readers",
        "order_metric": "gated",
        "order_note": "content-stream order, kept only where two reading-order analyses agree",
    },
    "layout-real": {
        "made_of": "real born-digital PDF pages (columns, tables, forms) rendered at 300 DPI; truth from the text layer, cross-checked",
        "order_metric": "gated",
        "order_note": "content-stream order, kept only where two reading-order analyses agree",
    },
    "degraded-real": {
        "made_of": "other pages of the same real PDFs, run through the bench scan-degradation simulator; truth moved with the image",
        "order_metric": "gated",
        "order_note": "as clean-real",
    },
}
NO_REAL_SOURCE = {
    "phone-v1": "no real photographs with exact truth exist; the row is a simulated camera capture of a generated page",
    "receipts-v1": "no unseen real receipts with exact truth exist (CORD v2 is fully used); the row is synthetic text with a simulated camera",
}


def _unlock(root: Path) -> None:
    for p in [root, *root.rglob("*")]:
        p.chmod(p.stat().st_mode | stat.S_IWUSR)


def add_real(a) -> None:
    """Add the real-sourced sets (screenshot from --web, the PDF sets from --pdf) to a sealed tree and rewrite SEALED.json and
    SHA256SUMS. Every file of the original sets keeps its hash; the old SHA256SUMS digest is recorded."""
    root = Path(a.root).expanduser()
    old_sums = (root / "SHA256SUMS").read_text()
    old_digest = hashlib.sha256(old_sums.encode()).hexdigest()
    _unlock(root)
    meta = json.loads((root / "SEALED.json").read_text())
    sources = {"screenshot-real": Path(a.web).expanduser() if a.web else None}
    for n in ("clean-real", "layout-real", "degraded-real"):
        sources[n] = Path(a.pdf).expanduser() / n if a.pdf else None
    for n, src in sources.items():
        if src is None or not (src / "pages").is_dir():
            continue
        if (root / n).exists():
            raise SystemExit(f"{n} already in the sealed tree")
        shutil.copytree(src, root / n, ignore=shutil.ignore_patterns("page.html"))
        pages = len(list((root / n / "pages").iterdir()))
        meta["sets"][n] = {"pages": pages, **REAL_KINDS[n], "real_images": False, "real_sourced": True}
        print(f"{n}: {pages} pages added")
    meta["no_real_source"] = NO_REAL_SOURCE
    meta["real_sets_added"] = {"previous_sha256sums_digest": old_digest}
    (root / "SEALED.json").write_text(json.dumps(meta, indent=1) + "\n")
    digest = write_sums(root)
    new_lines = set((root / "SHA256SUMS").read_text().splitlines())
    kept = [ln for ln in old_sums.splitlines() if ln.split("  ", 1)[1] != "SEALED.json"]
    if not set(kept) <= new_lines:
        raise SystemExit("a file of the original sets changed")
    lock(root)
    print(f"previous SHA256SUMS digest {old_digest}\nSHA256SUMS {digest}")


def main(a) -> None:
    {"build": build, "verify": verify, "add-real": add_real}[a.action](a)
