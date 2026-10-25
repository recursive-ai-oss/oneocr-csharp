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
            for i in range(max(1, len(w) - 7)):
                seen.add(" ".join(w[i : i + 8]))
    return seen


def verify(a) -> None:
    root = Path(a.root).expanduser()
    names = a.sets or list(sets.SETS)
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
        truths = sorted((d / "truth").glob("*.json"))
        images = sorted((d / "pages").iterdir())
        if len(truths) != sets.SETS[n].count or len(images) != len(truths):
            bad.append(f"{n}: {len(truths)} truths, {len(images)} images, expected {sets.SETS[n].count}")
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
                for i in range(max(1, len(w) - 7)):
                    new_windows.add(" ".join(w[i : i + 8]))
        reused = len(new_windows & old_windows)
        same_pages = sum(1 for rel, h in listed.items() if rel.startswith(f"{n}/pages/") and h in old_hashes)
        if reused or same_pages:
            bad.append(f"{n}: {reused} eight-word runs and {same_pages} page images also in earlier sets")
        print(f"{n}: {len(truths)} pages, {words} words, ink share {share:.4f}, size mismatches {size_bad}, "
              f"runs shared with earlier sets {reused}, identical images {same_pages}")
    if bad:
        print("FAILED:\n  " + "\n  ".join(bad), file=sys.stderr)
        raise SystemExit(1)
    print("sealed set verifies")


def main(a) -> None:
    (build if a.action == "build" else verify)(a)
