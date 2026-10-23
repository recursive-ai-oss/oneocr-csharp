"""Checks that a planned/rendered line dataset keeps train and eval apart.

Usage: uv run python -m gen.check DATA_DIR [PAGE_TRUTH_DIR]

Asserts: eval fonts are families the training set never uses (and none of the page-evaluation families appears anywhere);
validation and eval lines come from books the training set never reads; no val/eval text of 20+ characters equals a training
text (short fragments and generated strings that coincide by chance are counted, not failed); the held-out degradation ranges
do not overlap the training ranges; the calibration split (if any) shares no text of 20+ characters with validation or evaluation;
every label file lists exactly the ids of its plan. With a page truth directory it also
counts 8-word sequences shared between the training texts and the page evaluation texts (must be zero).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PAGE_EVAL_FAMILIES = {"gelasio", "carlito", "caladea", "sourceserif4", "sourcesans3", "ibmplexserif", "ibmplexsans"}


def read(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def disjoint(a: list, b: list) -> bool:
    """Ranges given as [lo, hi] or a list of them; True when no interval of a meets one of b."""
    ra = a if isinstance(a[0], list) else [a]
    rb = b if isinstance(b[0], list) else [b]
    return all(x[1] <= y[0] or y[1] <= x[0] for x in ra for y in rb)


def main() -> None:
    root = Path(sys.argv[1])
    base = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "plan.manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    fonts = {f["file"]: f for f in yaml.safe_load((base / "data/fonts.yaml").read_text())["fonts"]}
    assert not {f["file"].split("/")[1] for f in fonts.values()} & PAGE_EVAL_FAMILIES, "page-eval family in fonts.yaml"
    plans = {s: read(root / f"{s}.plan.jsonl") for s in cfg["sets"]}
    fam = {s: {fonts[r["font"]]["family"] for r in rows} for s, rows in plans.items()}
    train_fam = fam["train"]
    for s, spec in cfg["sets"].items():
        if spec["fonts"] == "heldout":
            assert not train_fam & fam[s], (s, train_fam & fam[s])
            assert all(fonts[r["font"]]["split"] == "heldout" for r in plans[s])
    train_books = set(manifest["sets"]["train"]["books"])
    train_texts = {r["text"] for r in plans["train"] if r["src"].startswith("book")}
    train_synth = {r["text"] for r in plans["train"] if not r["src"].startswith("book")}
    for s, spec in cfg["sets"].items():
        if s == "train":
            continue
        assert not train_books & set(manifest["sets"][s]["books"]), f"{s} shares books with train"
        # short fragments ("and", "of the") coincide between different books by nature; longer texts must not
        long_dup = [t for t in train_texts & {r["text"] for r in plans[s]} if len(t) >= 20]
        assert not long_dup, f"{s} shares book text with train: {long_dup[:3]}"
        # generated strings (a short number, a code) can coincide by chance; that is not a leak, but it is counted
        dup = len(train_synth & {r["text"] for r in plans[s] if not r["src"].startswith("book")})
        print(f"{s}: {dup} generated strings also occur in train")
        if spec["degrade"] == "heldout":
            if "gpu_degrade" in cfg:  # every sampled quantity of the GPU stage has held-out ranges that miss the training ranges
                g = cfg["gpu_degrade"]
                assert set(g["heldout"]["ranges"]) == set(g["train"]["ranges"])
                for k, r in g["heldout"]["ranges"].items():
                    assert disjoint(r, g["train"]["ranges"][k]), k
                continue
            for k in ("em", "blur", "noise", "contrast", "jpeg_q"):
                assert disjoint(cfg["degrade"]["heldout"][k], cfg["degrade"]["train"][k]), k
    if "calib" in plans:  # the calibration split shares no line (20+ characters) with validation or evaluation, nor any book with training
        cal = {r["text"] for r in plans["calib"] if len(r["text"]) >= 20}
        for other in plans:
            if other not in ("calib", "train"):
                both = cal & {r["text"] for r in plans[other]}
                assert not both, f"calib shares text with {other}: {sorted(both)[:3]}"
        n_fonts = len({r["font"] for r in plans["calib"]})
        print(f"calib: {len(plans['calib'])} lines, {n_fonts} font files, no text shared with the other sets")
    for s, rows in plans.items():
        lab = [json.loads(x)["id"] for x in (root / s / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
        assert lab == [r["id"] for r in rows], s
        assert all(r["src"].startswith(("book:", "synthetic:")) for r in rows)
    print("ok: train families", len(train_fam), "| eval families", sorted(fam["eval"]))
    if len(sys.argv) > 2:
        words = [t.split() for t in train_texts]
        grams = {" ".join(w[i : i + 8]) for w in words for i in range(len(w) - 7)}
        shared = 0
        for tp in sorted(Path(sys.argv[2]).glob("*.txt")):
            w = tp.read_text(encoding="utf-8").split()
            shared += sum(" ".join(w[i : i + 8]) in grams for i in range(len(w) - 7))
        print("8-word sequences shared with page evaluation text:", shared)
        assert shared == 0


if __name__ == "__main__":
    main()
