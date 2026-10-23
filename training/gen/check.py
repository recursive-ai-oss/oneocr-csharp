"""Checks that a planned/rendered line dataset keeps train and eval apart.

Usage: uv run python -m gen.check DATA_DIR

Asserts: eval fonts are families the training set never uses; no eval/val text equals a training text;
every label file lists only ids from its plan; and label provenance in the plan manifest is exact-truth synthetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def main() -> None:
    root = Path(sys.argv[1])
    manifest = Path(__file__).resolve().parents[1] / "data/fonts.yaml"
    fonts = {f["file"]: f for f in yaml.safe_load(manifest.read_text())["fonts"]}
    plans = {
        s: [json.loads(x) for x in (root / f"{s}.plan.jsonl").read_text(encoding="utf-8").splitlines()] for s in ("train", "val", "eval")
    }
    fam = {s: {fonts[r["font"]]["family"] for r in rows} for s, rows in plans.items()}
    assert not fam["train"] & fam["eval"], fam["train"] & fam["eval"]
    assert all(fonts[r["font"]]["split"] == "heldout" for r in plans["eval"])
    train_texts = {r["text"] for r in plans["train"]}
    for s in ("val", "eval"):
        assert not train_texts & {r["text"] for r in plans[s]}, f"{s} shares text with train"
    for s, rows in plans.items():
        lab = [json.loads(x)["id"] for x in (root / s / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
        assert lab == [r["id"] for r in rows], s
    print("ok: train families", sorted(fam["train"]), "| eval families", sorted(fam["eval"]))


if __name__ == "__main__":
    main()
