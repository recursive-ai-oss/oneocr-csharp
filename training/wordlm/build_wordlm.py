#!/usr/bin/env python3
"""Builds the word frequency file (format: docs/spec/word-lm-v1.md) from Project Gutenberg text.

    python3 training/wordlm/build_wordlm.py --texts DIR --out wordlm-v1.bin [--langs all|en] [--min-count 3]

DIR holds pg<ID>.txt for the books of training/data/texts.yaml (the files `python -m data.texts fetch` writes); each file is
checked against the SHA-256 in the manifest. Only books whose split is `train` are read, the same books the character language model
(training/charlm) is built from, so the held-out books and the two books the page evaluation sets are set in never reach the model.

A token is a space-delimited word of a paragraph with the punctuation around it removed. It is counted only when every character is
a Latin letter (so numbers, codes, words with an apostrophe or a hyphen and non-Latin words are not in the table), after the simple
lowercase mapping of each character (one code point to one; characters whose lowercase is not a single code point stay as they are).
Words seen fewer than --min-count times are left out, and so are words longer than 40 UTF-8 bytes.
Standard library only; the output is deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "charlm"))
import build_charlm as B  # noqa: E402

MAGIC = b"OCWF"
VERSION = 1
BLOCK = 16
MAX_WORD_BYTES = 40
NON_LATIN_LANGS = {"el", "ru", "bg", "sr"}


def is_latin_letter(ch: str) -> bool:
    return unicodedata.category(ch)[0] == "L" and unicodedata.name(ch, "").startswith("LATIN")


def strip_marks(tok: str) -> str:
    a, b = 0, len(tok)
    while a < b and not tok[a].isalpha():
        a += 1
    while b > a and not tok[b - 1].isalpha():
        b -= 1
    return tok[a:b]


def quant(count: int) -> int:
    """Quarter-bits: the count is about 2 ** (q / 4)."""
    return max(1, min(255, round(4 * math.log2(count))))


def count_words(texts: Path, manifest: Path, langs: str) -> tuple[Counter, int]:
    counts: Counter = Counter()
    books = 0
    for b in B.read_manifest(manifest):
        if b.get("split") != "train" or b["language"] in NON_LATIN_LANGS:
            continue
        if langs == "en" and b["language"] != "en":
            continue
        raw = (texts / f"pg{b['id']}.txt").read_bytes()
        if hashlib.sha256(raw).hexdigest() != b["sha256"]:
            sys.exit(f"SHA-256 mismatch for pg{b['id']}.txt")
        books += 1
        for para in B.paragraphs(B.body_of(raw)):
            for tok in para.split(" "):
                core = strip_marks(tok)
                if core and all(is_latin_letter(c) for c in core):
                    counts["".join(B.fold(c) for c in core)] += 1
    print(f"{books} training books, {len(counts)} word types, {sum(counts.values())} tokens", file=sys.stderr)
    return counts, books


def write(counts: Counter, out: Path, min_count: int) -> None:
    total = sum(counts.values())
    kept = sorted(
        ((w.encode("utf-8"), c) for w, c in counts.items() if c >= min_count and len(w.encode("utf-8")) <= MAX_WORD_BYTES),
        key=lambda e: e[0],
    )
    # Words that are not in the table are treated as seen about half as often as the rarest word that is.
    q_missing = max(1, quant(min_count) - 4)
    blocks, data = [], bytearray()
    prev = b""
    for i, (w, c) in enumerate(kept):
        if i % BLOCK == 0:
            blocks.append(len(data))
            prev = b""
        shared = 0
        while shared < min(len(prev), len(w), 255) and prev[shared] == w[shared]:
            shared += 1
        data += bytes([shared, len(w) - shared]) + w[shared:] + bytes([quant(c)])
        prev = w
    header = MAGIC + struct.pack("<HHIIBB", VERSION, 0, total, len(kept), quant(total), q_missing)
    header += struct.pack("<I", len(blocks)) + b"".join(struct.pack("<I", o) for o in blocks)
    out.write_bytes(header + bytes(data))
    print(f"wrote {out}: {out.stat().st_size} bytes, {len(kept)} words (count >= {min_count}), {total} tokens", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "texts.yaml")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--langs", choices=["all", "en"], default="all", help="Latin-script books of every language, or English only")
    ap.add_argument("--min-count", type=int, default=3)
    args = ap.parse_args()
    counts, _ = count_words(args.texts, args.manifest, args.langs)
    write(counts, args.out, args.min_count)


if __name__ == "__main__":
    main()
