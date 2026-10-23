"""Builds the recognizer charset (Latin + Greek + Cyrillic + common punctuation and currency).

The charset is part of the model contract: once a model ships against `charset-v1.txt` the file is
never edited or reordered; a different set becomes `charset-v2.txt`.

File format: one entry per line, UTF-8, line 1 is `<blank>` (the CTC blank, class 0); the rest are
NFC strings, one code point each, in code point order, with the space as the first real class.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

BLANK = "<blank>"

# (first, last) inclusive code point ranges; unassigned code points are dropped by the category check.
RANGES: list[tuple[int, int]] = [
    (0x0020, 0x007E),  # Basic Latin
    (0x00A1, 0x00FF),  # Latin-1 Supplement (soft hyphen removed below)
    (0x0100, 0x017F),  # Latin Extended-A: Central/Eastern European, Turkish, Baltic
    (0x01A0, 0x01A1),  # Vietnamese horned O
    (0x01AF, 0x01B0),  # Vietnamese horned U
    (0x0218, 0x021B),  # Romanian comma-below
    (0x1EA0, 0x1EF9),  # Vietnamese precomposed letters (Latin Extended Additional)
    (0x0386, 0x0386),  # Greek with tonos / dialytika
    (0x0388, 0x038A),
    (0x038C, 0x038C),
    (0x038E, 0x03A1),
    (0x03A3, 0x03CE),
    (0x0400, 0x045F),  # Cyrillic: Russian, Ukrainian, Belarusian, Serbian, Macedonian, Bulgarian
    (0x0490, 0x0491),  # Ukrainian Ghe with upturn
    (0x2013, 0x2014),  # en/em dash
    (0x2018, 0x201E),  # quotes
    (0x2020, 0x2022),  # dagger, double dagger, bullet
    (0x2026, 0x2026),  # ellipsis
    (0x2030, 0x2030),  # per mille
    (0x2032, 0x2033),  # prime, double prime
    (0x2039, 0x203A),  # single angle quotes
    (0x20A3, 0x20A4),  # franc, lira
    (0x20A9, 0x20AB),  # won, shekel, dong
    (0x20AC, 0x20AC),  # euro
    (0x20B4, 0x20B4),  # hryvnia
    (0x20B9, 0x20BA),  # rupee, Turkish lira
    (0x20BD, 0x20BD),  # ruble
    (0x2116, 0x2116),  # numero sign
    (0x2122, 0x2122),  # trade mark
    (0x2190, 0x2193),  # arrows
    (0x2212, 0x2212),  # minus
    (0x2248, 0x2248),
    (0x2260, 0x2260),
    (0x2264, 0x2265),
]
EXCLUDE = {0x00AD}


def build() -> list[str]:
    out = [BLANK]
    seen: set[str] = set()
    for lo, hi in RANGES:
        for cp in range(lo, hi + 1):
            if cp in EXCLUDE:
                continue
            ch = chr(cp)
            if unicodedata.category(ch)[0] not in "LNPSZ" or unicodedata.category(ch) in ("Cn", "Co"):
                continue
            if unicodedata.normalize("NFC", ch) != ch or ch in seen:
                continue
            seen.add(ch)
            out.append(ch)
    return out


def load(path: str | Path) -> list[str]:
    entries = Path(path).read_text(encoding="utf-8").split("\n")
    if entries and entries[-1] == "":
        entries.pop()
    assert entries[0] == BLANK and len(set(entries)) == len(entries)
    return entries


def script_of(ch: str) -> str:
    """Coarse script of one charset entry: latin, greek, cyrillic, or common (digits, punctuation...)."""
    name = unicodedata.name(ch, "")
    for s in ("LATIN", "GREEK", "CYRILLIC"):
        if name.startswith(s):
            return s.lower()
    return "common"


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/charset-v1.txt")
    entries = build()
    out.write_text("\n".join(entries) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for ch in entries[1:]:
        counts[script_of(ch)] = counts.get(script_of(ch), 0) + 1
    print(f"{out}: {len(entries)} classes incl. blank; {counts}")


if __name__ == "__main__":
    main()
