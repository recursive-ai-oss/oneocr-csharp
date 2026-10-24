"""Line-level scoring shared by every engine, and the word records that confidence calibration uses.

Text is normalised the way the page-level scorer does, so an engine is not charged for a font's glyph choice:
NFC, typographic quotes and dashes folded to ASCII, the ligatures fi/fl split, whitespace runs collapsed.
Case and punctuation are kept. CER is the edit distance of the normalised strings over the characters of the normalised
reference (spaces included); WER is the edit distance of the word sequences over the reference's word count. Both are micro
averages over a set's lines.

Hypothesis records are JSONL, one per line: {"id", "ref", "hyp", "words": [{"text", "conf"}]} (`words` only for our own models).
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from data import charset as cs

FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-", "\u00a0": " ",
})  # fmt: skip


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text).translate(FOLD).replace("\ufb01", "fi").replace("\ufb02", "fl")
    return " ".join(t.split())


def line_script(text: str) -> str:
    """latin / greek / cyrillic by the majority of letters; digits and symbols only count as latin."""
    n = {"latin": 0, "greek": 0, "cyrillic": 0}
    for ch in text:
        s = cs.script_of(ch) if ch.strip() else "common"
        if s in n:
            n[s] += 1
    best = max(n, key=lambda k: n[k])
    return best if n[best] else "latin"


def is_ascii_line(text: str) -> bool:
    """Plain English-style text: ASCII plus typographic quotes and dashes, which engines are not charged for."""
    return all(ord(c) < 128 for c in normalize(text))


def slice_of(text: str) -> str:
    """latin-ascii, latin-ext (accents, symbols beyond ASCII), greek or cyrillic."""
    s = line_script(text)
    if s != "latin":
        return s
    return "latin-ascii" if is_ascii_line(text) else "latin-ext"


def read_hyps(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


def score(rows: list[dict], raw: bool = False) -> dict:
    """Micro CER / WER over records with `ref` and `hyp`."""
    ce = ct = we = wt = exact = 0
    for r in rows:
        ref, hyp = (r["ref"], r["hyp"]) if raw else (normalize(r["ref"]), normalize(r["hyp"]))
        ce += Levenshtein.distance(ref, hyp)
        ct += len(ref)
        rw, hw = ref.split(), hyp.split()
        we += Levenshtein.distance(rw, hw)
        wt += len(rw)
        exact += ref == hyp
    return {"cer": ce / max(ct, 1), "wer": we / max(wt, 1), "line_acc": exact / max(len(rows), 1), "lines": len(rows),
            "chars": ct, "words": wt}  # fmt: skip


def score_by_slice(rows: list[dict]) -> dict[str, dict]:
    """Scores of all lines and of each script slice (by the reference text)."""
    out = {"all": score(rows)}
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(slice_of(normalize(r["ref"])), []).append(r)
    for k, g in sorted(groups.items()):
        out[k] = score(g)
    return out
