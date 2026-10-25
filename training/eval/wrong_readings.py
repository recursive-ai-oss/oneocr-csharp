"""What the wrongly read words of an engine output look like: confusions, heights, contrast, amounts, currency marks, scripts.

Takes the words the recall-CER attribution calls "wrong" (a recognised box touches the word and reads something else), aligns the
closest touching text with the truth by edit operations, and tallies the characters lost by confusion, kind of edit, word height,
crop contrast and shape (amounts with `.` or `,`, currency marks, look-alike letters from another script). Rates divide by all truth
characters in the same slice. Truth and hypothesis are in the benchmark layout, pages beside them.

    cd bench && uv run python ../training/eval/wrong_readings.py TRUTH_DIR HYP_DIR PAGES_DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from oneocr_bench.score import Levenshtein, normalize  # noqa: E402

from eval.recall_attribution import attribute_page, box, touches  # noqa: E402

CURRENCY = set("Rp$€₽¥£₩₫₹฿")
SEPS = set(".,:;'")
HEIGHT_BINS = [0, 16, 24, 40, 1e9]
CONTRAST_BINS = [0, 60, 90, 130, 1e9]
AMOUNT = re.compile(r"^[A-Za-z]{0,3}[\d.,]*\d[\d.,]*$")
LOOKALIKE_DIGIT = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8", "Z": "2", "g": "9", "q": "9", "G": "6"}


def script(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    n = unicodedata.name(ch, "").split()[0]
    return n if n in ("LATIN", "CYRILLIC", "GREEK") else "OTHER"


def kind(t: str, h: str) -> str:
    """One edit operation (truth char t or '', hypothesis char h or '') in one class."""
    if t and h:
        if t.lower() == h.lower():
            return "case"
        if unicodedata.normalize("NFD", t)[0] == unicodedata.normalize("NFD", h)[0]:
            return "accent"
        if t in SEPS and h in SEPS:
            return "separator swap"
        if t.isdigit() and h.isdigit():
            return "digit-digit"
        if (t.isdigit() and LOOKALIKE_DIGIT.get(h) == t) or (h.isdigit() and LOOKALIKE_DIGIT.get(t) == h):
            return "letter-digit look-alike"
        if t.isdigit() or h.isdigit():
            return "digit-other"
        if script(h) in ("CYRILLIC", "GREEK") and script(t) == "LATIN":
            return "Latin->Cyrillic/Greek"
        if t.isalpha() and h.isalpha():
            return "letter-letter"
        return "symbol"
    c = t or h
    what = "dropped" if t else "inserted"
    if c in SEPS:
        return f"separator {what}"
    if c in CURRENCY and not c.isalpha():
        return f"currency mark {what}"
    return f"{'digit' if c.isdigit() else 'letter' if c.isalpha() else 'symbol'} {what}"


def contrast(img: np.ndarray, b: list[float]) -> tuple[float, str]:
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    c = img[max(y0, 0) : max(y1, 1), max(x0, 0) : max(x1, 1)]
    if c.size < 4:
        return 0.0, "?"
    lo, hi = np.percentile(c, [5, 95])
    # polarity: more than half of the crop is dark and the ink (extreme tail) is the light side
    return float(hi - lo), ("light-on-dark" if np.median(c) < (lo + hi) / 2 else "dark-on-light")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("hyp")
    ap.add_argument("pages")
    ap.add_argument("--json")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    all_chars = collections.Counter()  # slice -> truth chars
    lost = collections.Counter()  # slice -> chars lost in wrong words
    words = collections.Counter()
    conf: collections.Counter = collections.Counter()
    ops = collections.Counter()
    rows = []
    for tf in sorted(Path(a.truth).glob("*.json")):
        tw = json.loads(tf.read_text())["words"]
        hf = Path(a.hyp) / tf.name
        hw = json.loads(hf.read_text())["words"] if hf.exists() else []
        pg = next(Path(a.pages).glob(tf.stem + ".*"), None)
        img = np.asarray(Image.open(pg).convert("L")) if pg else None
        for w in tw:
            n = sum(len(t) for t in normalize(w["text"]))
            b = box(w)
            sl = [f"h {lo}-{hi:g}" for lo, hi in zip(HEIGHT_BINS, HEIGHT_BINS[1:], strict=False) if lo <= b[3] - b[1] < hi]
            if img is not None:
                cc, pol = contrast(img, b)
                sl += [f"contrast {lo}-{hi:g}" for lo, hi in zip(CONTRAST_BINS, CONTRAST_BINS[1:], strict=False) if lo <= cc < hi] + [pol]
            sl.append("amount" if AMOUNT.match(w["text"].strip()) and any(ch.isdigit() for ch in w["text"]) else "other shape")
            if len(w["text"].strip()) == 1:
                sl.append("single character")
            for s in sl:
                all_chars[s] += n
        for t, w, miss, c, texts, _paired in attribute_page(tw, hw):
            if c != "wrong" or not miss:
                continue
            read = min(texts, key=lambda x: Levenshtein.distance(t, x)) if texts else ""
            b = box(w)
            sl = [f"h {lo}-{hi:g}" for lo, hi in zip(HEIGHT_BINS, HEIGHT_BINS[1:], strict=False) if lo <= b[3] - b[1] < hi]
            if img is not None:
                cc, pol = contrast(img, b)
                sl += [f"contrast {lo}-{hi:g}" for lo, hi in zip(CONTRAST_BINS, CONTRAST_BINS[1:], strict=False) if lo <= cc < hi] + [pol]
            sl.append("amount" if AMOUNT.match(w["text"].strip()) and any(ch.isdigit() for ch in w["text"]) else "other shape")
            if len(t) == 1:
                sl.append("single character")
            for s in sl:
                lost[s] += miss
                words[s] += 1
            mixed = len({script(ch) for ch in read if script(ch)} - {"OTHER"}) > 1
            for tag, i1, i2, j1, j2 in Levenshtein.opcodes(t, read):
                if tag == "equal":
                    continue
                for k in range(max(i2 - i1, j2 - j1)):
                    tc = t[i1 + k] if i1 + k < i2 else ""
                    hc = read[j1 + k] if j1 + k < j2 else ""
                    conf[(tc or "∅", hc or "∅")] += 1
                    ops[kind(tc, hc)] += 1
            if mixed:
                ops["(words with mixed-script reading)"] += 1
            hb = next((box(h) for h in hw if normalize(h["text"]) and read in normalize(h["text"]) and touches(b, box(h))), None)
            rows.append({"page": tf.stem, "truth": t, "read": read, "miss": miss, "h": round(b[3] - b[1], 1), "box": b, "hbox": hb})
    total_lost = sum(r["miss"] for r in rows)
    print(f"wrong words: {len(rows)}, chars lost {total_lost}")
    print("edit kinds (operations):", ", ".join(f"{k} {v}" for k, v in ops.most_common()))
    print("top confusions (truth->read):", " ".join(f"{t}>{h}:{n}" for (t, h), n in conf.most_common(a.top)))
    print("slice: wrong words, chars lost, truth chars, loss rate % in the slice")
    for s in sorted(all_chars):
        print(f"  {s:22s} {words[s]:4d} {lost[s]:5d} {all_chars[s]:6d} {100 * lost[s] / max(all_chars[s], 1):5.2f}")
    if a.json:
        Path(a.json).write_text(
            json.dumps({"rows": rows, "confusions": [[t, h, n] for (t, h), n in conf.most_common()]}, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
