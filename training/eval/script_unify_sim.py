"""What recall-CER and CER would be if a word never mixed look-alike letters of two scripts (a simulation on engine output).

The recognizer has one class set for Latin, Greek and Cyrillic, which share look-alike letters (Latin T, Greek Τ, Cyrillic Т). A word
such as "Тax" (Cyrillic Т) costs a character although every glyph is right. This rewrites each hypothesis word so that its minority
script's look-alikes take the majority script's letter (ties go to Latin, the script of the sets that matter first), leaves words
with no look-alike pair alone, and scores the original and the rewritten text with the benchmark's own functions.

    cd bench && uv run python ../training/eval/script_unify_sim.py TRUTH_DIR HYP_DIR
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from oneocr_bench.score import content_errors, normalize  # noqa: E402

# Look-alike pairs, Latin letter <- Cyrillic / Greek letters drawn alike in common fonts.
LATIN_OF = {}
for lat, others in {
    "A": "АΑ", "B": "ВΒ", "C": "С", "E": "ЕΕ", "H": "НΗ", "I": "ІΙ", "J": "Ј", "K": "КΚ", "M": "МΜ", "N": "Ν", "O": "ОΟ", "P": "РΡ",
    "S": "Ѕ", "T": "ТΤ", "X": "ХΧ", "Y": "ΥҮ", "Z": "Ζ", "a": "а", "c": "с", "e": "е", "i": "і", "j": "ј", "o": "оο", "p": "рρ",
    "s": "ѕ", "x": "х", "y": "у", "v": "ν",
}.items():  # fmt: skip
    for o in others:
        LATIN_OF[o] = lat
OTHER_OF: dict[str, dict[str, str]] = {"CYRILLIC": {}, "GREEK": {}}
for o, lat in LATIN_OF.items():
    OTHER_OF[unicodedata.name(o).split()[0]].setdefault(lat, o)


def script(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    n = unicodedata.name(ch, "").split()[0]
    return n if n in ("LATIN", "CYRILLIC", "GREEK") else None


def unify(word: str) -> str:
    cnt: dict[str, int] = {}
    for ch in word:
        s = script(ch)
        if s:
            cnt[s] = cnt.get(s, 0) + 1
    if len(cnt) < 2:
        return word
    major = max(cnt, key=lambda s: (cnt[s], s == "LATIN"))
    out = []
    for ch in word:
        s = script(ch)
        if s and s != major:
            if major == "LATIN" and ch in LATIN_OF:
                ch = LATIN_OF[ch]
            elif major != "LATIN" and s == "LATIN" and ch in OTHER_OF[major]:
                ch = OTHER_OF[major][ch]
        out.append(ch)
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("hyp")
    a = ap.parse_args()
    tot = {"ref": 0, "ce0": 0, "cm0": 0, "ce1": 0, "cm1": 0, "changed": 0, "words": 0}
    for tf in sorted(Path(a.truth).glob("*.txt")):
        ref = normalize(tf.read_text(encoding="utf-8"))
        hf = Path(a.hyp) / tf.name
        hyp = normalize(hf.read_text(encoding="utf-8")) if hf.exists() else []
        new = [unify(w) for w in hyp]
        e0, e1 = content_errors(ref, hyp), content_errors(ref, new)
        tot["ref"] += sum(map(len, ref))
        tot["ce0"] += e0[0]
        tot["cm0"] += e0[2]
        tot["ce1"] += e1[0]
        tot["cm1"] += e1[2]
        tot["changed"] += sum(x != y for x, y in zip(hyp, new, strict=True))
        tot["words"] += len(hyp)
    r = tot["ref"]
    print(f"{tot['changed']}/{tot['words']} words rewritten")
    cer = f"{100 * tot['ce0'] / r:.2f} -> {100 * tot['ce1'] / r:.2f}"
    rec = f"{100 * tot['cm0'] / r:.2f} -> {100 * tot['cm1'] / r:.2f}"
    print(f"order-free CER {cer}   recall-CER {rec}")


if __name__ == "__main__":
    main()
