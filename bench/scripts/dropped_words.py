"""Words that one run has and another does not, by page, with whether the truth had them.

    uv run python scripts/dropped_words.py --truth TRUTH_DIR --before DIR --after DIR [--scorer v3.1] [--show 40]

Compares the word sidecars (<id>.json) of the two runs by text and box, lists the words the first run has that the second lacks (and the
reverse), and counts how many of the dropped words were right (the truth page had the text more often than the after-run reads it).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import normalize, truth_text


def key(w: dict) -> tuple:
    return (w["text"], *(round(v) for v in w["box"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--scorer", default="v3.1")
    ap.add_argument("--show", type=int, default=40)
    a = ap.parse_args()
    td, bd, ad = Path(a.truth), Path(a.before), Path(a.after)
    rows = []
    good = bad = 0
    for tf in sorted(td.glob("*.txt")):
        stem = tf.stem
        fb, fa = bd / f"{stem}.json", ad / f"{stem}.json"
        if not fb.exists() or not fa.exists():
            continue
        wb, wa = json.loads(fb.read_text())["words"], json.loads(fa.read_text())["words"]
        kb, ka = Counter(key(w) for w in wb), Counter(key(w) for w in wa)
        gone = kb - ka
        if not gone:
            continue
        ref = Counter(normalize(truth_text(td, stem, a.scorer), a.scorer == "v3.1"))
        after_text = Counter(" ".join(normalize(w["text"], a.scorer == "v3.1")) for w in wa)
        for w in wb:
            k = key(w)
            if gone[k] > 0:
                gone[k] -= 1
                t = " ".join(normalize(w["text"], a.scorer == "v3.1"))
                right = ref[t] > after_text[t]  # the truth wanted another copy of this text that the after run lacks
                good += not right
                bad += right
                x0, y0, x1, y1 = w["box"]
                rows.append((stem, w["text"], round(w.get("conf", w.get("confidence", 0)), 2), round(x1 - x0), round(y1 - y0), "WAS RIGHT" if right else "extra/wrong"))
    print(f"dropped words: {good + bad}; they were wrong or extra: {good}; they were right (truth lost a match): {bad}")
    for r in rows[: a.show]:
        print(*r)
    shown = [r for r in rows if r[5] == "WAS RIGHT"][: a.show]
    if shown:
        print("-- right ones --")
        for r in shown:
            print(*r)


if __name__ == "__main__":
    main()
