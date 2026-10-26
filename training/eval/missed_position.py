# ruff: noqa: E501
"""Where our missed words sit relative to the words we did find: a line nothing was found on, the end of a found line, or its inside.

Reads the `--json` output of page_causes.py (with `all_words`), the truth and the engine's word sidecars. For every truth word of class
`missed` it looks at the recognized boxes on the same text line (vertical overlap of at least half the truth word's height):

  no-line      no recognized box on that line near the word (the line, or this part of it, was not detected)
  line-start   found boxes only to the right of the word (a clipped line start or a dropped first word)
  line-end     found boxes only to the left (clipped end or dropped last word)
  interior     found boxes on both sides (the word was dropped from the middle of a line: a blank reading or a filter)

    python eval/missed_position.py CAUSES.json BENCH_ROOT OUT_ROOT [--sets ...]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def box(w: dict) -> list[float]:
    q = w.get("ink_quad") or w.get("ink") or w.get("box") or w["quad"]
    return [float(v) for v in q] if len(q) == 4 else [min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("causes")
    ap.add_argument("bench")
    ap.add_argument("out")
    ap.add_argument("--sets", nargs="*")
    a = ap.parse_args()
    res = json.loads(Path(a.causes).read_text())
    for name in a.sets or res:
        cnt: Counter = Counter()
        chars: Counter = Counter()
        pages: dict[str, list] = {}
        for p, i, t, _h, c, _e, hh in res[name]["all_words"]:
            if c == "missed":
                pages.setdefault(p, []).append((i, t, hh))
        for p, ws in pages.items():
            tw = [w for w in json.loads((Path(a.bench) / name / "truth" / f"{p}.json").read_text())["words"] if " ".join(w["text"].split())]
            hw = [[float(v) for v in w["box"]] for w in json.loads((Path(a.out) / name / f"{p}.json").read_text())["words"]]
            for i, t, h in ws:
                b = box(tw[i])
                hgt = max(b[3] - b[1], 1)
                same = [x for x in hw if min(b[3], x[3]) - max(b[1], x[1]) >= 0.5 * hgt]
                left = any(x[2] <= b[0] + 0.3 * hgt for x in same)
                right = any(x[0] >= b[2] - 0.3 * hgt for x in same)
                k = "interior" if left and right else "line-end" if left else "line-start" if right else "no-line"
                cnt[k] += 1
                chars[k] += len(t)
        n = sum(cnt.values())
        print(f"{name}: {n} missed words: " + ", ".join(f"{k} {cnt[k]}w/{chars[k]}c" for k in ("no-line", "line-start", "line-end", "interior")))


if __name__ == "__main__":
    main()
