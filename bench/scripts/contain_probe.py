"""How many leftover (extra) hypothesis words sit inside another word's box, and how many correct words would a containment rule hit.

    uv run python scripts/contain_probe.py --root EVAL_ROOT --out OUT_ROOT --sets A B ... [--maxlen 2] [--frac 0.6]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import normalize, truth_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--maxlen", type=int, default=2)
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--stroke-conf", type=float, default=0.5, help="also report one-character words below this confidence, by box width over height")
    a = ap.parse_args()
    for s in a.sets:
        td = Path(a.root) / s / "truth"
        hit_extra = hit_ok = n_extra = 0
        for tf in sorted(td.glob("*.txt")):
            ref = Counter(normalize(truth_text(td, tf.stem, "v3.1"), True))
            f = Path(a.out) / s / f"{tf.stem}.json"
            if not f.exists():
                continue
            words = json.loads(f.read_text())["words"]
            seen: Counter = Counter()
            for i, w in enumerate(words):
                t = " ".join(normalize(w["text"], True))
                seen[t] += 1
                is_extra = seen[t] > ref[t]
                n_extra += is_extra
                x0, y0, x1, y1 = w["box"]
                if len(w["text"]) == 1 and w.get("conf", 1) < a.stroke_conf:
                    print(s, tf.stem, "STROKE-EXTRA" if is_extra else "STROKE-OK", repr(w["text"]), "w/h", round((x1 - x0) / max(y1 - y0, 1), 2), "conf", round(w.get("conf", 0), 2))
                if len(w["text"]) > a.maxlen:
                    continue
                area = max((x1 - x0) * (y1 - y0), 1e-9)
                for j, o in enumerate(words):
                    if j == i or len(o["text"]) <= len(w["text"]):
                        continue
                    ox0, oy0, ox1, oy1 = o["box"]
                    inter = max(min(x1, ox1) - max(x0, ox0), 0) * max(min(y1, oy1) - max(y0, oy0), 0)
                    if inter / area >= a.frac:
                        if is_extra:
                            hit_extra += 1
                            print(s, tf.stem, "EXTRA", repr(w["text"]), "in", repr(o["text"]), round(w.get("conf", 0), 2))
                        else:
                            hit_ok += 1
                            print(s, tf.stem, "OK-WORD", repr(w["text"]), "in", repr(o["text"]), round(w.get("conf", 0), 2))
                        break
        print(f"== {s}: extras {n_extra}, extras inside another word {hit_extra}, correct words inside another word {hit_ok}")


if __name__ == "__main__":
    main()
