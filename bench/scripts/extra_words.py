"""Words a run reports that another does not (by text and box), with the tagged value they fall into, if any.

    uv run python scripts/extra_words.py --eval EVAL_ROOT --set fin-tables --a RUN_ROOT --b RUN_ROOT [--show 30]

Lists the words of --b missing in --a, how many lie in no tagged value, in a quantity cell, or in an amount/date/account value's region.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench import amounts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--show", type=int, default=30)
    x = ap.parse_args()
    count: Counter = Counter()
    shown = 0
    for t in sorted((Path(x.eval) / x.set / "truth").glob("*.json")):
        truth = json.loads(t.read_text())
        fa, fb = Path(x.a) / x.set / f"{t.stem}.json", Path(x.b) / x.set / f"{t.stem}.json"
        if not fa.exists() or not fb.exists():
            continue
        wa, wb = json.loads(fa.read_text())["words"], json.loads(fb.read_text())["words"]
        have = {(w["text"], *(round(v) for v in w["box"])) for w in wa}
        for w in wb:
            if (w["text"], *(round(v) for v in w["box"])) in have:
                continue
            c = amounts._centre(w)
            where = "no tag"
            for g in truth["tags"]:
                if amounts._inside(c, amounts.value_region(truth["words"], g["words"])):
                    where = g["cls"]
                    break
            count[where] += 1
            if where not in ("no tag", "qty") and shown < x.show:
                shown += 1
                print(t.stem, w["text"], round(w["conf"], 2), [round(v) for v in w["box"]], where, g["value"])
    print(dict(count))


if __name__ == "__main__":
    main()
