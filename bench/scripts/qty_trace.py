"""Where the tagged quantity cells (mostly lone digits) of the financial sets are lost.

    uv run python scripts/qty_trace.py --eval EVAL_ROOT --set fin-scan [--set ...] --run RUN_ROOT [--off RUN_ROOT_NOISE_OFF] [--show 20]

For every `qty` tag (and, with --cls, the other classes) that the run does not read in place, says whether the same page read with the
noise filter off reads it (`lost to the noise filter`), whether no hypothesis word lies in its region at all (`undetected`), or the
reading is wrong (`misread`). Counts per set; --show lists the first pages and values of each cause.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench import amounts
from oneocr_bench.finreport import diagnose


def words(d: Path, stem: str) -> list[dict] | None:
    f = d / f"{stem}.json"
    return json.loads(f.read_text())["words"] if f.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--off")
    ap.add_argument("--cls", default="qty")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    total: Counter = Counter()
    for s in a.set:
        tdir = Path(a.eval) / s / "truth"
        run, off = Path(a.run) / s, (Path(a.off) / s if a.off else None)
        c: Counter = Counter()
        shown: dict = {}
        for t in sorted(tdir.glob("*.json")):
            truth = json.loads(t.read_text())
            hw = words(run, t.stem)
            if hw is None:
                continue
            page = amounts.score_page(truth, hw)
            if not page:
                continue
            ow = words(off, t.stem) if off else None
            opage = amounts.score_page(truth, ow) if ow is not None else None
            for i, d in enumerate(page["decisions"]):
                if d["cls"] not in a.cls.split(","):
                    continue
                c["values"] += 1
                if d["read"]:
                    c["read"] += 1
                    continue
                tag = next(x for x in truth["tags"] if x["words"] == d["words"])
                if opage and opage["decisions"][i]["read"]:
                    cause = "lost to the noise filter"
                else:
                    cause = diagnose(truth, ow if ow is not None else hw, tag)[0]
                    if cause == "undetected" and ow is not None:
                        cause = "undetected (filter off too)"
                c[cause] += 1
                if len(shown.setdefault(cause, [])) < a.show:
                    tw = truth["words"]
                    shown[cause].append((t.stem, "".join(tw[j]["text"] for j in d["words"]), [round(v) for v in tw[d["words"][0]].get("box", [])]))
        print(s, dict(c))
        for k, v in shown.items():
            print("   ", k, v)
        total.update(c)
    print("TOTAL", dict(total))


if __name__ == "__main__":
    main()
