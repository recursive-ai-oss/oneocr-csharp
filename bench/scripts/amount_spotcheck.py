"""Print a random sample of tagged values on which the in-place (box) amount rule and the text-only rule disagree, for a manual check.

    python scripts/amount_spotcheck.py --eval ROOT --runs RUNS [--n 20] [--seed 145] [--engines a b ...]

For each item: the engine, set and page, the truth value and class, both verdicts, and the hypothesis words nearest to the truth word
(text, and the offset of their centre from the truth word's centre in units of the truth word's height). The reviewer decides whether
the in-place verdict is the right one: the value is exactly read where it is printed, or it is not. When fewer than N items
disagree, agreeing items fill the sample.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oneocr_bench import amounts  # noqa: E402
from oneocr_bench.amounts import CLASSES, _centre  # noqa: E402
from oneocr_bench.finreport import discover  # noqa: E402


def diagnose(a) -> None:
    """Per engine, for single-word values: share read in place, share whose exact text is on the same row but outside the box rule's reach
    (what a granularity penalty would look like), share whose text is only elsewhere on the page, share absent. The box rule is
    penalising granularity when the second column is large for some engines and not for others."""
    from oneocr_bench.amounts import _inside, _key, _region

    ev = Path(a.eval).expanduser()
    print("| engine | values | in place % | same row, outside reach % | elsewhere on the page % | absent % |")
    print("|---|---|---|---|---|---|")
    for name, dirs in sorted(discover(Path(a.runs).expanduser(), "ours").items()):
        if a.engines and name not in a.engines:
            continue
        c = {"in": 0, "row": 0, "else": 0, "none": 0}
        for s, hyp in sorted(dirs.items()):
            if a.sets and s not in a.sets:
                continue
            for t in sorted((ev / s / "truth").glob("*.json")):
                side = hyp / f"{t.stem}.json"
                if not side.exists():
                    continue
                truth = json.loads(t.read_text())
                hw = json.loads(side.read_text())["words"]
                keys = [_key(w["text"]) for w in hw]
                cs = [_centre(w) for w in hw]
                for tag in truth.get("tags", []):
                    if tag["cls"] not in CLASSES or len(tag["words"]) != 1:
                        continue
                    tw = truth["words"][tag["words"][0]]
                    k, reg, tc = _key(tw["text"]), _region(tw), _centre(tw)
                    h = max(1.0, tw["box"][3] - tw["box"][1])
                    same = [j for j, x in enumerate(keys) if x == k]
                    if any(_inside(cs[j], reg) for j in same):
                        c["in"] += 1
                    elif any(abs(cs[j][1] - tc[1]) <= 0.6 * h for j in same):
                        c["row"] += 1
                    elif same:
                        c["else"] += 1
                    else:
                        c["none"] += 1
        n = sum(c.values())
        if n:
            print(f"| {name} | {n} | " + " | ".join(f"{100 * c[x] / n:.1f}" for x in ("in", "row", "else", "none")) + " |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=145)
    ap.add_argument("--engines", nargs="*")
    ap.add_argument("--sets", nargs="*", help="limit to these sets")
    ap.add_argument("--diagnose", action="store_true", help="instead of a sample, print per engine where the right text sits relative to each value")
    a = ap.parse_args()
    ev = Path(a.eval).expanduser()
    if a.diagnose:
        return diagnose(a)
    configs = discover(Path(a.runs).expanduser(), "ours")
    items = []
    for name, dirs in sorted(configs.items()):
        if a.engines and name not in a.engines:
            continue
        for s, hyp in sorted(dirs.items()):
            if a.sets and s not in a.sets:
                continue
            for t in sorted((ev / s / "truth").glob("*.json")):
                side = hyp / f"{t.stem}.json"
                if not side.exists() or not (hyp / f"{t.stem}.txt").exists():
                    continue
                truth = json.loads(t.read_text())
                hw = json.loads(side.read_text())["words"]
                box = amounts.score_page(truth, hw)
                txt = amounts.score_page_text(truth, (hyp / f"{t.stem}.txt").read_text())
                if not box:
                    continue
                for b, x in zip(box["decisions"], txt["decisions"]):
                    if b["cls"] in CLASSES:
                        items.append((name, s, t.stem, b, x["read"], truth, hw))
    dis = [i for i in items if i[3]["read"] != i[4]]
    agree = [i for i in items if i[3]["read"] == i[4]]
    print(f"{len(items)} values, {len(dis)} where the two rules disagree")
    rng = random.Random(a.seed)
    pick = rng.sample(dis, min(a.n, len(dis)))
    pick += rng.sample(agree, max(0, a.n - len(pick)))
    for k, (name, s, pid, b, txt_read, truth, hw) in enumerate(pick, 1):
        tw = truth["words"][b["words"][0]]
        tc = _centre(tw)
        h = max(1.0, tw["box"][3] - tw["box"][1])
        near = sorted(hw, key=lambda w: (_centre(w)[0] - tc[0]) ** 2 + (_centre(w)[1] - tc[1]) ** 2)[:4]
        print(f"\n[{k}] {name} {s}/{pid} {b['cls']} truth {' '.join(truth['words'][i]['text'] for i in b['words'])!r}: in-place {b['read']}, text-only {txt_read}")
        for w in near:
            c = _centre(w)
            print(f"      hyp {w['text']!r} dx {(c[0] - tc[0]) / h:+.1f} dy {(c[1] - tc[1]) / h:+.1f}" + "")


if __name__ == "__main__":
    main()
