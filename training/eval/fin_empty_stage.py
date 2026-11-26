"""Which stage loses a tagged value that comes back with nothing read in its region.

Needs a run made with `OneOcr.Bench run --words 1 --stages 1`: next to each `<id>.json` word sidecar there is `<id>.stages.json`, what
the page reader saw (lines it found, words it read) and dropped (faint-text filter, noise filter), all in input-image coordinates.

For every tagged value whose region holds no word of the final output (the scorer's rule, `oneocr_bench.amounts`), the value is
put in the first of these causes that applies:

  no line         no line the reader found overlaps the value's region (detector or line finding)
  line, no read   a line overlaps, but the recognizer wrote nothing that is a word in the region
  faint-drop      a word was read in the region and the faint-text recovery filter removed it
  noise-drop      a word was read in the region and the noise filter removed it
  other           words were read in the region and are not in the output (duplicate removal, a second reading chosen by the angle search)

    uv run python -m eval.fin_empty_stage --eval DIR --run DIR [--bench PATH] [--out report.json] [--list N]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


HEIGHTS = [(8, "<8 px"), (10, "8-10 px"), (12, "10-12 px"), (16, "12-16 px"), (1e9, ">=16 px")]


def _inside(c, region):
    xs = [p[0] for p in region]
    ys = [p[1] for p in region]
    return min(xs) <= c[0] <= max(xs) and min(ys) <= c[1] <= max(ys)


def _overlap(b, region):
    xs = [p[0] for p in region]
    ys = [p[1] for p in region]
    return min(b[2], max(xs)) > max(b[0], min(xs)) and min(b[3], max(ys)) > max(b[1], min(ys))


def _centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def classify(stages: list[dict], region) -> str:
    """The cause for one empty value; `stages` are the entries of the page's final reading."""
    if not any(e["stage"].startswith("strip") and _overlap(e["box"], region) for e in stages):
        return "no line"
    near = [e for e in stages if (e["stage"] in ("read", "faint-drop") or e["stage"].startswith("junk-drop")) and _inside(_centre(e["box"]), region)]
    if not near:
        return "line, no read"
    if any(e["stage"] == "faint-drop" for e in near):
        return "faint-drop"
    rules = sorted({e["stage"].split(":")[1] for e in near if e["stage"].startswith("junk-drop")})
    if rules:
        return "noise-drop:" + "+".join(rules)
    return "other"


def _final_reading(entries: list[dict], hyp: list[dict]) -> list[dict]:
    """The entries of the reading the page's output came from: the angle search reads a page several times and keeps one."""
    by: dict[int, list[dict]] = {}
    for e in entries:
        by.setdefault(e["reading"], []).append(e)
    best, best_n = None, -1
    for r in sorted(by):
        reads = [(e["text"], _centre(e["box"])) for e in by[r] if e["stage"] == "read"]
        n = 0
        for h in hyp:
            hc = _centre(h["box"])
            if any(t == h["text"] and abs(c[0] - hc[0]) < 4 and abs(c[1] - hc[1]) < 4 for t, c in reads):
                n += 1
        if n > best_n:
            best, best_n = r, n
    return by.get(best, [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="set directory with truth/")
    ap.add_argument("--run", required=True, help="run directory of that set")
    ap.add_argument("--bench", default=str(Path(__file__).resolve().parents[2] / "bench"))
    ap.add_argument("--qty", action="store_true", help="count the quantity cells instead of the money, date and account values")
    ap.add_argument("--out")
    ap.add_argument("--list", type=int, default=0, help="print this many example values per cause")
    a = ap.parse_args()
    sys.path.insert(0, a.bench)
    from oneocr_bench import amounts

    from oneocr_bench.score import _aabb

    causes: Counter = Counter()
    misread: Counter = Counter()
    by_height: dict = {label: [0, 0] for _, label in HEIGHTS}
    kinds: Counter = Counter()
    examples: dict[str, list] = {}
    total = 0
    for tf in sorted((Path(a.eval) / "truth").glob("*.json")):
        t = json.loads(tf.read_text())
        side = Path(a.run) / (tf.stem + ".json")
        stg = Path(a.run) / (tf.stem + ".stages.json")
        if not side.exists() or not stg.exists() or not t.get("tags"):
            continue
        hyp = json.loads(side.read_text())["words"]
        trace = json.loads(stg.read_text())
        entries = trace["entries"]
        final = _final_reading(entries, hyp)
        uniq: dict = {}
        for g in t["tags"]:
            if (g["cls"] in amounts.CLASSES) == (not a.qty):
                uniq.setdefault(tuple(g["words"]), len(uniq))
        keys = list(uniq)
        truth_words = t["words"]
        got = amounts._read_unique(truth_words, hyp, [{"words": list(k)} for k in keys])
        for k, ok in zip(keys, got):
            region = amounts.value_region(truth_words, list(k))
            want = "".join(amounts._nospace(truth_words[i]["text"]) for i in k)
            total += 1
            height = max(_aabb(truth_words[i], "box")[3] - _aabb(truth_words[i], "box")[1] for i in k)
            hb = next(label for limit, label in HEIGHTS if height < limit)
            by_height[hb][0] += 1
            by_height[hb][1] += int(ok)
            own = [h for h in hyp if _inside(amounts._centre(h), region)]
            if own:
                if not ok:
                    drops = sorted({e["stage"] for e in final if (e["stage"] == "faint-drop" or e["stage"].startswith("junk-drop")) and _inside(_centre(e["box"]), region)})
                    misread["misread, a word was dropped in the region: " + "+".join(drops) if drops else "misread, nothing dropped"] += 1
                continue
            c = classify(final, region)
            causes[c] += 1
            examples.setdefault(c, []).append((tf.stem, want, [round(x) for x in (min(p[0] for p in region), min(p[1] for p in region), max(p[0] for p in region), max(p[1] for p in region))]))
    report = {"values": total, "empty": sum(causes.values()), "causes": dict(causes), "misread": dict(misread),
              "read_by_truth_word_height": {k: f"{v[1]}/{v[0]}" for k, v in by_height.items()}}
    print(json.dumps(report))
    if a.list:
        for c, ex in examples.items():
            print(c, ex[: a.list])
    if a.out:
        Path(a.out).write_text(json.dumps({**report, "examples": {c: ex[:50] for c, ex in examples.items()}}, indent=1))


if __name__ == "__main__":
    main()
