"""Amount exact match (scorer v3.2, in place) of one run on one financial set, overall and per class.

    uv run python -m eval.fin_amount_rate --eval SETDIR --run RUNDIR [--bench PATH]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bench", default=str(Path(__file__).resolve().parents[2] / "bench"))
    a = ap.parse_args()
    sys.path.insert(0, a.bench)
    from oneocr_bench import amounts

    n: Counter = Counter()
    ok: Counter = Counter()
    for tf in sorted((Path(a.eval) / "truth").glob("*.json")):
        t = json.loads(tf.read_text())
        side = Path(a.run) / (tf.stem + ".json")
        if not side.exists() or not t.get("tags"):
            continue
        hyp = json.loads(side.read_text())["words"]
        page = amounts._page(t["tags"], amounts.read_values(t["words"], hyp, t["tags"]))
        for c, v in page["values"].items():
            n[c] += v
            ok[c] += page["read"][c]
    money = [c for c in amounts.CLASSES]
    tot = sum(n[c] for c in money)
    got = sum(ok[c] for c in money)
    print(f"amount exact match {100 * got / max(tot, 1):.2f}% ({got}/{tot})  " + "  ".join(f"{c} {ok[c]}/{n[c]}" for c in sorted(n)))


if __name__ == "__main__":
    main()
