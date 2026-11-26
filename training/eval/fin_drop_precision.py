"""How many words a page-stage filter drops are real: the words the faint-text and noise filters removed, against the truth words.

Reads a run made with `OneOcr.Bench run --words 1 --stages 1`. A dropped word is real when a truth word with the same text (as the scorer
normalises it) has its centre inside the dropped word's box, or the other way round; it is junk otherwise.

    uv run python -m eval.fin_drop_precision --eval DIR --run DIR [--bench PATH] [--rows out.jsonl]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fin_empty_stage import _centre, _final_reading  # noqa: E402


def _in(c, b):
    return b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bench", default=str(Path(__file__).resolve().parents[2] / "bench"))
    ap.add_argument("--rows")
    a = ap.parse_args()
    sys.path.insert(0, a.bench)
    from oneocr_bench import amounts
    from oneocr_bench.score import _aabb

    rows = []
    for tf in sorted((Path(a.eval) / "truth").glob("*.json")):
        t = json.loads(tf.read_text())
        side = Path(a.run) / (tf.stem + ".json")
        stg = Path(a.run) / (tf.stem + ".stages.json")
        if not side.exists() or not stg.exists():
            continue
        hyp = json.loads(side.read_text())["words"]
        entries = _final_reading(json.loads(stg.read_text())["entries"], hyp)
        tw = [(_aabb(w, "box"), amounts._nospace(w["text"])) for w in t["words"]]
        money = set()
        for g in t.get("tags") or []:
            for i in g["words"]:
                money.add(i)
        for e in entries:
            if not (e["stage"] == "faint-drop" or e["stage"].startswith("junk-drop")):
                continue
            hb = e["box"]
            hc = _centre(hb)
            txt = amounts._nospace(e["text"])
            real = tagged = False
            ink = amount = False
            for i, (b, s) in enumerate(tw):
                if _in(hc, b) or _in(_centre(b), hb):
                    ink = True
                    amount |= i in money
                    if s == txt:
                        real = True
                        tagged = i in money
                        break
            rows.append({"page": tf.stem, "stage": e["stage"], "text": e["text"], "conf": e["conf"], "h": hb[3] - hb[1], "w": hb[2] - hb[0],
                         "real": real, "tagged": tagged, "ink": ink, "amount_ink": amount, "box": hb})
    c = Counter((r["stage"], r["real"], r["ink"]) for r in rows)
    print("(stage, text equals a truth word, lies on a truth word's ink): count")
    for k in sorted(c):
        print(k, c[k])
    print("on the ink of an amount, date or account word:", Counter(r["stage"] for r in rows if r["amount_ink"]))
    print("... and the text equals the truth word:", Counter(r["stage"] for r in rows if r["tagged"]))
    print("real and amount-tagged:", Counter(r["stage"] for r in rows if r["tagged"]))
    if a.rows:
        Path(a.rows).write_text("\n".join(json.dumps(r) for r in rows))


if __name__ == "__main__":
    main()
