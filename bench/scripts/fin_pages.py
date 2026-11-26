"""Amount misses per page and by capture condition for one run of one financial set.

    uv run python scripts/fin_pages.py --eval EVAL_ROOT/SET --run RUN_DIR [--top 8]

Prints, for a run (`<id>.txt` and `<id>.json` word boxes) scored against `EVAL_ROOT/SET/truth`:
  - the pages with the most missed values, with their capture, skew, dpi, table style and miss causes;
  - the miss rate by skew bucket and by dpi (scan captures), and the share of misses that are `merged` (several values or lines read as one run).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench import amounts  # noqa: E402
from oneocr_bench.amounts import CLASSES  # noqa: E402
from oneocr_bench.finreport import diagnose  # noqa: E402


def bucket(skew: float) -> str:
    a = abs(skew)
    return "<0.5" if a < 0.5 else "0.5-2" if a < 2 else "2-4" if a < 4 else "4-6" if a < 6 else ">=6"


def edge_shape(want: str, got: str) -> str:
    """How a missed value differs from the truth: an end character lost (`lead`/`trail`), only separators or signs differ (`punct`), or other."""
    if not got:
        return "nothing"
    if want.startswith(got) and len(got) < len(want):
        return "trail"
    if want.endswith(got) and len(got) < len(want):
        return "lead"
    if got.startswith(want) or got.endswith(want):
        return "extra"
    if "".join(c for c in want if c.isalnum()) == "".join(c for c in got if c.isalnum()):
        return "punct"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--page", help="list the missed values of this page: truth, cause, what the engine read in the region")
    ap.add_argument("--show", type=int, default=40)
    a = ap.parse_args()
    tdir, run = Path(a.eval) / "truth", Path(a.run)
    rows = []
    by_skew: dict = defaultdict(lambda: [0, 0])
    by_dpi: dict = defaultdict(lambda: [0, 0])
    by_style: dict = defaultdict(lambda: [0, 0])
    tot = Counter()
    shapes: Counter = Counter()
    shape_skew: Counter = Counter()
    for t in sorted(tdir.glob("*.json")):
        truth = json.loads(t.read_text())
        side = run / f"{t.stem}.json"
        hw = json.loads(side.read_text())["words"] if side.exists() else []
        page = amounts.score_page(truth, hw)
        if not page:
            continue
        causes: Counter = Counter()
        n = 0
        shown = 0
        for d in page["decisions"]:
            if d["cls"] not in CLASSES:
                continue
            n += 1
            tag = next(x for x in truth["tags"] if x["words"] == d["words"])
            deg = truth.get("degradation", {})
            sk, dpi = deg.get("skew_deg"), deg.get("dpi")
            keys = []
            if sk is not None:
                keys.append((by_skew, bucket(sk)))
            if dpi is not None:
                keys.append((by_dpi, str(dpi)))
            keys.append((by_style, f"{truth.get('doc')}/{truth.get('style')}"))
            for tab, k in keys:
                tab[k][0] += 1
                tab[k][1] += int(not d["read"])
            if not d["read"]:
                cause, want, got = diagnose(truth, hw, tag)
                causes[cause] += 1
                shape = edge_shape(want, got)
                shapes[shape] += 1
                sk0 = truth.get("degradation", {}).get("skew_deg")
                if sk0 is not None:
                    shape_skew[(bucket(sk0), shape)] += 1
                if a.page == t.stem and shown < a.show:
                    shown += 1
                    tw = truth["words"]
                    reg = amounts.value_region(tw, tag["words"])
                    near = sorted((h for h in hw if amounts._inside(amounts._centre(h), reg)), key=lambda h: amounts._centre(h)[0])
                    ti = [tw[i].get("ink") or tw[i]["box"] for i in tag["words"]]
                    tx = (min(b[0] for b in ti), min(b[1] for b in ti), max(b[2] for b in ti), max(b[3] for b in ti))
                    hx = [(round(h["box"][0]), round(h["box"][2]), h["text"]) for h in near]
                    print(f"  MISS {cause}: want {want!r} got {got!r} truth-ink x {tx[0]:.0f}-{tx[2]:.0f} y {tx[1]:.0f}-{tx[3]:.0f}; hyp x {hx}")
        deg = truth.get("degradation", {})
        rows.append((sum(causes.values()), t.stem, n, truth.get("capture"), deg.get("skew_deg"), deg.get("dpi"), truth.get("doc"), truth.get("style"), dict(causes)))
        tot.update(causes)
    rows.sort(reverse=True)
    print("worst pages: misses id values capture skew dpi doc style causes")
    for r in rows[: a.top]:
        print(*r)
    for name, tab in (("skew", by_skew), ("dpi", by_dpi), ("doc/style", by_style)):
        print(f"by {name}: " + "; ".join(f"{k} {v[1]}/{v[0]} ({100 * v[1] / v[0]:.1f}%)" for k, v in sorted(tab.items())))
    print("shapes", dict(shapes))
    if shape_skew:
        print("shape by skew", "; ".join(f"{k[0]}:{k[1]}={v}" for k, v in sorted(shape_skew.items())))
    print("causes", dict(tot), "total misses", sum(tot.values()))


if __name__ == "__main__":
    main()
