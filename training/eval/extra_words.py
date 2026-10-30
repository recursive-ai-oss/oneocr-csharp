"""Extra and missed words per set, for comparing two or more engine runs that differ in one stage (for example the detector).

    cd bench && uv run python ../training/eval/extra_words.py --truth-root ROOT --sets A B C --run base=DIR1 --run new=DIR2

DIR<k>/<set>/<id>.txt are the engines' page texts. A hypothesis word is "extra" when it has no truth word left after the scorer's
pairing (word_outcomes in oneocr_bench.score, scorer v3.1 text): inserted text, or printed text the truth does not cover, so on sets
with partial truth (receipts) only the change between runs means anything. "paired" is a word read wrongly. Reports totals per
run and per set; with `--pages` also the pages whose extra count differs between the first run and the others.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from oneocr_bench.score import normalize, truth_text, word_outcomes  # noqa: E402


def count(truth_dir: Path, hyp_dir: Path) -> dict:
    tot = {"hyp_words": 0, "extra": 0, "paired": 0, "exact": 0, "truth_words": 0}
    per_page = {}
    for t in sorted(truth_dir.glob("*.txt")):
        h = hyp_dir / t.name
        text = h.read_text(encoding="utf-8", errors="replace") if h.exists() else ""
        ref = normalize(truth_text(truth_dir, t.stem, "v3.1"), True)
        hyp = normalize(text, True)
        out = word_outcomes(ref, hyp)
        c = {k: out.count(k) for k in ("extra", "paired", "exact")}
        per_page[t.stem] = c["extra"]
        tot["hyp_words"] += len(hyp)
        tot["truth_words"] += len(ref)
        for k, v in c.items():
            tot[k] += v
    return {**tot, "per_page": per_page}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth-root", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--pages", action="store_true")
    a = ap.parse_args()
    runs = [r.split("=", 1) for r in a.run]
    print(f"{'set':26s}" + "".join(f"{n:>28s}" for n, _ in runs))
    print(f"{'':26s}" + "".join(f"{'hyp  extra  paired':>28s}" for _ in runs))
    for s in a.sets:
        res = []
        for _, d in runs:
            hd = Path(d) / s
            res.append(count(Path(a.truth_root) / s / "truth", hd) if hd.exists() else None)
        print(f"{s:26s}" + "".join(f"{(str(r['hyp_words']).rjust(8) + str(r['extra']).rjust(8) + str(r['paired']).rjust(8)) if r else 'missing':>28s}" for r in res))
        if a.pages and res[0]:
            for pid, e0 in res[0]["per_page"].items():
                for (n, _), r in zip(runs[1:], res[1:], strict=False):
                    if r and r["per_page"].get(pid, e0) != e0:
                        print(f"    {pid}: {runs[0][0]} {e0} -> {n} {r['per_page'][pid]}")


if __name__ == "__main__":
    main()
