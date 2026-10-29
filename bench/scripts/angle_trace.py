"""One row per page from a bench log made with `--angle force --angletrace 1`: the first reading's confident share and score, the
best angle's score, and the missed characters (recall-CER numerator) of the page without and with the search, so that a trigger for
the angle search can be chosen from what a forced search gains.

    uv run python scripts/angle_trace.py --truth SET/truth --log OUT/SET.log --base DIR --forced DIR [--scorer v3]
        [--shares 0.6,0.7,0.8,0.9]

Prints the rows, then for each trigger share the pages it would search, the missed characters it would keep (searching = the forced
output, not searching = the base output) and the number of pages that pay for a search.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import score_page, truth_text

MARGIN = 1.1


def parse(log: Path) -> dict[str, dict]:
    pages: dict[str, dict] = {}
    cur: dict = {}
    pend: dict = {}
    for line in log.read_text(errors="replace").splitlines():
        m = re.match(r"\s+angle cover ([\d.]+) ink ([\d.]+)", line)
        if m:
            pend = {"cover": float(m[1]), "ink": float(m[2])}
            continue
        m = re.match(r"\s+angle weak\? share ([\d.]+) chars (\d+) words (\d+)", line)
        if m:
            cur = {"share": float(m[1]), "chars": int(m[2]), "words": int(m[3]), "scores": {}, **pend}
            continue
        m = re.match(r"\s+angle first ([\d.]+)", line)
        if m and cur:
            cur["first"] = float(m[1])
            continue
        m = re.match(r"\s+angle (-?\d+) (-?[\d.]+)$", line)
        if m and cur:
            cur["scores"][int(m[1])] = float(m[2])
            continue
        m = re.match(r"(\w+): \d+ ms", line)
        if m and cur:
            pages.setdefault(m[1], cur)  # the log repeats each page's trace; keep the first
            cur = {}
    return pages


def misses(truth: Path, d: Path, stem: str, scorer: str) -> tuple[int, int]:
    h = d / f"{stem}.txt"
    s = score_page(truth_text(truth, stem, scorer), h.read_text(encoding="utf-8", errors="replace") if h.exists() else "")
    return s["char_misses"], s["ref_chars"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--forced", required=True)
    ap.add_argument("--scorer", default="v3")
    ap.add_argument("--covers", default="0.5,0.6,0.7,0.8,0.85,0.9,0.95")
    ap.add_argument("--shares", default="0.6,0.7,0.8,0.85,0.9,0.95,1.01")
    a = ap.parse_args()
    truth = Path(a.truth)
    pages = parse(Path(a.log))
    rows = []
    total = 0
    for stem, p in sorted(pages.items()):
        b, n = misses(truth, Path(a.base), stem, a.scorer)
        f, _ = misses(truth, Path(a.forced), stem, a.scorer)
        total += n
        best = max(p["scores"].values(), default=0.0)
        rows.append({"stem": stem, "n": n, "base": b, "forced": f, "gain_ratio": best / p["first"] if p.get("first") else 0.0, **p})
    print("page    share  cover   ink  chars first   best/first  base forced")
    for r in sorted(rows, key=lambda r: r["base"] - r["forced"], reverse=True)[:30]:
        print(f"{r['stem']:7s} {r['share']:.3f} {r.get('cover', -1):.3f} {r.get('ink', -1):.3f} {r['chars']:5d} {r.get('first', 0):7.1f} {r['gain_ratio']:7.2f}  {r['base']:5d} {r['forced']:5d}")
    print("\nworst (forced search loses):")
    for r in sorted(rows, key=lambda r: r["forced"] - r["base"], reverse=True)[:8]:
        print(f"{r['stem']:7s} {r['share']:.3f} {r.get('cover', -1):.3f} {r.get('ink', -1):.3f} {r['chars']:5d} {r.get('first', 0):7.1f} {r['gain_ratio']:7.2f}  {r['base']:5d} {r['forced']:5d}")
    allbase = sum(r["base"] for r in rows)
    allforced = sum(r["forced"] for r in rows)
    print(f"\nrecall-CER: no search {100 * allbase / total:.3f}  forced {100 * allforced / total:.3f}  ({len(rows)} pages)")
    for c in (float(x) for x in a.covers.split(",")):
        def searched(r, c=c):
            return r["share"] < 0.6 or r.get("cover", 1) < c
        sel = [r for r in rows if searched(r)]
        kept = sum(r["forced"] if searched(r) else r["base"] for r in rows)
        paid = sum(1 for r in sel if r["forced"] < r["base"])
        print(f"trigger share < 0.6 or cover < {c:.2f}: searches {len(sel):3d} pages ({100 * len(sel) / len(rows):.0f}%), recall-CER {100 * kept / total:.3f}, pages helped {paid}")
    for t in (float(x) for x in a.shares.split(",")):
        sel = [r for r in rows if r["share"] < t]
        kept = sum(r["forced"] if r["share"] < t else r["base"] for r in rows)
        paid = sum(1 for r in sel if r["forced"] < r["base"])
        print(f"trigger share < {t:.2f}: searches {len(sel):3d} pages ({100 * len(sel) / len(rows):.0f}%), recall-CER {100 * kept / total:.3f}, pages helped {paid}")


if __name__ == "__main__":
    main()
