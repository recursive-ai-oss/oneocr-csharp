"""Crop geometry against truth line boxes, and where edge-deletion misreads sit in it.

    uv run python scripts/crop_geometry.py --truth SET/truth --geom DIR [--hyp DIR] [--scorer v3.1]

DIR holds <id>.json per page from `OneOcr.Bench cropgeom` (the strips the page reader reads and the crop each gets). Truth words are
grouped into printed lines with the grouping `truthcrop` uses; each truth line is matched to the strip that overlaps it most. Per line
(px and as a fraction of the truth line's ink height H):

  strip height, strip top/bottom offset (strip edge minus truth edge, so positive at the top = the strip starts below the truth ink),
  crop top/bottom/left/right margin (truth ink edge to crop edge, positive = room), left/right ink clip (positive = strip ends inside).

With --hyp (a run's <id>.txt directory) every edge-deletion misread of the scorer's own pairing is found on its page and the same
numbers are printed for its line, so the distribution of ordinary lines and that of the lines with deletions can be compared.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from error_table import analyse  # noqa: E402
from oneocr_bench.score import normalize  # noqa: E402


def truth_words(path: Path):
    d = json.loads(path.read_text())
    return [(w["text"], w.get("ink") or w["box"]) for w in d["words"]]


def group(words):
    order = sorted(range(len(words)), key=lambda i: (words[i][1][1] + words[i][1][3]) / 2)
    lines: list[list[int]] = []

    def bounds(l):
        b = [words[i][1] for i in l]
        return [min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)]

    for i in order:
        b = words[i][1]
        home = None
        for l in lines:
            lb = bounds(l)
            h = min(b[3] - b[1], lb[3] - lb[1])
            if min(b[3], lb[3]) - max(b[1], lb[1]) < 0.5 * h:
                continue
            gap = max(b[0] - lb[2], lb[0] - b[2])
            nearest = min(max(b[0] - words[j][1][2], words[j][1][0] - b[2]) for j in l)
            if nearest <= 2.5 * (b[3] - b[1]) or gap < 0:
                home = l
                break
        if home is None:
            lines.append([i])
        else:
            home.append(i)
    for l in lines:
        l.sort(key=lambda j: words[j][1][0])
    return lines, bounds


def line_metrics(tb, strip):
    """tb = truth line ink box [x0,y0,x1,y1]; strip = dict(ink=[x,y,w,h], crop=[x,y,w,h])."""
    H = tb[3] - tb[1]
    ix, iy, iw, ih = strip["ink"]
    cx, cy, cw, ch = strip["crop"]
    m = {
        "H": H,
        "strip_h": ih,
        "strip_top": iy - tb[1],
        "strip_bot": tb[3] - (iy + ih),
        "clip_l": ix - tb[0],
        "clip_r": tb[2] - (ix + iw),
        "crop_top": tb[1] - cy,
        "crop_bot": (cy + ch) - tb[3],
        "crop_l": tb[0] - cx,
        "crop_r": (cx + cw) - tb[2],
        "crop_h": ch,
    }
    return m


def match(tb, strips):
    best, bs = None, 0.0
    for s in strips:
        ix, iy, iw, ih = s["ink"]
        v = min(tb[3], iy + ih) - max(tb[1], iy)
        h = min(tb[2], ix + iw) - max(tb[0], ix)
        if v <= 0 or h <= 0:
            continue
        score = v * h
        if score > bs and v >= 0.4 * min(tb[3] - tb[1], ih):
            best, bs = s, score
    return best


def det_metrics(tb, dets):
    """The detected line box that overlaps the truth line most: truth ink edge to detector edge (positive = the box reaches past the ink)."""
    best, bs = None, 0.0
    for d in dets:
        v = min(tb[3], d[3]) - max(tb[1], d[1])
        h = min(tb[2], d[2]) - max(tb[0], d[0])
        if v > 0 and h > 0 and v * h > bs and v >= 0.4 * min(tb[3] - tb[1], d[3] - d[1]):
            best, bs = d, v * h
    if best is None:
        return None
    return {"H": tb[3] - tb[1], "det_l": tb[0] - best[0], "det_r": best[2] - tb[2], "det_t": tb[1] - best[1], "det_b": best[3] - tb[3]}


def summarize_det(name, rows):
    print(f"\n{name}  ({len(rows)} lines)  detector box edge minus truth ink edge, positive = box reaches past ink")
    for k in ["det_l", "det_r", "det_t", "det_b"]:
        v = sorted(r[k] for r in rows)
        f = sorted(r[k] / r["H"] for r in rows)
        q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]  # noqa: E731
        short = sum(1 for r in rows if r[k] < -0.5)
        print(f"  {k:6s} px p10 {q(v, .1):6.1f} p50 {q(v, .5):5.1f} p90 {q(v, .9):5.1f} | /H p10 {q(f, .1):6.3f} p50 {q(f, .5):6.3f} p90 {q(f, .9):6.3f} | ink outside box: {short} ({100 * short / len(rows):.1f}%)")


def summarize(name, rows):
    if not rows:
        print(f"{name}: no lines")
        return
    print(f"\n{name}  ({len(rows)} lines)")
    print("  metric        px: p10   p50   p90 |  /H: p10    p50    p90")
    for k in ["strip_h", "strip_top", "strip_bot", "clip_l", "clip_r", "crop_top", "crop_bot", "crop_l", "crop_r", "crop_h"]:
        v = sorted(r[k] for r in rows)
        f = sorted(r[k] / r["H"] for r in rows)

        def q(a, p):
            return a[min(len(a) - 1, int(p * len(a)))]

        print(f"  {k:10s} {q(v, .1):7.1f} {q(v, .5):5.1f} {q(v, .9):5.1f} | {q(f, .1):6.3f} {q(f, .5):6.3f} {q(f, .9):6.3f}")
    clipped = sum(1 for r in rows if r["clip_l"] > 0.5 or r["clip_r"] > 0.5)
    short = sum(1 for r in rows if r["strip_h"] < 0.9 * r["H"])
    print(f"  lines with an end clipped by >0.5 px: {clipped} ({100 * clipped / len(rows):.1f}%); strip shorter than 0.9 H: {short} ({100 * short / len(rows):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--geom", required=True)
    ap.add_argument("--hyp")
    ap.add_argument("--scorer", default="v3.1")
    a = ap.parse_args()
    truth, geom = Path(a.truth), Path(a.geom)
    all_rows, edge_rows, det_rows = [], [], []
    perpage = {}
    unmatched = total = 0
    for g in sorted(geom.glob("*.json")):
        t = truth / g.name
        if not t.exists():
            continue
        words = truth_words(t)
        lines, bounds = group(words)
        gj = json.loads(g.read_text())
        strips = [s for s in gj["strips"] if not s["tilted"]]
        for l in lines:
            dm = det_metrics(bounds(l), gj.get("detected", []))
            if dm is not None:
                det_rows.append(dm)
        info = []
        for l in lines:
            tb = bounds(l)
            s = match(tb, strips)
            total += 1
            if s is None:
                unmatched += 1
                info.append((l, tb, None))
                continue
            m = line_metrics(tb, s)
            all_rows.append(m)
            info.append((l, tb, m))
        perpage[g.stem] = (words, info)
    print(f"truth lines {total}, without a strip {unmatched}")
    summarize_det("detector boxes vs truth ink (lines whose box matches)", det_rows)
    summarize("all matched lines", all_rows)
    # A strip that spans several truth lines of a row (tab stops, table cells) says nothing about its ends: keep lines whose strip
    # and truth extents agree to within one line height.
    same = [r for r in all_rows if abs(r["clip_l"]) <= r["H"] and abs(r["clip_r"]) <= r["H"]]
    summarize("lines whose strip and truth extents agree within 1 H", same)
    summarize("  of those, strip shorter than 0.9 H", [r for r in same if r["strip_h"] < 0.9 * r["H"]])
    summarize("  of those, ink clipped >0.5 px at an end", [r for r in same if r["clip_l"] > 0.5 or r["clip_r"] > 0.5])
    over = [r for r in same if r["clip_l"] < -2 or r["clip_r"] < -2]
    cut = [r for r in same if r["clip_l"] > 0.25 * r["H"] or r["clip_r"] > 0.25 * r["H"]]
    print(f"\n  strips reaching more than 2 px past the truth ink: {len(over)}; ends cut by more than 0.25 H: {len(cut)}  (of {len(same)} lines with agreeing extents)")
    all_rows = same
    if a.hyp:
        rows, _, _ = analyse(truth, Path(a.hyp), a.scorer)
        def norm(s):
            w = normalize(s, a.scorer == "v3.1")
            return " ".join(w)

        edge = [r for r in rows if r[1] == "misread:deletion at word edge (letter)"]
        print(f"\nedge-deletion misreads: {len(edge)}")
        lost = 0
        for stem, cause, tw, hw, ch, wd in edge:
            if stem not in perpage:
                lost += 1
                continue
            words, info = perpage[stem]
            for l, tb, m in info:
                if m is None:
                    continue
                cands = [j for j in l if norm(words[j][0]) == tw]
                if not cands:
                    continue
                j = cands[0]
                side = "first" if j == l[0] else "last" if j == l[-1] else "inner"
                print(f"  {stem} {tw!r} -> {hw!r}: {side} word of its line; truth line ink {[round(x) for x in tb]} word ink {[round(x) for x in words[j][1]]}; " + ", ".join(f"{k} {v:.1f}" for k, v in m.items()))
                edge_rows.append({**m, "side": side, "which": "left" if hw and tw.endswith(hw) else "right"})
                break
            else:
                lost += 1
        print(f"  not located on a matched line: {lost}")
        summarize("lines holding an edge-deletion misread", edge_rows)
        for side in ("first", "last", "inner"):
            n = sum(1 for r in edge_rows if r["side"] == side)
            print(f"  word is the {side} word of its line: {n}")
        n = sum(1 for r in edge_rows if r["which"] == "left")
        print(f"  first letter lost: {n}, last letter lost: {len(edge_rows) - n}")


if __name__ == "__main__":
    main()
