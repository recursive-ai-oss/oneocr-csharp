# ruff: noqa: E501, B905
"""Break a page-level OCR error down by cause, from engine output and exact word truth.

Each truth word (ink quad) is compared with the recognized words' boxes:
  missed    no recognized box covers it (>= 30% of its area)
  matched   one recognized box with IoU >= 0.5 (error = word edit distance: recognition on a well-found word)
  merged    its box also covers a neighbouring truth word (fusion across a gap or a line)
  geometry  covered but not matched (box clipped, loose, or split in pieces)
Characters lost per class are summed, so the table says where the page CER comes from. Matched words are
further sliced by the truth word's tilt (relative to the page skew the engine removed) and by its height.

Inputs: <truth-dir> with <id>.json (words with ink_quad/quad and text) and <out-dir> with <id>.json (words with box, text).
Run: python eval/page_diag.py --eval BENCH_ROOT --out ENGINE_OUTPUT_ROOT
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


class Levenshtein:  # words are short, so the plain DP is fast enough and keeps this script dependency-free
    @staticmethod
    def distance(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    for a, b in (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "-")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def aabb(q):
    if len(q) == 4:
        return list(q)
    return [min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])]


def inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0)


def area(a):
    return max(a[2] - a[0], 0) * max(a[3] - a[1], 0)


def angle(q):
    if len(q) == 4:
        return 0.0
    return math.degrees(math.atan2(q[3] - q[1], q[2] - q[0]))


def diag(truth_dir: Path, out_dir: Path, key: str = "ink_quad"):
    cls = defaultdict(lambda: [0, 0, 0])  # words, truth chars, char errors
    fp_chars = 0
    total_chars = 0
    bins_tilt = defaultdict(lambda: [0, 0, 0])
    bins_h = defaultdict(lambda: [0, 0, 0])
    pages = {}
    allt = defaultdict(lambda: defaultdict(int))
    for tf in sorted(truth_dir.glob("*.json")):
        t = json.loads(tf.read_text())
        of = out_dir / tf.name
        hw = json.loads(of.read_text())["words"] if of.exists() else []
        skew = 0.0
        pf = out_dir / f"{tf.stem}.page.json"
        if pf.exists():
            skew = json.loads(pf.read_text())["pages"][0].get("skew", 0.0)
        tw = [w for w in t["words"] if norm(w["text"])]
        def geom(w):
            return w.get(key) or w.get("ink") or w.get("quad") or w["box"]

        tb = [aabb(geom(w)) for w in tw]
        hb = [[float(v) for v in w["box"]] for w in hw]
        hx = [norm(w["text"]) for w in hw]
        angs = [angle(geom(w)) for w in tw]
        # the engine removes a global skew; residual tilt is the truth angle relative to its page median
        med = sorted(angs)[len(angs) // 2] if angs else 0.0
        # residual tilt = truth angle minus what the engine removed (sign chosen per page: the one leaving less)
        res_a = [abs(a - skew) for a in angs]
        res_b = [abs(a + skew) for a in angs]
        sgn = 1.0 if sorted(res_a)[len(res_a) // 2] <= sorted(res_b)[len(res_b) // 2] else -1.0
        used_h = set()
        page_err = page_chars = 0
        for i, w in enumerate(tw):
            text = norm(w["text"])
            n = len(text)
            ov = [(j, inter(tb[i], hb[j])) for j in range(len(hb))]
            cover = sum(o for _, o in ov) / max(area(tb[i]), 1e-9)
            main = [j for j, o in ov if o / max(area(tb[i]), 1e-9) > 0.3]
            for j in main:
                used_h.add(j)
            h = tb[i][3] - tb[i][1]
            if cover < 0.3 or not main:
                c, err = "missed", n
            else:
                best = max(main, key=lambda j: inter(tb[i], hb[j]) / max(area(tb[i]) + area(hb[j]) - inter(tb[i], hb[j]), 1e-9))
                iou = inter(tb[i], hb[best]) / max(area(tb[i]) + area(hb[best]) - inter(tb[i], hb[best]), 1e-9)
                others = sum(1 for k in range(len(tb)) if k != i and inter(tb[k], hb[best]) > 0.5 * area(tb[k]))
                if others:
                    c = "merged"
                    err = Levenshtein.distance(text, hx[best])  # includes the neighbour's characters
                    err = min(err, n)  # count at most this word's own characters
                elif iou >= 0.5:
                    c = "matched"
                    err = Levenshtein.distance(text, hx[best])
                    err = min(err, n)
                    r = abs(angs[i] - med)
                    bt = "<0.5" if r < 0.5 else "<1" if r < 1 else "<2" if r < 2 else "<4" if r < 4 else ">=4"
                    bins_tilt[bt][0] += 1
                    bins_tilt[bt][1] += n
                    bins_tilt[bt][2] += err
                    bh = "<14" if h < 14 else "<20" if h < 20 else "<30" if h < 30 else "<45" if h < 45 else ">=45"
                    bins_h[bh][0] += 1
                    bins_h[bh][1] += n
                    bins_h[bh][2] += err
                else:
                    c = "geometry"
                    cat = " ".join(hx[j] for j in sorted(main, key=lambda j: hb[j][0]))
                    err = min(Levenshtein.distance(text, cat), n)
            r = abs(angs[i] - sgn * skew)
            bt2 = "<0.5" if r < 0.5 else "<1" if r < 1 else "<2" if r < 2 else "<4" if r < 4 else ">=4"
            allt[bt2][c] += err
            allt[bt2]["n"] += n
            allt[bt2]["w"] += 1
            cls[c][0] += 1
            cls[c][1] += n
            cls[c][2] += err
            page_err += err
            page_chars += n
        fp = sum(len(hx[j]) for j in range(len(hb)) if j not in used_h)
        fp_chars += fp
        total_chars += page_chars
        pages[tf.stem] = (page_err + fp) / max(page_chars, 1)
    return cls, fp_chars, total_chars, allt, bins_h, pages


def show(name, truth_root: Path, out_root: Path):
    cls, fp, tot, bt, bh, pages = diag(truth_root / "truth" if (truth_root / "truth").exists() else truth_root, out_root)
    err = sum(v[2] for v in cls.values()) + fp
    print(f"== {name}: truth chars {tot}, attributed errors {err} ({100 * err / tot:.2f}%)")
    for c in ("missed", "merged", "geometry", "matched"):
        w, n, e = cls[c]
        print(f"  {c:9s} words {w:5d} chars {n:6d} errors {e:6d} = {100 * e / tot:6.2f}% of truth chars ({100 * e / max(err, 1):5.1f}% of error)")
    print(f"  extra     recognized chars outside any truth word {fp} = {100 * fp / tot:.2f}%")
    print("  ALL words by residual tilt vs removed skew (deg): words, char-error rate, share by class")
    for k in ("<0.5", "<1", "<2", "<4", ">=4"):
        v = bt.get(k)
        if v:
            e = sum(v[c] for c in ("missed", "merged", "geometry", "matched"))
            print(f"    {k:5s} {v['w']:6d}w  CER {100 * e / max(v['n'], 1):6.2f}%  " + " ".join(f"{c} {100 * v[c] / max(e, 1):.0f}%" for c in ("missed", "merged", "geometry", "matched")))
    print("  matched words by ink height (px):", {k: f"{v[0]}w {100 * v[2] / max(v[1], 1):.2f}%" for k, v in sorted(bh.items())})
    worst = sorted(pages.items(), key=lambda kv: -kv[1])[:5]
    print("  worst pages:", ", ".join(f"{k} {100 * v:.1f}%" for k, v in worst))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="bench root with <set>/truth")
    ap.add_argument("--out", required=True, help="engine output root with <set>/")
    ap.add_argument("--sets", nargs="+", default=["clean300-v1", "degraded-v1", "screenshot-v1", "layout-v1", "phone-v1", "receipts-v1"])
    a = ap.parse_args()
    for s in a.sets:
        show(s, Path(a.eval) / s, Path(a.out) / s)
