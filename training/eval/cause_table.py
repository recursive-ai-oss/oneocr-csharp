# ruff: noqa: E501
"""Compact side-by-side cause tables from `page_causes.py --json` outputs of two engines (ours and a reference).

For each set: CER/WER percentage points per cause for both engines, then the truth words the reference reads exactly and we do
not, grouped by our cause and (for misreads) by error kind, with examples. "Gain" columns are the characters/words that would
be saved if those words were read as the reference reads them, scaled like page_causes.py to the official metric.

    python eval/cause_table.py OURS.json REF.json [--sets ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from page_causes import kind, ops  # noqa: E402

CLASSES = ("missed", "merged-gap", "merged-line", "merged-lines", "split-space", "split-other", "box", "matched")


def scales(r: dict) -> tuple[float, float]:
    tc, tw, o = r["truth_chars"], r["truth_words"], r["official"]
    att = max(r["attributed_errors"], 1)
    wrong = sum(v["wrong_words"] for v in r["classes"].values())
    if r["set"].startswith("receipts"):
        return o["recall_cer"] / (max(att - r["extra"]["chars"], 1) / tc), o["recall_wer"] * tw / max(wrong, 1)
    return o["cer"] / (att / tc), o["wer"] * tw / max(wrong + r["extra"]["words"], 1)


def pp(r: dict) -> dict[str, tuple[float, float]]:
    tc, tw = r["truth_chars"], r["truth_words"]
    cs, ws = scales(r)
    out = {}
    for c in CLASSES:
        v = r["classes"].get(c)
        if v:
            out[c] = (100 * v["errors"] / tc * cs, 100 * v["wrong_words"] / tw * ws)
    e = r["extra"]
    out["extra"] = (100 * e["chars"] / tc * cs, 100 * e["words"] / tw * ws)
    if r["set"].startswith("receipts"):
        out["extra"] = (0.0, 0.0)  # recall metrics: extras are free
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("ref")
    ap.add_argument("--sets", nargs="*")
    a = ap.parse_args()
    ours, ref = json.loads(Path(a.ours).read_text()), json.loads(Path(a.ref).read_text())
    for name in a.sets or ours:
        o, r = ours[name], ref[name]
        po, pr = pp(o), pp(r)
        oo, ro = o["official"], r["official"]
        rec = name.startswith("receipts")
        m = ("recall_cer", "recall_wer") if rec else ("cer", "wer")
        print(f"\n### {name}: ours CER/WER {100 * oo[m[0]]:.2f}/{100 * oo[m[1]]:.2f} order {100 * oo['order']:.2f} (order-only {100 * o['order_only']:.2f}) | ref {100 * ro[m[0]]:.2f}/{100 * ro[m[1]]:.2f} order {100 * ro['order']:.2f} (order-only {100 * r['order_only']:.2f})")
        print("| cause | ours CER-pp | ours WER-pp | ref CER-pp | ref WER-pp |\n|---|---|---|---|---|")
        for c in (*CLASSES, "extra"):
            if c in po or c in pr:
                x, y = po.get(c, (0, 0)), pr.get(c, (0, 0))
                print(f"| {c} | {x[0]:.3f} | {x[1]:.3f} | {y[0]:.3f} | {y[1]:.3f} |")
        k = o["kinds"]
        kt = max(sum(k.values()), 1)
        print("ours misread kinds:", ", ".join(f"{a_} {100 * b / kt:.0f}%" for a_, b in sorted(k.items(), key=lambda x: -x[1])), "| ref:", ", ".join(f"{a_} {100 * b / max(sum(r['kinds'].values()), 1):.0f}%" for a_, b in sorted(r["kinds"].items(), key=lambda x: -x[1])))
        # truth words the reference reads exactly and we do not
        rw = {(p, i): (cl, err) for p, i, _t, _h, cl, err, _hh in r["all_words"]}
        cs, ws = scales(o)
        tc, tw = o["truth_chars"], o["truth_words"]
        agg = defaultdict(lambda: [0, 0])
        kinds = defaultdict(Counter)
        ex = defaultdict(list)
        ours_ok_ref_bad = [0, 0]
        both = [0, 0]
        for p, i, t, h, cl, err, hh in o["all_words"]:
            rcl, rerr = rw.get((p, i), ("missed", len(t)))
            if err > 0 and rerr == 0:
                agg[cl][0] += 1
                agg[cl][1] += err
                if cl in ("matched", "box"):
                    for op in ops(t, h)[: len(t)]:
                        kinds[cl][kind(op)] += 1
                if len(ex[cl]) < 6:
                    ex[cl].append(f"{t!r}->{h!r} (h{hh})")
            elif err == 0 and rerr > 0:
                ours_ok_ref_bad[0] += 1
                ours_ok_ref_bad[1] += rerr
            elif err > 0 and rerr > 0:
                both[0] += 1
                both[1] += err
        print("truth words ref reads right but we do not (by our cause: words, char errors -> CER-pp / WER-pp if fixed):")
        for cl, (n, e) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
            kd = ", ".join(f"{a_} {b}" for a_, b in kinds[cl].most_common(5))
            print(f"  - {cl}: {n}w, {e}c -> {100 * e / tc * cs:.3f} / {100 * n / tw * ws:.3f}" + (f" [{kd}]" if kd else "") + "; e.g. " + "; ".join(ex[cl][:4]))
        print(f"  words we read right, ref not: {ours_ok_ref_bad[0]}w/{ours_ok_ref_bad[1]}c; both wrong: {both[0]}w/{both[1]}c")
        ex_o, ex_r = o["extra"], r["extra"]
        print(f"extras ours {ex_o['words']}w/{ex_o['chars']}c vs ref {ex_r['words']}w/{ex_r['chars']}c")


if __name__ == "__main__":
    main()
