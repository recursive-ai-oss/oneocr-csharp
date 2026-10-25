# ruff: noqa: E501, B905
"""Where a page-level OCR error comes from, per set: a finer cousin of page_diag.py.

Each truth word (ink box) is compared with the recognized words' boxes and put in one class; the characters it costs are
attributed to that class (the same attribution as page_diag.py), then recognition errors on well-found words are split by kind
(case, accents, digits, punctuation, letters) and sliced by height, background polarity, confidence and the page's font.
Extra recognized words (no truth word under them) are sliced by height, confidence and shape. Reading-order damage is measured
apart from reading errors: the truth indices of the matched words, in the order the engine returned them, against the truth order.

  missed          no recognized box covers 30% of the word
  merged-gap      ... and the fused text is exactly the neighbours joined without a space (a space not read)
  merged-line     one recognized box also covers a neighbour on the same line (fusion across a gap)
  merged-lines    ... a neighbour on another line (fusion across lines)
  split-space     two or more recognized boxes each cover 30% of the word and read together give the word (a space inserted)
  split-other     ... pieces that also differ from the word
  box             a single box, IoU < 0.5 (clipped or loose), text still read from it
  matched         one box with IoU >= 0.5: the edit distance is a recognition error

The attributed character errors approximate the official content-error pairing (bag of words, extras included); every share
is also given scaled to the official CER and WER of the same run, so "gain" columns are in the metric's own percentage points.

Run: python eval/page_causes.py --eval BENCH_ROOT --out ENGINE_OUTPUT_ROOT [--sets ...] [--json OUT.json]
ENGINE_OUTPUT_ROOT/<set>/ holds <id>.txt and <id>.json (words with box, text, conf) from `OneOcr.Bench run --words 1`.
Needs rapidfuzz and Pillow (the bench environment has both).
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from rapidfuzz.distance import Levenshtein as Lev

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from oneocr_bench.score import normalize, score_page  # noqa: E402

SETS = ["clean300-v1", "degraded-v1", "screenshot-v1", "layout-v1", "phone-v1", "receipts-v1"]
FOLD = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2212": "-"})


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).translate(FOLD).split())


def aabb(q):
    if len(q) == 4:
        return [float(v) for v in q]
    return [min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])]


def inter(a, b):
    return max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0)


def area(a):
    return max(a[2] - a[0], 0) * max(a[3] - a[1], 0)


def ops(a: str, b: str):
    """Edit operations turning truth a into hypothesis b: (kind, truth char, hyp char)."""
    out = []
    for tag, i1, i2, j1, j2 in Lev.opcodes(a, b):
        t, h = a[i1:i2], b[j1:j2]
        if tag == "equal":
            continue
        if tag == "replace":
            for k in range(max(len(t), len(h))):
                out.append(("sub" if k < len(t) and k < len(h) else "del" if k < len(t) else "ins", t[k] if k < len(t) else "", h[k] if k < len(h) else ""))
        elif tag == "delete":
            out.extend(("del", c, "") for c in t)
        else:
            out.extend(("ins", "", c) for c in h)
    return out


def strip_accents(c: str) -> str:
    return "".join(x for x in unicodedata.normalize("NFD", c) if not unicodedata.combining(x))


def kind(op) -> str:
    k, t, h = op
    ref = t or h
    if k == "sub" and t.lower() == h.lower():
        return "case"
    if k == "sub" and strip_accents(t) == strip_accents(h):
        return "accent"
    if ref.isdigit() or (k == "sub" and h.isdigit()):
        return "digit"
    if ref == " ":
        return "space"
    if not ref.isalnum():
        return "punct"
    return "letter"


def polarity(img, box) -> str:
    """'light-on-dark' when the ring around the box is darker than the ink-side median, else 'dark-on-light'."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    w, h = img.size
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return "dark-on-light"
    px = img.crop((max(x0 - 3, 0), max(y0 - 3, 0), min(x1 + 3, w), min(y1 + 3, h))).get_flattened_data()
    vals = sorted(px)
    median = vals[len(vals) // 2]
    return "light-on-dark" if median < 110 else "dark-on-light"


def hbin(h: float) -> str:
    return "<8" if h < 8 else "<12" if h < 12 else "<16" if h < 16 else "<24" if h < 24 else "<40" if h < 40 else ">=40"


def cbin(c: float) -> str:
    return "<0.5" if c < 0.5 else "<0.9" if c < 0.9 else "<0.99" if c < 0.99 else ">=0.99"


def shape(t: str) -> str:
    if len(t) == 1 and not t.isalnum():
        return "single punct"
    if len(t) == 1:
        return "single char"
    if all(not c.isalnum() for c in t):
        return "punct run"
    if any(c.isdigit() for c in t):
        return "has digit"
    return "word"


def lis_len(seq):
    tails = []
    for v in seq:
        i = bisect.bisect_left(tails, v)
        if i == len(tails):
            tails.append(v)
        else:
            tails[i] = v
    return len(tails)


def runs(seq, tw):
    """The truth blocks in the order the engine returned them, as runs [block, words]. A page read in order gives 0,1,2,..."""
    out = []
    for i in seq:
        b = tw[i].get("block", -1)
        if out and out[-1][0] == b:
            out[-1][1] += 1
        else:
            out.append([b, 1])
    return out[:30]


def analyse(eval_root: Path, out_root: Path, name: str) -> dict:
    truth_dir, pages_dir, hyp_dir = eval_root / name / "truth", eval_root / name / "pages", out_root / name
    cls = defaultdict(lambda: [0, 0, 0])  # words, truth chars, char errors
    cls_wrong_words = Counter()
    kinds = Counter()
    by_h, by_pol, by_conf, by_font, by_tpl, by_cat, by_shape = (defaultdict(lambda: [0, 0, 0]) for _ in range(7))
    miss = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # slice -> key -> words, chars (words with no recognized box under them)
    examples = defaultdict(list)
    extras = defaultdict(lambda: [0, 0])  # key -> words, chars
    confusions = Counter()
    breaks = Counter()
    wrong_words = []
    page_rows = {}
    tot_chars = tot_words = 0
    off_ce = off_we = off_oe = off_ref = off_cm = off_wm = 0
    order_only = 0
    extra_chars = extra_words = 0
    for tf in sorted(truth_dir.glob("*.json")):
        t = json.loads(tf.read_text())
        hyp_json = hyp_dir / tf.name
        hw = json.loads(hyp_json.read_text())["words"] if hyp_json.exists() else []
        hyp_txt = (hyp_dir / f"{tf.stem}.txt").read_text() if (hyp_dir / f"{tf.stem}.txt").exists() else ""
        sp = score_page(t["text"], hyp_txt)
        off_ce += sp["char_errors"]; off_we += sp["word_errors"]; off_oe += sp["order_errors"]; off_ref += sp["ref_words"]
        off_cm += sp["char_misses"]; off_wm += sp["word_misses"]
        tw = [w for w in t["words"] if norm(w["text"])]
        tb = [aabb(w.get("ink_quad") or w.get("ink") or w["box"]) for w in tw]
        hb = [[float(v) for v in w["box"]] for w in hw]
        hx = [norm(w["text"]) for w in hw]
        hc = [float(w.get("conf", 1.0)) for w in hw]
        img = None
        for ext in ("png", "jpg", "jpeg"):
            p = pages_dir / f"{tf.stem}.{ext}"
            if p.exists():
                img = Image.open(p).convert("L")
                break
        font = t.get("font", "-")
        tpl = t.get("template") or t.get("kind") or (t.get("degradation") and json.dumps(t["degradation"], sort_keys=True)[:40]) or "-"
        used = set()
        page_cls = Counter()
        page_chars = 0
        match_of_hyp = {}
        for i, w in enumerate(tw):
            text = norm(w["text"])
            n = len(text)
            a = area(tb[i])
            ov = [(j, inter(tb[i], hb[j])) for j in range(len(hb))]
            main = [j for j, o in ov if a > 0 and o / a > 0.3]
            used.update(main)
            h = tb[i][3] - tb[i][1]
            err = n
            c = "missed"
            sub_ops = None
            conf = None
            if main:
                best = max(main, key=lambda j: inter(tb[i], hb[j]) / max(a + area(hb[j]) - inter(tb[i], hb[j]), 1e-9))
                iou = inter(tb[i], hb[best]) / max(a + area(hb[best]) - inter(tb[i], hb[best]), 1e-9)
                nb = [k for k in range(len(tb)) if k != i and area(tb[k]) > 0 and inter(tb[k], hb[best]) > 0.5 * area(tb[k])]
                if nb:
                    same = any(abs((tb[k][1] + tb[k][3]) / 2 - (tb[i][1] + tb[i][3]) / 2) < 0.6 * max(h, 1) for k in nb)
                    c = "merged-line" if same else "merged-lines"
                    group = sorted([i] + nb, key=lambda k: tb[k][0])
                    if same and "".join(norm(tw[k]["text"]) for k in group) == hx[best]:
                        c = "merged-gap"  # the pieces are all there, only the space between them was not read
                    err = n  # the official pairing charges the whole word: it is unmatched and the fused word is extra
                elif len(main) >= 2:
                    pieces = [hx[j] for j in sorted(main, key=lambda j: hb[j][0])]
                    c = "split-space" if "".join(pieces) == text else "split-other"
                    if c == "split-space":
                        ch = pieces[0][-1]
                        breaks["after punct " + ch if not ch.isalnum() else "inside word (letters/digits)"] += 1
                    err = n  # a word cut in pieces is wrong as a word: all of its characters are charged
                elif iou >= 0.5:
                    c = "matched"
                    err = min(Lev.distance(text, hx[best]), n)
                    sub_ops = ops(text, hx[best])
                    conf = hc[best]
                else:
                    c = "box"
                    err = min(Lev.distance(text, hx[best]), n)
                    sub_ops = ops(text, hx[best])
                    conf = hc[best]
                for j in main:
                    match_of_hyp.setdefault(j, i)
            cls[c][0] += 1; cls[c][1] += n; cls[c][2] += err
            if c == "missed":
                pol0 = polarity(img, tb[i]) if img is not None else "?"
                for sl, key in (("height", hbin(h)), ("polarity", pol0), ("category", w.get("category", "-")), ("shape", shape(text))):
                    miss[sl][key][0] += 1; miss[sl][key][1] += n
            if err > 0 and c in ("matched", "box", "split-space", "split-other", "merged-gap") and len(examples[c]) < 400:
                examples[c].append([tf.stem, text, " ".join(hx[j] for j in sorted(main, key=lambda j: hb[j][0])), round(h, 1), font])
            if err > 0:
                cls_wrong_words[c] += 1
                if len(wrong_words) < 200000:
                    wrong_words.append((tf.stem, text, hx[best] if main else "", c))
            page_cls[c] += err
            page_chars += n
            if c in ("matched", "box") and err > 0 and sub_ops is not None:
                pol = polarity(img, tb[i]) if img is not None else "?"
                for op in sub_ops[:n]:
                    kinds[kind(op)] += 1
                    if op[0] != "ins" or True:
                        confusions[(op[1] or "∅", op[2] or "∅")] += 1
                for table, key in ((by_h, hbin(h)), (by_pol, pol), (by_conf, cbin(conf)), (by_font, font), (by_tpl, tpl), (by_shape, shape(text))):
                    table[key][0] += 1; table[key][1] += n; table[key][2] += err
            if c in ("matched", "box"):
                pol = polarity(img, tb[i]) if img is not None else "?"
                for table, key in ((by_h, hbin(h)), (by_pol, pol), (by_conf, cbin(conf)), (by_font, font), (by_tpl, tpl), (by_shape, shape(text))):
                    if err == 0:
                        table[key][0] += 1; table[key][1] += n
            if "category" in w:
                by_cat[w["category"]][0] += 1; by_cat[w["category"]][1] += n; by_cat[w["category"]][2] += err
        # extras: recognized boxes not covered by any truth word
        pe = 0
        for j in range(len(hb)):
            if j in used:
                continue
            cov = sum(inter(hb[j], b) for b in tb) / max(area(hb[j]), 1e-9)
            if cov > 0.3:
                continue
            hh = hb[j][3] - hb[j][1]
            for key in (f"h {hbin(hh)}", f"conf {cbin(hc[j])}", f"shape {shape(hx[j])}"):
                extras[key][0] += 1; extras[key][1] += len(hx[j])
            extra_chars += len(hx[j]); extra_words += 1; pe += len(hx[j])
        # order: truth indices in the order the engine returned its words
        seq = []
        for j in range(len(hb)):
            ti = match_of_hyp.get(j)
            if ti is not None and (not seq or seq[-1] != ti):
                seq.append(ti)
        seq = list(dict.fromkeys(seq))
        order_only += 2 * (len(seq) - lis_len(seq))
        tot_chars += page_chars; tot_words += len(tw)
        page_rows[tf.stem] = {
            "chars": page_chars, "attributed_cer": (sum(page_cls.values()) + pe) / max(page_chars, 1), "extra_chars": pe,
            "classes": dict(page_cls), "cer": sp["char_errors"] / max(sp["ref_chars"], 1), "wer": sp["word_errors"] / max(sp["ref_words"], 1),
            "order": sp["order_errors"] / max(sp["ref_words"], 1), "order_only": 2 * (len(seq) - lis_len(seq)) / max(len(tw), 1),
            "block_runs": runs(seq, tw), "meta": f"{tpl}/{font}", "words_found": len(hb), "truth_words": len(tw),
        }
    attributed = sum(v[2] for v in cls.values()) + extra_chars
    return {
        "set": name, "truth_chars": tot_chars, "truth_words": tot_words, "attributed_errors": attributed,
        "official": {"cer": off_ce / max(tot_chars, 1), "wer": off_we / max(off_ref, 1), "order": off_oe / max(off_ref, 1),
                     "recall_cer": off_cm / max(tot_chars, 1), "recall_wer": off_wm / max(off_ref, 1)},
        "classes": {k: {"words": v[0], "chars": v[1], "errors": v[2], "wrong_words": cls_wrong_words[k]} for k, v in cls.items()},
        "extra": {"chars": extra_chars, "words": extra_words, "by": {k: v for k, v in sorted(extras.items())}},
        "missed_slices": {a: dict(b) for a, b in miss.items()}, "examples": dict(examples), "kinds": dict(kinds), "breaks": dict(breaks.most_common(12)), "order_only": order_only / max(tot_words, 1),
        "slices": {n: {k: v for k, v in sorted(t.items())} for n, t in (("height", by_h), ("polarity", by_pol), ("conf", by_conf), ("font", by_font), ("template", by_tpl), ("category", by_cat), ("shape", by_shape))},
        "confusions": [[a, b, n] for (a, b), n in confusions.most_common(25)], "pages": page_rows,
    }


def show(r: dict) -> None:
    tc, tw, o = r["truth_chars"], r["truth_words"], r["official"]
    att = max(r["attributed_errors"], 1)
    partial = r["set"].startswith("receipts")  # truth covers part of the page: judged on recall, extras are free
    if partial:
        scale = o["recall_cer"] / (max(att - r["extra"]["chars"], 1) / tc)
        wscale = o["recall_wer"] * tw / max(sum(v["wrong_words"] for v in r["classes"].values()), 1)
    else:
        scale = o["cer"] / (att / tc) if att else 1.0
        wscale = o["wer"] * tw / max(sum(v["wrong_words"] for v in r["classes"].values()) + r["extra"]["words"], 1)
    print(f"\n===== {r['set']}: {tw} truth words, {tc} chars | official CER {100 * o['cer']:.2f} WER {100 * o['wer']:.2f} order {100 * o['order']:.2f}"
          f" recall-CER {100 * o['recall_cer']:.2f} recall-WER {100 * o['recall_wer']:.2f} | attributed {100 * r['attributed_errors'] / tc:.2f}% (scale to official CER x{scale:.2f})")
    print("  cause                   words  wrong   %truth-chars  CER-pp(scaled)  WER-pp(scaled)")
    for c in ("missed", "merged-gap", "merged-line", "merged-lines", "split-space", "split-other", "box", "matched"):
        v = r["classes"].get(c)
        if v:
            print(f"  {c:22s} {v['words']:6d} {v['wrong_words']:6d}   {100 * v['errors'] / tc:8.3f}     {100 * v['errors'] / tc * scale:8.3f}      {100 * v['wrong_words'] / tw * wscale:8.3f}")
    e = r["extra"]
    print(f"  {'extra words':22s} {e['words']:6d} {e['words']:6d}   {100 * e['chars'] / tc:8.3f}     {100 * e['chars'] / tc * scale:8.3f}      {100 * e['words'] / tw * wscale:8.3f}")
    print(f"  reading-order-only (approx. 2*(matched - LIS)/truth words): {100 * r['order_only']:.2f}% vs official order {100 * o['order']:.2f}% (includes reading errors)")
    k = r["kinds"]
    kt = max(sum(k.values()), 1)
    mt = r["classes"].get("matched", {"errors": 0})["errors"] + r["classes"].get("box", {"errors": 0})["errors"]
    print("  recognition error kinds (matched+box words; share of those character ops):", ", ".join(f"{a} {100 * b / kt:.0f}%" for a, b in sorted(k.items(), key=lambda x: -x[1])), f"  [recognition = {100 * mt / tc * scale:.3f} CER-pp]")
    for sl in ("height", "polarity", "conf", "shape", "font", "template", "category"):
        d = r["slices"][sl]
        if not d or (sl in ("font", "template", "category") and len(d) > 14 and sl != "category"):
            continue
        rows = []
        for key, (w, n, er) in d.items():
            rows.append((er / tc, f"{key}: {w}w {100 * er / max(n, 1):.2f}% ({100 * er / tc * scale:.3f}pp)"))
        print(f"  by {sl}:", "; ".join(x[1] for x in sorted(rows, reverse=True)[:8]))
    for sl, d in r.get("missed_slices", {}).items():
        print(f"  missed words by {sl}:", "; ".join(f"{k}: {v[0]}w/{v[1]}c" for k, v in sorted(d.items(), key=lambda kv: -kv[1][1])[:8]))
    for c, ex in r.get("examples", {}).items():
        print(f"  examples {c} (page, truth, read, height, font):", " | ".join(f"{e[0]} {e[1]!r}->{e[2]!r} h{e[3]}" for e in ex[:10]))
    print("  where split-space breaks fall:", r.get("breaks"))
    print("  extras:", "; ".join(f"{k} {v[0]}w/{v[1]}c" for k, v in r["extra"]["by"].items()))
    print("  top confusions (truth->hyp):", " ".join(f"{a}>{b}:{n}" for a, b, n in r["confusions"][:14]))
    worst = sorted(r["pages"].items(), key=lambda kv: -kv[1]["cer"] * kv[1]["chars"])[:8]
    print("  worst pages by CER contribution (share of the set's official char errors):")
    tot_err = sum(p["cer"] * p["chars"] for p in r["pages"].values()) or 1
    for pid, p in worst:
        dom = max(p["classes"].items(), key=lambda kv: kv[1])[0] if p["classes"] and max(p["classes"].values()) > 0 else "-"
        print(f"    {pid}: CER {100 * p['cer']:.2f} ({100 * p['cer'] * p['chars'] / tot_err:.0f}% of set) WER {100 * p['wer']:.2f} order {100 * p['order']:.2f} order-only {100 * p['order_only']:.2f}"
              f" extra {p['extra_chars']}c, dominant {dom} {p['classes'].get(dom, 0)}c; {p['meta']}; found {p['words_found']}/{p['truth_words']} words")
    by_tpl = defaultdict(lambda: [0, 0.0, 0])
    for p in r["pages"].values():
        v = by_tpl[p["meta"].split("/")[0]]
        v[0] += 1
        v[1] += p["order_only"] * p["truth_words"]
        v[2] += p["truth_words"]
    print("  order-only damage by template/kind (pages, share of truth words):", "; ".join(f"{k}: {v[0]}p {100 * v[1] / max(v[2], 1):.1f}%" for k, v in sorted(by_tpl.items(), key=lambda kv: -kv[1][1])[:8]))
    wo = sorted(r["pages"].items(), key=lambda kv: -kv[1]["order_only"] * kv[1]["truth_words"])[:5]
    print("  worst pages by order-only damage:", "; ".join(f"{k} {100 * p['order_only']:.1f}% ({p['meta']})" for k, p in wo))
    for k, p in wo[:3]:
        print(f"    {k} block runs (block:words): " + " ".join(f"{b}:{n}" for b, n in p["block_runs"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=SETS)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = {}
    for s in a.sets:
        res[s] = analyse(Path(a.eval), Path(a.out), s)
        show(res[s])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
