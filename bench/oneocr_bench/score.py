"""Score OCR hypotheses against truth text: order-free CER/WER and reading-order error.

Normalisation (applied to truth and hypothesis alike):
  1. Unicode NFC;
  2. typographic quotes/dashes folded to ASCII (' " -) and the ligatures fi/fl split,
     so an engine is not charged for a font's choice of glyph;
  3. all whitespace runs collapse to one space; text is split into words on whitespace.
Case and punctuation are kept.

Order-free content metrics (cer, wer): words are treated as a bag. Words present in
both sides cancel exactly; each leftover truth word is paired with the closest leftover
hypothesis word (greedy, smallest edit distance first); unpaired leftovers are
deletions / insertions.
  wer = max(#leftover truth words, #leftover hyp words) / #truth words
  cer = (sum of edit distances of pairs + chars of unpaired words) / #truth chars (no spaces)
A word split or merged by the engine therefore costs a little more than in an ordered alignment.

Reading-order error (order_err): normalised word-level edit distance of the two word
sequences in order (substitutions + insertions + deletions) / #truth words. It also
contains recognition errors; on a page with few of them it is dominated by order.

    python -m oneocr_bench score --truth DIR --hyp DIR [--json OUT]
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-",
    "\u00a0": " ",
})


def normalize(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text).translate(FOLD).replace("\ufb01", "fi").replace("\ufb02", "fl")
    return t.split()


def content_errors(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """(char errors, word errors, char misses, word misses) of the bag-of-words comparison.

    Misses count only what is wrong on the truth side (a truth word missing or misread; extra hypothesis
    words are free), for sets whose truth does not cover everything printed on the page.
    """
    rc, hc = Counter(ref), Counter(hyp)
    r_left = list((rc - hc).elements())
    h_left = list((hc - rc).elements())
    word_err = max(len(r_left), len(h_left))
    char_err = 0
    char_miss = 0
    if r_left and h_left:
        dist_fn = Levenshtein.distance
        pairs = sorted((dist_fn(r_left[i], h_left[j]), i, j) for i in range(len(r_left)) for j in range(len(h_left)))
        used_r: set[int] = set()
        used_h: set[int] = set()
        for dist, i, j in pairs:
            if i in used_r or j in used_h:
                continue
            used_r.add(i)
            used_h.add(j)
            char_err += min(dist, max(len(r_left[i]), len(h_left[j])))
            char_miss += min(dist, len(r_left[i]))
            if len(used_r) == len(r_left) or len(used_h) == len(h_left):
                break
        char_err += sum(len(w) for k, w in enumerate(r_left) if k not in used_r)
        char_miss += sum(len(w) for k, w in enumerate(r_left) if k not in used_r)
        char_err += sum(len(w) for k, w in enumerate(h_left) if k not in used_h)
    else:
        char_err = sum(len(w) for w in r_left) + sum(len(w) for w in h_left)
        char_miss = sum(len(w) for w in r_left)
    return char_err, word_err, char_miss, len(r_left)


def score_page(truth: str, hyp: str) -> dict:
    ref, hy = normalize(truth), normalize(hyp)
    ce, we, cm, wm = content_errors(ref, hy)
    return {
        "ref_words": len(ref), "ref_chars": sum(map(len, ref)),
        "char_errors": ce, "word_errors": we, "char_misses": cm, "word_misses": wm,
        "order_errors": Levenshtein.distance(ref, hy),
    }


def _aabb(w: dict) -> list[float]:
    if "box" in w:
        return [float(v) for v in w["box"]]
    q = w["quad"]
    return [min(q[0::2]), min(q[1::2]), max(q[0::2]), max(q[1::2])]


def box_matches(truth: list[list[float]], hyp: list[list[float]], thr: float = 0.5) -> int:
    """One-to-one matches at IoU >= thr, taking the highest-IoU pairs first (text is not compared)."""
    if not truth or not hyp:
        return 0
    t, h = np.array(truth), np.array(hyp)
    iw = np.minimum(t[:, None, 2], h[None, :, 2]) - np.maximum(t[:, None, 0], h[None, :, 0])
    ih = np.minimum(t[:, None, 3], h[None, :, 3]) - np.maximum(t[:, None, 1], h[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area_t = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    area_h = (h[:, 2] - h[:, 0]) * (h[:, 3] - h[:, 1])
    iou = inter / np.maximum(area_t[:, None] + area_h[None, :] - inter, 1e-9)
    ti, hi = np.nonzero(iou >= thr)
    order = np.argsort(-iou[ti, hi], kind="stable")
    used_t: set[int] = set()
    used_h: set[int] = set()
    n = 0
    for k in order:
        a, b = int(ti[k]), int(hi[k])
        if a in used_t or b in used_h:
            continue
        used_t.add(a)
        used_h.add(b)
        n += 1
    return n


def score_dir(truth_dir: Path, hyp_dir: Path) -> dict:
    """Micro-averaged over pages; a missing hypothesis file counts as empty output.

    Word-detection F1 at IoU 0.5 is added when every page has a sidecar `<id>.json` in the hypothesis directory:
    {"words": [{"box": [x0, y0, x1, y1]} | {"quad": [x1, y1, ..., x4, y4]}, ...]} in the pixel coordinates of the
    page image (a `"text"` member is allowed and ignored). Truth quads are compared as their axis-aligned boxes.
    """
    pages = {}
    det = {"tp": 0, "truth": 0, "hyp": 0}
    have_boxes = True
    for t in sorted(truth_dir.glob("*.txt")):
        h = hyp_dir / t.name
        text = h.read_text(encoding="utf-8", errors="replace") if h.exists() else ""
        pages[t.stem] = score_page(t.read_text(encoding="utf-8"), text)
        side = hyp_dir / f"{t.stem}.json"
        if not side.exists():
            have_boxes = False
        elif have_boxes:
            tb = [_aabb(w) for w in json.loads((truth_dir / f"{t.stem}.json").read_text())["words"]]
            hb = [_aabb(w) for w in json.loads(side.read_text())["words"]]
            det["tp"] += box_matches(tb, hb)
            det["truth"] += len(tb)
            det["hyp"] += len(hb)
    tot = {k: sum(p[k] for p in pages.values()) for k in next(iter(pages.values()))}
    r = {
        "pages": len(pages),
        "missing": sum(not (hyp_dir / f"{p}.txt").exists() for p in pages),
        "cer": tot["char_errors"] / tot["ref_chars"],
        "wer": tot["word_errors"] / tot["ref_words"],
        "order_err": tot["order_errors"] / tot["ref_words"],
        "cer_recall": tot["char_misses"] / tot["ref_chars"],
        "wer_recall": tot["word_misses"] / tot["ref_words"],
        "det": None,
        "totals": tot,
        "per_page": pages,
    }
    if have_boxes and det["truth"]:
        p = det["tp"] / det["hyp"] if det["hyp"] else 0.0
        rec = det["tp"] / det["truth"]
        r["det"] = {"precision": p, "recall": rec, "f1": 2 * p * rec / (p + rec) if p + rec else 0.0, **det}
    return r


def summary_line(r: dict) -> str:
    f1 = f"  F1@0.5 {r['det']['f1']:.4f} (P {r['det']['precision']:.3f} R {r['det']['recall']:.3f})" if r["det"] else ""
    return (f"pages {r['pages']} (missing {r['missing']})  CER {r['cer']:.4%}  WER {r['wer']:.4%}  "
            f"order {r['order_err']:.4%}  recall-CER {r['cer_recall']:.4%}  recall-WER {r['wer_recall']:.4%}{f1}")


def main(truth: str, hyp: str, out: str | None) -> None:
    r = score_dir(Path(truth), Path(hyp))
    print(summary_line(r))
    if out:
        Path(out).write_text(json.dumps(r, indent=1) + "\n")


def main_root(root: str, out: str | None) -> None:
    """One command, every set: score <root>/<set>/<id>.txt for each set that has a directory there."""
    from .sets import SETS, set_dir

    res = {}
    print(f"{'set':14} {'category':22} {'pages':>5} {'CER %':>7} {'WER %':>7} {'order %':>8} {'rCER %':>7} {'rWER %':>7} {'F1@.5':>6}")
    for name, spec in SETS.items():
        h = Path(root) / name
        if not h.is_dir():
            continue
        r = score_dir(set_dir(name) / "truth", h)
        res[name] = {k: v for k, v in r.items() if k != "per_page"} | {"category": spec.category}
        f1 = f"{r['det']['f1']:.3f}" if r["det"] else "-"
        print(f"{name:14} {spec.category:22} {r['pages']:>5} {r['cer'] * 100:>7.3f} {r['wer'] * 100:>7.3f} {r['order_err'] * 100:>8.3f} "
              f"{r['cer_recall'] * 100:>7.3f} {r['wer_recall'] * 100:>7.3f} {f1:>6}")
    if out:
        Path(out).write_text(json.dumps(res, indent=1) + "\n")
