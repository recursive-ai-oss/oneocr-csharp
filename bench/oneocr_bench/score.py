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

Word confidence (optional): when every hypothesis sidecar word has `text` and a confidence (`confidence`, or `conf` as written
by the C# bench), the scorer adds expected calibration error; see `word_outcomes` and `ece`.

Scorer versions (`--scorer`): v2 builds the truth word sequence from the truth text `<id>.txt`. v3 builds it from the truth's own word
boxes (`<id>.json`) wherever they exist, see `v3_truth_words`; a truth directory without word boxes scores identically under both.
Normalisation, the bag pairing and the order distance are the same in both.

Word boxes: truth `words` carry `box` (the browser's layout rectangle) and, in bench-v2 sets, `ink` (tight to the printed ink).
`--boxes ink` (default) scores F1 against `ink` where the truth has it, `--boxes layout` against `box`.

    python -m oneocr_bench score --truth DIR --hyp DIR [--boxes ink|layout] [--scorer v2|v3] [--json OUT]
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-",
    "\u00a0": " ",
})


def normalize(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text).translate(FOLD).replace("\ufb01", "fi").replace("\ufb02", "fl")
    return t.split()


SCORERS = ("v2", "v3")
DEFAULT_SCORER = "v2"

# v3 joins a word box to its list neighbour when their layout boxes are at most this far apart horizontally (pixels plus a share of the
# box height) and overlap vertically by at least half of the shorter box.
TOUCH_PX = 0.5
TOUCH_HEIGHT = 0.02
TOUCH_OVERLAP = 0.5


def _is_mark(text: str) -> bool:
    """A word made only of punctuation or symbols (Unicode categories P* and S*), such as `,` `)` `-` `/` `$`."""
    return bool(text) and all(unicodedata.category(c)[0] in "PS" for c in text)


def _touch(a: list[float], b: list[float]) -> bool:
    h = min(a[3] - a[1], b[3] - b[1])
    if h <= 0 or min(a[3], b[3]) - max(a[1], b[1]) < TOUCH_OVERLAP * h:
        return False
    return max(a[0] - b[2], b[0] - a[2]) <= TOUCH_PX + TOUCH_HEIGHT * max(a[3] - a[1], b[3] - b[1])


def v3_truth_words(words: list[dict]) -> list[str]:
    """The truth word sequence of scorer v3, from a truth sidecar's `words` (in their order), as unnormalised strings.

    Every box word is a truth word, except that a box whose text is only punctuation or symbols is joined to each neighbour in the list
    (previous, next) whose layout box touches it: the ink is continuous there, so a reader sees one word (`Intelligence` `,` is
    `Intelligence,`, `1` `/` `2` is `1/2`). Two letter or digit boxes are never joined, so a word the page wraps at a hyphen or dash stays
    one word per line (`breakfast-` and `room`), as does any other pair of separate pieces. Boxes whose normalised text is empty are dropped.
    """
    ws = [w for w in words if normalize(w["text"])]
    boxes = [_aabb(w, "box") for w in ws]
    out: list[str] = []
    joined = False  # the previous box was joined to this one
    for i, w in enumerate(ws):
        if joined:
            out[-1] += w["text"]
        else:
            out.append(w["text"])
        joined = (i + 1 < len(ws) and (_is_mark("".join(normalize(w["text"]))) or _is_mark("".join(normalize(ws[i + 1]["text"]))))
                  and _touch(boxes[i], boxes[i + 1]))
    return out


def truth_text(truth_dir: Path, stem: str, scorer: str = DEFAULT_SCORER) -> str:
    """The truth of one page as text: `<id>.txt` (v2), or under v3 the box words of `<id>.json` joined by newlines when it has any."""
    txt = (truth_dir / f"{stem}.txt").read_text(encoding="utf-8")
    if scorer == "v2":
        return txt
    if scorer != "v3":
        raise ValueError(f"unknown scorer {scorer!r}: expected one of {SCORERS}")
    side = truth_dir / f"{stem}.json"
    if side.exists():
        words = v3_truth_words(json.loads(side.read_text()).get("words") or [])
        if words:
            return "\n".join(words)
    return txt


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


ECE_BINS = 15


def word_outcomes(ref: list[str], hyp: list[str]) -> list[str]:
    """How each hypothesis word is matched to the truth, the way `content_errors` pairs words.

    "exact": identical to a truth word (identical words cancel one for one; among equal hypothesis words the earlier ones are
    the ones that match). Leftover truth and hypothesis words are then paired closest first, as in `content_errors`:
    "paired" is a hypothesis word read wrongly (paired with a truth word it is not equal to), "extra" a leftover with no
    truth word left (inserted text, or printed text the truth does not cover).
    """
    rc = Counter(ref)
    taken: Counter = Counter()
    out = [""] * len(hyp)
    for j, w in enumerate(hyp):
        if taken[w] < rc[w]:
            taken[w] += 1
            out[j] = "exact"
    left_h = [j for j, o in enumerate(out) if not o]
    left_r = list((rc - Counter(hyp)).elements())
    for j in left_h:
        out[j] = "extra"
    if left_r and left_h:
        d = cdist(left_r, [hyp[j] for j in left_h], scorer=Levenshtein.distance)
        used_r: set[int] = set()
        used_h: set[int] = set()
        m = len(left_h)
        for flat in np.argsort(d, axis=None, kind="stable"):  # smallest distance first, then truth index, then hypothesis index
            i, k = divmod(int(flat), m)
            if i in used_r or k in used_h:
                continue
            used_r.add(i)
            used_h.add(k)
            out[left_h[k]] = "paired"
            if len(used_r) == len(left_r) or len(used_h) == m:
                break
    return out


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error: equal-width bins on [0, 1], sum of (bin share) * |bin accuracy - bin mean confidence|."""
    if len(conf) == 0:
        return float("nan")
    idx = np.minimum((np.clip(conf, 0.0, 1.0) * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.sum() / len(conf) * abs(correct[sel].mean() - conf[sel].mean())
    return float(total)


def _confidence(w: dict) -> float | None:
    v = w.get("confidence", w.get("conf"))
    return None if v is None else float(v)


def ece_summary(conf: list[float], status: list[str]) -> dict:
    c = np.array(conf)
    st = np.array(status)
    ok = (st == "exact").astype(float)
    m = st != "extra"
    return {
        "bins": ECE_BINS, "words": len(c), "accuracy": float(ok.mean()), "mean_confidence": float(c.mean()),
        "ece": ece(c, ok), "matched_words": int(m.sum()),
        "matched_accuracy": float(ok[m].mean()) if m.any() else float("nan"), "ece_matched": ece(c[m], ok[m]),
    }


def score_page(truth: str, hyp: str) -> dict:
    ref, hy = normalize(truth), normalize(hyp)
    ce, we, cm, wm = content_errors(ref, hy)
    return {
        "ref_words": len(ref), "ref_chars": sum(map(len, ref)),
        "char_errors": ce, "word_errors": we, "char_misses": cm, "word_misses": wm,
        "order_errors": Levenshtein.distance(ref, hy),
    }


def _aabb(w: dict, key: str = "box") -> list[float]:
    if key in w:
        return [float(v) for v in w[key]]
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


def score_dir(truth_dir: Path, hyp_dir: Path, boxes: str = "ink", scorer: str = DEFAULT_SCORER) -> dict:
    """Micro-averaged over pages; a missing hypothesis file counts as empty output.

    Word-detection F1 at IoU 0.5 is added when every page has a sidecar `<id>.json` in the hypothesis directory:
    {"words": [{"box": [x0, y0, x1, y1]} | {"quad": [x1, y1, ..., x4, y4]}, ...]} in the pixel coordinates of the
    page image (a `"text"` member is allowed and ignored for F1). Truth quads are compared as their axis-aligned boxes.
    `boxes` picks the truth box: "ink" (the tight box where the truth has one, as in bench-v2) or "layout".

    Expected calibration error is added when every sidecar word has `text` and a confidence: words are matched to the page's
    truth words by `word_outcomes`; "correct" is an exact match after normalisation. `ece` counts every hypothesis word
    (unmatched ones are wrong); `ece_matched` leaves out the "extra" words, which is what a partial truth (CORD) needs.

    `scorer` picks the truth word sequence (module docstring); detection F1 always uses the truth boxes as given.
    """
    if scorer not in SCORERS:
        raise ValueError(f"unknown scorer {scorer!r}: expected one of {SCORERS}")
    key = "ink" if boxes == "ink" else "box"
    pages = {}
    det = {"tp": 0, "truth": 0, "hyp": 0}
    have_boxes = True
    have_conf = True
    conf: list[float] = []
    status: list[str] = []
    for t in sorted(truth_dir.glob("*.txt")):
        h = hyp_dir / t.name
        text = h.read_text(encoding="utf-8", errors="replace") if h.exists() else ""
        truth = truth_text(truth_dir, t.stem, scorer)
        pages[t.stem] = score_page(truth, text)
        side = hyp_dir / f"{t.stem}.json"
        if not side.exists():
            have_boxes = have_conf = False
        else:
            hw = json.loads(side.read_text())["words"]
            if have_boxes and not (truth_dir / f"{t.stem}.json").exists():
                have_boxes = False  # a truth without word boxes: no detection score for the set
            if have_boxes:
                tb = [_aabb(w, key) for w in json.loads((truth_dir / f"{t.stem}.json").read_text())["words"]]
                hb = [_aabb(w) for w in hw]
                det["tp"] += box_matches(tb, hb)
                det["truth"] += len(tb)
                det["hyp"] += len(hb)
            if have_conf:
                if any("text" not in w or _confidence(w) is None for w in hw):
                    have_conf = False
                else:
                    rows = [(" ".join(normalize(w["text"])), _confidence(w)) for w in hw]
                    rows = [(k, c) for k, c in rows if k]
                    status += word_outcomes(normalize(truth), [k for k, _ in rows])
                    conf += [c for _, c in rows]
    if not pages:
        raise SystemExit(f"no truth pages (<id>.txt) in {truth_dir}")
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
    r["boxes"] = boxes
    r["scorer"] = scorer
    r["ece"] = ece_summary(conf, status) if have_conf and conf else None
    if have_boxes and det["truth"]:
        p = det["tp"] / det["hyp"] if det["hyp"] else 0.0
        rec = det["tp"] / det["truth"]
        r["det"] = {"precision": p, "recall": rec, "f1": 2 * p * rec / (p + rec) if p + rec else 0.0, **det}
    return r


def summary_line(r: dict) -> str:
    tag = "  [scorer v3]" if r.get("scorer") == "v3" else ""
    f1 = f"  F1@0.5 {r['det']['f1']:.4f} [{r['boxes']} boxes] (P {r['det']['precision']:.3f} R {r['det']['recall']:.3f})" if r["det"] else ""
    e = r.get("ece")
    ece_s = f"  ECE {e['ece']:.4f} (matched words only {e['ece_matched']:.4f}; {e['words']} words)" if e else ""
    return (f"pages {r['pages']} (missing {r['missing']})  CER {r['cer']:.4%}  WER {r['wer']:.4%}  "
            f"order {r['order_err']:.4%}  recall-CER {r['cer_recall']:.4%}  recall-WER {r['wer_recall']:.4%}{f1}{ece_s}{tag}")


def main(truth: str, hyp: str, out: str | None, boxes: str = "ink", scorer: str = DEFAULT_SCORER) -> None:
    r = score_dir(Path(truth), Path(hyp), boxes, scorer)
    print(summary_line(r))
    if out:
        Path(out).write_text(json.dumps(r, indent=1) + "\n")


def main_root(root: str, out: str | None, boxes: str = "ink", scorer: str = DEFAULT_SCORER) -> None:
    """One command, every set: score <root>/<set>/<id>.txt for each set that has a directory there."""
    from .sets import SETS, set_dir

    res = {}
    print(f"{'set':14} {'category':22} {'pages':>5} {'CER %':>7} {'WER %':>7} {'order %':>8} {'rCER %':>7} {'rWER %':>7} {'F1@.5':>6} {'ECE':>6} {'ECE-m':>6}")
    for name, spec in SETS.items():
        h = Path(root) / name
        if not h.is_dir():
            continue
        r = score_dir(set_dir(name) / "truth", h, boxes, scorer)
        res[name] = {k: v for k, v in r.items() if k != "per_page"} | {"category": spec.category}
        f1 = f"{r['det']['f1']:.3f}" if r["det"] else "-"
        e = f"{r['ece']['ece']:>6.4f} {r['ece']['ece_matched']:>6.4f}" if r["ece"] else f"{'-':>6} {'-':>6}"
        print(f"{name:14} {spec.category:22} {r['pages']:>5} {r['cer'] * 100:>7.3f} {r['wer'] * 100:>7.3f} {r['order_err'] * 100:>8.3f} "
              f"{r['cer_recall'] * 100:>7.3f} {r['wer_recall'] * 100:>7.3f} {f1:>6} {e}")
    if out:
        Path(out).write_text(json.dumps(res, indent=1) + "\n")
