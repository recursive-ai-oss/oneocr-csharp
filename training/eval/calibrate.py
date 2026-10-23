"""Word-confidence calibration: fit a per-script monotone piecewise-linear map, and measure expected calibration error.

A word is correct when it is exactly a word of the reference line: the hypothesis and reference word sequences are aligned by
edit distance and every hypothesis word in an `equal` block is correct (extra or misread words are not). The map goes from the
raw confidence (eval/confidence.py) to the probability of being correct; it is isotonic regression over quantile bins of the
calibration split, reduced to at most 64 knots with strictly increasing x in [0, 1] and non-decreasing y in [0, 1].

ECE: 15 equal-width bins over [0, 1], word level: sum over bins of (bin count / N) * |accuracy - mean confidence|.

    uv run python -m eval.calibrate fit --hyps calib.hyps.jsonl --out calibration.json
    uv run python -m eval.calibrate ece --calibration calibration.json --hyps A.hyps.jsonl [B.hyps.jsonl ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from eval.confidence import apply_map
from eval.scoring import read_hyps

SCRIPTS = {"latin": "Latin", "greek": "Greek", "cyrillic": "Cyrillic"}
MAX_KNOTS = 64


def word_samples(rows: list[dict]) -> list[tuple[str, float, bool]]:
    """(script, raw confidence, correct) for every hypothesis word."""
    out = []
    for r in rows:
        ref, hyp = r["ref"].split(), [w["text"] for w in r["words"]]
        ok = [False] * len(hyp)
        for op in Levenshtein.opcodes(ref, hyp):
            if op.tag == "equal":
                for j in range(op.dest_start, op.dest_end):
                    ok[j] = True
        out += [(SCRIPTS[w["script"]], w["conf"], ok[j]) for j, w in enumerate(r["words"])]
    return out


def pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: the non-decreasing fit to y."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y.tolist(), w.tolist(), strict=True):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wt, c = wts[-2] + wts[-1], cnt[-2] + cnt[-1]
            vals[-2:], wts[-2:], cnt[-2:] = [v], [wt], [c]
    return np.repeat(vals, cnt)


def fit_map(conf: np.ndarray, correct: np.ndarray, bins: int = MAX_KNOTS - 2) -> dict:
    """Knots of the monotone map. Samples are cut into `bins` quantile bins (fewer when raw values repeat); each bin gives
    one (mean confidence, accuracy) point, isotonic regression makes the accuracies non-decreasing, and end knots at 0 and 1 carry
    the outer values."""
    order = np.argsort(conf, kind="stable")
    c, k = conf[order], correct[order].astype(float)
    edges = np.unique(np.quantile(c, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, len(edges) - 2) if len(edges) > 1 else np.zeros(len(c), int)
    xs, ys, ws = [], [], []
    for b in range(max(len(edges) - 1, 1)):
        m = which == b
        if m.any():
            xs.append(float(c[m].mean()))
            ys.append(float(k[m].mean()))
            ws.append(float(m.sum()))
    ys = pav(np.array(ys), np.array(ws)).tolist()
    keep = [0]  # strictly increasing x
    for i in range(1, len(xs)):
        if xs[i] > xs[keep[-1]]:
            keep.append(i)
    xs, ys = [xs[i] for i in keep], [ys[i] for i in keep]
    if xs[0] > 0.0:
        xs, ys = [0.0] + xs, [ys[0]] + ys
    if xs[-1] < 1.0:
        xs, ys = xs + [1.0], ys + [ys[-1]]
    assert len(xs) <= MAX_KNOTS
    f32 = lambda v: float(f"{np.float32(min(max(v, 0.0), 1.0)):.9g}")  # noqa: E731 - exact in float32, short in JSON
    return {"x": [f32(v) for v in xs], "y": [f32(v) for v in ys]}


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def calibrated(samples: list[tuple[str, float, bool]], cal: dict) -> np.ndarray:
    return np.array([apply_map(c, cal["scripts"][s]) for s, c, _ in samples])


def ece_report(samples: list[tuple[str, float, bool]], cal: dict) -> dict:
    """ECE before and after, over all words and per script (scripts with at least 30 words)."""
    out = {}
    groups = {"all": samples} | {s: [x for x in samples if x[0] == s] for s in cal["scripts"]}
    for name, g in groups.items():
        if len(g) < 30:
            continue
        raw = np.array([x[1] for x in g])
        ok = np.array([x[2] for x in g], dtype=float)
        out[name] = {"words": len(g), "accuracy": float(ok.mean()), "mean_raw": float(raw.mean()),
                     "ece_raw": ece(raw, ok), "ece_calibrated": ece(calibrated(g, cal), ok)}  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--hyps", required=True)
    f.add_argument("--out", required=True)
    e = sub.add_parser("ece")
    e.add_argument("--calibration", required=True)
    e.add_argument("hyps", nargs="+")
    a = ap.parse_args()
    if a.cmd == "fit":
        samples = word_samples(read_hyps(a.hyps))
        cal = {"raw": "word-mean-max-run", "scripts": {}}
        for s in SCRIPTS.values():
            g = [x for x in samples if x[0] == s]
            cal["scripts"][s] = fit_map(np.array([x[1] for x in g]), np.array([x[2] for x in g]))
            print(s, len(g), "words,", len(cal["scripts"][s]["x"]), "knots, accuracy", round(sum(x[2] for x in g) / len(g), 4))
        Path(a.out).write_text(json.dumps(cal, indent=1), encoding="utf-8")
    else:
        cal = json.loads(Path(a.calibration).read_text(encoding="utf-8"))
        for h in a.hyps:
            print(h, json.dumps(ece_report(word_samples(read_hyps(h)), cal)))


if __name__ == "__main__":
    main()
