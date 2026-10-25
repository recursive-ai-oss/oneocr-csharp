"""Paired comparison of two recognizers on the same crops: CER and WER of each, their difference and a bootstrap interval.

    uv run python -m eval.paired_gate OLD_HYPS_DIR NEW_HYPS_DIR [--old-name 0.6.0 --new-name 0.6.1] [--sets a,b] [--json out.json]

Both directories hold <set>.hyps.jsonl (eval.model_hyps). Lines are paired by id; each bootstrap draw resamples lines with replacement
and recomputes both micro-averaged rates on the same draw, so the interval is the noise of the *difference* (the crops are shared).
A row is "worse" when the 95% interval of (new - old) lies entirely above zero, "better" when entirely below, else "same".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from eval.scoring import normalize, read_hyps


def per_line(rows: list[dict]) -> dict[str, tuple[int, int, int, int]]:
    out = {}
    for r in rows:
        ref, hyp = normalize(r["ref"]), normalize(r["hyp"])
        rw, hw = ref.split(), hyp.split()
        out[r["id"]] = (Levenshtein.distance(ref, hyp), len(ref), Levenshtein.distance(rw, hw), len(rw))
    return out


def compare(old: list[dict], new: list[dict], draws: int = 2000, seed: int = 0) -> dict:
    a, b = per_line(old), per_line(new)
    ids = sorted(a.keys() & b.keys())
    x = np.array([a[i] for i in ids], dtype=np.float64)
    y = np.array([b[i] for i in ids], dtype=np.float64)
    rate = lambda m, idx, c: m[idx, c].sum(axis=-1) / np.maximum(m[idx, c + 1].sum(axis=-1), 1)  # noqa: E731
    res = {"lines": len(ids)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(draws, len(ids)))
    for name, c in (("cer", 0), ("wer", 2)):
        o = x[:, c].sum() / max(x[:, c + 1].sum(), 1)
        n = y[:, c].sum() / max(y[:, c + 1].sum(), 1)
        d = rate(y, idx, c) - rate(x, idx, c)
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[name] = {"old": o, "new": n, "delta": n - o, "lo": lo, "hi": hi,
                     "verdict": "worse" if lo > 0 else "better" if hi < 0 else "same"}  # fmt: skip
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--old-name", default="old")
    ap.add_argument("--new-name", default="new")
    ap.add_argument("--sets", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    old, new = Path(a.old), Path(a.new)
    names = a.sets.split(",") if a.sets else sorted(p.name.removesuffix(".hyps.jsonl") for p in new.glob("*.hyps.jsonl") if (old / p.name).exists())  # noqa: E501
    print(f"| set | lines | {a.old_name} CER / WER % | {a.new_name} CER / WER % | change CER pp [95% CI] | change WER pp [95% CI] | verdict |\n|---|---:|---:|---:|---|---|---|")  # noqa: E501
    allr = {}
    for s in names:
        r = compare(read_hyps(old / f"{s}.hyps.jsonl"), read_hyps(new / f"{s}.hyps.jsonl"))
        allr[s] = r
        c, w = r["cer"], r["wer"]
        f = lambda m: f"{100 * m['delta']:+.2f} [{100 * m['lo']:+.2f}, {100 * m['hi']:+.2f}]"  # noqa: E731
        print(f"| {s} | {r['lines']} | {100 * c['old']:.2f} / {100 * w['old']:.2f} | {100 * c['new']:.2f} / {100 * w['new']:.2f} | {f(c)} | {f(w)} | CER {c['verdict']}, WER {w['verdict']} |")  # noqa: E501
    if a.json:
        Path(a.json).write_text(json.dumps(allr, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
