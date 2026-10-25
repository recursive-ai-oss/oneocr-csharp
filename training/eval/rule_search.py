# ruff: noqa: E501
"""Fit a page-conditional text threshold on held-out pages and confirm it on others.

The detector's text threshold is a manifest value, so a rule "pages where <condition on cheap page measures> use threshold X, the rest use
the default" gives, page by page, exactly the output of the manifest variant for X. Per-page error counts of every variant are computed
once with the benchmark's own page scorer (micro-averaged sums are exact), after which any rule is a cheap sum.

A condition is one or two terms `measure > T` / `measure < T` joined by AND (measures come from `eval.page_noise`). A rule is an ordered
list of (condition, variant); a page takes the first variant whose condition holds, else LO. Search: the best single rule, then (greedy)
a second rule on the pages the first did not take. Objective: mean of the target sets' CER (default degraded and phone). Constraint: no
fit set (CORD recall included) loses more than `tol` percentage points CER, WER or order against LO.

    python rule_search.py --fit-root GRID --fit-noise noise.json --fit NAME=TRUTH ... \
        --conf-root GRID2 --conf-noise noise2.json --conf NAME=TRUTH ... --lo t0.3-e0.4 --hi t0.4-e0.4,t0.5-e0.4,t0.6-e0.4 \
        --targets degraded-tune:degraded-conf,phone-tune:phone-conf --recall sel-cord,cord-conf
--targets pairs a fit set with its confirm set (FIT:CONF); only the fit sets enter the objective.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from oneocr_bench.score import score_page

KEYS = ("char_errors", "ref_chars", "word_errors", "ref_words", "order_errors", "char_misses", "word_misses")
MEASURES = ("sigma", "junk", "comps", "coverage", "height", "prob", "nlines", "paper", "contrast", "aspect")


def page_table(root: Path, variants: list[str], truth: dict[str, Path], noise: dict) -> dict:
    """{set: {"pages": [...], "x": {measure: array}, "v": {variant: (pages, 7) array}}}"""
    tab = {}
    for name, tdir in truth.items():
        pages = sorted(p.stem for p in tdir.glob("*.txt"))
        v = {}
        for var in variants:
            rows = []
            for p in pages:
                f = root / var / name / f"{p}.txt"
                h = f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""
                sc = score_page((tdir / f"{p}.txt").read_text(encoding="utf-8"), h)
                rows.append([sc[k] for k in KEYS])
            v[var] = np.array(rows, np.float64)
        x = {m: np.array([noise[name][p][m] for p in pages]) for m in MEASURES}
        tab[name] = {"pages": pages, "x": x, "v": v}
    return tab


def metrics(tab: dict, name: str, pick: np.ndarray, variants: list[str], recall: bool) -> np.ndarray:
    """(CER, WER, order) in percent for one set when page i takes variant pick[i] (an index into `variants`)."""
    if "stack" not in tab[name]:
        tab[name]["stack"] = np.stack([tab[name]["v"][v] for v in variants])  # (variant, page, 7)
    stack = tab[name]["stack"]
    s = stack[pick, np.arange(stack.shape[1])].sum(0)
    ce, we = (s[5], s[6]) if recall else (s[0], s[2])
    return 100 * np.array([ce / s[1], we / s[3], s[4] / s[3]])


def terms_for(tab: dict, measures: list[str], n_thr: int) -> list[tuple]:
    """`measure > T` and `measure < T` at T evenly spaced in rank over the pooled fit values."""
    terms = []
    for m in measures:
        pooled = np.unique(np.concatenate([t["x"][m] for t in tab.values()]))
        if len(pooled) < 2:
            continue
        mids = (pooled[:-1] + pooled[1:]) / 2
        pick = mids if len(mids) <= n_thr else mids[np.round(np.linspace(0, len(mids) - 1, n_thr)).astype(int)]
        for t in pick:
            terms += [(m, ">", float(t)), (m, "<", float(t))]
    return terms


def conditions(tab: dict, measures: list[str], n_thr: int, n_pair: int) -> list[tuple]:
    """Single terms (n_thr thresholds per measure) and AND pairs of terms on different measures (n_pair thresholds per measure)."""
    conds = [(a,) for a in terms_for(tab, measures, n_thr)]
    coarse = terms_for(tab, measures, n_pair)
    return conds + [(a, b) for a, b in itertools.combinations(coarse, 2) if a[0] != b[0]]


def holds(cond: tuple, x: dict) -> np.ndarray:
    ok = np.ones(len(next(iter(x.values()))), bool)
    for m, op, t in cond:
        ok &= (x[m] > t) if op == ">" else (x[m] < t)
    return ok


def evaluate(tab, rules, variants, lo, recall):
    """Per-set metrics for an ordered rule list [(cond, variant)]."""
    out = {}
    for n in tab:
        pick = np.full(len(tab[n]["pages"]), variants.index(lo))
        taken = np.zeros(len(pick), bool)
        for cond, var in rules:
            m = holds(cond, tab[n]["x"]) & ~taken
            pick[m] = variants.index(var)
            taken |= m
        out[n] = (metrics(tab, n, pick, variants, n in recall), int(taken.sum()), len(pick))
    return out


def fmt(cond: tuple) -> str:
    return " and ".join(f"{m} {op} {t:.4g}" for m, op, t in cond)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-root", required=True)
    ap.add_argument("--fit-noise", required=True)
    ap.add_argument("--fit", action="append", required=True, help="NAME=TRUTH_DIR")
    ap.add_argument("--conf-root")
    ap.add_argument("--conf-noise")
    ap.add_argument("--conf", action="append", default=[], help="NAME=TRUTH_DIR")
    ap.add_argument("--lo", required=True)
    ap.add_argument("--hi", required=True)
    ap.add_argument("--targets", required=True, help="FIT:CONF,... sets whose CER is the objective")
    ap.add_argument("--recall", default="sel-cord,cord-conf")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--n-thr", type=int, default=24)
    ap.add_argument("--n-pair", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=0.03, help="a rule must lower the objective by at least this many points")
    ap.add_argument("--min-pages", type=int, default=5, help="a rule must take at least this many pages of the target sets (no single-page fits)")
    ap.add_argument("--measures", default=",".join(MEASURES))
    ap.add_argument("--also", help="a variant to show next to LO in the report (e.g. v020)")
    ap.add_argument("--out")
    a = ap.parse_args()
    his = a.hi.split(",")
    variants = [a.lo, *his] + ([a.also] if a.also else [])
    recall = set(a.recall.split(","))
    fit_truth = {k: Path(v) for k, v in (s.split("=", 1) for s in a.fit)}
    targets = [t.split(":") for t in a.targets.split(",")]
    tfit = page_table(Path(a.fit_root), variants, fit_truth, json.loads(Path(a.fit_noise).read_text()))
    measures = a.measures.split(",")
    conds = conditions(tfit, measures, a.n_thr, a.n_pair)
    print(f"{len(conds)} conditions over {len(measures)} measures", flush=True)
    base = {n: metrics(tfit, n, np.zeros(len(tfit[n]["pages"]), int), variants, n in recall) for n in tfit}
    obj_sets = [t[0] for t in targets]

    def search(prev_rules):
        best = None
        prev = evaluate(tfit, prev_rules, variants, a.lo, recall)
        for cond in conds:
            for hi in his:
                res = evaluate(tfit, [*prev_rules, (cond, hi)], variants, a.lo, recall)
                harm = max(float((res[n][0] - base[n]).max()) for n in tfit)
                if harm > a.tol:
                    continue
                if sum(res[n][1] for n in obj_sets) - sum(prev[n][1] for n in obj_sets) < a.min_pages:
                    continue
                obj = float(np.mean([res[n][0][0] for n in obj_sets]))
                key = (obj, sum(r[1] for r in res.values()))  # ties: the rule that takes fewer pages
                if best is None or key < best[0]:
                    best = (key, cond, hi, res, harm)
        return best

    rules: list[tuple] = []
    cur = float(np.mean([base[n][0] for n in obj_sets]))
    print(f"LO {a.lo}: objective (mean CER of {obj_sets}) {cur:.3f}")
    for stage in range(2):
        b = search(rules)
        if b is None or b[0][0] >= cur - a.min_gain:
            break
        rules.append((b[1], b[2]))
        cur = b[0][0]
        print(f"rule {stage + 1}: if {fmt(b[1])} -> {b[2]}   objective {cur:.3f}, worst fit loss {b[4]:+.3f} pp", flush=True)
    if not rules:
        print("no rule improves the objective within the tolerance: recommend nothing")

    def report(label, tab, upto=None):
        used = rules if upto is None else rules[:upto]
        res = evaluate(tab, used, variants, a.lo, recall)
        print(f"\n{label}: CER / WER / order % (sel-cord and cord-conf: recall); pages taken by the rule")
        for n in tab:
            b0 = metrics(tab, n, np.zeros(len(tab[n]["pages"]), int), variants, n in recall)
            line = f"  {n:18s} LO {b0[0]:.3f} / {b0[1]:.3f} / {b0[2]:.3f}   rule {res[n][0][0]:.3f} / {res[n][0][1]:.3f} / {res[n][0][2]:.3f}   d {res[n][0][0] - b0[0]:+.3f} / {res[n][0][1] - b0[1]:+.3f} / {res[n][0][2] - b0[2]:+.3f}   taken {res[n][1]}/{res[n][2]}"
            if a.also:
                al = metrics(tab, n, np.full(len(tab[n]["pages"]), variants.index(a.also)), variants, n in recall)
                line += f"   {a.also} {al[0]:.3f} / {al[1]:.3f} / {al[2]:.3f}"
            print(line)

    report("FIT", tfit)
    if len(rules) > 1:
        report("FIT, first rule only", tfit, 1)
    if a.conf_root:
        conf_truth = {k: Path(v) for k, v in (s.split("=", 1) for s in a.conf)}
        tconf = page_table(Path(a.conf_root), variants, conf_truth, json.loads(Path(a.conf_noise).read_text()))
        report("CONFIRM", tconf)
        if len(rules) > 1:
            report("CONFIRM, first rule only", tconf, 1)
    if a.out:
        Path(a.out).write_text(json.dumps([{"cond": [list(t) for t in c], "variant": v} for c, v in rules], indent=1))


if __name__ == "__main__":
    main()
