"""Paired bootstrap of the CER and WER difference between two models' hypotheses on identical lines (eval.model_hyps outputs).

    uv run python -m eval.paired_bootstrap --old DIR [--old DIR2 ...] --new DIR --sets a,b,c [--draws 2000] [--out table.md]

Each `--old` directory is searched in order for `<set>.hyps.jsonl`. Lines are resampled with replacement (the same lines for both
models); the table gives the corpus CER/WER of each model and the 95% interval of new minus old, in percentage points. An interval
that excludes zero is a real change at that sample size.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rapidfuzz.distance import Levenshtein


def load(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for ln in path.read_text().splitlines():
        r = json.loads(ln)
        out[r["id"]] = (r["ref"], r["hyp"])
    return out


def stats(pairs: list[tuple[str, str]]) -> list[tuple[int, int, int, int]]:
    rows = []
    for ref, hyp in pairs:
        rw, hw = ref.split(), hyp.split()
        rows.append((Levenshtein.distance(ref, hyp), len(ref), Levenshtein.distance(rw, hw), len(rw)))
    return rows


def rate(rows, idx, k):
    e = sum(rows[i][k] for i in idx)
    n = sum(rows[i][k + 1] for i in idx)
    return 100.0 * e / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", action="append", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out")
    a = ap.parse_args()
    rng = random.Random(7)
    lines = ["| set | lines | old CER / WER % | new CER / WER % | dCER pp [95%] | dWER pp [95%] |", "|---|---:|---:|---:|---|---|"]
    for s in a.sets.split(","):
        op = next((Path(d) / f"{s}.hyps.jsonl" for d in a.old if (Path(d) / f"{s}.hyps.jsonl").exists()), None)
        npth = Path(a.new) / f"{s}.hyps.jsonl"
        if op is None or not npth.exists():
            lines.append(f"| {s} | missing | | | | |")
            continue
        o, n = load(op), load(npth)
        ids = sorted(set(o) & set(n))
        ro = stats([o[i] for i in ids])
        rn = stats([n[i] for i in ids])
        full = list(range(len(ids)))
        d = {0: [], 2: []}
        for _ in range(a.draws):
            idx = [rng.randrange(len(ids)) for _ in ids]
            for k in (0, 2):
                d[k].append(rate(rn, idx, k) - rate(ro, idx, k))
        ci = {}
        for k in (0, 2):
            v = sorted(d[k])
            ci[k] = (v[int(0.025 * a.draws)], v[int(0.975 * a.draws)], rate(rn, full, k) - rate(ro, full, k))
        f = lambda r, k: f"{rate(r, full, k):.2f}"  # noqa: E731
        c = lambda k: f"{ci[k][2]:+.2f} [{ci[k][0]:+.2f}, {ci[k][1]:+.2f}]"  # noqa: E731
        lines.append(f"| {s} | {len(ids)} | {f(ro, 0)} / {f(ro, 2)} | {f(rn, 0)} / {f(rn, 2)} | {c(0)} | {c(2)} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
