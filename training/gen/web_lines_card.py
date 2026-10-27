# ruff: noqa: E501
"""Data card of a rendered web line set: counts, sites, fonts, sizes, what was dropped and why.

    uv run python -m gen.web_lines_card --data OUT [--out card.md]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    root = Path(a.data)
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    pages = load(root / "pages.jsonl")
    out = ["# Web line set: data card", ""]
    out.append(f"Pages listed {len(pages)}; rendered ok {sum(p['result'] == 'ok' for p in pages)}; "
               f"results: {dict(Counter(p['result'][:40] for p in pages))}")  # fmt: skip
    drops: Counter = Counter()
    for p in pages:
        drops.update(p.get("drops", {}))
    kept_total = sum(p.get("kept_after_dedupe", p.get("kept", 0)) for p in pages)
    cand = sum(p.get("candidates", 0) for p in pages)
    out += ["", f"Candidate visual lines {cand}; kept after every check and the per-site text cap: {kept_total}", "", "| dropped | lines | share of candidates |", "|---|---|---|"]
    for k, v in drops.most_common():
        out.append(f"| {k} | {v} | {100 * v / max(1, cand):.1f}% |")
    for split in ("train", "heldout"):
        rows = load(root / split / "labels.jsonl")
        if not rows:
            continue
        out += ["", f"## {split}: {len(rows)} lines, {len({r['url'] for r in rows})} pages, {len({r['publisher'] for r in rows})} sites", ""]
        out.append("| site | pages | lines |")
        out.append("|---|---|---|")
        pc: Counter = Counter()
        lc: Counter = Counter()
        for r in rows:
            lc[r["publisher"]] += 1
        for pg in {r["url"]: r["publisher"] for r in rows}.values():
            pc[pg] += 1
        for k, v in lc.most_common():
            out.append(f"| {k} | {pc[k]} | {v} |")
        em = np.array([r["em_px"] for r in rows])
        chars = np.array([len(r["text"]) for r in rows])
        out += ["", f"em size (device px): p10 {np.percentile(em, 10):.1f}, median {np.median(em):.1f}, p90 {np.percentile(em, 90):.1f}; "
                f"characters per line: median {np.median(chars):.0f}, p90 {np.percentile(chars, 90):.0f}; crop width at 32 px: median {np.median([r['w'] for r in rows]):.0f}"]  # fmt: skip
        out.append(f"device-pixel ratio: {dict(Counter(r['dpr'] for r in rows))}; anti-aliasing: {dict(Counter(r['aa'] for r in rows))}")
        out.append(f"lines set in the stand-in font for a held-out family: {sum(r.get('substituted', False) for r in rows)}")
        out.append(f"lines with a viewport narrower than 500 px: {sum(r['viewport'][0] < 500 for r in rows)}")
        fam: Counter = Counter()
        for r in rows:
            for f in r["font"]:
                fam["Open Sans (stand-in)" if f == "HeldOutStandIn" else f] += 1
        out += ["", "Font families the browser used (lines containing each):", ""]
        out.append(", ".join(f"{k} {v}" for k, v in fam.most_common(60)))
        digits = sum(any(c.isdigit() for c in r["text"]) for r in rows)
        upper = sum(r["text"].isupper() for r in rows)
        out.append(f"\nlines with a digit: {digits}; all-capital lines: {upper}")
    out += ["", f"Run: {run.get('seconds')} s on {run.get('workers')} workers; urls sha256 {run.get('urls_sha256')}"]
    text = "\n".join(out) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
