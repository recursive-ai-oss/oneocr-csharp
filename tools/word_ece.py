"""Page-level word ECE for the engine's word sidecars.

    python tools/word_ece.py TRUTH_DIR OUTPUT_DIR [--bins 10] [--json result.json]

TRUTH_DIR holds `<id>.txt` (the scorer's reading-order truth). OUTPUT_DIR holds `<id>.json` sidecars written by
`OneOcr.Bench run --words 1`: {"words": [{"text": ..., "conf": calibrated, "conf_raw": raw}, ...]}.

A hypothesis word is counted as correct when its normalized text equals a still-unclaimed truth word of the same page (order
free, each truth word claimed once, the scorer's own normalization). Expected calibration error is the usual equal-width-bin
sum over bins of (bin share) * |accuracy - mean confidence|. Truth that omits printed words (receipts) makes read-but-unannotated
words count as wrong, so receipts ECE is pessimistic; read it as an upper bound.

This lives outside bench/ on purpose: the benchmark is frozen, and this only reads it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from oneocr_bench.score import normalize  # noqa: E402  (the scorer's own text normalization)


def collect(truth_dir: Path, out_dir: Path) -> list[tuple[float, float, int]]:
    """(calibrated, raw, correct) for every hypothesis word."""
    rows = []
    for t in sorted(truth_dir.glob("*.txt")):
        side = out_dir / f"{t.stem}.json"
        if not side.exists():
            raise SystemExit(f"missing word sidecar {side}")
        bag = Counter(normalize(t.read_text(encoding="utf-8")))
        for w in json.loads(side.read_text(encoding="utf-8"))["words"]:
            toks = normalize(w["text"])
            if not toks:
                continue
            key = " ".join(toks)
            ok = 0
            if bag[key] > 0:
                bag[key] -= 1
                ok = 1
            rows.append((float(w["conf"]), float(w["conf_raw"]), ok))
    return rows


def ece(pairs: list[tuple[float, int]], bins: int) -> float:
    if not pairs:
        return 0.0
    count = [0] * bins
    conf = [0.0] * bins
    acc = [0] * bins
    for c, ok in pairs:
        b = min(int(c * bins), bins - 1)
        count[b] += 1
        conf[b] += c
        acc[b] += ok
    n = len(pairs)
    return sum(count[b] / n * abs(acc[b] / count[b] - conf[b] / count[b]) for b in range(bins) if count[b])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("out")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = collect(Path(a.truth), Path(a.out))
    n = len(rows)
    res = {
        "words": n,
        "exact_match_rate": sum(r[2] for r in rows) / n if n else 0.0,
        "ece_calibrated": ece([(r[0], r[2]) for r in rows], a.bins),
        "ece_raw": ece([(r[1], r[2]) for r in rows], a.bins),
        "bins": a.bins,
    }
    print(json.dumps(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
