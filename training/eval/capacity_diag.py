# ruff: noqa: E501
"""Where a recognizer's line errors come from: character-level edit operations broken down by kind, so capacity-bound
confusions (look-alike glyph pairs) can be told from data gaps (spaces, punctuation, case, rare classes, particular fonts or sizes).

    uv run python -m eval.capacity_diag --hyps HYPS_DIR --sets a,b,c [--labels name=DIR ...] [--curve metrics.jsonl ...] --out report.md

HYPS_DIR/<set>.hyps.jsonl comes from eval.model_hyps. Optional --labels maps a set name to its line-set directory so the
per-font error rate can be reported (labels.jsonl carries the font file). --curve prints the validation curve of earlier runs: a
curve that is still falling at the end of training says "train longer" rather than "bigger".
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# glyph pairs a small network is expected to find hard whatever the data: they differ by a few pixels at 32 px height
LOOKALIKE = [
    set(s) for s in ("Il1|", "O0Q", "rn:m", "cl:d", "vu", "VW", "5S", "8B", "2Z", "gq9", "ao", "ec", ".,;:", "'`´’", "-–—‐−", '"”“„')
]


def align(ref: str, hyp: str) -> list[tuple[str, str, str]]:
    """Levenshtein alignment; ops are (kind, ref_char, hyp_char) with kind in match/sub/ins/del."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("match" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    return ops[::-1]


def klass(c: str) -> str:
    if c == " ":
        return "space"
    cat = unicodedata.category(c)
    if cat.startswith("L"):
        return "letter"
    if cat.startswith("N"):
        return "digit"
    if cat.startswith("P") or cat.startswith("S"):
        return "punct"
    return "other"


def lookalike(a: str, b: str) -> bool:
    if a.lower() == b.lower() and a != b:
        return True
    if unicodedata.normalize("NFD", a)[0] == unicodedata.normalize("NFD", b)[0]:
        return True  # accent only
    return any(a in s and b in s for s in LOOKALIKE)


def analyse(rows: list[dict], fonts: dict[str, str] | None) -> dict:
    kinds: Counter = Counter()
    conf: Counter = Counter()
    by_font: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_len: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    tot = 0
    err_lines = 0
    for r in rows:
        ref, hyp = r["ref"], r["hyp"]
        tot += len(ref)
        e = 0
        if ref != hyp:
            err_lines += 1
            for kind, a, b in align(ref, hyp):
                if kind == "match":
                    continue
                e += 1
                if kind == "sub":
                    if a.lower() == b.lower():
                        tag = "sub:case"
                    elif lookalike(a, b):
                        tag = "sub:lookalike"
                    else:
                        tag = f"sub:{klass(a)}->{klass(b)}"
                    conf[(a, b)] += 1
                else:
                    c = a if kind == "del" else b
                    tag = f"{kind}:{klass(c)}"
                    conf[(a or "∅", b or "∅")] += 1
                kinds[tag] += 1
        if fonts and r["id"] in fonts:
            f = by_font[fonts[r["id"]]]
            f[0] += e
            f[1] += len(ref)
        b = by_len["<=8" if len(ref) <= 8 else "<=24" if len(ref) <= 24 else ">24"]
        b[0] += e
        b[1] += len(ref)
    return {"chars": tot, "lines": len(rows), "err_lines": err_lines, "kinds": kinds, "conf": conf, "font": by_font, "len": by_len}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--labels", nargs="*", default=[], help="name=DIR of the line set (for per-font rates)")
    ap.add_argument("--curve", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    labels = dict(x.split("=", 1) for x in a.labels)
    out: list[str] = []
    for s in a.sets.split(","):
        rows = [json.loads(line) for line in (Path(a.hyps) / f"{s}.hyps.jsonl").read_text(encoding="utf-8").splitlines() if line]
        fonts = None
        if s in labels:
            fonts = {}
            for line in (Path(labels[s]) / "labels.jsonl").read_text(encoding="utf-8").splitlines():
                if line:
                    j = json.loads(line)
                    fonts[j["id"]] = j.get("font", "?")
        r = analyse(rows, fonts)
        n = r["chars"]
        total = sum(r["kinds"].values())
        out.append(
            f"## {s}: {r['lines']} lines, {n} chars, CER {100 * total / n:.2f}%, lines with an error {100 * r['err_lines'] / r['lines']:.1f}%\n"
        )
        out.append("| error kind | count | share of errors | pp of CER |\n|---|---:|---:|---:|")
        for k, v in r["kinds"].most_common():
            out.append(f"| {k} | {v} | {100 * v / max(total, 1):.1f}% | {100 * v / n:.3f} |")
        out.append("\ntop confusions (ref→hyp): " + ", ".join(f"{a!r}→{b!r}×{v}" for (a, b), v in r["conf"].most_common(14)))
        out.append("\nCER by line length: " + ", ".join(f"{k}: {100 * v[0] / max(v[1], 1):.2f}%" for k, v in sorted(r["len"].items())))
        if r["font"]:
            worst = sorted(((v[0] / max(v[1], 1), k, v[1]) for k, v in r["font"].items() if v[1] >= 300), reverse=True)[:8]
            out.append(
                "\nworst fonts (CER, ≥300 chars): " + "; ".join(f"{k.split('/')[-1]} {100 * c:.1f}% ({m} chars)" for c, k, m in worst)
            )
        out.append("")
    for c in a.curve:
        rows = [json.loads(line) for line in Path(c).read_text(encoding="utf-8").splitlines() if line]
        ev = [x for x in rows if "val" in x and isinstance(x["val"], dict)]
        out.append(f"## validation curve {c}\n")
        for x in ev[:: max(1, len(ev) // 14)] + ev[-1:]:
            out.append(f"- step {x.get('step')}: val CER {100 * x['val']['cer']:.3f}%")
        out.append("")
    Path(a.out).write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
