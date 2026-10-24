"""The line-level gate table: our recognizer against Tesseract on identical crops, per category.

    uv run python -m eval.gate --ours HYPS_DIR --tess TESS_DIR --out gate.md [--json gate.json]

HYPS_DIR/<set>.hyps.jsonl from eval.model_hyps; TESS_DIR/<set>/<model>-psm<N>-x<S>.hyps.jsonl from eval.tess_lines. For each row the
Tesseract configuration (page-segmentation mode and enlargement) with the lowest CER on that row is reported, per model
(best, fast): the engine's best case. Rows are the page categories (crops by truth word boxes, as the library's line finder would
cut them) and, for the held-out synthetic sets, slices by script. Tesseract reads Latin lines with eng, Greek with ell and
Cyrillic with rus. `latin-ext` lines (accents and symbols beyond ASCII) are outside eng's alphabet; the row is shown for
information and is not a gate row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.scoring import read_hyps, score, slice_of

CATEGORIES = [
    ("clean300-v1", "clean 300 DPI print"),
    ("degraded-v1", "degraded scans"),
    ("screenshot-v1", "screenshots and UI"),
    ("layout-v1", "complex layout"),
    ("phone-v1", "phone photos (word crops)"),
    ("receipts-v1", "receipts (word crops)"),
]
SYNTH = [("eval", "held-out fonts + degradation"), ("diag_fonts", "held-out fonts"), ("diag_degrade", "held-out degradation")]


def best_config(tess_dir: Path, model: str, ids: set[str]) -> tuple[str, dict]:
    best = ("", {"cer": 9e9})
    for f in sorted(tess_dir.glob(f"{model}-*.hyps.jsonl")):
        rows = [r for r in read_hyps(f) if r["id"] in ids]
        s = score(rows)
        if s["cer"] < best[1]["cer"]:
            best = (f.name.removesuffix(".hyps.jsonl").split("-", 1)[1], s)
    return best


def row(label: str, ours_rows: list[dict], tess_dir: Path, gate: bool = True) -> dict:
    ids = {r["id"] for r in ours_rows}
    o = score(ours_rows)
    b_cfg, b = best_config(tess_dir, "best", ids)
    f_cfg, f = best_config(tess_dir, "fast", ids)
    if o["cer"] < min(b["cer"], f["cer"]) and o["wer"] < min(b["wer"], f["wer"]):
        verdict = "win"
    elif o["cer"] < min(b["cer"], f["cer"]):
        verdict = "win CER, not WER"
    else:
        verdict = "LOSS"
    if not gate:
        verdict += " (info)"
    return {"row": label, "lines": o["lines"], "ours": o, "verdict": verdict,
            "tess_best": b | {"config": b_cfg}, "tess_fast": f | {"config": f_cfg}}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--tess", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    ours, tess = Path(a.ours), Path(a.tess)
    rows = []
    for name, label in CATEGORIES:
        rows.append(row(f"{label} [{name}]", read_hyps(ours / f"{name}.hyps.jsonl"), tess / name))
    for name, label in SYNTH:
        allr = read_hyps(ours / f"{name}.hyps.jsonl")
        groups: dict[str, list[dict]] = {}
        for r in allr:
            groups.setdefault(slice_of(r["ref"]), []).append(r)
        for sl in ("latin-ascii", "latin-ext", "greek", "cyrillic"):
            if sl in groups:
                rows.append(row(f"{label} [{name}] {sl}", groups[sl], tess / name, gate=sl != "latin-ext"))
    pct = lambda v: f"{100 * v:.2f}"  # noqa: E731
    head = "| category | lines | ours CER % | ours WER % | Tesseract best CER % | best WER % | best config"
    head += " | Tesseract fast CER % | fast WER % | fast config | result |"
    lines = [head, "|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|"]
    for r in rows:
        o, b, f = r["ours"], r["tess_best"], r["tess_fast"]
        lines.append(f"| {r['row']} | {r['lines']} | {pct(o['cer'])} | {pct(o['wer'])} | {pct(b['cer'])} | {pct(b['wer'])} | {b['config']} "
                     f"| {pct(f['cer'])} | {pct(f['wer'])} | {f['config']} | {r['verdict']} |")  # fmt: skip
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
