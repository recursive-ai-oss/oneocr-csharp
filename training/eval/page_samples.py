"""Word-confidence samples from page-level engine output, for fitting and checking the confidence map where it is judged.

The line-level calibration split sees crops cut at the truth boxes. The page gate sees words as the whole pipeline produces them:
joined and split words, boxes shifted by the detector, noise filtering. A map fitted only on crops can be off there, most on
receipts, so this turns a page run (`OneOcr.Bench run --words 1`, one `<id>.json` sidecar per page with `text`, `conf`, `conf_raw`)
into the hyps format of `eval.calibrate`, with an explicit `correct` flag per word.

A word is correct when it equals a truth word (the bench scorer's pairing, `oneocr_bench.score.word_outcomes`). Words the truth does
not cover at all ("extra": printed text the transcription left out, or inserted text) are left out by default, as in the scorer's
matched ECE, which is the figure used for sets with partial truth.

    uv run python -m eval.page_samples --truth SET/truth --engine OUT --out page.hyps.jsonl [--ids ids.txt] [--keep-extra]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from oneocr_bench.score import normalize, truth_text, word_outcomes  # noqa: E402


def page_rows(truth_dir: Path, engine_dir: Path, ids: list[str] | None, keep_extra: bool, scorer: str = "v3") -> list[dict]:
    rows = []
    for t in sorted(truth_dir.glob("*.txt")):
        if ids is not None and t.stem not in ids:
            continue
        side = engine_dir / f"{t.stem}.json"
        if not side.exists():
            continue
        words = [w for w in json.loads(side.read_text())["words"] if " ".join(normalize(w["text"]))]
        keys = [" ".join(normalize(w["text"])) for w in words]
        status = word_outcomes(normalize(truth_text(truth_dir, t.stem, scorer)), keys)
        kept = [(w, s) for w, s in zip(words, status, strict=True) if keep_extra or s != "extra"]
        rows.append({
            "id": t.stem, "ref": "", "hyp": " ".join(w["text"] for w, _ in kept),
            "words": [{"text": w["text"], "conf": float(w.get("conf_raw", w["conf"])), "script": "latin"} for w, _ in kept],
            "correct": [s == "exact" for _, s in kept],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", help="text file with page ids to use (whitespace separated); default all")
    ap.add_argument("--keep-extra", action="store_true")
    a = ap.parse_args()
    ids = Path(a.ids).read_text().split() if a.ids else None
    rows = page_rows(Path(a.truth), Path(a.engine), ids, a.keep_extra)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    n = sum(len(r["words"]) for r in rows)
    print(f"{len(rows)} pages, {n} words, accuracy {sum(sum(r['correct']) for r in rows) / max(n, 1):.4f}")


if __name__ == "__main__":
    main()
