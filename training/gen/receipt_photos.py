# ruff: noqa: E501
"""Receipt photos with transcriptions: split by image, bench-format page sets, and engine-line labels.

Reads a local directory of receipt photos, one label JSON per photo (`{"readable": bool, "lines": [{"text", "sure"}]}`, one entry per
printed line top to bottom; unreadable photos have no lines) and, for `align`, the engine's page model per photo
(`OneOcr.Bench run --pagejson 1`). The photos and labels are generated and transcribed outside this repository and are not
redistributed; see `datasets/LICENCES.md`.

    uv run python -m gen.receipt_photos split    --images DIR --labels DIR --out DIR [--dev 150 --calib 100 --seed S]
    uv run python -m gen.receipt_photos pageset  --images DIR --labels DIR --split DIR/split.json --part dev --out DIR/receipts-gen-dev
    uv run python -m gen.receipt_photos align    --images DIR --labels DIR --split DIR/split.json --engine DIR --part train --out DIR/lines

`split` writes split.json (image ids and SHA-256 per part: dev, calib, train; assignment by hash, so adding photos never moves one).
`pageset` writes pages/ and truth/ in the evaluation-set layout: `truth/<id>.txt` has the confidently transcribed lines (an engine
is scored on recall, so lines the transcriber was unsure of, and text it left out, never count against it).
`align` pairs each line the engine found with one transcribed line and writes `lines.jsonl` (records for `LineGen --page-lines`:
id, text, page, box) for the crops the engine itself cuts; tilted lines are rectified first, as the engine reads them. A line is
labelled with the whole transcribed line when the two are alike (similarity >= --min-sim), or, when the engine's line is one column
of a transcribed line, with the word-aligned part that matches it. Ties between two transcribed lines, matches to lines the
transcriber was unsure of, and lines nothing resembles are dropped, and every stage is counted in `stats.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import partial_ratio_alignment

PARTS = ("dev", "calib", "train")


def norm(t: str) -> str:
    return " ".join(unicodedata.normalize("NFC", t).split())


def load_labels(d: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(d.glob("*.json")):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        out[f.stem] = {"readable": bool(j.get("readable")), "lines": [{"text": norm(ln.get("text", "")), "sure": bool(ln.get("sure"))} for ln in j.get("lines", []) if norm(ln.get("text", ""))]}
    return out


def find_image(images: Path, pid: str) -> Path:
    return next(p for p in sorted(images.glob(pid + ".*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_split(a: argparse.Namespace) -> None:
    labels = load_labels(Path(a.labels))
    images = Path(a.images)
    ok = [k for k, v in labels.items() if v["readable"] and any(ln["sure"] for ln in v["lines"])]
    order = sorted(ok, key=lambda k: hashlib.sha256(f"{a.seed}:{k}".encode()).hexdigest())
    parts = {"dev": order[: a.dev], "calib": order[a.dev : a.dev + a.calib], "train": order[a.dev + a.calib :]}
    out = {"seed": a.seed, "dropped_unreadable": len(labels) - len(ok), "parts": {p: [{"id": k, "sha256": sha256(find_image(images, k))} for k in sorted(v)] for p, v in parts.items()}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "split.json").write_text(json.dumps(out, indent=1) + "\n")
    print({p: len(v) for p, v in parts.items()}, "unreadable dropped", out["dropped_unreadable"])


def part_ids(split: str, part: str) -> list[str]:
    return [r["id"] for r in json.loads(Path(split).read_text())["parts"][part]]


def cmd_pageset(a: argparse.Namespace) -> None:
    labels = load_labels(Path(a.labels))
    out = Path(a.out)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    for pid in part_ids(a.split, a.part):
        src = find_image(Path(a.images), pid)
        dst = out / "pages" / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        sure = [ln["text"] for ln in labels[pid]["lines"] if ln["sure"]]
        (out / "truth" / f"{pid}.txt").write_text("\n".join(sure) + "\n", encoding="utf-8")
    print(f"{len(part_ids(a.split, a.part))} pages -> {out}")


def engine_lines(page_json: Path) -> list[dict]:
    pg = json.loads(page_json.read_text())["pages"][0]
    return [{"text": norm(ln["text"]), "box": ln["bounds"], "quad": ln.get("quad")} for b in pg["blocks"] for ln in b["lines"] if norm(ln["text"])]


def candidates(etext: str, ref: list[dict], min_sim: float) -> list[tuple[float, int, int, int]]:
    """Per transcribed line, the best label for the engine's text: the whole line or, when the line is longer, the word-aligned part
    of it that matches best. (similarity, line index, start, end), best first, one per transcribed line."""
    out = []
    for j, r in enumerate(ref):
        t = r["text"]
        best = (Levenshtein.normalized_similarity(etext, t), 0, len(t))
        if len(t) > len(etext):
            al = partial_ratio_alignment(etext, t)
            if al is not None:
                s0, s1 = al.dest_start, al.dest_end
                while s0 > 0 and t[s0 - 1] != " ":
                    s0 -= 1
                while s1 < len(t) and t[s1] != " ":
                    s1 += 1
                sim = Levenshtein.normalized_similarity(etext, t[s0:s1])
                if sim > best[0]:
                    best = (sim, s0, s1)
        if best[0] >= min_sim:
            out.append((best[0], j, best[1], best[2]))
    return sorted(out, reverse=True)


def match_page(eng: list[dict], ref: list[dict], min_sim: float, min_chars: int) -> tuple[list[tuple[int, str]], dict[str, int]]:
    """(engine line index, label) pairs and a count per outcome. A transcribed span labels at most one engine line; an engine line whose
    two best spans read differently, or whose best span is in a line the transcriber was unsure of, gets no label."""
    n = {"engine_lines": len(eng), "full": 0, "part": 0, "tie": 0, "unsure_ref": 0, "no_match": 0, "short": 0}
    cands = []
    for i, e in enumerate(eng):
        if len(e["text"]) < min_chars:
            n["short"] += 1
            continue
        c = candidates(e["text"], ref, min_sim)
        if not c:
            n["no_match"] += 1
            continue
        if len(c) > 1 and c[1][0] >= c[0][0] - 0.03 and ref[c[1][1]]["text"][c[1][2] : c[1][3]] != ref[c[0][1]]["text"][c[0][2] : c[0][3]]:
            n["tie"] += 1
            continue
        cands.append((c[0][0], i, c))
    taken: dict[int, list[tuple[int, int]]] = {}
    chosen = []
    for _sim, i, c in sorted(cands, key=lambda x: -x[0]):
        for _s, j, s0, s1 in c[:3]:
            if any(s0 < b and a < s1 for a, b in taken.get(j, [])):
                continue
            if not ref[j]["sure"]:
                n["unsure_ref"] += 1
            else:
                taken.setdefault(j, []).append((s0, s1))
                n["full" if (s0, s1) == (0, len(ref[j]["text"])) else "part"] += 1
                chosen.append((i, ref[j]["text"][s0:s1]))
            break
        else:
            n["no_match"] += 1
    return chosen, n


def cmd_align(a: argparse.Namespace) -> None:
    from PIL import Image  # noqa: PLC0415

    from eval.page_lines import word_patch  # noqa: PLC0415

    labels = load_labels(Path(a.labels))
    out = Path(a.out)
    (out / "patches").mkdir(parents=True, exist_ok=True)
    rows = []
    total: dict[str, int] = {}
    for pid in part_ids(a.split, a.part):
        ep = Path(a.engine) / f"{pid}.page.json"
        if not ep.exists():
            continue
        eng = engine_lines(ep)
        chosen, n = match_page(eng, labels[pid]["lines"], a.min_sim, a.min_chars)
        for k, v in n.items():
            total[k] = total.get(k, 0) + v
        src = find_image(Path(a.images), pid)
        g = None
        for i, text in chosen:
            e = eng[i]
            if e["quad"]:
                if g is None:
                    with Image.open(src) as im:
                        g = im.convert("L")
                patch = out / "patches" / f"{pid}-e{i:03d}.png"
                box = word_patch(g, e["quad"], patch)
                rows.append({"id": f"{pid}-e{i:03d}", "text": text, "page": str(patch), "box": [round(v, 2) for v in box], "tilted": True})
            else:
                rows.append({"id": f"{pid}-e{i:03d}", "text": text, "page": str(src), "box": [round(v, 2) for v in e["box"]], "tilted": False})
    (out / "lines.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    total["labelled"] = len(rows)
    total["tilted"] = sum(r["tilted"] for r in rows)
    (out / "stats.json").write_text(json.dumps(total, indent=1) + "\n")
    print(total)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("split", "pageset", "align"):
        p = sub.add_parser(name)
        p.add_argument("--images", required=True)
        p.add_argument("--labels", required=True)
        p.add_argument("--out", required=True)
        if name == "split":
            p.add_argument("--dev", type=int, default=150)
            p.add_argument("--calib", type=int, default=100)
            p.add_argument("--seed", default="receipts-gen-1")
        else:
            p.add_argument("--split", required=True)
            p.add_argument("--part", choices=PARTS, required=True)
        if name == "align":
            p.add_argument("--engine", required=True)
            p.add_argument("--min-sim", type=float, default=0.8)
            p.add_argument("--min-chars", type=int, default=3)
    a = ap.parse_args()
    {"split": cmd_split, "pageset": cmd_pageset, "align": cmd_align}[a.cmd](a)


if __name__ == "__main__":
    main()
