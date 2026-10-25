# ruff: noqa: E501
"""How the engine's line boxes differ from truth line boxes on a page set with word-box truth, and line sets to read both.

    uv run python -m eval.receipt_crops --pages DIR --truth DIR --engine DIR --out DIR

PAGES/<id>.png, TRUTH/<id>.json ({"words": [{text, box}]}), ENGINE/<id>.page.json (the engine's page model). A truth line is a
group of truth words on one row (`eval.page_lines.group_lines`); an engine line is *complete* when its words fall on the words of
exactly one truth line and cover at least 80% of them (the rest are partial, merged or unmatched and are counted, not compared).

Writes OUT/truth-lines.jsonl and OUT/engine-lines.jsonl, line records for `LineGen --page-lines` with the same ids and the same
(truth) text, one cut from the truth boxes and one from the engine's line boxes, so a recognizer reads the same words from either
crop; and prints, over the complete lines, the engine line's height against the ink height of the truth line, its offsets to
the truth ink in line heights, how far the crop the library cuts (ink +-H/4 vertically, +-H/2 horizontally) reaches into the
neighbouring line's ink rows, and the page tilt from the truth quads.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from eval.page_lines import drop_wrapped, group_lines, page_skew
from eval.recall_attribution import attribute_page  # puts bench/ on the path


def ink_box(px: np.ndarray, box: list[float]) -> tuple[int, int, int, int]:
    """Ink box inside a truth box, the rule `LineGen --page-lines` uses (densest row run, gaps of up to 2 rows bridged)."""
    h, w = px.shape
    bx0, bx1 = max(0, min(int(math.floor(box[0])), w - 1)), max(1, min(int(math.ceil(box[2])), w))
    by0, by1 = max(0, min(int(math.floor(box[1])), h - 1)), max(1, min(int(math.ceil(box[3])), h))
    sub = px[by0:by1, bx0:bx1].astype(int)
    if sub.size == 0:
        return bx0, by0, bx1, by1
    thr = (sub.min() + sub.max()) // 2
    dark = sub < thr
    rows = dark.sum(axis=1)
    peak = int(rows.argmax())
    if rows[peak] == 0:
        return bx0, by0, bx1, by1
    min_row = max(1, int(0.04 * rows[peak]))
    top = bot = peak
    gap, y = 0, peak - 1
    while y >= 0 and gap <= 2:
        if rows[y] >= min_row:
            top, gap = y, 0
        else:
            gap += 1
        y -= 1
    gap, y = 0, peak + 1
    while y < len(rows) and gap <= 2:
        if rows[y] >= min_row:
            bot, gap = y, 0
        else:
            gap += 1
        y += 1
    cols = np.where(dark[top : bot + 1].any(axis=0))[0]
    if len(cols) == 0:
        return bx0, by0, bx1, by1
    return bx0 + int(cols[0]), by0 + top, bx0 + int(cols[-1]) + 1, by0 + bot + 1


def engine_lines(page_json: Path) -> list[dict]:
    pg = json.loads(page_json.read_text())["pages"][0]
    return [{"box": ln["bounds"], "words": [w["bounds"] for w in ln["words"]], "text": ln["text"]}
            for b in pg["blocks"] for ln in b["lines"]]  # fmt: skip


def inside(wb: list[float], tb: list[float]) -> bool:
    cx, cy = (wb[0] + wb[2]) / 2, (wb[1] + wb[3]) / 2
    return tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]


def quantiles(v: list[float]) -> str:
    if not v:
        return "n/a"
    a = np.array(v)
    return f"p10 {np.percentile(a, 10):.2f}  median {np.median(a):.2f}  p90 {np.percentile(a, 90):.2f}"


def main() -> None:
    from oneocr_bench.score import normalize  # on the path once eval.recall_attribution is imported
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_rows, e_rows = [], []
    st: dict[str, list[float]] = {k: [] for k in ("h_ratio", "top", "bottom", "left", "right", "t_contam", "e_contam")}
    n_t = n_complete = n_partial = n_other = 0
    skews: list[float] = []
    status_cost: dict[str, int] = {}
    status_words: dict[str, int] = {}
    ref_chars = 0
    for tp in sorted(Path(a.truth).glob("*.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        pid = t.get("id", tp.stem)
        ep = Path(a.engine) / f"{pid}.page.json"
        if not ep.exists():
            continue
        page = next(Path(a.pages).glob(pid + ".*"))
        px = np.array(Image.open(page).convert("L"))
        words = t["words"]
        if all("quad" in w for w in words):
            skews.append(abs(math.degrees(page_skew(words))))
        tlines, _ = drop_wrapped(words, group_lines(words))
        elines = engine_lines(ep)
        t_ink = [ink_box(px, [min(w["box"][0] for w in ln), min(w["box"][1] for w in ln), max(w["box"][2] for w in ln), max(w["box"][3] for w in ln)]) for ln in tlines]
        n_t += len(tlines)
        word_line = {id(w): i for i, ln in enumerate(tlines) for w in ln}
        used: set[int] = set()
        matched: dict[int, dict] = {}
        for e in elines:
            hits: dict[int, int] = {}
            miss = 0
            for wb in e["words"]:
                found = None
                for w in words:
                    if id(w) in word_line and inside(wb, w["box"]):
                        found = w
                        break
                if found is None:
                    miss += 1
                else:
                    hits[word_line[id(found)]] = hits.get(word_line[id(found)], 0) + 1
            if len(hits) == 1 and miss == 0:
                ti = next(iter(hits))
                if hits[ti] >= 0.8 * len(tlines[ti]) and ti not in used and hits[ti] == len(tlines[ti]) == len(e["words"]):
                    used.add(ti)
                    matched[ti] = e
                    continue
                n_partial += 1
            else:
                n_other += 1
        n_complete += len(matched)
        # recall-CER misses of the truth words, by the kind of engine line that holds the word
        engine_of: dict[int, str] = {}
        for ti, ln in enumerate(tlines):
            for w in ln:
                engine_of[id(w)] = "complete line" if ti in matched else "line cut differently"
        for w in words:
            engine_of.setdefault(id(w), "line cut differently")
        ref_chars += sum(len(x) for w in words for x in normalize(w["text"]))
        hw = [{"text": wd["text"], "box": wd["bounds"]} for pg in json.loads(ep.read_text())["pages"][:1] for b in pg["blocks"] for ln in b["lines"] for wd in ln["words"]]
        for _t, w, miss, cls, _tx, _p in attribute_page(words, hw):
            if miss:
                key = "complete line" if engine_of.get(id(w)) == "complete line" else f"cut differently, {cls}"
                status_cost[key] = status_cost.get(key, 0) + miss
                status_words[key] = status_words.get(key, 0) + 1
        for ti, e in matched.items():
            if e["box"][2] - e["box"][0] < 4 or e["box"][3] - e["box"][1] < 4:
                continue
            ln = tlines[ti]
            text = " ".join(w["text"] for w in ln)
            lid = f"{pid}-l{ti:03d}"
            tb = [min(w["box"][0] for w in ln), min(w["box"][1] for w in ln), max(w["box"][2] for w in ln), max(w["box"][3] for w in ln)]
            t_rows.append({"id": lid, "text": text, "page": str(page), "box": [round(v, 2) for v in tb]})
            e_rows.append({"id": lid, "text": text, "page": str(page), "box": [round(v, 2) for v in e["box"]]})
            ix0, iy0, ix1, iy1 = t_ink[ti]
            th = max(iy1 - iy0, 1)
            eh = e["box"][3] - e["box"][1]
            st["h_ratio"].append(eh / th)
            st["top"].append((e["box"][1] - iy0) / th)
            st["bottom"].append((e["box"][3] - iy1) / th)
            st["left"].append((e["box"][0] - ix0) / th)
            st["right"].append((e["box"][2] - ix1) / th)
            # share of the library's crop rows (ink +-H/4, H >= 0.85 x page median is ignored here) that fall on other lines' ink rows
            for key, (y0, y1) in (("t_contam", (iy0, iy1)), ("e_contam", (e["box"][1], e["box"][3]))):
                hh = y1 - y0
                c0, c1 = y0 - hh / 4, y1 + hh / 4
                over = 0.0
                for tj, (_, jy0, _, jy1) in enumerate(t_ink):
                    if tj != ti:
                        # another truth line's ink rows reaching into this crop, only where it shares columns with it
                        if min(c1, jy1) > max(c0, jy0) and min(ix1, t_ink[tj][2]) > max(ix0, t_ink[tj][0]):
                            over += min(c1, jy1) - max(c0, jy0)
                st[key].append(over / max(c1 - c0, 1))
    print(f"{n_t} truth lines; engine lines complete {n_complete} ({100 * n_complete / max(n_t, 1):.1f}%), "
          f"partial/other {n_partial}/{n_other}")  # fmt: skip
    print(f"page tilt from truth quads, degrees: {quantiles(skews)}; share above 0.5: {100 * np.mean(np.array(skews) > 0.5):.0f}%")
    print("recall-CER misses by the line the truth word sits in (pp of all truth characters):")
    for k in sorted(status_cost):
        print(f"  {k}: {status_words[k]} words, {status_cost[k]} chars, {100 * status_cost[k] / max(ref_chars, 1):.2f} pp")
    labels = {"h_ratio": "engine line height / truth ink height", "top": "engine top - truth ink top (ink heights)",
              "bottom": "engine bottom - truth ink bottom", "left": "engine left - truth ink left", "right": "engine right - truth ink right",
              "t_contam": "crop rows on other lines' ink, truth crop", "e_contam": "crop rows on other lines' ink, engine crop"}  # fmt: skip
    for k, lab in labels.items():
        print(f"  {lab}: {quantiles(st[k])}")
    (out / "truth-lines.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in t_rows), encoding="utf-8")
    (out / "engine-lines.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in e_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
