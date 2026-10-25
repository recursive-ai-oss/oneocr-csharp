# ruff: noqa: E501
"""What a hypothetical word filter would do on the recognized words: for each (max confidence, max length) say how many
recognized words it drops that sit on no truth word (extras: a gain), on a correctly read truth word (a loss), and on a
misread truth word (a small gain or loss). Counts are words and characters, per set.

Run: python eval/filter_sim.py --eval BENCH_ROOT --out ENGINE_OUTPUT_ROOT [--sets ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from page_causes import SETS, aabb, area, inter, norm


def collect(eval_root: Path, out_root: Path, name: str):
    rows = []  # (conf, length, height, kind) kind: extra | right | wrong
    for tf in sorted((eval_root / name / "truth").glob("*.json")):
        t = json.loads(tf.read_text())
        hp = out_root / name / tf.name
        hw = json.loads(hp.read_text())["words"] if hp.exists() else []
        tb = [aabb(w.get("ink_quad") or w.get("ink") or w["box"]) for w in t["words"] if norm(w["text"])]
        tt = [norm(w["text"]) for w in t["words"] if norm(w["text"])]
        for w in hw:
            hb = [float(v) for v in w["box"]]
            text = norm(w["text"])
            cov = [(inter(hb, b) / max(area(hb), 1e-9), k) for k, b in enumerate(tb)]
            cov = [c for c in cov if c[0] > 0.3]
            if not cov:
                kind = "extra"
            else:
                kind = "right" if any(tt[k] == text for _, k in cov) else "wrong"
            rows.append((float(w.get("conf", 1)), len(text), hb[3] - hb[1], kind, all(not ch.isalnum() for ch in text)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=SETS)
    a = ap.parse_args()
    for s in a.sets:
        rows = collect(Path(a.eval), Path(a.out), s)
        print(f"== {s}: {len(rows)} recognized words; extras {sum(r[3] == 'extra' for r in rows)}, wrong {sum(r[3] == 'wrong' for r in rows)}")
        for conf in (0.5, 0.9, 0.99, 1.01):
            sel = [r for r in rows if r[4] and r[0] < conf]
            c = {k: sum(1 for r in sel if r[3] == k) for k in ("extra", "right", "wrong")}
            print(f"  punctuation-only words, conf<{conf}: drops extra {c['extra']:4d} | right {c['right']:4d} | wrong {c['wrong']:4d}")
        for conf in (0.3, 0.5, 0.7, 0.9):
            for ln in (1, 2, 99):
                sel = [r for r in rows if r[0] < conf and r[1] <= ln]
                c = {k: sum(1 for r in sel if r[3] == k) for k in ("extra", "right", "wrong")}
                print(f"  conf<{conf} len<={ln:2d}: drops extra {c['extra']:4d} | right {c['right']:4d} | wrong {c['wrong']:4d}")
