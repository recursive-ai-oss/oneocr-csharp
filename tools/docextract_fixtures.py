"""Parity fixtures for the document-extraction forward pass, including pages longer than one window.

Run from the `training/` directory of a checkout that has the docextract modules (`models/docextract.py`, `export/docextract.py`,
`models/docextract_assemble.py`, `parity/docextract_fixtures.py`):

    uv run python ../tools/docextract_fixtures.py --seed 7 --out OUT_DIR

It exports an untrained model (random weights with non-trivial layer-norm parameters and biases), then writes `OUT_DIR/model.onm`,
`OUT_DIR/fixtures.json` and `OUT_DIR/data/*.f32` in the layout of `parity/docextract_fixtures.py`, and for every sample also the
pointer arrays (`group_ptr`, `kv_ptr`, global word indices) and the expected assembled JSON. Samples:

  words-24    random characters, unknown characters, 32-character words, every layer stored
  words-64    every layer stored
  words-150   heads only
  words-300   heads only
  words-513   two windows (512 + 129 words): the shortest page that needs windowing
  words-900   three windows

Windowing (docs/spec/docextract-v0.md): a page of more than `max_words` (512) words is run in windows of 512 words starting every 384
words; each word takes its tags, roles and pointer row from the window whose centre is nearest the word (the lower window on a tie);
pointers are windows-local indices shifted to page indices; the document-type logits are the mean of the windows' logits weighted by the
number of words each window owns.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from export.docextract import Reference, export, random_model, read_back
from models.docextract import CLASSES, DOC_TYPES, featurize
from models.docextract_assemble import ALLOWED, assemble
from parity.docextract_fixtures import FULL, HEADS, random_page, save

WINDOW, STRIDE = 512, 384


def windows(n: int) -> list[tuple[int, int]]:
    starts = [0]
    while starts[-1] + WINDOW < n:
        starts.append(starts[-1] + STRIDE)
    return [(s, min(s + WINDOW, n)) for s in starts]


def owner(n: int, wins: list[tuple[int, int]]) -> list[int]:
    out = []
    for w in range(n):
        best, bd = 0, None
        for k, (s, e) in enumerate(wins):
            d = abs(w + 0.5 - (s + e) / 2)
            if bd is None or d < bd:
                best, bd = k, d
        out.append(best)
    return out


def run_windowed(ref: Reference, chars, feats, bins, geo) -> dict:
    n = chars.shape[0]
    wins = windows(n)
    own = owner(n, wins)
    tags = torch.zeros(n, ref.t["tag_head.weight"].shape[0], dtype=ref.dtype)
    roles = torch.zeros(n, ref.t["role_head.weight"].shape[0], dtype=ref.dtype)
    ptr = np.zeros((2, n), dtype=np.int64)
    margin = np.zeros((2, n))
    owned = [0] * len(wins)
    docs = []
    for k, (s, e) in enumerate(wins):
        with torch.no_grad():
            o = ref.forward(chars[s:e], feats[s:e], bins[s:e], geo[s:e])
        docs.append(o["doc"].double())
        for w in range(s, e):
            if own[w] != k:
                continue
            tags[w] = o["tags"][w - s]
            roles[w] = o["roles"][w - s]
            for r in range(2):
                row = o["rel"][r, w - s].double().numpy()
                top = np.argsort(-row, kind="stable")[:2]
                ptr[r, w] = s + int(top[0])
                margin[r, w] = float(row[top[0]] - row[top[1]]) if len(row) > 1 else 1e9
            owned[k] += 1
    # document logits: weighted mean of the windows' logits, accumulated in window order
    acc = torch.zeros_like(docs[0])
    for k in range(len(wins)):
        acc += docs[k] * owned[k]
    doc = (acc / n).to(ref.dtype)
    return {"tags": tags, "roles": roles, "doc": doc, "ptr": ptr, "margin": margin, "windows": wins}


def pointer_matrix(ptr: np.ndarray, n: int) -> np.ndarray:
    rel = np.zeros((2, n, n), dtype=np.float32)
    for r in range(2):
        rel[r, np.arange(n), ptr[r]] = 1.0
    return rel


def sample(out: Path, sid: str, texts, boxes, w, h, ref32: Reference, ref64: Reference, names) -> dict:
    f = featurize(texts, boxes, w, h)
    chars = torch.from_numpy(f["chars"].astype(np.int64))
    feats, bins, geo = (torch.from_numpy(f[k]) for k in ("feats", "bins", "geo"))
    n = len(texts)
    entries, floors = [], {}
    wins = windows(n)
    if len(wins) == 1:
        with torch.no_grad():
            o32 = ref32.forward(chars, feats, bins, geo)
            o64 = ref64.forward(chars, feats.double(), bins, geo.double())
        for nme in names:
            save(out, sid, nme, o32[nme], entries, floors, o64[nme])
            f64 = out / "data" / f"{sid}.{nme}.f64"  # the float64 reference itself, for checking our fp32 error against the exact answer
            f64.write_bytes(o64[nme].numpy().astype("<f8").tobytes())
            entries[-1]["file64"] = f"data/{f64.name}"
        res = {k: o32[k].numpy() for k in ("tags", "roles", "doc", "rel")}
        ptr = np.stack([res["rel"][r].argmax(1) for r in range(2)])
        margin = np.zeros((2, n))
        for r in range(2):
            srt = np.sort(res["rel"][r].astype(np.float64), axis=1)
            margin[r] = srt[:, -1] - srt[:, -2] if n > 1 else 1e9
        assembled = assemble(res, texts, np.asarray(boxes, dtype=np.float64))
    else:
        m = run_windowed(ref32, chars, feats, bins, geo)
        for nme in ("tags", "roles", "doc"):
            save(out, sid, nme, m[nme], entries, floors, m[nme].double())  # no fp64 floor for merged windows
        ptr, margin = m["ptr"], m["margin"]
        res = {"tags": m["tags"].numpy(), "roles": m["roles"].numpy(), "doc": m["doc"].numpy(), "rel": pointer_matrix(ptr, n)}
        assembled = assemble(res, texts, np.asarray(boxes, dtype=np.float64))
    for nme, arr in (("feats", feats), ("geo", geo)):
        (out / "data" / f"{sid}.in_{nme}.f32").write_bytes(arr.numpy().astype("<f4").tobytes())
    return {
        "id": sid,
        "width": w,
        "height": h,
        "windows": [list(x) for x in wins],
        "words": [{"text": t, "box": b} for t, b in zip(texts, boxes, strict=True)],
        "featurize": {
            "unit": f["unit"],
            "chars": f["chars"].tolist(),
            "bins": f["bins"].tolist(),
            "feats_file": f"data/{sid}.in_feats.f32",
            "geo_file": f"data/{sid}.in_geo.f32",
        },
        "layers": entries,
        "fp32_vs_fp64_floor": floors,
        "group_ptr": ptr[0].tolist(),
        "kv_ptr": ptr[1].tolist(),
        "ptr_margin": [margin[0].tolist(), margin[1].tolist()],
        "assembled": assembled,
    }


def assembly_sample(out: Path, rng: random.Random, sid: str, n: int, doc: str) -> dict:
    """Assembly only: logits with planned structure (multi-word spans, line items, tax rows, key/value pairs) and the expected JSON."""
    g = np.random.default_rng(rng.randrange(1 << 30))
    texts, boxes, w, h = random_page(rng, n)
    pool = [c for c, name in enumerate(CLASSES) if name in ALLOWED[doc]] or list(range(len(CLASSES)))
    ntags = 1 + 2 * len(CLASSES)
    tags = g.normal(0, 1.0, (n, ntags)).astype(np.float32)
    roles = g.normal(0, 1.0, (n, 5)).astype(np.float32)
    prev = None
    for i in range(n):
        r = rng.random()
        if r < 0.45:
            prev = None
            t = 0
        elif prev is not None and r < 0.75:
            t = 2 + 2 * prev
        else:
            prev = rng.choice(pool)
            t = 1 + 2 * prev
        tags[i, t] += 5.0
    role_seq = []
    i = 0
    keys = []
    while i < n:
        r = rng.random()
        if r < 0.5:
            role_seq.append(0)
            i += 1
        else:
            klen = rng.randint(1, 2)
            vlen = rng.randint(1, 3)
            if i + klen + vlen > n:
                break
            keys.append(i)
            role_seq += [1] + [2] * (klen - 1) + [3] + [4] * (vlen - 1)
            i += klen + vlen
    role_seq += [0] * (n - len(role_seq))
    for i, t in enumerate(role_seq):
        roles[i, t] += 5.0
    rel = g.normal(0, 1.0, (2, n, n)).astype(np.float32)
    anchors = sorted(rng.sample(range(n), min(n, 5)))
    for i in range(n):
        a = rng.choice(anchors)
        rel[0, i, a if rng.random() < 0.8 else rng.randrange(n)] += 6.0
        rel[1, i, rng.choice(keys) if keys else i] += 6.0
    dl = g.normal(0, 1.0, 3).astype(np.float32)
    dl[DOC_TYPES.index(doc)] += 4.0
    for nme, arr in ("tags", tags), ("roles", roles), ("doc", dl), ("rel", rel):
        (out / "data" / f"{sid}.logits_{nme}.f32").write_bytes(arr.astype("<f4").tobytes())
    res = {"tags": tags, "roles": roles, "doc": dl, "rel": rel}
    return {
        "id": sid,
        "assemblyOnly": True,
        "doc": doc,
        "words": [{"text": t, "box": b} for t, b in zip(texts, boxes, strict=True)],
        "assembled": assemble(res, texts, np.asarray(boxes, dtype=np.float64)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    model = random_model(a.seed)
    export(model, out / "model.onm", "0.0.0-random", {"random_init_seed": a.seed})
    manifest, t = read_back(out / "model.onm")
    ref32 = Reference(manifest["config"], t)
    ref64 = Reference(manifest["config"], t, torch.float64)
    rng = random.Random(a.seed)
    samples = []
    for sid, n, names in (
        ("words-24", 24, FULL + HEADS),
        ("words-64", 64, FULL + HEADS),
        ("words-150", 150, HEADS),
        ("words-300", 300, HEADS),
        ("words-513", 513, ()),
        ("words-900", 900, ()),
    ):
        tx, bx, w, h = random_page(rng, n)
        samples.append(sample(out, sid, tx, bx, w, h, ref32, ref64, names))
    for k, (n, kind) in enumerate([(80, "invoice"), (120, "receipt"), (60, "other"), (200, "invoice"), (200, "receipt")]):
        samples.append(assembly_sample(out, rng, f"assembly-{k}-{kind}", n, kind))
    doc = {"architecture": manifest["architecture"], "version": manifest["version"], "window": WINDOW, "stride": STRIDE, "samples": samples}
    (out / "fixtures.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(len(samples), "samples")


if __name__ == "__main__":
    main()
