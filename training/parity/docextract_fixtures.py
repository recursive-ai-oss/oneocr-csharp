"""Forward parity fixtures for the docextract C# forward pass and assembler.

Usage: uv run python -m parity.docextract_fixtures --model model.onm --data val.pkl [cord-dev.pkl ...] --out OUT_DIR [--seed 5]
  or:  uv run python -m parity.docextract_fixtures --model model.onm --random --out OUT_DIR       (words from a seeded generator)

Layout of OUT_DIR (the recognizer fixtures' layout, with pages instead of lines):
  model.onm            the container the references were computed from
  fixtures.json        samples: id, words (text + box), page size, featurize outputs, layers [{name, shape, file, sha256}], fp32-vs-fp64
                       floor per layer, and for assembled samples the expected assembled JSON
  data/<id>.<layer>.f32  raw little-endian float32, row-major

Samples: three short pages with every layer stored, one 200-word page with only the heads stored (every file stays under 1 MB), one
page of random characters (unknown characters, 32-character words, single characters, ties in the position bins). References run in
fp32 on the container's fp16-rounded tensors; the fp64 run on the same tensors gives the error floor. Page samples are real words
from held-out pages with the labels dropped; nothing in a fixture is a training label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from export.docextract import Reference, read_back
from models.docextract import MAX_CHARS, N_CHARS, featurize
from models.docextract_assemble import assemble

FULL = ("char_pool", "emb", "pair_bias", "block.0", "block.1", "block.2", "block.3", "block.4", "final")
HEADS = ("tags", "roles", "doc", "rel")


def random_page(rng: random.Random, n: int) -> tuple[list[str], list[list[float]], int, int]:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-/€ÄÖÜßéèçŁłΑΒΓαβγДЖЯжя\u4e2d\u2603"
    texts, boxes = [], []
    y = 20.0
    x = 20.0
    for i in range(n):
        k = 32 + 5 if i == 0 else (1 if i % 7 == 0 else rng.randint(1, 14))
        texts.append("".join(rng.choice(alphabet) for _ in range(k)))
        w = 9.0 * min(k, 32)
        if x + w > 760:
            x, y = 20.0, y + 22.0
        boxes.append([x, y, x + w, y + 18.0])
        x += w + 8.0
    return texts, boxes, 800, int(y + 60)


def save(out: Path, sid: str, name: str, a: torch.Tensor, entries: list, floors: dict, a64: torch.Tensor) -> None:
    arr = a.detach().cpu().numpy().astype("<f4")
    f = out / "data" / f"{sid}.{name}.f32"
    f.write_bytes(arr.tobytes())
    assert f.stat().st_size < 1_000_000, f"{f} too large"
    d = (a.double() - a64).abs()
    ref = float(a64.abs().max())
    floors[name] = {"max_abs": float(d.max()), "ref_max_abs": ref, "max_abs_over_ref_max": float(d.max()) / max(ref, 1e-30)}
    entries.append({"name": name, "shape": list(arr.shape), "file": f"data/{f.name}", "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})


def sample(out: Path, sid: str, kind: str, texts, boxes, w, h, ref32: Reference, ref64: Reference, names, assembled: bool) -> dict:
    f = featurize(texts, boxes, w, h)
    chars = torch.from_numpy(f["chars"].astype(np.int64))
    feats, bins, geo = (torch.from_numpy(f[k]) for k in ("feats", "bins", "geo"))
    with torch.no_grad():
        o32 = ref32.forward(chars, feats, bins, geo)
        o64 = ref64.forward(chars, feats.double(), bins, geo.double())
    entries, floors = [], {}
    for nme in names:
        save(out, sid, nme, o32[nme], entries, floors, o64[nme])
    for nme, arr in (("feats", feats), ("geo", geo)):
        (out / "data" / f"{sid}.in_{nme}.f32").write_bytes(arr.numpy().astype("<f4").tobytes())
    rec = {
        "id": sid,
        "kind": kind,
        "width": w,
        "height": h,
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
    }
    if assembled:
        res = {k: o32[k].numpy() for k in ("tags", "roles", "doc", "rel")}
        res["tags"] = res["tags"].astype(np.float32)
        rec["assembled"] = assemble(res, texts, np.asarray(boxes, dtype=np.float64))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    shutil.copy(a.model, out / "model.onm")
    manifest, t = read_back(Path(a.model))
    ref32 = Reference(manifest["config"], t)
    ref64 = Reference(manifest["config"], t, torch.float64)
    rng = random.Random(a.seed)
    pages = []
    for path in a.data:
        d = pickle.load(open(path, "rb"))
        for src in ("generated", "cord"):
            exs = [e for e in d.get(src, {}).get("ocr", []) if len(e["texts"]) >= 64]
            if exs:
                pages.append((src, rng.sample(exs, min(3, len(exs)))))
    samples = []
    picks = [(s, e) for s, es in pages for e in es]
    rng.shuffle(picks)
    for i, (src, e) in enumerate(picks[:3]):
        n = [48, 64, 40][i]
        start = rng.randint(0, max(0, len(e["texts"]) - n))
        samples.append(
            sample(
                out,
                f"page{i}",
                src,
                e["texts"][start : start + n],
                e["boxes"][start : start + n].tolist(),
                e["width"],
                e["height"],
                ref32,
                ref64,
                FULL + HEADS,
                True,
            )
        )
    big = [e for _, es in pages for e in es if len(e["texts"]) >= 200]
    if big:
        e = big[0]
        samples.append(
            sample(
                out,
                "page-large",
                "generated",
                e["texts"][:200],
                e["boxes"][:200].tolist(),
                e["width"],
                e["height"],
                ref32,
                ref64,
                HEADS,
                True,
            )
        )
    tx, bx, w, h = random_page(rng, 24)
    samples.append(sample(out, "random", "random", tx, bx, w, h, ref32, ref64, FULL + HEADS, False))
    doc = {
        "architecture": manifest["architecture"],
        "version": manifest["version"],
        "model_sha256": hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
        "max_chars": MAX_CHARS,
        "n_chars": N_CHARS,
        "samples": samples,
    }
    (out / "fixtures.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    nlayers = sum(len(s["layers"]) for s in samples)
    nbytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{len(samples)} samples, {nlayers} layer files, {nbytes} bytes")


if __name__ == "__main__":
    main()
