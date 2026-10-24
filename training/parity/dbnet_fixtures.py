"""Forward parity fixtures for the dbnet-lite C# kernels.

Usage: uv run python -m parity.dbnet_fixtures --model model.onm --pages DIR_OF_PAGE_IMAGES --out OUT_DIR [--seed 5]

Layout of OUT_DIR (the recognizer fixtures' layout, with tiles instead of lines):
  model.onm            the container the references were computed from
  fixtures.json        samples: id, png, shape, layers [{name, shape, file, sha256}]; fp32-vs-fp64 floor per layer
  inputs/<id>.png      8-bit gray tiles (omitted for the noise tile)
  data/<id>.<layer>.f32  raw little-endian float32, row-major [C,H,W] (orient: [4])

Tiles are seeded crops of real pages (long side 1024 as at inference) chosen for ink, one seeded noise tile, and one larger crop that
stores only the head outputs (every file stays under 1 MB). References run in fp32 from the container's fp16-rounded, folded weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.detect import prepare, to_input
from export.dbnet import FoldedReference, read_tensors

HEADS = ("layout", "prob", "orient")


def ink_crop(canvas: np.ndarray, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    best, best_std = None, -1.0
    for _ in range(40):
        y = int(rng.integers(0, canvas.shape[0] - h + 1))
        x = int(rng.integers(0, canvas.shape[1] - w + 1))
        c = canvas[y : y + h, x : x + w]
        s = float(c.std())
        if s > best_std:
            best, best_std = c, s
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    shutil.copyfile(a.model, out / "model.onm")
    manifest, t = read_tensors(Path(a.model))
    tt = {k: torch.from_numpy(v) for k, v in t.items()}
    ref32 = FoldedReference(manifest["config"], tt)
    ref64 = FoldedReference(manifest["config"], tt, torch.float64)
    rng = np.random.default_rng(a.seed)
    pages = sorted(p for p in Path(a.pages).iterdir() if p.suffix in (".png", ".jpg"))
    picks = [pages[i] for i in sorted(rng.choice(len(pages), 4, replace=False))]
    tiles = []
    for k, p in enumerate(picks):
        canvas, _ = prepare(Image.open(p))
        tiles.append((f"tile{k}", ink_crop(canvas, 128, 160, rng), True, True))
    tiles.append(("noise", rng.integers(0, 256, (64, 96), dtype=np.uint8), False, True))
    canvas, _ = prepare(Image.open(picks[0]))
    tiles.append(("wide", ink_crop(canvas, 224, 288, rng), True, False))

    samples, floor = [], {}
    for sid, arr, save_png, all_layers in tiles:
        if save_png:
            Image.fromarray(arr).save(out / "inputs" / f"{sid}.png")
        x = to_input(arr)
        o32, o64 = ref32.forward(x), ref64.forward(x.double())
        layers = [("input", x[0])] + [(k, v[0]) for k, v in o32.items() if all_layers or k in HEADS]
        entries = []
        for name, v in layers:
            v = v.contiguous().float()
            f = out / "data" / f"{sid}.{name}.f32"
            f.write_bytes(v.numpy().astype("<f4").tobytes())
            assert f.stat().st_size < 1_000_000, f"{f} exceeds 1 MB"
            entries.append(
                {"name": name, "shape": list(v.shape), "file": f"data/{f.name}", "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
            )
            if name != "input":
                ref = o64[name][0]
                d = (o32[name][0].double() - ref).abs()
                s = floor.setdefault(name, {"max_abs": 0.0, "ref_max_abs": 0.0, "max_rel_above_1e-3_of_max": 0.0})
                s["max_abs"] = max(s["max_abs"], float(d.max()))
                s["ref_max_abs"] = max(s["ref_max_abs"], float(ref.abs().max()))
                big = ref.abs() > 1e-3 * ref.abs().max()
                if big.any():
                    s["max_rel_above_1e-3_of_max"] = max(s["max_rel_above_1e-3_of_max"], float((d[big] / ref.abs()[big]).max()))
        samples.append({"id": sid, "png": f"inputs/{sid}.png" if save_png else None, "shape": [1, *arr.shape], "layers": entries})
    doc = {
        "architecture": manifest["architecture"],
        "model": "model.onm",
        "model_sha256": hashlib.sha256((out / "model.onm").read_bytes()).hexdigest(),
        "dtype": "float32 little-endian, row-major",
        "input_normalisation": "(p / 255 - 0.5) / 0.5",
        "fp32_vs_fp64_floor": floor,
        "samples": samples,
    }
    (out / "fixtures.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"{len(samples)} samples -> {out}; floor max_abs: { {k: v['max_abs'] for k, v in floor.items()} }")


if __name__ == "__main__":
    main()
