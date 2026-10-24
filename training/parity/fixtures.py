"""Parity fixtures for the C# forward kernels.

Usage: uv run python -m parity.fixtures --model model.onm --data LINESET_DIR --out OUT_DIR [--n 50] [--max-width 640]

Layout of OUT_DIR:
  model.onm            the container the references were computed from
  fixtures.json        lines: id, png, text (truth), decoded (greedy, from the reference), layers [{name, shape, file, sha256}]
  lines/<id>.png       8-bit gray crops at the input height
  data/<id>.<layer>.f32  raw little-endian float32, row-major

Layers: input [1,H,W] (normalised), conv.<i> [C,H,W] (after ReLU and pool), lstm.l<k> [T,2*hidden], fc [T,classes] (crnn-v0);
stem.<i>/mix.<i>/merge [C,H,W], proj and block.<i> [T,D], fc [T,classes] (svtr-lite).
References run in fp32 from the container's fp16-rounded, BN-folded weights. fixtures.json also records
the fp32-vs-fp64 difference of each layer, which is the floor for any fp32 implementation's tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from export.container import read_container
from models.crnn import FoldedReference as CrnnReference
from models.crnn import greedy_decode
from models.svtr import FoldedReference as SvtrReference
from train.data import load_gray, normalize


def pick(rows: list[dict], n: int, max_width: int) -> list[dict]:
    """n lines of evenly spread widths (deterministic: sorted by width, then id)."""
    ok = sorted((r for r in rows if r["w"] <= max_width), key=lambda r: (r["w"], r["id"]))
    idx = np.linspace(0, len(ok) - 1, n).round().astype(int)
    return [ok[i] for i in sorted(set(idx.tolist()))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-v1", help="the same weights in container v1 (with calibration); copied next to model.onm")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-width", type=int, default=640)
    a = ap.parse_args()

    out, data = Path(a.out), Path(a.data)
    (out / "lines").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    shutil.copyfile(a.model, out / "model.onm")
    if a.model_v1:
        shutil.copyfile(a.model_v1, out / "model-v1.onm")
    manifest, tensors = read_container(a.model)
    t = {k: torch.from_numpy(v) for k, v in tensors.items()}
    Reference = SvtrReference if manifest["config"].get("arch") == "svtr-lite" else CrnnReference
    ref32 = Reference(manifest["config"], t)
    ref64 = Reference(manifest["config"], t, torch.float64)
    charset = manifest["charset"]

    rows = [json.loads(line) for line in (data / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line]
    lines = []
    floor: dict[str, dict] = {}
    for r in pick(rows, a.n, a.max_width):
        png = out / "lines" / (r["id"] + ".png")
        shutil.copyfile(data / "images" / (r["id"] + ".png"), png)
        x = torch.from_numpy(normalize(load_gray(png)))[None, None]
        o32, o64 = ref32.forward(x), ref64.forward(x)
        layers = [("input", x[0])] + list(o32.items())
        entries = []
        for name, arr in layers:
            arr = arr.contiguous().float()
            f = out / "data" / f"{r['id']}.{name}.f32"
            f.write_bytes(arr.numpy().astype("<f4").tobytes())
            assert f.stat().st_size < 1_000_000, f"{f} exceeds 1 MB"
            entries.append({"name": name, "shape": list(arr.shape), "file": f"data/{f.name}",
                            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})  # fmt: skip
            if name != "input":
                d = (o32[name].double() - o64[name]).abs()
                rel = d / o64[name].abs().clamp_min(1e-6)
                s = floor.setdefault(name, {"max_abs": 0.0, "max_rel_above_1e-3": 0.0})
                s["max_abs"] = max(s["max_abs"], float(d.max()))
                big = o64[name].abs() > 1e-3
                if big.any():
                    s["max_rel_above_1e-3"] = max(s["max_rel_above_1e-3"], float(rel[big].max()))
        lines.append({"id": r["id"], "png": f"lines/{png.name}", "text": r["text"], "width": r["w"],
                      "decoded": greedy_decode(o32["fc"], charset), "layers": entries})  # fmt: skip
    v1_entry = {}
    if a.model_v1:
        v1_entry = {"model_v1": "model-v1.onm", "model_v1_sha256": hashlib.sha256((out / "model-v1.onm").read_bytes()).hexdigest()}
    doc = {
        "architecture": manifest["architecture"],
        "model": "model.onm",
        "model_sha256": hashlib.sha256((out / "model.onm").read_bytes()).hexdigest(),
        "dtype": "float32 little-endian, row-major",
        **v1_entry,
        "fp32_vs_fp64_floor": floor,
        "lines": lines,
    }
    (out / "fixtures.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(lines)} lines -> {out}; fp32-vs-fp64 floor: {json.dumps(floor)}")


if __name__ == "__main__":
    main()
