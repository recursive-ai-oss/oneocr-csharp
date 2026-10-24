"""Exports a dbnet-lite checkpoint to model container v1 (architecture `dbnet-lite-v0`, see export/dbnet-lite-architecture.md).

Usage:
  uv run python -m export.dbnet --checkpoint best.pt --out model.onm [--version 0.1.0]
  uv run python -m export.dbnet --random-seed 7 --out model.onm

Batch norm is folded into the preceding convolution in fp64 and rounded once to fp16. The training-only threshold branch is not
exported. `FoldedReference` is the layer graph written out with plain torch ops on the container's (fp16-rounded) tensors: the
parity fixtures and the exporter's self-check use it, and it is the thing the C# forward pass must reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.detect import EXPAND_LAYOUT, LAYOUT_THRESHOLD, TEXT_THRESHOLD
from export.container import ALIGN, MAGIC, _pad
from models.dbnet import LAYOUT_CLASSES, DBNetLite, arch_id, count_macs, count_params

INPUT = {"height": 0, "channels": 1, "mean": 0.5, "std": 0.5, "invert": False}  # height 0: any multiple of 32
TRAIN_ONLY = ("thr4", "thr_proj", "thr2", "thr_out")


def fold(model: DBNetLite) -> OrderedDict[str, torch.Tensor]:
    """name -> fp32 tensor (fp64-folded, not yet fp16-rounded). Names are module paths: `<path>.weight`, `<path>.bias`."""
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    model = model.eval()

    def conv_bn(path: str, seq: nn.Sequential) -> None:
        conv, bn = seq[0], seq[1]
        w = conv.weight.detach().double()
        scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)
        out[f"{path}.weight"] = (w * scale.view(-1, 1, 1, 1)).float()
        out[f"{path}.bias"] = (bn.bias.detach().double() - bn.running_mean.detach().double() * scale).float()

    def plain(path: str, m: nn.Module) -> None:
        out[f"{path}.weight"] = m.weight.detach().float()
        out[f"{path}.bias"] = m.bias.detach().float()

    def dw(path: str, blk) -> None:
        conv_bn(f"{path}.dw", blk.dw)
        conv_bn(f"{path}.pw", blk.pw)

    conv_bn("stem", model.stem)
    for i, st in enumerate(model.stages):
        for j, blk in enumerate(st):
            dw(f"stage{i}.{j}", blk)
    for i, m in enumerate(model.lat):
        plain(f"lat{i}", m)
    dw("smooth8", model.smooth8)
    dw("smooth4", model.smooth4)
    for j, blk in enumerate(model.layout):
        dw(f"layout.{j}", blk)
    if model.cfg["layout16_blocks"] > 0:
        for j, blk in enumerate(model.layout16):
            dw(f"layout16.{j}", blk)
        plain("layout_mix", model.layout_mix)
    plain("layout_out", model.layout_out)
    dw("text4", model.text4)
    plain("stem_proj", model.stem_proj)
    dw("text2", model.text2)
    plain("prob_out", model.prob_out)
    plain("orient", model.orient)
    return out


def up2(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="nearest")


class FoldedReference:
    """The exported graph on plain tensors. `dtype` float64 gives the exact reference for error floors."""

    def __init__(self, cfg: dict, tensors: dict[str, torch.Tensor], dtype=torch.float32):
        self.cfg = cfg
        self.t = {k: v.to(dtype) for k, v in tensors.items()}

    def conv(self, name: str, x, stride=1, groups=1, relu=True):
        w, b = self.t[f"{name}.weight"], self.t[f"{name}.bias"]
        y = F.conv2d(x, w, b, stride, w.shape[-1] // 2, groups=groups)
        return F.relu(y) if relu else y

    def block(self, name: str, x, stride=1):
        y = self.conv(f"{name}.dw", x, stride, groups=x.shape[1])
        return self.conv(f"{name}.pw", y)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        c, o = self.cfg, OrderedDict()
        o["stem"] = s2 = self.conv("stem", x, 2)
        h, feats = s2, []
        for i, d in enumerate(c["depths"]):
            for j in range(d):
                h = self.block(f"stage{i}.{j}", h, 2 if j == 0 else 1)
            o[f"stage{i}"] = h
            feats.append(h)
        l4, l8, l16, l32 = (self.conv(f"lat{i}", f, relu=False) for i, f in enumerate(feats))
        p16 = l16 + up2(l32)
        o["p16"] = p16
        o["p8"] = p8 = self.block("smooth8", l8 + up2(p16))
        o["p4"] = p4 = self.block("smooth4", l4 + up2(p8))
        c = {"layout16_blocks": 0, "orient16": False, **c}
        if c["layout16_blocks"] > 0:
            g = p16
            for j in range(c["layout16_blocks"]):
                g = self.block(f"layout16.{j}", g)
            o["layout16"] = g
            o["layout_mix"] = mix = self.conv("layout_mix", g, relu=False)
            h = self.block("layout.0", p8) + up2(mix)
            for j in range(1, c["layout_blocks"]):
                h = self.block(f"layout.{j}", h)
        else:
            h = p8
            for j in range(c["layout_blocks"]):
                h = self.block(f"layout.{j}", h)
        o["layout_feat"] = h
        o["layout"] = self.conv("layout_out", h, relu=False)
        o["text4"] = t4 = self.block("text4", p4)
        o["text2"] = t2 = self.block("text2", up2(t4) + self.conv("stem_proj", s2, relu=False))
        o["prob"] = self.conv("prob_out", t2, relu=False)
        g = feats[3].mean((2, 3))
        if c["orient16"]:
            g = torch.cat([feats[2].mean((2, 3)), g], 1)
        o["orient"] = F.linear(g, self.t["orient.weight"], self.t["orient.bias"])
        return o


def write_container_v1(path: Path, manifest: dict, tensors: dict[str, np.ndarray]) -> None:
    """Container v1 (docs: model-container-v1). Same layout as the v0 writer; `calibration` is written empty (no word confidences)."""
    entries, blobs, offset = [], [], 0
    for name, arr in tensors.items():
        h = np.ascontiguousarray(arr, dtype="<f2")
        entries.append({"name": name, "dtype": "f16", "shape": list(h.shape), "offset": offset, "length": h.size * 2})
        blobs.append(h.tobytes())
        offset += h.size * 2 + _pad(h.size * 2)
    full = {k: manifest[k] for k in ["architecture", "name", "version", "license", "input", "charset", "config"]}
    full["tensors"] = entries
    full["calibration"] = {"raw": "word-mean-max-run", "scripts": {}}
    full["provenance"] = manifest.get("provenance", {})
    body = json.dumps(full, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out = bytearray(MAGIC + struct.pack("<II", 1, len(body)) + body)
    out += bytes(_pad(len(out)))
    for k, blob in enumerate(blobs):
        out += blob
        if k != len(blobs) - 1:
            out += bytes(_pad(len(blob)))
    path.write_bytes(bytes(out))


def read_tensors(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Minimal reader for checking our own file (fp16 widened to fp32)."""
    data = path.read_bytes()
    assert data[:8] == MAGIC
    version, mlen = struct.unpack_from("<II", data, 8)
    assert version == 1
    manifest = json.loads(data[16 : 16 + mlen])
    start = 16 + mlen
    start += _pad(start)
    t = {}
    for e in manifest["tensors"]:
        assert e["offset"] % ALIGN == 0
        n = int(np.prod(e["shape"]))
        t[e["name"]] = np.frombuffer(data, "<f2", n, start + e["offset"]).reshape(e["shape"]).astype(np.float32)
    return manifest, t


def export(
    model: DBNetLite, path: Path, version: str, provenance: dict, postprocess: dict | None = None, layout_trained: list | None = None
) -> dict:
    tensors = fold(model)
    manifest = {
        "architecture": arch_id(model.cfg),
        "name": "text-detector-layout",
        "version": version,
        "license": "MIT OR Apache-2.0",
        "input": INPUT,
        # a detector has no output alphabet; the layout class names stand in so the v1 reader's charset checks pass (index i + 1 = class i)
        "charset": ["<blank>", *LAYOUT_CLASSES],
        "config": {
            **model.cfg,
            "layout_classes": LAYOUT_CLASSES,
            # probabilities and growth factors the post-processing uses (spec: post-processing); picked on validation pages
            "postprocess": postprocess
            or {"text_threshold": TEXT_THRESHOLD, "text_expand": 1.0, "layout_threshold": LAYOUT_THRESHOLD, "layout_expand": EXPAND_LAYOUT},
            # classes the training data labelled; the other channels exist in the file but were never trained
            "layout_trained": layout_trained or LAYOUT_CLASSES,
        },
        "provenance": provenance,
    }
    write_container_v1(path, manifest, {k: v.numpy() for k, v in tensors.items()})
    return manifest


def random_model(seed: int, cfg: dict | None = None) -> DBNetLite:
    """Random init with non-trivial BN statistics so that folding is exercised."""
    torch.manual_seed(seed)
    m = DBNetLite(cfg)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.weight.data.uniform_(0.6, 1.4)
            mod.bias.data.normal_(0, 0.1)
            mod.running_mean.data.normal_(0, 0.2)
            mod.running_var.data.uniform_(0.5, 2.0)
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--random-seed", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--model-config", help="JSON of model config overrides for --random-seed")
    ap.add_argument("--postprocess", help="JSON {text_threshold,text_expand,layout_threshold,layout_expand}, default: eval.detect")
    ap.add_argument("--run-manifest", help="run_manifest.json of the training run, copied into provenance")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prov: dict = {}
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model = DBNetLite(ck["config"])
        model.load_state_dict(ck["model"])
        prov = {"step": ck["step"]}
        if a.run_manifest:
            prov["run"] = json.loads(Path(a.run_manifest).read_text())
    else:
        model = random_model(a.random_seed, json.loads(a.model_config) if a.model_config else None)
        prov = {"random_init_seed": a.random_seed, "note": "untrained, for parity fixtures only"}
    model = model.eval()
    trained = None
    if "run" in prov:
        from train.detect_data import CMASK

        names = [src["name"] for src in prov["run"]["config"]["sources"]]
        trained = [c for i, c in enumerate(LAYOUT_CLASSES) if any(CMASK[n][i] for n in names)]
    export(model, out, a.version, prov, json.loads(a.postprocess) if a.postprocess else None, trained)
    # fp16-rounded folded graph vs the PyTorch model in fp32
    manifest, t = read_tensors(out)
    ref = FoldedReference(manifest["config"], {k: torch.from_numpy(v) for k, v in t.items()})
    x = torch.randn(1, 1, 128, 160)
    with torch.no_grad():
        ref_out, m_out = ref.forward(x), model(x, train_heads=False)
    diffs = {k: float((ref_out[k] - m_out[k]).abs().max()) for k in ("prob", "layout", "orient")}
    macs = count_macs(model, 1024, 736)["total"]
    info = {
        "file": str(out), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "params_inference": count_params(model), "tensors": len(t), "macs_1024x736": macs,
        "fp16_folded_vs_fp32_model_max_abs": diffs,
    }  # fmt: skip
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
