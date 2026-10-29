"""Backward-pass parity fixtures for the C# fine-tuning of the document-extraction model (docextract-v0).

Usage (from `training/`, with the docextract modules of `models/` and `export/` present):

    uv run python -m parity.docextract_backward --model model.onm --out OUT_DIR [--seed 31]

`model.onm` is a docextract-v0 container (an untrained random-init one is fine, see `export.docextract --random-seed`). The references use
the container's fp16-rounded weights and the graph of `export.docextract.Reference` (plain torch ops on one page without padding), the
graph the C# forward pass reproduces. Everything is seeded and single-threaded on the CPU, so regenerating gives identical files.

Layout of OUT_DIR:
  model.onm                    the container the references come from
  fixtures.json                everything below: inputs, targets, losses, tolerances' floors, file index
  data/<name>.f32              raw little-endian float32, row-major

Two kinds of fixtures:
  * `model`: one page through the whole model with all five loss terms (tag, role, document, two pointer relations) at non-trivial
    weights, class weights on the tags and ignored words; the loss of each term, the gradient of every parameter (container names and
    shapes; the padding row 0 of `char_emb` is zero, as with `padding_idx=0`) and the gradient at the named intermediates
    (`char_pool`, `emb`, `pair_bias`, `block.i`, `final`).
  * `layers`: each backward kernel on its own with a seeded upstream gradient: layer norm, tanh GELU, softmax attention with a pair bias,
    the pair MLP, the character CNN with the embedding gather, the pointer scores, the cross-entropy with ignored rows and class weights.
Every expected tensor is stored from a float32 run and with `max_abs`, the largest difference of that float32 run to a float64 run (the
error PyTorch itself makes), so C# is held to a multiple of PyTorch's own error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from export.docextract import Reference, read_back
from models.docextract import NREL, featurize, pair_features
from parity.docextract_fixtures import random_page

LOSS_WEIGHTS = {"tag": 1.0, "role": 0.5, "doc": 0.25, "item": 0.75, "kv": 1.25}
FORMAT = "oneocr-docextract-backward-1"


class Out:
    def __init__(self, root: Path):
        self.root = root
        (root / "data").mkdir(parents=True)
        self.index: dict[str, dict] = {}

    def put(self, group: str, name: str, t32: torch.Tensor, t64: torch.Tensor | None = None) -> None:
        """Stores t32 (float32 run); with t64 also the float32-vs-float64 floor."""
        a = t32.detach().cpu().numpy().astype("<f4")
        key = f"{group}.{name}".replace("/", "_")
        f = self.root / "data" / f"{key}.f32"
        f.write_bytes(a.tobytes())
        e = {"shape": list(a.shape), "file": f"data/{f.name}", "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        if t64 is not None:
            d = (t32.detach().double() - t64.detach()).abs()
            e["max_abs"] = float(d.max()) if d.numel() else 0.0
            e["ref_max_abs"] = float(t64.detach().abs().max()) if d.numel() else 0.0
        self.index[key] = e

    def ints(self, group: str, name: str, a) -> None:
        a = np.asarray(a, dtype="<i4")
        key = f"{group}.{name}"
        f = self.root / "data" / f"{key}.i32"
        f.write_bytes(a.tobytes())
        self.index[key] = {"shape": list(a.shape), "file": f"data/{f.name}", "dtype": "i32"}


def leaf(t: dict[str, torch.Tensor], dtype) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone().to(dtype).requires_grad_(True) for k, v in t.items()}


def ce(logits: torch.Tensor, target: torch.Tensor, weight=None) -> torch.Tensor:
    return F.cross_entropy(logits, target, weight=weight, ignore_index=-100, reduction="mean")


def model_fixture(out: Out, cfg: dict, tensors: dict, seed: int) -> dict:
    rng = random.Random(seed)
    n = 40
    texts, boxes, w, h = random_page(rng, n)
    f = featurize(texts, boxes, w, h)
    chars = torch.from_numpy(f["chars"].astype(np.int64))
    feats, bins, geo = (torch.from_numpy(f[k]) for k in ("feats", "bins", "geo"))
    g = torch.Generator().manual_seed(seed)

    def targets(classes: int, ignore: float = 0.25) -> torch.Tensor:
        t = torch.randint(0, classes, (n,), generator=g)
        return torch.where(torch.rand(n, generator=g) < ignore, torch.full_like(t, -100), t)

    tag_t, role_t = targets(cfg["tags"]), targets(cfg["roles"])
    item_t, kv_t = targets(n), targets(n)
    doc_t = int(torch.randint(0, cfg["doc_types"], (1,), generator=g))
    class_w = 0.5 + torch.rand(cfg["tags"], generator=g)
    for name, arr in (("tags", tag_t), ("roles", role_t), ("item", item_t), ("kv", kv_t)):
        out.ints("model", f"target_{name}", arr.numpy())
    out.put("model", "tag_class_weights", class_w)

    results = {}
    for dtype, label in ((torch.float32, "f32"), (torch.float64, "f64")):
        t = leaf(tensors, dtype)
        ref = Reference(cfg, t, dtype)
        o = ref.forward(chars, feats.to(dtype), bins, geo.to(dtype))
        inter = {k: o[k] for k in ("char_pool", "emb", "pair_bias", "final", *[f"block.{i}" for i in range(cfg["layers"])])}
        for v in inter.values():
            v.retain_grad()
        cw = class_w.to(dtype)
        losses = {
            "tag": ce(o["tags"], tag_t, cw),
            "role": ce(o["roles"], role_t),
            "doc": F.cross_entropy(o["doc"][None], torch.tensor([doc_t])),
            "item": ce(o["rel"][0], item_t),
            "kv": ce(o["rel"][1], kv_t),
        }
        total = sum(LOSS_WEIGHTS[k] * v for k, v in losses.items())
        total.backward()
        grads = {k: v.grad.detach().clone() for k, v in t.items()}
        grads["char_emb.weight"][0] = 0  # padding_idx = 0
        results[label] = {
            "losses": {k: float(v) for k, v in losses.items()},
            "total": float(total),
            "grads": grads,
            "inter": {k: v.grad.detach().clone() for k, v in inter.items()},
            "outputs": {k: o[k].detach().clone() for k in ("tags", "roles", "doc", "rel")},
        }
    r32, r64 = results["f32"], results["f64"]
    for k, v in r32["grads"].items():
        out.put("model.grad", k, v, r64["grads"][k])
    for k, v in r32["inter"].items():
        out.put("model.dinter", k, v, r64["inter"][k])
    for k, v in r32["outputs"].items():
        out.put("model.out", k, v, r64["outputs"][k])
    for name, arr in (("feats", feats), ("geo", geo)):
        out.put("model", f"in_{name}", arr)
    return {
        "words": [{"text": t, "box": b} for t, b in zip(texts, boxes, strict=True)],
        "width": w,
        "height": h,
        "n": n,
        "chars": f["chars"].tolist(),
        "bins": f["bins"].tolist(),
        "unit": f["unit"],
        "doc_target": doc_t,
        "loss_weights": LOSS_WEIGHTS,
        "losses_f32": r32["losses"],
        "losses_f64": r64["losses"],
        "total_f32": r32["total"],
        "total_f64": r64["total"],
    }


def layer_fixtures(out: Out, cfg: dict, tensors: dict, seed: int) -> dict:
    """Each kernel alone, float32 with a float64 floor. Shapes avoid multiples of 16 where the kernels have tails."""
    g = torch.Generator().manual_seed(seed + 1)
    info = {}

    def rnd(*shape, dtype=torch.float32, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(dtype)

    def run(fn, inputs: dict, gy: torch.Tensor, wrt: list[str]) -> dict:
        """fn(**inputs) -> tensor; returns grads of `wrt` inputs for float32 and float64."""
        res = {}
        for dtype in (torch.float32, torch.float64):
            ins = {k: (v.detach().clone().to(dtype).requires_grad_(k in wrt) if v.is_floating_point() else v) for k, v in inputs.items()}
            y = fn(**ins)
            y.backward(gy.to(dtype))
            res[dtype] = ({k: ins[k].grad.detach().clone() for k in wrt}, y.detach().clone())
        return res

    def store(group: str, inputs: dict, gy: torch.Tensor, res: dict, wrt: list[str]) -> None:
        for k, v in inputs.items():
            if v.is_floating_point():
                out.put(group, f"in.{k}", v)
            else:
                out.ints(group, f"in.{k}", v.numpy())
        out.put(group, "gy", gy)
        out.put(group, "y", res[torch.float32][1], res[torch.float64][1])
        for k in wrt:
            out.put(group, f"d.{k}", res[torch.float32][0][k], res[torch.float64][0][k])

    # layer norm (rows 13, width 192 = model width, tail-free; and width 50 with a tail)
    for d in (cfg["d"], 50):
        x, wgt, b = rnd(13, d, scale=2.0) + 0.3, 0.7 + torch.rand(d, generator=g) * 0.6, rnd(d, scale=0.1)
        gy = rnd(13, d)
        res = run(lambda x, w, b: F.layer_norm(x, (x.shape[-1],), w, b, 1e-5), {"x": x, "w": wgt, "b": b}, gy, ["x", "w", "b"])
        store(f"layers.layer_norm{d}", {"x": x, "w": wgt, "b": b}, gy, res, ["x", "w", "b"])

    # tanh GELU
    x = rnd(9, 77, scale=2.5)
    gy = rnd(9, 77)
    res = run(lambda x: F.gelu(x, approximate="tanh"), {"x": x}, gy, ["x"])
    store("layers.gelu", {"x": x}, gy, res, ["x"])

    # one attention head with a pair bias: scores = q k^T / sqrt(dh) + bias, softmax, times v (n odd, dh = 32)
    for n in (37, 64):
        dh = cfg["d"] // cfg["heads"]
        q, k, v, bias = rnd(n, dh), rnd(n, dh), rnd(n, dh), rnd(n, n)
        gy = rnd(n, dh)
        fn = lambda q, k, v, bias: torch.softmax(q @ k.T / dh**0.5 + bias, -1) @ v  # noqa: E731
        res = run(fn, {"q": q, "k": k, "v": v, "bias": bias}, gy, ["q", "k", "v", "bias"])
        store(f"layers.attention{n}", {"q": q, "k": k, "v": v, "bias": bias}, gy, res, ["q", "k", "v", "bias"])

    # pair MLP: pb = pair2(relu(pair1(pair_features(geo)))) permuted to [channels, n, n], weights only
    n = 19
    geo = torch.cumsum(torch.rand(n, 4, generator=g) * 3, 0)
    geo[:, 2:] = geo[:, :2] + 1 + torch.rand(n, 2, generator=g) * 4
    w1, b1, w2, b2 = (tensors[k] for k in ("pair1.weight", "pair1.bias", "pair2.weight", "pair2.bias"))
    gy = rnd(w2.shape[0], n, n)

    def pair(geo, w1, b1, w2, b2):
        pf = pair_features(geo[None])[0]
        return F.linear(F.relu(F.linear(pf, w1, b1)), w2, b2).permute(2, 0, 1)

    res = run(pair, {"geo": geo, "w1": w1, "b1": b1, "w2": w2, "b2": b2}, gy, ["w1", "b1", "w2", "b2"])
    store("layers.pair_mlp", {"geo": geo, "w1": w1, "b1": b1, "w2": w2, "b2": b2}, gy, res, ["w1", "b1", "w2", "b2"])

    # pointer scores: rel = q k^T / sqrt(r) + bias
    n, r = 29, cfg["rel_dim"]
    q, k, bias = rnd(n, r), rnd(n, r), rnd(n, n)
    gy = rnd(n, n)
    res = run(lambda q, k, bias: q @ k.T / r**0.5 + bias, {"q": q, "k": k, "bias": bias}, gy, ["q", "k", "bias"])
    store("layers.pointer", {"q": q, "k": k, "bias": bias}, gy, res, ["q", "k", "bias"])

    # cross-entropy with ignored rows and class weights
    rows, classes = 23, 95
    z = rnd(rows, classes, scale=3.0)
    tgt = torch.randint(0, classes, (rows,), generator=g)
    tgt[::4] = -100
    wt = 0.5 + torch.rand(classes, generator=g)
    for tag, weight in (("plain", None), ("weighted", wt)):
        zz = {k: z.detach().clone().to(dt).requires_grad_(True) for k, dt in (("f32", torch.float32), ("f64", torch.float64))}
        loss = {k: ce(v, tgt, None if weight is None else weight.to(v.dtype)) for k, v in zz.items()}
        for v in loss.values():
            v.backward()
        out.put(f"layers.ce_{tag}", "in.z", z)
        out.ints(f"layers.ce_{tag}", "in.target", tgt.numpy())
        if weight is not None:
            out.put(f"layers.ce_{tag}", "in.weight", weight)
        out.put(f"layers.ce_{tag}", "loss", loss["f32"].reshape(1), loss["f64"].reshape(1))
        out.put(f"layers.ce_{tag}", "d.z", zz["f32"].grad, zz["f64"].grad)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    torch.set_num_threads(1)
    root = Path(a.out)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    out = Out(root)
    shutil.copy(a.model, root / "model.onm")
    manifest, tensors = read_back(Path(a.model))
    cfg = manifest["config"]
    page = model_fixture(out, cfg, tensors, a.seed)
    layer_fixtures(out, cfg, tensors, a.seed)
    doc = {
        "format": FORMAT,
        "seed": a.seed,
        "architecture": manifest["architecture"],
        "model_sha256": hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
        "max_chars": manifest["input"]["max_chars"],
        "page": page,
        "tensors": out.index,
    }
    (root / "fixtures.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"{len(out.index)} tensors, {total} bytes")


if __name__ == "__main__":
    main()
