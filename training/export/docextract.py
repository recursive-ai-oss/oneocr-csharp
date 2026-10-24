"""Exports a docextract checkpoint to model container v1 (architecture `docextract-v0`, see export/docextract-architecture.md).

Usage:
  uv run python -m export.docextract --checkpoint best.pt --out model.onm [--version 0.1.0] [--run-manifest run_manifest.json]
  uv run python -m export.docextract --random-seed 7 --out model.onm        (untrained, for parity fixtures)

Tensors keep the module's names (`<path>.weight`, `<path>.bias`). The model has no batch norm, so nothing is folded: the weights are
rounded once to fp16. `Reference` is the graph written with plain torch ops on the container's tensors, for one page without padding;
the parity fixtures and the exporter's self-check use it, and it is what the C# forward pass must reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from export.container import write_container
from models.docextract import (
    CLASSES,
    DOC_TYPES,
    MAX_CHARS,
    N_CHARS,
    NBINS,
    NFEATS,
    NPAIR,
    NREL,
    ROLES,
    UNK_CHAR,
    DocExtract,
    charset_ids,
    count_params,
    pair_features,
)

ARCHITECTURE = "docextract-v0"


def charset_list() -> list[str]:
    cs = charset_ids()
    out = ["<blank>"] + [None] * len(cs)
    for c, i in cs.items():
        out[i] = c
    return out


def tensors_of(model: DocExtract) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((k, v.detach().float().cpu()) for k, v in model.state_dict().items())


def manifest_for(model: DocExtract, version: str, provenance: dict) -> dict:
    charset = charset_list()
    # the unknown-character id is the first id past the alphabet; the library reads it from the manifest and requires this
    assert UNK_CHAR == len(charset), f"unk_char {UNK_CHAR} != charset length {len(charset)}"
    assert model.cfg["max_words"] == 512, "input.max_words and config.max_words must agree with the window rule"
    assert model.cfg["tags"] == 1 + 2 * model.cfg["classes"] and model.cfg["d"] % model.cfg["heads"] == 0
    return {
        "architecture": ARCHITECTURE,
        "name": "document-extraction",
        "version": version,
        "license": "MIT OR Apache-2.0",
        # words in, no pixels: the "input" block names the featurization limits instead of an image size
        "input": {"kind": "words", "max_words": model.cfg["max_words"], "max_chars": MAX_CHARS, "bins": NBINS, "features": NFEATS},
        # the character alphabet of the word embedding: charset-v1 (class 0 = padding); id 622 = unknown character
        "charset": charset,
        "config": {**model.cfg, "classes_list": CLASSES, "roles_list": ROLES, "doc_types_list": DOC_TYPES, "unk_char": UNK_CHAR},
        "provenance": provenance,
    }


def export(model: DocExtract, path: Path, version: str, provenance: dict) -> dict:
    t = tensors_of(model)
    m = manifest_for(model, version, provenance)
    m["calibration"] = {"raw": "word-mean-max-run", "scripts": {}}  # no confidence calibration for this model
    write_container(path, m, {k: v.numpy() for k, v in t.items()}, version=1)
    return m


class Reference:
    """Forward pass of one page on plain tensors. dtype float64 gives the exact reference for error floors."""

    def __init__(self, cfg: dict, tensors: dict[str, torch.Tensor], dtype=torch.float32):
        self.cfg = cfg
        self.t = {k: v.to(dtype) for k, v in tensors.items()}
        self.dtype = dtype

    def lin(self, name, x):
        return F.linear(x, self.t[f"{name}.weight"], self.t[f"{name}.bias"])

    def ln(self, name, x):
        return F.layer_norm(x, (x.shape[-1],), self.t[f"{name}.weight"], self.t[f"{name}.bias"], 1e-5)

    def forward(self, chars, feats, bins, geo) -> dict:
        """chars [N, L] long, feats [N, NFEATS], bins [N, 4] long, geo [N, 4]; returns every intermediate named as in the spec."""
        c, t, dt = self.cfg, self.t, self.dtype
        n = chars.shape[0]
        o: dict[str, torch.Tensor] = {}
        e = t["char_emb.weight"][chars]  # [N, L, E]
        valid = (chars != 0).to(dt)  # [N, L]
        cv = F.conv1d(e.transpose(1, 2), t["char_conv.weight"], t["char_conv.bias"], padding=1)  # [N, C, L]
        h = F.relu(cv) * valid[:, None, :]
        o["char_pool"] = h.amax(-1)
        w = self.lin("char_proj", o["char_pool"])
        x = (
            w
            + t["pos_x0.weight"][bins[:, 0]]
            + t["pos_x1.weight"][bins[:, 2]]
            + t["pos_y0.weight"][bins[:, 1]]
            + t["pos_y1.weight"][bins[:, 3]]
        )
        x = x + self.lin("feat_proj", feats.to(dt))
        o["emb"] = x = self.ln("emb_ln", x)
        pf = pair_features(geo.to(dt)[None])[0]  # [N, N, NPAIR]
        pb = self.lin("pair2", F.relu(self.lin("pair1", pf))).permute(2, 0, 1)  # [L*H + R, N, N]
        o["pair_bias"] = pb
        nh, d = c["heads"], c["d"]
        dh = d // nh
        for i in range(c["layers"]):
            p = f"blocks.{i}"
            q, k, v = self.lin(f"{p}.qkv", self.ln(f"{p}.ln1", x)).view(n, 3, nh, dh).permute(1, 2, 0, 3)  # each [H, N, dh]
            s = q @ k.transpose(-1, -2) / dh**0.5 + pb[i * nh : (i + 1) * nh]
            a = torch.softmax(s, -1) @ v  # [H, N, dh]
            x = x + self.lin(f"{p}.proj", a.transpose(0, 1).reshape(n, d))
            x = x + self.lin(f"{p}.fc2", F.gelu(self.lin(f"{p}.fc1", self.ln(f"{p}.ln2", x)), approximate="tanh"))
            o[f"block.{i}"] = x
        o["final"] = x = self.ln("final_ln", x)
        o["tags"] = self.lin("tag_head", x)
        o["roles"] = self.lin("role_head", x)
        o["doc"] = self.lin("doc_head", x.mean(0))
        r = c["rel_dim"]
        q = self.lin("rel_q", x).view(n, NREL, r).permute(1, 0, 2)
        k = self.lin("rel_k", x).view(n, NREL, r).permute(1, 0, 2)
        o["rel"] = q @ k.transpose(-1, -2) / r**0.5 + pb[c["layers"] * nh :]
        return o


def random_model(seed: int, cfg: dict | None = None) -> DocExtract:
    """Random init with non-trivial layer-norm parameters and biases so that every tensor is exercised."""
    torch.manual_seed(seed)
    m = DocExtract(cfg)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("ln1.weight") or name.endswith("ln2.weight") or "_ln." in name and name.endswith("weight"):
                p.uniform_(0.7, 1.3)
            elif name.endswith("bias"):
                p.normal_(0, 0.05)
        for head in (m.tag_head, m.role_head, m.doc_head, m.rel_q, m.rel_k):  # the 0.02 init leaves outputs near zero; widen them
            head.weight.mul_(8.0)
        m.pair2.weight.mul_(4.0)
        m.char_emb.weight[0].zero_()
    return m.eval()


def read_back(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    from export.container import read_container

    manifest, t = read_container(path)
    return manifest, {k: torch.from_numpy(v) for k, v in t.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--random-seed", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--run-manifest")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prov: dict = {}
    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        model = DocExtract(ck["config"])
        model.load_state_dict(ck["model"])
        prov = {"step": ck["step"]}
        if a.run_manifest:
            prov["run"] = json.loads(Path(a.run_manifest).read_text())
    else:
        model = random_model(a.random_seed)
        prov = {"random_init_seed": a.random_seed, "note": "untrained, for parity fixtures only"}
    model = model.eval()
    export(model, out, a.version, prov)
    manifest, t = read_back(out)
    ref = Reference(manifest["config"], t)
    g = torch.Generator().manual_seed(1)
    n = 30
    chars = torch.randint(0, N_CHARS, (n, MAX_CHARS), generator=g)
    chars[:, 0] = chars[:, 0].clamp_min(1)
    chars = chars * (torch.arange(MAX_CHARS)[None] < torch.randint(1, MAX_CHARS, (n, 1), generator=g))
    feats = torch.randn(n, NFEATS, generator=g) * 0.3
    bins = torch.randint(0, NBINS, (n, 4), generator=g)
    geo = torch.cumsum(torch.rand(n, 4, generator=g) * 3, 0)
    geo[:, 2:] = geo[:, :2] + 1 + torch.rand(n, 2, generator=g) * 4
    with torch.no_grad():
        r = ref.forward(chars, feats, bins, geo)
        mo = model(chars[None], feats[None], bins[None], geo[None], torch.ones(1, n, dtype=torch.bool))
    diffs = {k: float((r[k] - mo[k][0]).abs().max()) for k in ("tags", "roles", "rel")}
    info = {
        "file": str(out),
        "bytes": out.stat().st_size,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "params": count_params(model),
        "tensors": len(t),
        "fp16_size_mb": round(count_params(model) * 2 / 1e6, 3),
        "fp16_reference_vs_fp32_model_max_abs": diffs,
        "classes": len(CLASSES),
        "pair_features": NPAIR,
    }
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
