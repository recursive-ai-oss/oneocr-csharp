"""svtr-lite: convolutional stem + global self-attention blocks -> linear -> CTC.

A line recognizer in the family of SVTR (Du et al., "SVTR: Scene Text Recognition with a Single Visual Model", 2022): the image
is turned into patch-like tokens by strided convolutions, mixed locally and globally by transformer blocks, merged along the
height and read out with CTC, with no recurrent layer. Two things differ from the paper, both to fit a CPU runtime that has
variable-width lines and a small budget:
  * local mixing is a width convolution (a dense conv1d, kernel 3) before every attention layer instead of windowed attention. The
    convolution is also the only source of position information: a stack of them gives each token its neighbourhood, which is all
    CTC ordering needs, and nothing is tied to a fixed input width (the paper's learned positional table is);
  * attention is global over the width only: the height has been merged away before the blocks (the paper keeps 2-D tokens
    through the first stages).
The reduction to a token sequence follows the PP-OCR technical reports' lightweight recognizers (strided conv stem, height merge).
Everything is randomly initialised.

Layer types: dense convolution (3x3 incl. strides, and 1-D k=3), batch norm (folded at export), GELU (tanh form), layer norm,
linear, multi-head self-attention (softmax(QK^T / sqrt(d)) V), residual add.

Tensors are named as a container would name them: `stem.{0,1}`, `mix.{i}`, `merge`, `proj`, `block.{i}.{local,norm1,qkv,out,norm2,fc1,fc2}`,
`norm`, `fc`.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CONFIG = {
    "arch": "svtr-lite",
    "stem": [32, 64],
    "mix_blocks": 2,
    "merge": 128,
    "dim": 192,
    "blocks": 6,
    "heads": 4,
    "mlp": 2,
    "classes": 622,
}

INPUT_HEIGHT = 32
BN_EPS = 1e-5
LN_EPS = 1e-5


def output_length(width: int, config: dict | None = None) -> int:
    """Two stride-2 convolutions with kernel 3 and padding 1: ceil(ceil(w / 2) / 2)."""
    return (((width + 1) // 2) + 1) // 2


def conv_bn(cin: int, cout: int, stride=(1, 1)) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(cout, eps=BN_EPS), nn.GELU(approximate="tanh")
    )


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: int):
        super().__init__()
        self.heads = heads
        self.local = nn.Conv1d(dim, dim, 3, padding=1)
        self.norm0 = nn.LayerNorm(dim, eps=LN_EPS)
        self.norm1 = nn.LayerNorm(dim, eps=LN_EPS)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=LN_EPS)
        self.fc1 = nn.Linear(dim, mlp * dim)
        self.fc2 = nn.Linear(mlp * dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """x: [B, T, D]; mask: [B, 1, 1, T] bool, True where a key is a real column."""
        x = x + F.gelu(self.local(self.norm0(x).transpose(1, 2)).transpose(1, 2), approximate="tanh")
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.out(a.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate="tanh"))


class SVTRLite(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = copy.deepcopy(config or DEFAULT_CONFIG)
        c = self.config
        s0, s1 = c["stem"]
        self.stem = nn.ModuleList([conv_bn(1, s0, (2, 2)), conv_bn(s0, s1, (2, 2))])
        self.mix = nn.ModuleList([conv_bn(s1, s1) for _ in range(c["mix_blocks"])])
        self.merge = conv_bn(s1, c["merge"], (2, 1))
        self.proj = nn.Linear(c["merge"] * 4, c["dim"])
        self.blocks = nn.ModuleList([Block(c["dim"], c["heads"], c["mlp"]) for _ in range(c["blocks"])])
        self.norm = nn.LayerNorm(c["dim"], eps=LN_EPS)
        self.fc = nn.Linear(c["dim"], c["classes"])

    def out_len(self, width: int) -> int:
        return output_length(width)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.stem:
            x = m(x)
        for m in self.mix:
            x = x + m(x)
        x = self.merge(x)  # [B, C, 4, T]
        b, c, h, t = x.shape
        return self.proj(x.permute(0, 3, 1, 2).reshape(b, t, c * h))  # channel-major flatten of (C, H)

    def forward(self, x: torch.Tensor, widths: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, 1, 32, W] normalised; widths: true widths when padded. Returns logits [B, T, classes]."""
        seq = self.tokens(x)
        mask = None
        if widths is not None:
            lengths = torch.tensor([output_length(int(w)) for w in widths], device=seq.device)
            mask = (torch.arange(seq.shape[1], device=seq.device).view(1, -1) < lengths.view(-1, 1)).view(-1, 1, 1, seq.shape[1])
        for blk in self.blocks:
            seq = blk(seq, mask)
        return self.fc(self.norm(seq))


def macs_per_column(config: dict, t: int = 80) -> dict:
    """Multiply-accumulates per output column; the attention term grows with the sequence length t (80 is a 320 px line)."""
    s0, s1 = config["stem"]
    d, m = config["dim"], config["mlp"]
    stem = s0 * 9 * 16 * 2 + s1 * s0 * 9 * 8
    mix = config["mix_blocks"] * s1 * s1 * 9 * 8
    merge = config["merge"] * s1 * 9 * 4
    proj = config["merge"] * 4 * d
    per_block = d * d * 3 + 4 * d * d + 2 * m * d * d + 2 * t * d
    total = stem + mix + merge + proj + config["blocks"] * per_block + d * config["classes"]
    return {
        "stem": stem,
        "mix": mix,
        "merge": merge,
        "proj": proj,
        "blocks": config["blocks"] * per_block,
        "fc": d * config["classes"],
        "total": total,
        "t": t,
    }


def fold_batch_norm(model: SVTRLite) -> dict[str, torch.Tensor]:
    """Container tensors in fp32: batch norms folded into their convolutions (in fp64, rounded once); the rest as trained."""
    sd: dict[str, torch.Tensor] = {}

    def fold(name: str, blk: nn.Sequential) -> None:
        conv, bn = blk[0], blk[1]
        scale = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
        sd[f"{name}.weight"] = (conv.weight.double() * scale.view(-1, 1, 1, 1)).float()
        sd[f"{name}.bias"] = (bn.bias.double() - bn.running_mean.double() * scale).float()

    for i, m in enumerate(model.stem):
        fold(f"stem.{i}", m)
    for i, m in enumerate(model.mix):
        fold(f"mix.{i}", m)
    fold("merge", model.merge)
    for prefix, mod in [("proj", model.proj), ("norm", model.norm), ("fc", model.fc)]:
        for k, v in mod.state_dict().items():
            sd[f"{prefix}.{k}"] = v.clone()
    for i, blk in enumerate(model.blocks):
        for k, v in blk.state_dict().items():
            sd[f"block.{i}.{k}"] = v.clone()
    return {k: v.detach().cpu().float() for k, v in sd.items()}


def load_folded(config: dict, tensors: dict[str, torch.Tensor]) -> SVTRLite:
    """An SVTRLite whose batch norms are exact identities and whose convolutions are the container's folded tensors."""
    m = SVTRLite(config)
    sd = {}
    groups = (
        [(f"stem.{i}", f"stem.{i}") for i in range(2)]
        + [(f"mix.{i}", f"mix.{i}") for i in range(config["mix_blocks"])]
        + [("merge", "merge")]
    )
    for name, _ in groups:
        sd[f"{name}.0.weight"] = tensors[f"{name}.weight"]
        bias = tensors[f"{name}.bias"]
        sd[f"{name}.1.weight"] = torch.ones_like(bias)
        sd[f"{name}.1.bias"] = bias
        sd[f"{name}.1.running_mean"] = torch.zeros_like(bias)
        sd[f"{name}.1.running_var"] = torch.full_like(bias, 1.0 - BN_EPS)
        sd[f"{name}.1.num_batches_tracked"] = torch.tensor(0)
    for k, v in tensors.items():
        if k.startswith(("proj.", "norm.", "fc.")):
            sd[k] = v
        elif k.startswith("block."):
            _, i, rest = k.split(".", 2)
            sd[f"blocks.{i}.{rest}"] = v
    m.load_state_dict(sd)
    return m.eval()


class FoldedReference:
    """fp32 (or fp64) forward over the container's tensors with every stage's output exposed, for parity fixtures."""

    def __init__(self, config: dict, tensors: dict[str, torch.Tensor], dtype: torch.dtype = torch.float32):
        self.config = config
        self.dtype = dtype
        self.t = {k: v.to(dtype) for k, v in tensors.items()}

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [1, 1, 32, W] normalised. Returns {'stem.0': [C,H,W], ..., 'merge', 'proj': [T,D], 'block.i': [T,D], 'fc': [T,classes]}."""
        c, t = self.config, self.t
        out: dict[str, torch.Tensor] = {}
        x = x.to(self.dtype)

        def cbg(name: str, x: torch.Tensor, stride) -> torch.Tensor:
            return F.gelu(F.conv2d(x, t[f"{name}.weight"], t[f"{name}.bias"], stride=stride, padding=1), approximate="tanh")

        for i in range(2):
            x = cbg(f"stem.{i}", x, (2, 2))
            out[f"stem.{i}"] = x[0].clone()
        for i in range(c["mix_blocks"]):
            x = x + cbg(f"mix.{i}", x, (1, 1))
            out[f"mix.{i}"] = x[0].clone()
        x = cbg("merge", x, (2, 1))
        out["merge"] = x[0].clone()
        b, ch, h, w = x.shape
        seq = F.linear(x.permute(0, 3, 1, 2).reshape(b, w, ch * h), t["proj.weight"], t["proj.bias"])
        out["proj"] = seq[0].clone()
        d, heads = c["dim"], c["heads"]

        def ln(name: str, v: torch.Tensor) -> torch.Tensor:
            return F.layer_norm(v, (d,), t[f"{name}.weight"], t[f"{name}.bias"], LN_EPS)

        for i in range(c["blocks"]):
            p = f"block.{i}"
            loc = F.conv1d(ln(f"{p}.norm0", seq).transpose(1, 2), t[f"{p}.local.weight"], t[f"{p}.local.bias"], padding=1).transpose(1, 2)
            seq = seq + F.gelu(loc, approximate="tanh")
            q, k, v = (
                F.linear(ln(f"{p}.norm1", seq), t[f"{p}.qkv.weight"], t[f"{p}.qkv.bias"])
                .view(b, w, 3, heads, d // heads)
                .permute(2, 0, 3, 1, 4)
            )
            att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // heads), -1) @ v
            seq = seq + F.linear(att.transpose(1, 2).reshape(b, w, d), t[f"{p}.out.weight"], t[f"{p}.out.bias"])
            hmid = F.gelu(F.linear(ln(f"{p}.norm2", seq), t[f"{p}.fc1.weight"], t[f"{p}.fc1.bias"]), approximate="tanh")
            seq = seq + F.linear(hmid, t[f"{p}.fc2.weight"], t[f"{p}.fc2.bias"])
            out[p] = seq[0].clone()
        out["fc"] = F.linear(ln("norm", seq), t["fc.weight"], t["fc.bias"])[0].clone()
        return out
