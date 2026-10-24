"""DBNet-style text detector with a layout head, small enough for CPU kernels.

Text detection follows Liao et al., "Real-time Scene Text Detection with
Differentiable Binarization" (AAAI 2020, arXiv:1911.08947): a probability map
of shrunk text regions, a threshold map, and the approximate binary map
sigmoid(k (P - T)) used only to train. At inference only the probability map
is used, so the threshold branch is dropped from the exported file.

Changes for a CPU budget (all conv, depthwise conv, ReLU, nearest upsample, add):
  * a depthwise-separable backbone built from random initialisation;
  * a light FPN (1x1 laterals, nearest top-down add) instead of concatenating
    four 256-channel maps;
  * the text map is produced at stride 2, which words of about 10 px need.

The layout head predicts, per region class, a shrunk "core" map at stride 8,
the same idea as DB applied to boxes: each connected component of a class map
is one region, so no non-maximum suppression is needed. An orientation head
(global average pool of the stride-32 feature, one linear layer) classifies
the page rotation 0/90/180/270.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LAYOUT_CLASSES = ["text", "title", "list", "table", "figure", "header_footer", "form_field", "checkbox"]

DEFAULT = {
    "arch": "dbnet-lite",
    "stem": 16,
    "widths": [32, 72, 144, 224],  # stride 4, 8, 16, 32
    "depths": [2, 3, 4, 3],
    "fpn": 56,
    "text": 32,
    "layout": 112,
    "layout_blocks": 3,
    "classes": len(LAYOUT_CLASSES),
    # dbnet-lite-v1 additions; the defaults reproduce the dbnet-lite-v0 graph exactly
    "layout16_blocks": 0,  # > 0: a stride-16 context branch feeding the layout head
    "layout16": 128,
    "orient16": False,  # True: orientation from the mean of the stride-16 and stride-32 features
}


def arch_id(cfg: dict) -> str:
    return "dbnet-lite-v1" if cfg.get("layout16_blocks", 0) > 0 or cfg.get("orient16") else "dbnet-lite-v0"


def conv_bn(cin, cout, k=3, s=1, groups=1, relu=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DW(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, each followed by batch norm and ReLU."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = conv_bn(cin, cin, 3, stride, groups=cin)
        self.pw = conv_bn(cin, cout, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


def up2(x):
    return F.interpolate(x, scale_factor=2, mode="nearest")


class DBNetLite(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        c = {**DEFAULT, **(cfg or {})}
        self.cfg = c
        self.stem = conv_bn(1, c["stem"], 3, 2)
        stages = []
        cin = c["stem"]
        for w, d in zip(c["widths"], c["depths"], strict=False):
            blocks = [DW(cin, w, 2)] + [DW(w, w) for _ in range(d - 1)]
            stages.append(nn.Sequential(*blocks))
            cin = w
        self.stages = nn.ModuleList(stages)
        f = c["fpn"]
        self.lat = nn.ModuleList([nn.Conv2d(w, f, 1) for w in c["widths"]])
        self.smooth8 = DW(f, f)
        self.smooth4 = DW(f, f)
        # layout head on the stride-8 pyramid level
        lay = [DW(f, c["layout"])] + [DW(c["layout"], c["layout"]) for _ in range(c["layout_blocks"] - 1)]
        self.layout = nn.Sequential(*lay)
        self.layout_out = nn.Conv2d(c["layout"], c["classes"], 1)
        if c["layout16_blocks"] > 0:
            n, w16 = c["layout16_blocks"], c["layout16"]
            self.layout16 = nn.Sequential(DW(f, w16), *[DW(w16, w16) for _ in range(n - 1)])
            self.layout_mix = nn.Conv2d(w16, c["layout"], 1)
        # text head: stride-4 level, up to stride 2 together with the stem feature
        self.text4 = DW(f, c["text"])
        self.stem_proj = nn.Conv2d(c["stem"], c["text"], 1)
        self.text2 = DW(c["text"], c["text"])
        self.prob_out = nn.Conv2d(c["text"], 1, 1)
        # training-only threshold branch (same shape, dropped at export)
        self.thr4 = DW(f, c["text"])
        self.thr_proj = nn.Conv2d(c["stem"], c["text"], 1)
        self.thr2 = DW(c["text"], c["text"])
        self.thr_out = nn.Conv2d(c["text"], 1, 1)
        self.orient = nn.Linear(c["widths"][-1] + (c["widths"][2] if c["orient16"] else 0), 4)

    def forward(self, x, train_heads: bool = True):
        s2 = self.stem(x)
        feats = []
        h = s2
        for st in self.stages:
            h = st(h)
            feats.append(h)
        l4, l8, l16, l32 = (conv(f) for conv, f in zip(self.lat, feats, strict=True))
        p16 = l16 + up2(l32)
        p8 = self.smooth8(l8 + up2(p16))
        p4 = self.smooth4(l4 + up2(p8))
        if self.cfg["layout16_blocks"] > 0:
            h = self.layout[0](p8) + up2(self.layout_mix(self.layout16(p16)))
            for blk in list(self.layout)[1:]:
                h = blk(h)
        else:
            h = self.layout(p8)
        layout = self.layout_out(h)
        g = feats[3].mean((2, 3))
        if self.cfg["orient16"]:
            g = torch.cat([feats[2].mean((2, 3)), g], 1)
        t = self.text2(up2(self.text4(p4)) + self.stem_proj(s2))
        out = {"prob": self.prob_out(t), "layout": layout, "orient": self.orient(g)}
        if train_heads:
            th = self.thr2(up2(self.thr4(p4)) + self.thr_proj(s2))
            out["thresh"] = self.thr_out(th)
        return out


def count_macs(model: DBNetLite, h: int, w: int) -> dict:
    """Multiply-accumulates of the inference graph (no threshold branch) for a 1xHxW input."""
    macs = {"total": 0}
    skip = ("thr4", "thr_proj", "thr2", "thr_out")
    hooks = []

    def hook(name):
        def f(m, inp, out):
            if name.split(".")[0] in skip:
                return
            if isinstance(m, nn.Conv2d):
                n = out.numel() // out.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            elif isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
            else:
                return
            macs[name] = n
            macs["total"] += n

        return f

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook(name)))
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, h, w, device=next(model.parameters()).device), train_heads=False)
    model.train(was)
    for hk in hooks:
        hk.remove()
    return macs


def count_params(model: DBNetLite) -> int:
    """Stored parameters of the inference graph: conv weights, one folded bias per conv, the linear layer."""
    n = 0
    for name, m in model.named_modules():
        if name.split(".")[0] in ("thr4", "thr_proj", "thr2", "thr_out"):
            continue
        if isinstance(m, nn.Conv2d):
            n += m.weight.numel() + m.out_channels
        elif isinstance(m, nn.Linear):
            n += m.weight.numel() + m.bias.numel()
    return n


if __name__ == "__main__":
    m = DBNetLite()
    for h, w in ((1024, 736), (1024, 1024)):
        mm = count_macs(m, h, w)
        print(h, w, f"{mm['total'] / 1e9:.3f} GMACs")
    print("stored parameters:", count_params(m), "fp16 bytes:", 2 * count_params(m))
