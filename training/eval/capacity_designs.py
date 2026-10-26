# ruff: noqa: E501
"""Size, compute and single-thread CPU latency of recognizer designs, from the model configs.

    uv run python -m eval.capacity_designs [--width 1000] [--repeat 5]

Latency is torch on one CPU thread with fp32 weights, a proxy for the C# kernels (which run the same fp32 arithmetic but are
not the same code): it ranks designs, it does not predict milliseconds. Parameters count the tensors a container would hold
(batch norm folded away); fp16 size is two bytes each.
"""

from __future__ import annotations

import argparse
import copy
import time

import torch

from models import build, fold, macs_per_column
from models.crnn import DEFAULT_CONFIG as CRNN_BASE

CLASSES = 622


def crnn(conv: list[tuple[int, list[int] | None]], hidden: int, layers: int = 2) -> dict:
    return {
        "conv": [{"out": o, "kernel": [3, 3], "padding": [1, 1], "pool": p} for o, p in conv],
        "lstm": {"hidden": hidden, "layers": layers},
        "classes": CLASSES,
    }


S4 = [[2, 2], [2, 2], None, [2, 1], [2, 1], [2, 1]]  # width stride 4 (production)
S2 = [[2, 2], [2, 1], None, [2, 1], [2, 1], [2, 1]]  # width stride 2: twice the time steps for the same pixels

DESIGNS: dict[str, dict] = {
    "crnn-v0 (0.6.0)": copy.deepcopy(CRNN_BASE) | {"classes": CLASSES},
    # same widths, finer time axis: does the stride-4 step lose narrow or touching glyphs?
    "A0 crnn stride 2, same widths": crnn(list(zip([32, 64, 128, 128, 192, 192], S2, strict=True)), 192),
    # wider, stride 4
    "A1 crnn wide": crnn(list(zip([48, 96, 192, 192, 256, 256], S4, strict=True)), 256),
    # stride 2 and wider, sized for the latency limit
    "A2 crnn stride 2, wide": crnn(list(zip([32, 64, 128, 160, 224, 224], S2, strict=True)), 256),
    "B1 svtr-lite wide": {
        "arch": "svtr-lite",
        "stem": [48, 96],
        "mix_blocks": 2,
        "merge": 160,
        "dim": 256,
        "blocks": 8,
        "heads": 8,
        "mlp": 2,
        "classes": CLASSES,
    },
    "B2 svtr-lite medium": {
        "arch": "svtr-lite",
        "stem": [32, 64],
        "mix_blocks": 2,
        "merge": 128,
        "dim": 256,
        "blocks": 6,
        "heads": 8,
        "mlp": 2,
        "classes": CLASSES,
    },
}


def params(model: torch.nn.Module) -> int:
    return sum(v.numel() for v in fold(model).values())


@torch.no_grad()
def latency_ms(model: torch.nn.Module, width: int, repeat: int) -> float:
    model.eval()
    x = torch.randn(1, 1, 32, width)
    for _ in range(2):
        model(x)
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        model(x)
        ts.append(time.perf_counter() - t)
    return 1000 * sorted(ts)[len(ts) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(1)
    print(
        f"| design | params | fp16 MB | GMACs per 32x{a.width} line | T | 1-thread ms | x baseline |\n|---|---:|---:|---:|---:|---:|---:|"
    )
    base = None
    for name, cfg in DESIGNS.items():
        m = build(cfg)
        p = params(m)
        macs = macs_per_column(m.config, m.out_len(a.width)) if cfg.get("arch") else macs_per_column(m.config)
        t = m.out_len(a.width)
        if cfg.get("arch"):  # svtr: the shared estimate leaves out each block's width convolution
            total = (macs["total"] * t + cfg["blocks"] * 3 * cfg["dim"] ** 2 * t) / 1e9
        else:
            total = macs["total"] * t / 1e9
        ms = latency_ms(m, a.width, a.repeat)
        base = base or ms
        print(f"| {name} | {p / 1e6:.2f}M | {2 * p / 1e6:.1f} | {total:.2f} | {t} | {ms:.1f} | {ms / base:.2f} |")


if __name__ == "__main__":
    main()
