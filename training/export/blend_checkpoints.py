"""Linear interpolation between two checkpoints of one architecture: w = (1 - alpha) * base + alpha * tuned.

A fine-tune that gains on a new kind of data can lose a little on what the base model did well; interpolating the weights (Wortsman et
al., "Robust fine-tuning of zero-shot models", 2022, applied to our own checkpoints) trades the two off with one number, picked on
held-out data. Integer buffers (batch-norm counters) are taken from the tuned checkpoint.

    uv run python -m export.blend_checkpoints BASE.pt TUNED.pt OUT.pt --alpha 0.5
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("tuned")
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, required=True)
    a = ap.parse_args()
    b = torch.load(a.base, map_location="cpu", weights_only=False)
    t = torch.load(a.tuned, map_location="cpu", weights_only=False)
    assert b["config"] == t["config"], "different architectures"
    out = dict(t)
    out["model"] = {
        k: ((1 - a.alpha) * b["model"][k].float() + a.alpha * v.float()).to(v.dtype) if v.is_floating_point() else v
        for k, v in t["model"].items()
    }
    out["blend"] = {"alpha": a.alpha, "base_step": b.get("step"), "tuned_step": t.get("step")}
    torch.save(out, a.out)


if __name__ == "__main__":
    main()
