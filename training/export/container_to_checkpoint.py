"""Turns a recognizer container back into a training checkpoint, so a released model can be fine-tuned.

The container holds batch norm folded into the conv weights, so the checkpoint's batch norms are exact identities
(`models.crnn.load_folded`); train with `train.freeze_bn: true` so they stay identities, and the next export folds them back
unchanged. The weights are the container's fp16 values widened to fp32.

    uv run python -m export.container_to_checkpoint IN.onm OUT.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from export.container import read_container
from models import load_folded


def main() -> None:
    manifest, tensors = read_container(sys.argv[1])
    model = load_folded(manifest["config"], {k: torch.from_numpy(v) for k, v in tensors.items()})
    torch.save({"model": model.state_dict(), "step": 0, "best": float("inf"), "config": manifest["config"], "charset": manifest["charset"],
                "manifest": {"started_from": Path(sys.argv[1]).name, "name": manifest.get("name"), "version": manifest.get("version")}},
               sys.argv[2])  # fmt: skip
    print(f"{sys.argv[2]}: {sum(v.size for v in tensors.values())} parameters from {manifest['name']} {manifest['version']}")


if __name__ == "__main__":
    main()
