"""Turns the clean packs of the evaluation sets into degraded 32 px crops, once, with logged parameters.

Usage: uv run python -m degrade.materialize CONFIG DATA_DIR [SET ...]

DATA_DIR/clean/<set>/ (LineGen --clean) -> DATA_DIR/<set>/{images/<id>.png, labels.jsonl, params.jsonl}. Which side of the config's
gpu_degrade section a set uses (training or held-out ranges) comes from its `degrade` entry. The seed is the config seed and the
set's position in the config, and batches are taken in plan order, so the output is reproducible on the same device and torch
version. Training data is not materialised: it is degraded on the fly, with fresh draws every step.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

from data import charset as cs
from degrade.gpu import degrade, param_records
from train.data import PackSet

BASE = Path(__file__).resolve().parents[1]
BATCH = 250


def main() -> None:
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    root = Path(sys.argv[2])
    only = set(sys.argv[3:])
    charset = cs.load(BASE / cfg["charset"])
    for i, (name, spec) in enumerate(cfg["sets"].items()):
        if name.endswith("train") or (only and name not in only):
            continue
        pack = PackSet(root / "clean" / name, charset)
        out = root / name
        (out / "images").mkdir(parents=True, exist_ok=True)
        g = torch.Generator(device="cuda")
        g.manual_seed(cfg["seed"] * 1000 + i)
        sides = [cfg["gpu_degrade"][k] for k in ((["train", "heldout"]) if spec["degrade"] == "all" else [spec["degrade"]])]
        labels, params = [], []
        for s in range(0, len(pack), BATCH):
            idx = list(range(s, min(s + BATCH, len(pack))))
            x, w = pack.batch(idx)
            # "all": even samples use the training ranges, odd samples the held-out ranges
            parts = [(slice(k, None, len(sides)), sd) for k, sd in enumerate(sides)]
            img = [None] * len(idx)
            w32 = torch.zeros(len(idx), dtype=torch.long)
            plist = [None] * len(idx)
            for sl, sd in parts:
                sub = list(range(len(idx)))[sl]
                im, ww, pp = degrade(x[sub].cuda(), w[sub].cuda(), sd, g)
                im = im.cpu().numpy()
                recs = param_records(pp, [pack.ids[idx[j]] for j in sub])
                for n, j in enumerate(sub):
                    img[j] = im[n]
                    w32[j] = ww[n]
                    plist[j] = recs[n]
            for k, j in enumerate(idx):
                png = out / "images" / (pack.ids[j] + ".png")
                Image.fromarray(img[k][0, :, : int(w32[k])], mode="L").save(png)
                labels.append({"id": pack.ids[j], "text": pack.texts[j], "w": int(w32[k]), "h": 32, "font": pack.fonts[j],
                               "sha256": hashlib.sha256(png.read_bytes()).hexdigest()})  # fmt: skip
            params += plist
        (out / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels), encoding="utf-8")
        (out / "params.jsonl").write_text("".join(json.dumps(r) + "\n" for r in params), encoding="utf-8")
        print(f"{name}: {len(labels)} crops ({spec['degrade']} ranges) -> {out}", flush=True)


if __name__ == "__main__":
    main()
