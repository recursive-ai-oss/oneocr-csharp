"""Smoke check of the targeted-error loop and the reader-resampling degradation on a few dozen synthetic lines, on the CPU.

    uv run python -m train.check_adaptive

Builds a tiny clean pack (text drawn with PIL's default font), runs the degradation with the lines-v6 `adapt` side (resampling,
thermal effects, binned draws), measures a random-initialised model on it, publishes two weight versions and draws indices from them.
Exits non-zero if a shape, a weight or a determinism check fails.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from data import charset as cs
from degrade.gpu import degrade
from models import build, default_config
from train.adaptive import Adaptive
from train.data import PackSet

BASE = Path(__file__).resolve().parents[1]
TEXTS = ["Total 12.50", "VAT 20% 3,400", "Hello world", "2026-10-24", "INV-0042/B", "Subtotal 1 250", "abc def", "No. 7731"]


def make_pack(root: Path, n: int, fonts: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows, blob, off = [], bytearray(), 0
    for i in range(n):
        t = TEXTS[i % len(TEXTS)] + (" x" * (i % 3))
        im = Image.new("L", (24 * len(t) // 2 + 20, 48), 255)
        ImageDraw.Draw(im).text((8, 14), t, fill=0)
        a = np.asarray(im, dtype=np.uint8)
        rows.append({"id": f"s{i}", "text": t, "w": a.shape[1], "h": 48, "off": off, "font": fonts[i % len(fonts)], "sha256": ""})
        blob += a.tobytes()
        off += a.size
    (root / "clean.u8").write_bytes(bytes(blob))
    (root / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    cfg = yaml.safe_load((BASE / "configs/lines-v6.yaml").read_text(encoding="utf-8"))
    acfg = yaml.safe_load((BASE / "configs/crnn-v8.yaml").read_text(encoding="utf-8"))["sources"][0]["adaptive"]
    acfg = {**acfg, "every": 10}
    side = cfg["gpu_degrade"]["adapt"]
    charset = cs.load(BASE / "data/charset-v1.txt")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fonts = ["ofl/a/A.ttf", "ofl/b/B.ttf", "ofl/c/C.ttf"]
        make_pack(tmp / "train", 64, fonts)
        make_pack(tmp / "stream", 32, fonts)
        pack, stream = PackSet(tmp / "train", charset), PackSet(tmp / "stream", charset)
        g = torch.Generator()
        g.manual_seed(1)
        x, w = pack.batch(list(range(16)))
        out, w32, params = degrade(x, w, side, g)
        assert out.shape[:3] == (16, 1, 32) and out.dtype == torch.uint8, out.shape
        assert params["resample"].max() >= 1 and "dropout" in params and params["rs_shift"].shape == (16, 2, 2)
        torch.manual_seed(0)
        model = build(default_config("crnn-v0", len(charset)))
        ad = Adaptive(pack, stream, charset, acfg, side, 8, tmp)
        for n, p in ad.base_probs.items():
            assert abs(sum(p) - 1) < 1e-6, n
        ad.publish(0, ad.initial())
        state, report = ad.measure(model, 1, "cpu", ad.versions[0])
        ad.publish(1, state)
        assert set(report["kind_error"]) and all(abs(sum(p) - 1) < 1e-6 for p in state["bins"].values())
        a, b = ad.sample(12, 32), ad.sample(12, 32)
        assert a == b and max(a) < len(pack), "sampling must be a function of the step"
        bins = ad.bins_for(1)
        out2, _, p2 = degrade(x, w, {**side, "_bins": bins}, g)
        assert out2.shape[2] == 32
        fresh = Adaptive(pack, stream, charset, acfg, side, 8, tmp)
        assert fresh.load(1) and np.allclose(fresh.versions[1]["cum"], ad.versions[1]["cum"]), "a resumed run must see the same weights"
        print("adaptive loop ok:", json.dumps({k: report[k] for k in ("char_error", "kind_error")})[:200])


if __name__ == "__main__":
    main()
