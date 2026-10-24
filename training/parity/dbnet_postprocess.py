"""Reference boxes for the C# post-processing of the page detector.

Usage: uv run python -m parity.dbnet_postprocess --fixtures DIR [--seed 11]

Reads the head outputs stored by `parity.dbnet_fixtures` in DIR (`data/<tile>.prob.f32`, `.layout.f32`, `.orient.f32`) and adds
seeded synthetic maps (rectangles of high logits with noise, diagonal neighbours, single cells, edge contact). Each case is decoded
with `eval.detect.decode` at scale 1 and written to DIR/postprocess.json together with the map files in DIR/postprocess/. The C#
side must find the same lines, regions and orientation (it also clips boxes to the page, which `decode` does not).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.detect import decode
from models.dbnet import LAYOUT_CLASSES


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return (e / e.sum()).astype(np.float32)


def synthetic(rng: np.random.Generator, ph: int, pw: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = rng.normal(-2.5, 0.8, (ph, pw)).astype(np.float32)
    for _ in range(int(rng.integers(3, 9))):
        h, w = int(rng.integers(2, 7)), int(rng.integers(8, pw // 2))
        y, x = int(rng.integers(0, ph - h + 1)), int(rng.integers(0, pw - w + 1))
        prob[y : y + h, x : x + w] = rng.uniform(0.3, 4.0, (h, w))
    for _ in range(6):  # single cells and diagonal neighbours: not lines, not joined
        y, x = int(rng.integers(0, ph - 1)), int(rng.integers(0, pw - 1))
        prob[y, x] = 3.0
        if rng.random() < 0.5:
            prob[y + 1, x + 1] = 3.0
    prob[: int(rng.integers(2, 5)), : int(rng.integers(6, 20))] = 2.0  # touching the page edge
    lh, lw = ph // 4, pw // 4
    layout = rng.normal(-2.5, 0.8, (len(LAYOUT_CLASSES), lh, lw)).astype(np.float32)
    for c in range(len(LAYOUT_CLASSES)):
        for _ in range(int(rng.integers(0, 4))):
            h, w = int(rng.integers(1, max(2, lh // 2))), int(rng.integers(1, max(2, lw // 2)))
            y, x = int(rng.integers(0, lh - h + 1)), int(rng.integers(0, lw - w + 1))
            layout[c, y : y + h, x : x + w] = rng.uniform(0.2, 4.0, (h, w))
    orient = rng.normal(0, 2.0, 4).astype(np.float32)
    return prob, layout, orient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    root = Path(a.fixtures)
    out = root / "postprocess"
    out.mkdir(exist_ok=True)
    doc = json.loads((root / "fixtures.json").read_text())
    rng = np.random.default_rng(a.seed)
    cases = []

    def add(name: str, prob: np.ndarray, layout: np.ndarray, orient: np.ndarray) -> None:
        ph, pw = prob.shape
        m = {"prob": sigmoid(prob), "layout": sigmoid(layout), "orient": softmax(orient), "scale": 1.0}
        res = decode(m)
        (out / f"{name}.prob.f32").write_bytes(prob.astype("<f4").tobytes())
        (out / f"{name}.layout.f32").write_bytes(layout.astype("<f4").tobytes())
        cases.append(
            {
                "name": name,
                "prob_shape": [ph, pw],
                "layout_shape": list(layout.shape),
                "orient": [float(v) for v in orient],
                "expected_orient": [float(v) for v in res["orient"]],
                "lines": [[float(v) for v in b] for b in res["text"]],
                "line_scores": [float(v) for v in res["text_scores"]],
                "regions": [{"class": int(c), "score": float(s), "box": [float(v) for v in b]} for c, s, b in res["layout"]],
            }
        )

    for s in doc["samples"]:
        got = {}
        for layer in s["layers"]:
            if layer["name"] in ("prob", "layout", "orient"):
                got[layer["name"]] = np.frombuffer((root / layer["file"]).read_bytes(), "<f4").reshape(layer["shape"])
        add(f"fixture-{s['id']}", got["prob"][0], got["layout"], got["orient"])
    for k, (ph, pw) in enumerate([(64, 80), (96, 128), (128, 96), (200, 160), (48, 256), (256, 64)]):
        add(f"synthetic-{k}", *synthetic(rng, ph, pw))
    (root / "postprocess.json").write_text(json.dumps({"classes": LAYOUT_CLASSES, "cases": cases}, indent=1), encoding="utf-8")
    print(f"{len(cases)} cases -> {root / 'postprocess.json'}")


if __name__ == "__main__":
    main()
