"""Greedy CTC decoding with the word confidences of the container contract.

Raw word confidence: for each emitted character (a run of equal non-blank arg-max steps) take the largest softmax probability of
its class over the run; a word's raw confidence is the mean of that value over the word's characters. Words are the maximal
runs of non-whitespace emitted characters. Confidence is the calibrated probability that the word is exactly right, obtained by
passing the raw value through a per-script monotone piecewise-linear map (`apply_map`).

Hypothesis records written here: {"id", "ref", "hyp", "words": [{"text", "conf", "script"}]}.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch

from eval.scoring import line_script
from train.data import LineSet, to_input


def decode_chars(logits: torch.Tensor, charset: list[str]) -> list[tuple[str, float]]:
    """logits [T, classes] -> [(character, run-max probability)] in emission order."""
    p = torch.softmax(logits.float(), -1)
    ids = p.argmax(-1).tolist()
    top = p.max(-1).values.tolist()
    out: list[tuple[str, float]] = []
    prev, best = 0, 0.0
    for t, i in enumerate(ids):
        if i != prev:
            if prev != 0:
                out.append((charset[prev], best))
            best = 0.0
        if i != 0:
            best = max(best, top[t])
        prev = i
    if prev != 0:
        out.append((charset[prev], best))
    return out


def words_of(chars: list[tuple[str, float]]) -> list[dict]:
    """Words with raw confidence (mean of their characters' values) and script."""
    words, cur = [], []
    for ch, c in chars + [(" ", 0.0)]:
        if ch.strip():
            cur.append((ch, c))
        elif cur:
            text = "".join(x[0] for x in cur)
            words.append({"text": text, "conf": float(np.float32(np.mean([x[1] for x in cur]))), "script": line_script(text)})
            cur = []
    return words


@torch.no_grad()
def hypotheses(model, data: LineSet, charset: list[str], device: str = "cuda", batch_size: int = 128) -> list[dict]:
    model.eval()
    order = sorted(range(len(data)), key=lambda i: data.widths[i])
    rows: list[dict | None] = [None] * len(data)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        x, widths = data.batch(idx)
        logits = model(to_input(x, device), widths).float().cpu()
        for k, i in enumerate(idx):
            chars = decode_chars(logits[k, : model.out_len(int(widths[k]))], charset)
            rows[i] = {"id": data.ids[i], "ref": data.texts[i], "hyp": "".join(c for c, _ in chars), "words": words_of(chars)}
    return rows  # type: ignore[return-value]


def write_hyps(rows: list[dict], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def apply_map(x: float, knots: dict) -> float:
    """Monotone piecewise-linear map through (x[i], y[i]); constant beyond the first and last knot. Container rule: float64
    `y0 + (y1 - y0) * ((r - x0) / (x1 - x0))` without FMA (Python floats), cast to float32 at the end."""
    xs, ys = knots["x"], knots["y"]
    if x <= xs[0]:
        y = ys[0]
    elif x >= xs[-1]:
        y = ys[-1]
    else:
        i = bisect.bisect_right(xs, x)
        y = ys[i - 1] + (ys[i] - ys[i - 1]) * ((x - xs[i - 1]) / (xs[i] - xs[i - 1]))
    return float(np.float32(y))
