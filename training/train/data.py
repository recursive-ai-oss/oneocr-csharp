"""Dataset loading for rendered line crops (images/<id>.png + labels.jsonl) and batching by width."""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MEAN, STD = 0.5, 0.5


def normalize(gray_u8: np.ndarray) -> np.ndarray:
    """Container input rule: x = p / 255, then (x - mean) / std."""
    return ((gray_u8.astype(np.float32) / 255.0) - MEAN) / STD


def to_input(x: torch.Tensor, device: str) -> torch.Tensor:
    """Container input rule on the device: x = p / 255, then (x - mean) / std."""
    return ((x.to(device, non_blocking=True).float() / 255.0) - MEAN) / STD


def load_gray(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        assert im.mode == "L", f"{path}: expected 8-bit gray, got {im.mode}"
        return np.asarray(im, dtype=np.uint8)


class LineSet:
    def __init__(self, root: str | Path, charset: list[str], limit: int | None = None):
        root = Path(root)
        rows = [json.loads(line) for line in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line]
        if limit:
            rows = rows[:limit]
        index = {c: i for i, c in enumerate(charset)}
        with ThreadPoolExecutor(16) as ex:
            self.images = list(ex.map(lambda r: load_gray(root / "images" / (r["id"] + ".png")), rows))
        self.texts = [r["text"] for r in rows]
        self.ids = [r["id"] for r in rows]
        self.targets = [[index[c] for c in t] for t in self.texts]
        self.widths = [im.shape[1] for im in self.images]

    def __len__(self) -> int:
        return len(self.images)

    def batch(self, idx: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """uint8 [B, 1, H, Wmax] padded by replicating each image's last column (paper colour); true widths. Use `to_input`."""
        wmax = max(self.widths[i] for i in idx)
        x = np.empty((len(idx), 1, self.images[idx[0]].shape[0], wmax), dtype=np.uint8)
        for k, i in enumerate(idx):
            im = self.images[i]
            x[k, 0, :, : im.shape[1]] = im
            x[k, 0, :, im.shape[1] :] = im[:, -1:]
        return torch.from_numpy(x), torch.tensor([self.widths[i] for i in idx])

    def epoch_batches(self, batch_size: int, rng: random.Random) -> list[list[int]]:
        """Width-bucketed batches: sort a shuffled pool in chunks so padding stays small."""
        order = list(range(len(self)))
        rng.shuffle(order)
        chunk = batch_size * 50
        batches = []
        for s in range(0, len(order), chunk):
            part = sorted(order[s : s + chunk], key=lambda i: self.widths[i])
            batches += [part[b : b + batch_size] for b in range(0, len(part), batch_size)]
        rng.shuffle(batches)
        return batches


def edit_distance(a: str | list, b: str | list) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
