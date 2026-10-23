"""Line-level CER / WER of a CRNN on a rendered line set (greedy CTC decoding)."""

from __future__ import annotations

import torch

from models.crnn import CRNN, greedy_decode, output_length
from train.data import LineSet, edit_distance


@torch.no_grad()
def evaluate(model: CRNN, data: LineSet, charset: list[str], device: str = "cuda", batch_size: int = 128,
             limit: int | None = None) -> dict:  # fmt: skip
    model.eval()
    n = min(len(data), limit or len(data))
    order = sorted(range(n), key=lambda i: data.widths[i])
    char_err = char_tot = word_err = word_tot = exact = 0
    for s in range(0, n, batch_size):
        idx = order[s : s + batch_size]
        x, widths = data.batch(idx)
        logits = model(x.to(device), widths).float().cpu()
        for k, i in enumerate(idx):
            hyp = greedy_decode(logits[k, : output_length(int(widths[k]), model.config)], charset)
            ref = data.texts[i]
            char_err += edit_distance(hyp, ref)
            char_tot += len(ref)
            word_err += edit_distance(hyp.split(), ref.split())
            word_tot += len(ref.split())
            exact += hyp == ref
    return {"cer": char_err / char_tot, "wer": word_err / word_tot, "line_acc": exact / n, "lines": n}
