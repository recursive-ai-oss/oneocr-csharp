"""How often a reader drops the first or last character of a line, on labelled crops of a given padding.

Counts, over lines of at least 3 characters, the share whose reading loses the first character, loses the last character, or is
wrong elsewhere (alignment by edit operations: a deletion at the very start or end of the truth). The two end rates are the check
for the failure where a word read from a strip that covers it fully still comes back one letter short ("Putih" -> "Puti").

    uv run python -m eval.end_drops --checkpoint best.pt --data DIR --sets recv_val,ends_val,cord_dev_box [--compare other.pt]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rapidfuzz.distance import Levenshtein

from models import build
from models.crnn import greedy_decode
from train.data import LineSet, to_input


@torch.no_grad()
def rates(model, data: LineSet, charset: list[str], device: str, batch_size: int = 128) -> dict:
    model.eval()
    order = sorted(range(len(data)), key=lambda i: data.widths[i])
    n = first = last = other = exact = 0
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        x, widths = data.batch(idx)
        logits = model(to_input(x, device), widths).float().cpu()
        for k, i in enumerate(idx):
            ref = data.texts[i]
            if len(ref) < 3:
                continue
            hyp = greedy_decode(logits[k, : model.out_len(int(widths[k]))], charset)
            n += 1
            if hyp == ref:
                exact += 1
                continue
            ops = Levenshtein.opcodes(ref, hyp)
            lost_first = any(o.tag == "delete" and o.src_start == 0 for o in ops)
            lost_last = any(o.tag == "delete" and o.src_end == len(ref) for o in ops)
            first += lost_first
            last += lost_last
            other += not (lost_first or lost_last)
    return {"lines": n, "exact": exact / max(n, 1), "first_deleted": first / max(n, 1), "last_deleted": last / max(n, 1),
            "other_errors": other / max(n, 1)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sets", required=True, help="comma list of set directories under --data")
    ap.add_argument("--compare", help="a second checkpoint, reported next to the first")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    models = {}
    for tag, path in (("checkpoint", a.checkpoint), ("compare", a.compare)):
        if path:
            ck = torch.load(path, map_location=a.device, weights_only=False)
            m = build(ck["config"]).to(a.device)
            m.load_state_dict(ck["model"])
            models[tag] = (m, ck["charset"])
    for s in a.sets.split(","):
        for tag, (m, charset) in models.items():
            r = rates(m, LineSet(Path(a.data) / s, charset), charset, a.device)
            print(json.dumps({"set": s, "model": tag, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}}))


if __name__ == "__main__":
    main()
