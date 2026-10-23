"""Line-level CER / WER of a CRNN on a rendered line set (greedy CTC decoding)."""

from __future__ import annotations

import torch

from models import build
from models.crnn import greedy_decode
from train.data import LineSet, edit_distance, to_input


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: LineSet, charset: list[str], device: str = "cuda", batch_size: int = 128,
             limit: int | None = None, errors: list | None = None) -> dict:  # fmt: skip
    model.eval()
    n = min(len(data), limit or len(data))
    order = sorted(range(n), key=lambda i: data.widths[i])
    char_err = char_tot = word_err = word_tot = exact = 0
    for s in range(0, n, batch_size):
        idx = order[s : s + batch_size]
        x, widths = data.batch(idx)
        logits = model(to_input(x, device), widths).float().cpu()
        for k, i in enumerate(idx):
            hyp = greedy_decode(logits[k, : model.out_len(int(widths[k]))], charset)
            ref = data.texts[i]
            if errors is not None and hyp != ref:
                errors.append({"id": data.ids[i], "ref": ref, "hyp": hyp})
            char_err += edit_distance(hyp, ref)
            char_tot += len(ref)
            word_err += edit_distance(hyp.split(), ref.split())
            word_tot += len(ref.split())
            exact += hyp == ref
    return {"cer": char_err / char_tot, "wer": word_err / word_tot, "line_acc": exact / n, "lines": n}


def main() -> None:
    """uv run python -m eval.lines --checkpoint final.pt --data DIR --sets val,eval  (prints one JSON object per set)"""
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--errors", help="directory for <set>.errors.jsonl (misread lines)")
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cuda", weights_only=False)
    model = build(ck["config"]).cuda()
    model.load_state_dict(ck["model"])
    for s in a.sets.split(","):
        errs: list = []
        r = evaluate(model, LineSet(Path(a.data) / s, ck["charset"]), ck["charset"], errors=errs)
        if a.errors:
            Path(a.errors).mkdir(parents=True, exist_ok=True)
            (Path(a.errors) / f"{s.replace('/', '_')}.errors.jsonl").write_text(
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in errs), encoding="utf-8"
            )
        print(json.dumps({s: r}))


if __name__ == "__main__":
    main()
