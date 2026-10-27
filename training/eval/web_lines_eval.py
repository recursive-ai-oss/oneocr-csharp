"""Line CER / WER of a recognizer container on a web line set, sliced by what the pixels came from.

    uv run python -m eval.web_lines_eval --model model-v1.onm --data OUT/heldout [--threads 4] [--out report.json]

Reads the exported container, so the weights are the fp16-rounded, batch-norm-folded ones the library runs, with fp32 compute.
Slices: publisher, device-pixel ratio, anti-aliasing, font size in device pixels, font family. CER is micro-averaged (edit
distance over reference characters). A CPU job: it takes the container, not a training checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from export.container import read_container
from models import load_folded
from models.crnn import greedy_decode
from train.data import LineSet, edit_distance, to_input


def size_bucket(px: float) -> str:
    for hi in (10, 12, 14, 17, 22, 32):
        if px < hi:
            return f"<{hi}px"
    return ">=32px"


@torch.no_grad()
def run(model, data: LineSet, charset: list[str], batch_size: int = 64) -> list[dict]:
    order = sorted(range(len(data)), key=lambda i: data.widths[i])
    out: list[dict | None] = [None] * len(data)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        x, widths = data.batch(idx)
        logits = model(to_input(x, "cpu"), widths).float()
        for k, i in enumerate(idx):
            hyp = greedy_decode(logits[k, : model.out_len(int(widths[k]))], charset)
            ref = data.texts[i]
            out[i] = {
                "id": data.ids[i],
                "ref": ref,
                "hyp": hyp,
                "ce": edit_distance(hyp, ref),
                "we": edit_distance(hyp.split(), ref.split()),
                "n": len(ref),
                "nw": len(ref.split()),
            }
    return out  # type: ignore[return-value]


def summarize(items: list[dict]) -> dict:
    n = sum(i["n"] for i in items)
    nw = sum(i["nw"] for i in items)
    exact = sum(i["ce"] == 0 for i in items)
    return {
        "lines": len(items),
        "chars": n,
        "cer": round(100 * sum(i["ce"] for i in items) / max(1, n), 3),
        "wer": round(100 * sum(i["we"] for i in items) / max(1, nw), 3),
        "line_acc": round(100 * exact / max(1, len(items)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--errors", help="write misread lines here (jsonl)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    manifest, tensors = read_container(a.model)
    charset = manifest["charset"]
    model = load_folded(manifest["config"], {k: torch.from_numpy(v) for k, v in tensors.items()}).eval()
    data = LineSet(a.data, charset)
    rows = [json.loads(line) for line in (Path(a.data) / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line]
    meta = {r["id"]: r for r in rows}
    t0 = time.time()
    res = run(model, data, charset)
    report = {"model": {"name": manifest.get("name"), "version": manifest.get("version")}, "data": a.data, "all": summarize(res),
              "seconds": round(time.time() - t0, 1), "threads": a.threads}  # fmt: skip
    slices = {"publisher": lambda m: m["publisher"], "dpr": lambda m: str(m["dpr"]), "aa": lambda m: m["aa"],
              "size": lambda m: size_bucket(m["em_px"]), "font": lambda m: ",".join(m["font"][:1]) or "?"}  # fmt: skip
    for name, fn in slices.items():
        groups = defaultdict(list)
        for r in res:
            groups[fn(meta[r["id"]])].append(r)
        report[name] = {k: summarize(v) for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])) if len(v) >= 20}
    if a.errors:
        Path(a.errors).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res if r["ce"]), encoding="utf-8")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
