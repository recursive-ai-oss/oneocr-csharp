"""Calibration fixture: raw and calibrated word confidences for held-out lines, from the exported (fp16, folded) weights.

    uv run python -m parity.calibration_fixture --model model-v1.onm --data EVAL_SET_DIR --out OUT_DIR [--n 200]

OUT_DIR/calibration.json: {raw, scripts (the map), lines: [{id, png, text, decoded, words: [{index, text, script, raw, raw_bits,
calibrated, calibrated_bits}]}]}; OUT_DIR/lines/<id>.png. Bits are the float32 patterns as hex. The reference forward runs in
fp32 and the raw confidence is the mean over the word's character runs (largest softmax probability of the character's class over
its run), rounded to float32; a C# implementation's own forward differs by float rounding, so compare raw with a tolerance and the
map alone exactly: calibrated = float32(y0 + (y1 - y0) * ((r - x0) / (x1 - x0))) in float64 without FMA, constant outside the knots.
Lines: a third each of Latin (most), Greek and Cyrillic from the held-out set, evenly spread over the set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch

from eval.confidence import apply_map, decode_chars, words_of
from eval.scoring import line_script, normalize
from export.container import read_container
from models.crnn import FoldedReference
from train.data import load_gray
from train.data import normalize as pixels

NAMES = {"latin": "Latin", "greek": "Greek", "cyrillic": "Cyrillic"}


def bits(v: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', v))[0]:08x}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    manifest, tensors = read_container(a.model)
    cal = manifest["calibration"]["scripts"]
    ref = FoldedReference(manifest["config"], {k: torch.from_numpy(v) for k, v in tensors.items()})
    charset = manifest["charset"]
    data, out = Path(a.data), Path(a.out)
    (out / "lines").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (data / "labels.jsonl").read_text(encoding="utf-8").splitlines() if x]
    by: dict[str, list[dict]] = {k: [] for k in NAMES}
    for r in rows:
        if r["w"] <= 800:
            by[line_script(normalize(r["text"]))].append(r)
    quota = {"latin": a.n // 2, "greek": a.n // 4, "cyrillic": a.n - a.n // 2 - a.n // 4}
    lines = []
    for sc, q in quota.items():
        pool = by[sc]
        for i in np.linspace(0, len(pool) - 1, q).round().astype(int):
            r = pool[int(i)]
            png = out / "lines" / f"{r['id']}.png"
            shutil.copyfile(data / "images" / f"{r['id']}.png", png)
            x = torch.from_numpy(pixels(load_gray(png)))[None, None]
            logits = ref.forward(x)["fc"]
            chars = decode_chars(logits, charset)
            words = []
            for k, w in enumerate(words_of(chars)):
                raw = w["conf"]
                y = apply_map(raw, cal[NAMES[w["script"]]]) if NAMES[w["script"]] in cal else raw
                words.append({"index": k, "text": w["text"], "script": NAMES[w["script"]], "raw": raw, "raw_bits": bits(raw),
                              "calibrated": y, "calibrated_bits": bits(y)})  # fmt: skip
            decoded = "".join(c for c, _ in chars)
            lines.append({"id": r["id"], "png": f"lines/{png.name}", "text": r["text"], "decoded": decoded, "words": words})
    doc = {"raw": manifest["calibration"]["raw"], "scripts": cal, "lines": lines}
    (out / "calibration.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    n_words = sum(len(ln["words"]) for ln in lines)
    print(f"{len(lines)} lines, {n_words} words -> {out}/calibration.json")


if __name__ == "__main__":
    main()
