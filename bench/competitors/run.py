"""Run PaddleOCR or docTR (released pip packages, stock models) over a directory of pages, as a black box.

Run it with the Python of the engine's own virtual environment (see setup.sh), never with the benchmark's:

    venv-paddle/bin/python competitors/run.py --engine paddle --variant v5-mobile --pages P --out O --threads 1
    venv-doctr/bin/python  competitors/run.py --engine doctr  --variant default   --pages P --out O --threads 0

--threads 1 pins the engine to one CPU thread; 0 leaves the engine's own default. One process handles one set, so
the peak RSS it reports belongs to that set and configuration (it includes model load).

Writes <id>.txt (the engine's own line order, words of a line joined by spaces), <id>.json (word boxes in page
pixels, see the scorer) and timing.json. Time per page covers reading the file through the engine's documented
entry point and recognising it; model load is reported on its own.

Variants (names from each project's public documentation):
  paddle: v5-mobile = PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec, v5-server = PP-OCRv5_server_det + PP-OCRv5_server_rec,
          v5-mobile-en = PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec, v5-server-en = PP-OCRv5_server_det + en_PP-OCRv5_mobile_rec,
          v6-default = the package's default pipeline (PP-OCRv6_medium).
          The multilingual v5 recognizers leave out the spaces between words on small or degraded lines ("stillamiable,still");
          the project's documentation describes the English v5 recognizer as the one that fixes omitted spaces, so the -en
          variants are the fair English rows. The multilingual rows stay for reference.
  doctr:  default = ocr_predictor() with no arguments; mobile = db_mobilenet_v3_large + crnn_mobilenet_v3_small.
Paddle's document orientation, unwarping and text-line orientation steps are switched off (the sets are upright);
docTR keeps its documented defaults. Paddle runs with enable_mkldnn=False because the default (oneDNN) path
raises an error with the pinned paddlepaddle on the worker.

Paddle's word boxes are its own (the documented return_word_box option: one box per token, which are joined into whitespace-
separated words); a line without them falls back to cutting the line box in proportion to the character counts of its words.
docTR returns word boxes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path


def join_tokens(tokens: list, boxes: list) -> list[dict]:
    """Paddle's per-token boxes to whitespace-separated words: tokens are joined until one is blank or ends in a space (", ")."""
    words, cur, box = [], "", None
    for tok, b in zip(tokens, boxes):
        piece = str(tok).strip()
        if piece:
            cur += piece
            box = list(map(float, b)) if box is None else [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
        if (not piece or str(tok)[-1:].isspace()) and cur:
            words.append({"text": cur, "box": box})
            cur, box = "", None
    if cur:
        words.append({"text": cur, "box": box})
    return words


def load_paddle(variant: str, threads: int, det_side: int):
    from paddleocr import PaddleOCR

    kw = dict(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device="cpu",
              enable_mkldnn=False)  # the default (oneDNN) path fails with paddlepaddle 3.3.1 on this CPU
    if threads:
        kw["cpu_threads"] = threads
    if det_side:
        kw |= dict(text_det_limit_type="max", text_det_limit_side_len=det_side)
    if variant == "v5-mobile":
        kw |= dict(text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="PP-OCRv5_mobile_rec")
    elif variant == "v5-server":
        kw |= dict(text_detection_model_name="PP-OCRv5_server_det", text_recognition_model_name="PP-OCRv5_server_rec")
    elif variant == "v5-mobile-en":
        kw |= dict(text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="en_PP-OCRv5_mobile_rec")
    elif variant == "v5-server-en":
        kw |= dict(text_detection_model_name="PP-OCRv5_server_det", text_recognition_model_name="en_PP-OCRv5_mobile_rec")
    elif variant != "v6-default":
        raise SystemExit(f"unknown paddle variant {variant}")
    ocr = PaddleOCR(**kw)

    def run(path: Path):
        res = list(ocr.predict(str(path), return_word_box=True))[0].json
        res = res.get("res", res)
        out_lines, words = [], []
        tw, tb = res.get("text_word") or [], res.get("text_word_boxes") or []
        for i, (text, box) in enumerate(zip(res["rec_texts"], res["rec_boxes"])):
            if not str(text).strip():
                continue
            out_lines.append(str(text))
            if i < len(tw) and i < len(tb) and len(tw[i]) == len(tb[i]) and tw[i]:
                words += join_tokens(tw[i], tb[i])
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            toks = str(text).split()
            total = sum(len(t) for t in toks) + max(len(toks) - 1, 0)  # spaces count as one character
            x = x0
            for j, t in enumerate(toks):
                w = (x1 - x0) * len(t) / total
                words.append({"text": t, "box": [round(x, 1), y0, round(x + w, 1), y1]})
                x += w + ((x1 - x0) / total if j < len(toks) - 1 else 0)
        return out_lines, words

    return run


def load_doctr(variant: str, threads: int, det_side: int):
    import torch

    if threads:
        torch.set_num_threads(threads)
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    if variant == "default":
        model = ocr_predictor(pretrained=True)
    elif variant == "mobile":
        model = ocr_predictor(det_arch="db_mobilenet_v3_large", reco_arch="crnn_mobilenet_v3_small", pretrained=True)
    else:
        raise SystemExit(f"unknown doctr variant {variant}")

    def run(path: Path):
        with torch.inference_mode():
            result = model(DocumentFile.from_images(str(path)))
        page = result.pages[0]
        h, w = page.dimensions
        out_lines, words = [], []
        for block in page.blocks:
            for line in block.lines:
                out_lines.append(" ".join(wd.value for wd in line.words))
                for wd in line.words:
                    (ax, ay), (bx, by) = wd.geometry
                    words.append({"text": wd.value, "box": [round(ax * w, 1), round(ay * h, 1), round(bx * w, 1), round(by * h, 1)]})
        return out_lines, words

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["paddle", "doctr"], required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--det-side", type=int, default=0, help="paddle: cap the longer image side for detection at N pixels (0 = stock setting)")
    a = ap.parse_args()

    files = sorted(p for p in Path(a.pages).iterdir() if p.suffix in (".png", ".jpg"))
    if a.limit:
        files = files[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    load0 = os.getloadavg()
    t0 = time.perf_counter()
    run = (load_paddle if a.engine == "paddle" else load_doctr)(a.variant, a.threads, a.det_side)
    load_s = time.perf_counter() - t0

    times = {}
    for p in files:
        t = time.perf_counter()
        lines, words = run(p)
        times[p.stem] = time.perf_counter() - t
        (out / f"{p.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out / f"{p.stem}.json").write_text(json.dumps({"words": words}) + "\n", encoding="utf-8")

    vals = list(times.values())
    info = {
        "engine": a.engine, "variant": a.variant, "threads": a.threads or "default", "det_side": a.det_side, "pages": len(files),
        "model_load_s": load_s, "first_page_s": vals[0] if vals else 0,
        "mean_s_per_page": sum(vals) / len(vals) if vals else 0,
        "mean_s_per_page_after_first": sum(vals[1:]) / len(vals[1:]) if len(vals) > 1 else 0,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cpu": next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")), platform.processor()),
        "cores": os.cpu_count(), "loadavg_before": load0, "loadavg_after": os.getloadavg(),
        "per_page_s": times,
    }
    (out / "timing.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "per_page_s"}))


if __name__ == "__main__":
    main()
