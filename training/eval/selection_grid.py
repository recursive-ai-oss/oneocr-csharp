# ruff: noqa: E501
"""Held-out selection of the detector's text post-processing (text_threshold, text_expand) through the real engine.

The frozen evaluation sets are for the final report only. This builds two selection sets that no training run reads:
  sel-pages   generator validation pages (held-out fonts, books, templates, degradation) with exact page text
  sel-cord    CORD v2 dev receipts (official dev split; partial truth, so scored on recall)
then, for every (threshold, expand) pair, exports the detector with those values in its manifest, runs the C# bench over both
sets and prints CER / WER / order (recall-CER on receipts).

    # build once (python, any env)
    python selection_grid.py build --pages PAGES_ROOT --cord DATASETS_ROOT --out SELDIR [--n 120]
    # grid (run from bench/ so oneocr_bench imports; the exporter runs in the training project)
    python ../training/eval/selection_grid.py grid --checkpoint CKPT --run-manifest RUN/run_manifest.json --sel SELDIR \
        --recognizer MODEL.onm --work DIR --pairs 0.3:0.5,0.3:0.6 [--layout-expand 0.85]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
BENCH = TRAIN.parent / "bench"


def build(a: argparse.Namespace) -> None:
    out = Path(a.out)
    root = Path(a.pages) / "val"
    files = sorted((root / "truth").glob("*.json"))
    step = max(1, len(files) // a.n)
    d = out / "sel-pages"
    (d / "pages").mkdir(parents=True, exist_ok=True)
    (d / "truth").mkdir(parents=True, exist_ok=True)
    n = 0
    for tf in files[::step][: a.n]:
        t = json.loads(tf.read_text())
        pg = t["pages"][0]
        img = root / t["synthetic"]["image"]["file"]
        words = [{"text": w["text"], "box": w["bounds"], "block": bi} for bi, b in enumerate(pg["blocks"]) for ln in b["lines"] for w in ln["words"]]
        text = "\n".join(" ".join(w["text"] for w in ln["words"]) for b in pg["blocks"] for ln in b["lines"])
        link = d / "pages" / (tf.stem + img.suffix)
        link.unlink(missing_ok=True)
        link.symlink_to(img)
        (d / "truth" / (tf.stem + ".txt")).write_text(text, encoding="utf-8")
        (d / "truth" / (tf.stem + ".json")).write_text(json.dumps({"words": words}, ensure_ascii=False), encoding="utf-8")
        n += 1
    print("sel-pages", n)
    sys.path.insert(0, str(TRAIN))
    from datasets.cord_crops import words_of  # official dev split only; never test

    ds = Path(a.cord) / "cord" / "v2"
    rows = [json.loads(x) for x in (ds / "index" / "dev.jsonl").read_text().splitlines()]
    ann = {r["id"]: r for r in map(json.loads, (ds / "annotations" / "dev.jsonl").read_text().splitlines())}
    d = out / "sel-cord"
    (d / "pages").mkdir(parents=True, exist_ok=True)
    (d / "truth").mkdir(parents=True, exist_ok=True)
    for r in rows:
        pid = r["id"].replace("/", "-")
        ws = words_of(json.loads(ann[r["id"]]["ground_truth"]))
        link = d / "pages" / (pid + Path(r["file"]).suffix)
        link.unlink(missing_ok=True)
        link.symlink_to(ds / r["file"])
        (d / "truth" / (pid + ".txt")).write_text(" ".join(w["text"] for w in ws), encoding="utf-8")
        (d / "truth" / (pid + ".json")).write_text(json.dumps({"words": [{"text": w["text"], "box": w["box"]} for w in ws]}), encoding="utf-8")
    print("sel-cord", len(rows))


def grid(a: argparse.Namespace) -> None:
    sys.path.insert(0, str(BENCH))
    from oneocr_bench.gate import build_bench  # noqa: PLC0415
    from oneocr_bench.score import score_dir  # noqa: PLC0415

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    dll = build_bench(work, a.dotnet)
    rows = []
    for pair in a.pairs.split(","):
        thr, exp = pair.split(":")
        x = work / f"t{thr}-e{exp}"
        x.mkdir(exist_ok=True)
        post = json.dumps({"text_threshold": float(thr), "text_expand": float(exp), "layout_threshold": 0.6, "layout_expand": a.layout_expand})
        subprocess.run(
            ["uv", "run", "--project", str(TRAIN), "python", "-m", "export.dbnet", "--checkpoint", a.checkpoint, "--run-manifest", a.run_manifest,
             "--version", a.version, "--exclude-classes", "form_field,checkbox", "--postprocess", post, "--out", str(x / "model.onm")],
            check=True, cwd=TRAIN, capture_output=True,
        )  # fmt: skip
        res = {}
        for s in ("sel-pages", "sel-cord"):
            out = x / s
            out.mkdir(exist_ok=True)
            subprocess.run(
                [a.dotnet, str(dll), "run", "--model", a.recognizer, "--detector", str(x / "model.onm"), "--pages", str(Path(a.sel) / s / "pages"),
                 "--out", str(out), "--degree", "4", "--repeat", "1", "--words", "1", "--pagejson", "1"],
                check=True, stdout=subprocess.DEVNULL,
            )  # fmt: skip
            r = score_dir(Path(a.sel) / s / "truth", out, "layout")
            res[s] = r
        p, c = res["sel-pages"], res["sel-cord"]
        rows.append((thr, exp, p["cer"], p["wer"], p["order_err"], c["cer_recall"], c["wer_recall"]))
        print(f"thr {thr} expand {exp}: pages CER {100 * p['cer']:.3f} WER {100 * p['wer']:.3f} order {100 * p['order_err']:.3f} | cord recall-CER {100 * c['cer_recall']:.2f} recall-WER {100 * c['wer_recall']:.2f}", flush=True)
    (work / "grid.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pages", required=True)
    b.add_argument("--cord", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=120)
    g = sub.add_parser("grid")
    for k in ("checkpoint", "run_manifest", "sel", "recognizer", "work", "pairs"):
        g.add_argument("--" + k.replace("_", "-"), required=True)
    g.add_argument("--version", default="0.2.1")
    g.add_argument("--layout-expand", type=float, default=0.85)
    g.add_argument("--dotnet", default="dotnet")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else grid(a)
