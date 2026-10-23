"""Evaluates trained recognizers on every line set and measures CPU cost; writes one JSON report and a markdown table.

Usage: uv run python -m eval.bakeoff --out DIR --sets-dir LINES_V2_DIR --bench-dir BENCH_LINES_DIR NAME=checkpoint.pt [...]

Each model is evaluated the way it ships: batch norm folded, weights rounded to fp16, computed in fp32. Sets: the materialised
synthetic sets of the line set (val, eval = held-out fonts and degradation, diag_fonts, diag_degrade) and, per page set, the crops
from eval/make_bench_lines.sh. Checkpoints are chosen on validation only; nothing here selects on the page sets.
Latency is torch on one CPU thread (a proxy for the library's kernels, not a measurement of them): median of 30 forward passes of
a 1 x 1 x 32 x W line.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from eval.lines import evaluate
from models import build, fold, load_folded, macs_per_column
from train.data import LineSet

PAGE_SETS = ["clean300-v1", "degraded-v1", "screenshot-v1", "layout-v1", "phone-v1", "receipts-v1"]
SYNTH = ["val", "eval", "diag_fonts", "diag_degrade"]


def shipped_model(ck: dict):
    """The trained model as the container holds it: folded, fp16-rounded, run in fp32."""
    m = build(ck["config"])
    m.load_state_dict(ck["model"])
    t = {k: v.half().float() for k, v in fold(m.eval()).items()}
    return load_folded(ck["config"], t).eval(), m.eval()


@torch.no_grad()
def latency_ms(model, width: int, reps: int = 30) -> float:
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    m = model.cpu().eval()
    x = torch.randn(1, 1, 32, width)
    for _ in range(5):
        m(x)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        m(x)
        ts.append(time.perf_counter() - t)
    torch.set_num_threads(prev)
    return statistics.median(ts) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets-dir", required=True, help="line set directory holding the materialised synthetic sets")
    ap.add_argument("--bench-dir", required=True, help="output of eval/make_bench_lines.sh")
    ap.add_argument("models", nargs="+", help="NAME=checkpoint.pt")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    for item in a.models:
        name, _, path = item.partition("=")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        charset = ck["charset"]
        folded, raw = shipped_model(ck)
        n_params = sum(v.numel() for v in fold(raw).values())
        r: dict = {
            "step": ck["step"],
            "container_parameters": n_params,
            "fp16_mb": round(n_params * 2 / 1e6, 2),
            "macs_per_column": macs_per_column(ck["config"]),
            "cpu_ms_w320": round(latency_ms(folded, 320), 2),
            "cpu_ms_w1000": round(latency_ms(folded, 1000), 2),
            "sets": {},
        }
        dev = "cuda"
        folded = folded.to(dev)
        raw = raw.to(dev)
        todo = {s: Path(a.sets_dir) / s for s in SYNTH}
        for s in PAGE_SETS:
            for variant in ("finder",):
                if (Path(a.bench_dir) / s / variant).exists():
                    todo[f"{s}/{variant}"] = Path(a.bench_dir) / s / variant
        for label, d in todo.items():
            ds = LineSet(d, charset)
            res = evaluate(folded, ds, charset, dev)
            if label in ("val", "eval"):
                res["fp32_cer"] = evaluate(raw, ds, charset, dev)["cer"]
            r["sets"][label] = res
            print(name, label, json.dumps(res), flush=True)
        report[name] = r
    (out / "bakeoff.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    names = list(report)
    labels = list(next(iter(report.values()))["sets"])
    lines = ["| set | lines | " + " | ".join(f"{n} CER % | {n} WER %" for n in names) + " |", "|---|---:|" + "---:|---:|" * len(names)]
    for lab in labels:
        row = f"| {lab} | {report[names[0]]['sets'][lab]['lines']} | "
        row += " | ".join(f"{100 * report[n]['sets'][lab]['cer']:.2f} | {100 * report[n]['sets'][lab]['wer']:.2f}" for n in names)
        lines.append(row + " |")
    for key in ("container_parameters", "fp16_mb", "cpu_ms_w320", "cpu_ms_w1000"):
        lines.append(f"| {key} | | " + " | ".join(f"{report[n][key]} | " for n in names) + " |")
    lines.append("| MACs per column (total) | | " + " | ".join(f"{report[n]['macs_per_column']['total']} | " for n in names) + " |")
    (out / "bakeoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
