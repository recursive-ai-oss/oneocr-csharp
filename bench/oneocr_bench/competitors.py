"""Merge competitor runs into one Markdown table: accuracy from the scorer, timing from each run's timing.json.

    python -m oneocr_bench competitors-report ROOT [--eval DIR] [--timing DIR]

ROOT holds one directory per configuration, each with <set>/{<id>.txt, <id>.json, timing.json}. A configuration
named NAME-default (all pages, the engine's own threading) may have a sibling NAME-1t (first pages only, one CPU
thread); a configuration without either suffix has no thread control (Apple Vision). --eval is the directory with
<set>/truth (default: the benchmark data directory).
--timing takes the timing.json files from another tree with the same layout (short timing-only runs on a quiet
machine) instead of from ROOT.
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import data_dir
from .score import score_dir
from .sets import SETS


def report(root: str, eval_dir: str | None, timing: str | None = None) -> str:
    base = Path(eval_dir) if eval_dir else data_dir()
    rootp = Path(root)
    timep = Path(timing) if timing else rootp
    names = sorted({d.name.removesuffix("-default").removesuffix("-1t") for d in rootp.iterdir() if d.is_dir()})
    lines = ["| category (set) | engine / configuration | CER % | WER % | order % | recall-CER % | recall-WER % | F1@0.5 | s/page 1 thread | s/page default | peak RSS MB (1t / default) | load1 before / after (default run) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for set_name, spec in SETS.items():
        for name in names:
            main = rootp / f"{name}-default" if (rootp / f"{name}-default").is_dir() else rootp / name
            one = rootp / f"{name}-1t"
            hyp = main / set_name
            if not hyp.is_dir():
                continue
            s = score_dir(base / set_name / "truth", hyp)
            tmain = timep / main.name / set_name / "timing.json"
            tone = timep / one.name / set_name / "timing.json"
            tm = json.loads((tmain if tmain.exists() else hyp / "timing.json").read_text())
            t1 = json.loads(tone.read_text()) if tone.exists() else None
            f1 = f"{s['det']['f1']:.3f}" if s["det"] else "-"
            sec1 = f"{t1['mean_s_per_page']:.2f} (n={t1['pages']})" if t1 else "-"
            rss1 = f"{t1['peak_rss_mb']:.0f}" if t1 else "-"
            load = f"{tm['loadavg_before'][0]:.1f} / {tm['loadavg_after'][0]:.1f}" if "loadavg_before" in tm else "-"
            lines.append(f"| {spec.category} ({set_name}) | {name} | {s['cer'] * 100:.2f} | {s['wer'] * 100:.2f} | {s['order_err'] * 100:.2f} | "
                         f"{s['cer_recall'] * 100:.2f} | {s['wer_recall'] * 100:.2f} | {f1} | {sec1} | {tm['mean_s_per_page']:.2f} (n={tm['pages']}) | "
                         f"{rss1} / {tm['peak_rss_mb']:.0f} | {load} |")
    return "\n".join(lines) + "\n"
