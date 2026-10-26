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
from .sets import category, discover


def report(root: str, eval_dir: str | None, timing: str | None = None) -> str:
    base = Path(eval_dir) if eval_dir else data_dir()
    rootp = Path(root)
    timep = Path(timing) if timing else rootp
    names = sorted({d.name.removesuffix("-default").removesuffix("-1t") for d in rootp.iterdir() if d.is_dir()})
    lines = ["| category (set) | engine / configuration | CER % | WER % | order % | recall-CER % | recall-WER % | F1@0.5 | s/page 1 thread | s/page default | peak RSS MB (1t / default) | load1 before / after (default run) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for set_name in discover(base):
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
            lines.append(f"| {category(set_name)} ({set_name}) | {name} | {s['cer'] * 100:.2f} | {s['wer'] * 100:.2f} | {s['order_err'] * 100:.2f} | "
                         f"{s['cer_recall'] * 100:.2f} | {s['wer_recall'] * 100:.2f} | {f1} | {sec1} | {tm['mean_s_per_page']:.2f} (n={tm['pages']}) | "
                         f"{rss1} / {tm['peak_rss_mb']:.0f} | {load} |")
    return "\n".join(lines) + "\n"


def freeze(srcs: list[str], eval_dir: str, out: str, info_only: list[str]) -> None:
    """Copy competitor hypotheses into a self-contained directory and score them with the current scorer on the frozen sets.

    Each source is ROOT or PREFIX=ROOT; ROOT holds one directory per configuration (NAME, NAME-default or NAME-1t; the 1t ones
    are timing subsets and are skipped) with <set>/<id>.txt, <id>.json and timing.json. The result is OUT/hyp/<PREFIX+NAME>/<set>/...,
    OUT/scores.json (what `gate --competitors` reads) and OUT/SHA256SUMS. A configuration whose name contains a word of
    `info_only` stays in the table but does not set the baseline.
    """
    import hashlib
    import shutil

    outp = Path(out).expanduser()
    base = Path(eval_dir).expanduser()
    outp.mkdir(parents=True, exist_ok=True)
    configs: dict = {}
    for src in srcs:
        prefix, _, root = src.rpartition("=")
        rootp = Path(root).expanduser()
        for d in sorted(p for p in rootp.iterdir() if p.is_dir() and not p.name.endswith("-1t")):
            name = prefix + d.name.removesuffix("-default")
            entry = {"gate": not any(w in name for w in info_only), "sets": {}}
            for set_name in discover(base):
                hyp = d / set_name
                if not hyp.is_dir():
                    continue
                s = score_dir(base / set_name / "truth", hyp, "ink")
                entry["sets"][set_name] = {k: s[k] for k in ("cer", "wer", "order_err", "cer_recall", "wer_recall", "pages", "missing")} | {
                    "f1": s["det"]["f1"] if s["det"] else None}
                dst = outp / "hyp" / name / set_name
                dst.mkdir(parents=True, exist_ok=True)
                for f in hyp.iterdir():
                    if f.suffix in (".txt", ".json"):
                        shutil.copyfile(f, dst / f.name)
            if entry["sets"]:
                configs[name] = entry
                print(f"{name}: {len(entry['sets'])} sets", flush=True)
    scorer = Path(__file__).with_name("score.py")
    (outp / "scores.json").write_text(json.dumps({
        "eval": str(base), "scorer_sha256": hashlib.sha256(scorer.read_bytes()).hexdigest(), "boxes": "ink", "configs": configs}, indent=1) + "\n")
    lines = []
    for f in sorted(p for p in outp.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(outp)}")
    (outp / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"wrote {outp / 'scores.json'} and SHA256SUMS ({len(lines)} files)")
