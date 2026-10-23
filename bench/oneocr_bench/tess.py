"""Run the Tesseract binary (stock models, black box) over an evaluation set and score it.

For each model (best, fast) and page-segmentation mode, every page is run as its own process, one at a time, with
one thread (OMP_THREAD_LIMIT=1): wall time per page includes process start and model load, which is what
`tesseract page.png out` costs a caller; peak RSS is the child's maximum resident set (wait4). The same
invocation asks for the documented `txt` and `tsv` outputs, so word boxes for the detection score come from the
binary's own TSV output. Accuracy does not depend on the thread setting, so the best mode per model (lowest CER,
then WER; for the receipts set the lowest recall-CER, because its truth does not cover the whole receipt) is
re-run once more with default threading to time it and to check the text is identical.

    python -m oneocr_bench tess --set degraded-v1 --psms 3 4 6 11
    python -m oneocr_bench tess-report            # merge the per-set results into one table
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from .paths import data_dir
from .score import score_dir
from .sets import SETS, page_images, set_dir


def _loadavg() -> list[float]:
    return [round(x, 2) for x in os.getloadavg()]


def _cpu() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def run_one(tess: str, model: str, psm: int, img: Path, out_base: Path, threads1: bool) -> tuple[float, int]:
    env = dict(os.environ)
    if threads1:
        env["OMP_THREAD_LIMIT"] = "1"
    else:
        env.pop("OMP_THREAD_LIMIT", None)
    t0 = time.perf_counter()
    p = subprocess.Popen([tess, model, str(img), str(out_base), "-l", "eng", "--psm", str(psm), "txt", "tsv"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, status, ru = os.wait4(p.pid, 0)
    dt = time.perf_counter() - t0
    if status != 0:
        raise RuntimeError(f"tesseract failed on {img.name} (status {status})")
    return dt, ru.ru_maxrss  # KiB on Linux


def tsv_to_sidecar(tsv: Path, out: Path) -> None:
    """Level-5 (word) rows of Tesseract's TSV as the scorer's box sidecar."""
    words = []
    with open(tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["level"] == "5" and (r["text"] or "").strip():
                x, y, w, h = (int(r[k]) for k in ("left", "top", "width", "height"))
                words.append({"text": r["text"], "box": [x, y, x + w, y + h]})
    out.write_text(json.dumps({"words": words}) + "\n")
    tsv.unlink()


def main(set_name: str, models: list[str], psms: list[int], tess: str, results: str | None) -> None:
    pages = page_images(set_name)
    truth = set_dir(set_name) / "truth"
    res_dir = Path(results) if results else data_dir() / "results" / set_name
    res_dir.mkdir(parents=True, exist_ok=True)
    ver = subprocess.run([tess, "best", "--version"], check=False, capture_output=True, text=True)
    env_info = {"cpu": _cpu(), "cores": os.cpu_count(), "loadavg_before": _loadavg(),
                "tesseract": (ver.stderr or ver.stdout).splitlines()[:1], "set": set_name,
                "category": SETS[set_name].category, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    key = "cer_recall" if set_name == "receipts-v1" else "cer"
    rows, loads_during = [], []
    for model in models:
        for psm in psms:
            hyp = res_dir / "hyp" / f"{model}-psm{psm}-1thread"
            hyp.mkdir(parents=True, exist_ok=True)
            times, rss = [], []
            for img in pages:
                dt, kb = run_one(tess, model, psm, img, hyp / img.stem, True)
                tsv_to_sidecar(hyp / f"{img.stem}.tsv", hyp / f"{img.stem}.json")
                times.append(dt)
                rss.append(kb)
            loads_during.append(_loadavg()[0])
            s = score_dir(truth, hyp)
            row = {"model": model, "psm": psm, "1thread": {
                "s_per_page": sum(times) / len(times), "max_s": max(times), "peak_rss_mb": max(rss) / 1024,
                "cer": s["cer"], "wer": s["wer"], "order_err": s["order_err"], "cer_recall": s["cer_recall"],
                "wer_recall": s["wer_recall"], "f1": s["det"]["f1"] if s["det"] else None,
                "precision": s["det"]["precision"] if s["det"] else None,
                "recall": s["det"]["recall"] if s["det"] else None}}
            rows.append(row)
            print(json.dumps(row), flush=True)

    best = {m: min((r for r in rows if r["model"] == m), key=lambda r: (r["1thread"][key], r["1thread"]["wer"]))["psm"]
            for m in models}
    for model in models:
        psm = best[model]
        row = next(r for r in rows if r["model"] == model and r["psm"] == psm)
        hyp = res_dir / "hyp" / f"{model}-psm{psm}-default"
        hyp.mkdir(parents=True, exist_ok=True)
        times, rss = [], []
        for img in pages:
            dt, kb = run_one(tess, model, psm, img, hyp / img.stem, False)
            (hyp / f"{img.stem}.tsv").unlink()
            times.append(dt)
            rss.append(kb)
        loads_during.append(_loadavg()[0])
        row["default"] = {"s_per_page": sum(times) / len(times), "max_s": max(times), "peak_rss_mb": max(rss) / 1024}
        row["thread_outputs_identical"] = all(
            (res_dir / "hyp" / f"{model}-psm{psm}-1thread" / f"{p.stem}.txt").read_bytes() == (hyp / f"{p.stem}.txt").read_bytes()
            for p in pages)
        link = data_dir() / "results" / f"best-{model}" / set_name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(res_dir / "hyp" / f"{model}-psm{psm}-1thread")
    env_info.update(loadavg_during=loads_during, loadavg_after=_loadavg())
    (res_dir / "tesseract.json").write_text(json.dumps({"env": env_info, "pages": len(pages), "rows": rows, "best_psm": best,
                                                         "selected_by": key}, indent=1) + "\n")
    print(json.dumps({"set": set_name, "best_psm": best}))


def report() -> None:
    """One markdown table over every set that has results: best-of-PSM rows."""
    lines = ["| category (set) | model | PSM | CER % | WER % | order % | recall-CER % | F1@0.5 | s/page 1 thread | s/page default | peak RSS MB (1t / default) | load1 before / max during / after |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, spec in SETS.items():
        p = data_dir() / "results" / name / "tesseract.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        env = d["env"]
        load = f"{env['loadavg_before'][0]} / {max(env['loadavg_during'])} / {env['loadavg_after'][0]}"
        for model, psm in d["best_psm"].items():
            r = next(r for r in d["rows"] if r["model"] == model and r["psm"] == psm)
            a, b = r["1thread"], r["default"]
            f1 = f"{a['f1']:.3f}" if a["f1"] is not None else "-"
            lines.append(f"| {spec.category} ({name}) | {model} | {psm} | {a['cer'] * 100:.2f} | {a['wer'] * 100:.2f} | {a['order_err'] * 100:.2f} | "
                         f"{a['cer_recall'] * 100:.2f} | {f1} | {a['s_per_page']:.2f} | {b['s_per_page']:.2f} | "
                         f"{a['peak_rss_mb']:.0f} / {b['peak_rss_mb']:.0f} | {load} |")
    out = "\n".join(lines) + "\n"
    (data_dir() / "results" / "tesseract-table.md").write_text(out)
    print(out)
