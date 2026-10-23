"""Run the Tesseract binary (stock models, black box) over the page set and score it.

For each model (best, fast), page-segmentation mode and thread setting, every page is run as its own
process, one at a time: wall time per page includes process start and model load, which is what
`tesseract page.png out` costs a caller. Peak RSS is the child's maximum resident set (wait4).
Accuracy does not depend on the thread setting, so hypotheses are scored once per (model, psm)
from the single-thread run; the default-threading run is timed and compared for equality.

    python -m oneocr_bench tess --models best fast --psms 3 4 6
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

from .paths import data_dir
from .score import score_dir


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
    p = subprocess.Popen([tess, model, str(img), str(out_base), "-l", "eng", "--psm", str(psm)], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, status, ru = os.wait4(p.pid, 0)
    dt = time.perf_counter() - t0
    if status != 0:
        raise RuntimeError(f"tesseract failed on {img.name} (status {status})")
    return dt, ru.ru_maxrss  # KiB on Linux


def main(models: list[str], psms: list[int], set_name: str, tess: str, results: str | None) -> None:
    root = data_dir()
    pages = sorted((root / set_name / "pages").glob("*.png"))
    truth = root / set_name / "truth"
    res_dir = Path(results) if results else root / "results" / set_name
    res_dir.mkdir(parents=True, exist_ok=True)
    env_info = {"cpu": _cpu(), "cores": os.cpu_count(), "loadavg_before": _loadavg(),
                "tesseract": subprocess.run([tess, "best", "--version"], capture_output=True, text=True).stderr.splitlines()[:1]
                or subprocess.run([tess, "best", "--version"], capture_output=True, text=True).stdout.splitlines()[:1]}
    rows = []
    for model in models:
        for psm in psms:
            row = {"model": model, "psm": psm}
            for label, one in (("1thread", True), ("default", False)):
                hyp = res_dir / "hyp" / f"{model}-psm{psm}-{label}"
                hyp.mkdir(parents=True, exist_ok=True)
                times, rss = [], []
                loads = [_loadavg()[0]]
                for img in pages:
                    dt, kb = run_one(tess, model, psm, img, hyp / img.stem, one)
                    times.append(dt)
                    rss.append(kb)
                loads.append(_loadavg()[0])
                s = score_dir(truth, hyp)
                row[label] = {"s_per_page": sum(times) / len(times), "max_s": max(times),
                              "peak_rss_mb": max(rss) / 1024, "load1_start_end": loads,
                              "cer": s["cer"], "wer": s["wer"], "order_err": s["order_err"]}
            row["thread_outputs_identical"] = all(
                (res_dir / "hyp" / f"{model}-psm{psm}-1thread" / f"{p.stem}.txt").read_bytes()
                == (res_dir / "hyp" / f"{model}-psm{psm}-default" / f"{p.stem}.txt").read_bytes()
                for p in pages)
            rows.append(row)
            print(json.dumps(row), flush=True)
    env_info["loadavg_after"] = _loadavg()

    best = {}
    for model in models:
        cand = [r for r in rows if r["model"] == model]
        best[model] = min(cand, key=lambda r: (r["1thread"]["cer"], r["1thread"]["wer"]))["psm"]
    (res_dir / "tesseract.json").write_text(json.dumps({"env": env_info, "pages": len(pages), "rows": rows,
                                                         "best_psm": best}, indent=1) + "\n")
    lines = ["| model | psm | CER % | WER % | order % | s/page 1 thread | s/page default | peak RSS MB (1t / default) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a, b = r["1thread"], r["default"]
        star = " *" if best[r["model"]] == r["psm"] else ""
        lines.append(f"| {r['model']} | {r['psm']}{star} | {a['cer'] * 100:.3f} | {a['wer'] * 100:.3f} | {a['order_err'] * 100:.3f} | "
                     f"{a['s_per_page']:.2f} | {b['s_per_page']:.2f} | {a['peak_rss_mb']:.0f} / {b['peak_rss_mb']:.0f} |")
    (res_dir / "tesseract.md").write_text("\n".join(lines) + "\n\n* best psm by CER (then WER) per model\n")
    print("\n".join(lines))
