"""One-command gate table: our engine against Tesseract best-of and fast, per category, on the frozen sets.

    # accuracy (reads the pages once per set; run it nice'd when the machine is quiet)
    python -m oneocr_bench gate --model MODEL.onm [--detector DET.onm]

    # speed and memory (refuses unless this job holds the machine lock, see below)
    python -m oneocr_bench gate --model MODEL.onm [--detector DET.onm] --timing

Accuracy mode runs `OneOcr.Bench run --words 1 --pagejson 1` on every set, scores with the bench scorer (CER, WER,
reading order, ink F1@0.5, ECE / ECE-m; receipts on recall-CER / recall-WER because CORD truth is partial) and
writes `accuracy.json`. `--timing` mode runs the engine without sidecars, per set and per degree (1 and the default
min(4, cores)), three times, and writes `timing.json`. The timed process is the shipped command line, `oneocr ocr`,
NativeAOT-published (see `build_tool`); accuracy mode keeps `OneOcr.Bench` because it writes the word-box sidecars. Either mode then rewrites `gate.md` and `gate.json` from
whatever of the two files exist, so the table carries a speed section only after a timing run.

Tesseract numbers come from the existing `tess` results (`results/<set>/tesseract.json` and its hypothesis
directories); `--rerun-tess` runs them again first. The baseline for each metric is the best page-segmentation mode
for that metric of the `best` model (the hardest baseline), as in the Tesseract tables of bench/README.md.

Scorer versions: `--scorer v3` (default) judges on truth words taken from the truth boxes, `--scorer v2` on the truth text; the table is
computed for both and the other version's verdicts are listed beside the judged ones (see bench/README.md, Scoring). Tesseract's
best-of values are rescored from its hypothesis directories under v3, the competitors from a frozen directory made by
`competitors-freeze` (`--rescore` adds v3 to an older one).

Verdicts
  quality      beat the best engine measured, per metric, on CER, WER and reading order (recall-CER / recall-WER on
               receipts, whose order is shown but not judged). The candidates are Tesseract best-of and, with
               --competitors DIR, every gated competitor configuration of a frozen results directory. On clean print an
               error within 0.2 percentage points above the baseline is a tie and passes.
  calibration  ECE <= 0.05 (receipts: ECE is an upper bound because the truth is partial; ECE-m is shown).
  speed        faster than Tesseract fast; slower than fast but faster than best, with the quality row won, is an
               "exception" (reported, decided elsewhere). Memory: peak RSS at the default degree at most 300 MB.

The timing lock: the job itself must have written ~/WORKER-BENCH.lock with a `pid=<shell pid>` token naming one of
this process's ancestors, e.g. in the job's shell:

    set -o noclobber; echo "oneocr gate pid=$$ $(date -Is)" > ~/WORKER-BENCH.lock || exit 75
    trap 'rm -f ~/WORKER-BENCH.lock' EXIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from . import tess as tess_mod
from .paths import BENCH, data_dir
from .score import SCORERS, score_dir
from .sets import ORDER_INFO_ONLY, category, discover, is_clean, is_real, is_receipts

LOCK = Path(os.environ.get("ONEOCR_LOCK", "~/WORKER-BENCH.lock")).expanduser()
TIE_PP = 0.2  # clean print may tie within this many percentage points
ECE_GATE = 0.05
MEM_CAP_MB = 300  # peak RSS per page at the default degree
CLEAN = "clean300-v1"
RECEIPTS = "receipts-v1"


# --------------------------------------------------------------------------- environment

def _ancestors() -> set[int]:
    pids, p = set(), os.getpid()
    while p > 1:
        pids.add(p)
        try:
            stat = Path(f"/proc/{p}/stat").read_text()
            p = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def lock_held_by_this_job() -> str | None:
    """None when this job's own shell holds the lock; otherwise the reason it does not."""
    if not LOCK.exists():
        return f"{LOCK} does not exist"
    text = LOCK.read_text().strip()
    m = re.search(r"\bpid=(\d+)", text)
    if not m:
        return f"{LOCK} has no pid= token ({text!r}): it belongs to another job"
    if int(m.group(1)) not in _ancestors():
        return f"{LOCK} is held by another job ({text!r})"
    return None


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _load() -> list[float]:
    return [round(x, 2) for x in os.getloadavg()]


def _sha(path: str | None) -> str | None:
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_info() -> dict:
    return {"cpu": _cpu(), "cores": os.cpu_count(), "host": platform.node(), "loadavg": _load()}


# --------------------------------------------------------------------------- running our engine

def build_bench(work: Path, dotnet: str) -> Path:
    out = work / "bin"
    log = work / "build.log"
    r = subprocess.run([dotnet, "build", str(BENCH / "OneOcr.Bench"), "-c", "Release", "-v", "quiet", "-clp:NoSummary", "-o", str(out)],
                       capture_output=True, text=True)
    log.write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"building OneOcr.Bench failed, see {log}\n" + "\n".join((r.stdout + r.stderr).splitlines()[-15:]))
    return out / "OneOcr.Bench.dll"


def run_engine(dotnet: str, dll: Path, args: argparse.Namespace, set_name: str, out: Path, degree: int, sidecars: bool) -> tuple[float, dict]:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [dotnet, str(dll), "run", "--model", args.model, "--pages", str(Path(args.eval) / set_name / "pages"),
           "--out", str(out), "--degree", str(degree), "--repeat", "1"]
    if args.detector:
        cmd += ["--detector", args.detector]
    if args.wordlm:
        cmd += ["--wordlm", args.wordlm]
    if getattr(args, "wordfreq", None):
        cmd += ["--wordfreq", args.wordfreq]
    if sidecars:
        cmd += ["--words", "1", "--pagejson", "1"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise SystemExit(f"OneOcr.Bench failed on {set_name} (exit {r.returncode}):\n{r.stderr[-1500:]}")
    return wall, json.loads((out / "timing.json").read_text())


def _host_rid() -> str:
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("aarch64", "arm64") else "x64"
    system = {"linux": "linux", "darwin": "osx", "windows": "win"}.get(platform.system().lower(), "linux")
    return f"{system}-{arch}"


def _auto_isa(rid: str) -> str | None:
    """The x86-64 level the NativeAOT tool is compiled for: v4 (AVX-512) when this host has it, else v3 (AVX2 + FMA). Arm needs none."""
    if not rid.endswith("-x64"):
        return None
    try:
        flags = set(re.search(r"^flags\s*:\s*(.*)$", Path("/proc/cpuinfo").read_text(), re.M).group(1).split())
    except (OSError, AttributeError):
        return "x86-64-v3"
    return "x86-64-v4" if {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"} <= flags else "x86-64-v3"


def build_tool(work: Path, dotnet: str, isa: str) -> tuple[Path, dict]:
    """Publishes `oneocr` (src/OneOcr.Tool) as a NativeAOT executable, the way a user would (`dotnet publish -r <rid>`), and returns it
    with the build description for the report. `isa` is an `IlcInstructionSet` value, `auto` for the host's best, or `default` for the
    project's own default. The models are given at run time (`--model`, `--detector`), so the pinned bundled files are not needed."""
    rid = _host_rid()
    chosen = _auto_isa(rid) if isa == "auto" else (None if isa == "default" else isa)
    out = work / "tool"
    cmd = [dotnet, "publish", str(BENCH.parent / "src" / "OneOcr.Tool"), "-c", "Release", "-r", rid, "-o", str(out), "-v", "quiet", "-clp:NoSummary"]
    if chosen:
        cmd.append(f"-p:IlcInstructionSet={chosen}")
    log = work / "build-tool.log"
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(r.stdout + r.stderr)
    exe = out / ("OneOcr.Tool.exe" if rid.startswith("win") else "OneOcr.Tool")
    if r.returncode != 0 or not exe.exists():
        raise SystemExit(f"publishing src/OneOcr.Tool failed, see {log}\n" + "\n".join((r.stdout + r.stderr).splitlines()[-15:]))
    return exe, {"process": "oneocr ocr (src/OneOcr.Tool, NativeAOT publish)", "rid": rid, "instruction_set": chosen or "project default", "sha256": _sha(str(exe))}


def _pins() -> dict[str, str]:
    """Role -> SHA-256 of the files the library pins as its bundled models (src/OneOcr.Models/models.json), when the checkout has it."""
    f = Path(__file__).resolve().parent.parent.parent / "src" / "OneOcr.Models" / "models.json"
    try:
        return {m["role"]: m["sha256"] for m in json.loads(f.read_text())["models"]}
    except (OSError, ValueError, KeyError):
        return {}


def model_files(args: argparse.Namespace) -> dict:
    """The files that produced the rows: path, SHA-256 and whether it is the bundled pin, for recognizer, detector and word character model."""
    pins = _pins()
    out: dict = {}
    for key, role, path in (("recognizer", "recognizer", args.model), ("detector", "detector", args.detector), ("word_model", "language-model", args.wordlm),
                            ("word_freq", "word-frequencies", getattr(args, "wordfreq", None))):
        sha = _sha(path)
        out[key] = {"file": path, "sha256": sha, "is_bundled_pin": (sha == pins.get(role)) if sha and pins else None}
    return out


def _models_line(mf: dict) -> str:
    names = {"recognizer": "recognizer", "detector": "detector", "word_model": "char LM", "word_freq": "word frequencies"}
    parts = []
    for k, label in names.items():
        e = mf[k]
        if not e["sha256"]:
            parts.append(f"{label}: none" + (" (classical finder)" if k == "detector" else ""))
        else:
            pin = "" if e["is_bundled_pin"] is None else (" = bundled pin" if e["is_bundled_pin"] else " (not the bundled pin)")
            parts.append(f"{label} `{e['sha256']}`{pin}")
    return "Models: " + "; ".join(parts) + "."


def run_tool(exe: Path, args: argparse.Namespace, set_name: str, out: Path, degree: int) -> tuple[float, dict]:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe), "ocr", "--model", args.model, "--in", str(Path(args.eval) / set_name / "pages"), "--out", str(out), "--degree", str(degree),
           "--stats", str(out / "timing.json")]
    if args.detector:
        cmd += ["--detector", args.detector]
    if args.wordlm:
        cmd += ["--wordlm", args.wordlm]
    if getattr(args, "wordfreq", None):
        cmd += ["--wordfreq", args.wordfreq]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise SystemExit(f"oneocr ocr failed on {set_name} (exit {r.returncode}):\n{r.stderr[-1500:]}")
    return wall, json.loads((out / "timing.json").read_text())


# --------------------------------------------------------------------------- Tesseract baselines

def tess_baselines(set_name: str, truth: Path, tess_results: Path, scorer: str = "v2") -> dict | None:
    p = tess_results / set_name / "tesseract.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if scorer != "v2":
        # tesseract.json holds the numbers of the scorer it ran with (v2); every (model, PSM) run is scored again from its own output.
        rows = []
        for r in d["rows"]:
            hyp = tess_results / set_name / "hyp" / f"{r['model']}-psm{r['psm']}-1thread"
            if not hyp.is_dir():
                print(f"warning: {hyp} not found: no Tesseract baseline under scorer {scorer} for {set_name}", file=sys.stderr)
                return None
            s = score_dir(truth, hyp, "ink", scorer)
            rows.append(r | {"1thread": r["1thread"] | {k: s[k] for k in ("cer", "wer", "order_err", "cer_recall", "wer_recall")}})
        d = d | {"rows": rows}
    out: dict = {"source": str(p), "env": d["env"], "pages": d["pages"], "models": {}}
    for model in ("best", "fast"):
        rows = [r for r in d["rows"] if r["model"] == model]
        if not rows:
            continue
        per = {}
        for k in ("cer", "wer", "order_err", "cer_recall", "wer_recall"):
            r = min(rows, key=lambda r: r["1thread"][k])
            per[k] = {"value": r["1thread"][k], "psm": r["psm"]}
        sel = d["best_psm"][model]
        row = next(r for r in rows if r["psm"] == sel)
        # F1 against the frozen ink boxes: rescore the selected run's own output with the current scorer.
        hyp = tess_results / set_name / "hyp" / f"{model}-psm{sel}-1thread"
        f1 = None
        if hyp.is_dir():
            s = score_dir(truth, hyp, "ink", scorer)
            f1 = s["det"]["f1"] if s["det"] else None
        out["models"][model] = {"best_per_metric": per, "selected_psm": sel, "selected_by": d["selected_by"], "f1_ink": f1,
                                "s_per_page_1thread": row["1thread"]["s_per_page"], "s_per_page_default": row.get("default", {}).get("s_per_page"),
                                "peak_rss_mb_1thread": row["1thread"]["peak_rss_mb"], "peak_rss_mb_default": row.get("default", {}).get("peak_rss_mb"),
                                "load": {"before": d["env"]["loadavg_before"], "during": d["env"]["loadavg_during"], "after": d["env"]["loadavg_after"]}}
    return out


# --------------------------------------------------------------------------- verdicts

def metric_keys(set_name: str, order_info: bool = False) -> list[tuple[str, str, bool]]:
    """(label, key in score result, judged); `order_info` shows the order row without judging it (a set whose order truth is not firm)."""
    if is_receipts(set_name):
        return [("recall-CER", "cer_recall", True), ("recall-WER", "wer_recall", True), ("order", "order_err", False)]
    return [("CER", "cer", True), ("WER", "wer", True), ("order", "order_err", not order_info)]


def load_competitors(path: str | None) -> dict | None:
    """scores.json of a frozen competitor-results directory: {config: {set: {metric: value}}} plus the gate flag per config."""
    if not path:
        return None
    p = Path(path).expanduser() / "scores.json"
    if not p.exists():
        raise SystemExit(f"{p} not found: build the directory with `competitors-freeze`")
    return json.loads(p.read_text())


def competitors_for(comp: dict | None, scorer: str) -> dict | None:
    """The frozen competitor scores as one scorer version sees them: {configs: {name: {gate, sets: {set: {metric: value}}}}}."""
    if comp is None or scorer == "v2":
        return comp
    configs = {}
    for cfg, c in comp["configs"].items():
        sets = {}
        for n, row in c["sets"].items():
            if scorer not in row:
                raise SystemExit(f"the competitor directory has no scorer {scorer} scores ({cfg}, {n}): run `competitors-freeze --rescore DIR --out DIR --eval EVAL`")
            sets[n] = row[scorer]
        configs[cfg] = c | {"sets": sets}
    return comp | {"configs": configs}


def candidates(set_name: str, key: str, tess: dict | None, comp: dict | None) -> list[tuple[str, float]]:
    """(engine and configuration, value) for one metric: Tesseract best-of and every gated competitor configuration."""
    out = []
    if tess and "best" in tess["models"]:
        b = tess["models"]["best"]["best_per_metric"][key]
        out.append((f"tesseract-best psm{b['psm']}", b["value"]))
    if comp:
        for cfg, c in comp["configs"].items():
            if c.get("gate", True) and set_name in c["sets"] and c["sets"][set_name].get(key) is not None:
                out.append((cfg, c["sets"][set_name][key]))
    return out


def quality(set_name: str, ours: dict, tess: dict | None, comp: dict | None = None, order_info: bool = False) -> dict:
    """Win / tie / loss per metric against the best value any measured engine reaches (Tesseract best-of included). Without a
    competitor directory the baseline is Tesseract best-of alone. Clean print may tie within TIE_PP percentage points."""
    rows = []
    ok = True
    for label, key, judged in metric_keys(set_name, order_info):
        o = ours[key]
        row = {"metric": label, "key": key, "ours": o, "judged": judged}
        cands = candidates(set_name, key, tess, comp)
        if tess and "fast" in tess["models"]:
            row["tess_fast"] = tess["models"]["fast"]["best_per_metric"][key]["value"]
        if tess and "best" in tess["models"]:
            row["tess_best"] = tess["models"]["best"]["best_per_metric"][key]["value"]
        if cands:
            engine, best = min(cands, key=lambda c: c[1])
            row.update(best=best, best_engine=engine, others={e: v for e, v in sorted(cands, key=lambda c: c[1])})
            diff_pp = (o - best) * 100
            if not judged:
                v = "info"
            elif o < best:
                v = "win"
            elif is_clean(set_name) and diff_pp <= TIE_PP:
                v = "tie"
            else:
                v = "loss"
            row["verdict"] = v
            ok = ok and v in ("win", "tie", "info")
        else:
            row["verdict"] = "no baseline"
            ok = False
        rows.append(row)
    return {"rows": rows, "pass": bool(ok), "baseline": "best competitor" if comp else "Tesseract best-of"}


def calibration(set_name: str, ours: dict) -> dict:
    e = ours.get("ece")
    if not e:
        return {"ece": None, "ece_matched": None, "verdict": "no confidences", "pass": False}
    v = "pass" if e["ece"] <= ECE_GATE else "fail"
    note = None
    if is_receipts(set_name):
        note = "ECE is an upper bound (partial truth)"
        if v == "fail" and e["ece_matched"] <= ECE_GATE:
            v = "unclear"
    return {"ece": e["ece"], "ece_matched": e["ece_matched"], "words": e["words"], "verdict": v, "pass": v == "pass", "note": note}


def speed_rows(timing: dict, tess_timing: dict | None, quality_by_set: dict) -> dict:
    """Per set and degree label: ours vs Tesseract fast and best, timed the same way (one process per run, start and model load
    included, mean of the runs). Our degree 1 row is compared with Tesseract at one OpenMP thread, the default-degree row with
    Tesseract at the same number of threads."""
    out = {}
    dd = str(timing["default_degree"])
    for name, t in timing["sets"].items():
        tt = ((tess_timing or {}).get("sets") or {}).get(name)
        rows = []
        for tag, deg, tkey in (("degree 1", "1", "1"), (f"default degree ({dd})", "default", dd)):
            r = t[deg]
            row = {"config": tag, "ours_ms": r["mean_ms_per_page"], "ours_rss_mb": r["mean_peak_rss_mb"],
                   "ours_ms_runs": r["ms_per_page_runs"], "ours_rss_runs": r["peak_rss_mb_runs"],
                   "ours_process_ms_per_page": r["mean_process_ms_per_page"], "model_load_ms": r["mean_model_load_ms"]}
            if tt and all(m in tt["models"] and tkey in tt["models"][m]["threads"] for m in ("best", "fast")):
                f, b = (tt["models"][m]["threads"][tkey] for m in ("fast", "best"))
                fast_ms, best_ms = f["mean_ms_per_page"], b["mean_ms_per_page"]
                fast_rss, best_rss = f["mean_peak_rss_mb"], b["mean_peak_rss_mb"]
                won = quality_by_set.get(name, {}).get("pass", False)
                ours = row["ours_process_ms_per_page"]
                if ours < fast_ms:
                    sv = "pass"
                elif ours < best_ms and won:
                    sv = "exception"
                else:
                    sv = "fail"
                row.update(fast_ms=fast_ms, best_ms=best_ms, fast_rss_mb=fast_rss, best_rss_mb=best_rss, speed=sv,
                           tess_psm={m: tt["models"][m]["psm"] for m in ("best", "fast")})
            else:
                row.update(speed="no Tesseract timing")
            if deg == "default":
                row["memory"] = "pass" if row["ours_rss_mb"] <= MEM_CAP_MB else "fail"
            else:
                row["memory"] = "info"
            rows.append(row)
        out[name] = rows
    return out


def time_tesseract(args, names: list[str], tess_results: Path, default_degree: int, out_dir: Path) -> None:
    """Tesseract (black-box binary, stock models) timed like our engine: one process per page, start and model load included,
    peak RSS from wait4, whole set per run, `repeats` runs, at one OpenMP thread and at the default degree. The PSM per model is
    the one `tess` selected for that set. Sets already in tess-timing.json are kept unless re-run."""
    path = out_dir / "tess-timing.json"
    doc = json.loads(path.read_text()) if path.exists() else {"sets": {}}
    doc.update(env=_env_info(), repeats=args.repeats, default_degree=default_degree,
               lock=LOCK.read_text().strip() if LOCK.exists() else "none (smoke)")
    during = doc.setdefault("load", {}).setdefault("during", [])
    doc["load"]["before"] = _load()
    scratch = Path(args.work).expanduser() / "tess-timing-tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    for n in names:
        pages = sorted(p for p in (Path(args.eval) / n / "pages").iterdir() if p.suffix in (".png", ".jpg"))
        if args.pages:
            pages = pages[: args.pages]
        meta = tess_results / n / "tesseract.json"
        psms = json.loads(meta.read_text())["best_psm"] if meta.exists() else {}
        entry: dict = {"pages": len(pages), "models": {}}
        for model in ("best", "fast"):
            psm = psms.get(model)
            if psm is None:
                psm = 6
                print(f"warning: no selected PSM for {model}/{n} in {meta}; using 6", file=sys.stderr)
            m: dict = {"psm": psm, "threads": {}}
            for threads in sorted({1, default_degree}):
                ms, rss = [], []
                for i in range(args.repeats):
                    why = lock_ok(args)
                    if why:
                        raise SystemExit(f"lost the lock during timing: {why}")
                    times, peaks = [], []
                    for img in pages:
                        dt, kb = tess_run_one(args.tess, model, psm, img, scratch / img.stem, threads)
                        times.append(dt)
                        peaks.append(kb)
                    ms.append(statistics.fmean(times) * 1000)
                    rss.append(max(peaks) / 1024)
                    during.append(_load()[0])
                    print(f"tesseract {n} {model} psm{psm} threads {threads} run {i}: {ms[-1]:.0f} ms/page, {rss[-1]:.0f} MB, load1 {during[-1]}", flush=True)
                m["threads"][str(threads)] = {"ms_per_page_runs": ms, "mean_ms_per_page": statistics.fmean(ms),
                                              "peak_rss_mb_runs": rss, "mean_peak_rss_mb": statistics.fmean(rss)}
            entry["models"][model] = m
        doc["sets"][n] = entry
        doc["load"]["after"] = _load()
        path.write_text(json.dumps(doc, indent=1) + "\n")  # after every set, so a lost job keeps what it measured


def tess_run_one(tess: str, model: str, psm: int, img: Path, out_base: Path, threads: int) -> tuple[float, int]:
    env = dict(os.environ, OMP_THREAD_LIMIT=str(threads))
    t0 = time.perf_counter()
    p = subprocess.Popen([tess, model, str(img), str(out_base), "-l", "eng", "--psm", str(psm)], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, status, ru = os.wait4(p.pid, 0)
    dt = time.perf_counter() - t0
    out_base.with_suffix(".txt").unlink(missing_ok=True)
    if status != 0:
        raise RuntimeError(f"tesseract failed on {img.name} (status {status})")
    return dt, ru.ru_maxrss  # KiB on Linux


# --------------------------------------------------------------------------- report

def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.2f}"


def render(out_dir: Path) -> None:
    acc_p, tim_p = out_dir / "accuracy.json", out_dir / "timing.json"
    acc = json.loads(acc_p.read_text()) if acc_p.exists() else None
    tim = json.loads(tim_p.read_text()) if tim_p.exists() else None
    tt_p = out_dir / "tess-timing.json"
    tt = json.loads(tt_p.read_text()) if tt_p.exists() else None
    report: dict = {"accuracy": None, "timing": None}
    md = ["# Gate table" + (" (SEALED set)" if acc and acc.get("sealed") else ""), ""]
    if acc and acc.get("sealed"):
        md += ["Sealed evaluation set: pages and seeds are not public; the SHA256SUMS digest below identifies the exact pages. "
               "A run on it counts once per milestone and nothing is tuned against it.", ""]
    if acc:
        md += [f"Accuracy: model `{acc['model_sha256'][:12]}…`" + (f", detector `{acc['detector_sha256'][:12]}…`" if acc["detector_sha256"] else ", no detector (classical finder)"),
               _models_line(acc["models"]) if acc.get("models") else "",
               f"engine {acc['runtime']}; eval sets `{acc['eval']}` (SHA256SUMS `{(acc['eval_sums_sha256'] or 'n/a')[:12]}…`); scorer {acc.get('scorer', 'v2')} (verdicts below; the other version is listed after the tables), boxes: ink.",
               f"{acc['env']['cpu']}, {acc['env']['cores']} threads; load before {acc['load_before']}, after {acc['load_after']}; started {acc['started']}.", "",
               "Quality: error rates in %. Baseline per metric: " + ("the best value any measured engine reaches (Tesseract best-of with its best PSM per metric, and the competitor configurations of the frozen results directory)" if acc.get("competitors") else "Tesseract best-of (best PSM per metric)") +
               ". Receipts: recall-CER / recall-WER (order is shown, not judged; so is the order of screenshot-real).", "",
               ]
        report["accuracy"] = {}
        # Sets we rendered and sets sourced from real captures are reported in separate tables.
        groups = [(t, [n for n in acc["sets"] if is_real(n) == real]) for t, real in (("Synthetic sets", False), ("Real-sourced sets", True))]
        groups = [(t, ns) for t, ns in groups if ns]
        for title, gnames in groups:
            if len(groups) > 1:
                md += [f"## {title}", ""]
            md += ["| category (set) | metric | ours | best | best engine / configuration | Tesseract best | Tesseract fast | verdict |", "|---|---|---|---|---|---|---|---|"]
            for name in gnames:
                d = acc["sets"][name]
                q = d["quality"]
                for r in q["rows"]:
                    md.append(f"| {category(name)} ({name}) | {r['metric']} | {_pct(r['ours'])} | {_pct(r.get('best'))} | {r.get('best_engine', '-')} | "
                              f"{_pct(r.get('tess_best'))} | {_pct(r.get('tess_fast'))} | {r['verdict']} |")
                report["accuracy"][name] = {"quality": q, "calibration": d["calibration"], "f1": d["f1"], "real": is_real(name)}
            if acc.get("competitors"):
                md += ["", "All engines, error rate % per metric (the best per row is in bold):", ""]
                for name in gnames:
                    rows = acc["sets"][name]["quality"]["rows"]
                    engines = sorted({e for r in rows for e in r.get("others", {})})
                    md += [f"{category(name)} ({name})", "", "| metric | ours | " + " | ".join(engines) + " |", "|---|---|" + "---|" * len(engines)]
                    for r in rows:
                        best = r.get("best")
                        cells = []
                        for e in engines:
                            v = r.get("others", {}).get(e)
                            cells.append("-" if v is None else (f"**{v * 100:.2f}**" if v == best else f"{v * 100:.2f}"))
                        md.append(f"| {r['metric']} | {_pct(r['ours'])} | " + " | ".join(cells) + " |")
                    md.append("")
            if any("other_scorer" in acc["sets"][n] for n in gnames):
                oth = acc["sets"][gnames[0]].get("other_scorer", {}).get("scorer", "v2")
                md += ["", f"Scorer {acc.get('scorer', 'v2')} against scorer {oth} (error rate %, best any engine, verdict):", "",
                       f"| category (set) | metric | ours {acc.get('scorer', 'v2')} | best {acc.get('scorer', 'v2')} | best engine {acc.get('scorer', 'v2')} | verdict {acc.get('scorer', 'v2')} | ours {oth} | best {oth} | best engine {oth} | verdict {oth} |", "|---|---|---|---|---|---|---|---|---|---|"]
                for name in gnames:
                    d = acc["sets"][name]
                    if "other_scorer" not in d:
                        continue
                    for r, o in zip(d["quality"]["rows"], d["other_scorer"]["quality"]["rows"]):
                        md.append(f"| {category(name)} ({name}) | {r['metric']} | {_pct(r['ours'])} | {_pct(r.get('best'))} | {r.get('best_engine', '-')} | {r['verdict']} | "
                                  f"{_pct(o['ours'])} | {_pct(o.get('best'))} | {o.get('best_engine', '-')} | {o['verdict']} |")
                    report["accuracy"][name]["other_scorer"] = d["other_scorer"]
            md += ["", "| category (set) | quality | pages | ECE | ECE-m | calibration (≤ 0.05) | F1@0.5 ink: ours | Tess best | Tess fast |", "|---|---|---|---|---|---|---|---|---|"]
            for name in gnames:
                d = acc["sets"][name]
                c, q, f = d["calibration"], d["quality"], d["f1"]
                ece = "-" if c["ece"] is None else f"{c['ece']:.4f}"
                ecem = "-" if c["ece_matched"] is None else f"{c['ece_matched']:.4f}"
                cv = c["verdict"] + (f" ({c['note']})" if c.get("note") else "")
                qv = "PASS" if q["pass"] else "FAIL"
                f1o = "-" if f["ours"] is None else f"{f['ours']:.3f}"
                f1b = "-" if f["tess_best"] is None else f"{f['tess_best']:.3f}"
                f1f = "-" if f["tess_fast"] is None else f"{f['tess_fast']:.3f}"
                md.append(f"| {category(name)} ({name}) | {qv} | {d['pages']} | {ece} | {ecem} | {cv} | {f1o} | {f1b} | {f1f} |")
            md.append("")
    else:
        md += ["No accuracy run yet (`gate` without `--timing`).", ""]
    if tim:
        qmap = {n: d["quality"] for n, d in acc["sets"].items()} if acc else {}
        sp = speed_rows(tim, tt, qmap)
        report["timing"] = {"env": tim["env"], "sets": sp, "load": tim["load"]}
        md += ["## Speed and memory", "",
               _models_line(tim["models"]) if tim.get("models") else "",
               f"{tim['env']['cpu']}, {tim['env']['cores']} threads. Load (1 min) before {tim['load']['before']}, during (after each run) "
               f"min {min(tim['load']['during'])} / max {max(tim['load']['during'])}, after {tim['load']['after']}; lock held by the job: {tim['lock']}. "
               f"Each cell is the mean of {tim['repeats']} runs (one `{(tim.get('engine') or {}).get('process', 'OneOcr.Bench run')}` process per run, {(tim.get('engine') or {}).get('rid', '')} "
               f"instruction set {(tim.get('engine') or {}).get('instruction_set', 'JIT')}, start and model load included, one pass over the set). Speed: faster than Tesseract fast; "
               "slower than fast but faster than best with the quality row won is an exception, not a pass. "
               f"Memory: peak RSS at the default degree at most {MEM_CAP_MB} MB (degree 1 is shown for information). "
               f"Tesseract timed the same way (stock models, PSM per set as selected by `tess`, one process per page, OMP_THREAD_LIMIT = 1 / {tim['default_degree']}, "
               f"{(tt or {}).get('repeats', '-')} runs; load before {((tt or {}).get('load') or {}).get('before')}, after {((tt or {}).get('load') or {}).get('after')}). "
               "`page ms` (model loaded once) is information only.", "",
               "| category | config | ours page ms (info) | ours process ms/page | Tess fast | Tess best | speed | ours RSS MB | Tess fast | Tess best | memory |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, rows in sp.items():
            for r in rows:
                fm = "-" if "fast_ms" not in r else f"{r['fast_ms']:.0f}"
                bm = "-" if "best_ms" not in r else f"{r['best_ms']:.0f}"
                fr = "-" if "fast_rss_mb" not in r else f"{r['fast_rss_mb']:.0f}"
                br = "-" if "best_rss_mb" not in r else f"{r['best_rss_mb']:.0f}"
                md.append(f"| {category(name)} ({name}) | {r['config']} | {r['ours_ms']:.0f} | {r['ours_process_ms_per_page']:.0f} | {fm} | {bm} | {r['speed']} | "
                          f"{r['ours_rss_mb']:.0f} | {fr} | {br} | {r['memory']} |")
        md.append("")
    (out_dir / "gate.json").write_text(json.dumps(report, indent=1) + "\n")
    (out_dir / "gate.md").write_text("\n".join(md))
    print("\n".join(md))


# --------------------------------------------------------------------------- modes

def _eval_sets(args) -> list[str]:
    names = args.sets or discover(Path(args.eval))
    if not names:
        raise SystemExit(f"no sets (<set>/pages and <set>/truth) under {args.eval}")
    for n in names:
        if not (Path(args.eval) / n / "pages").is_dir():
            raise SystemExit(f"{Path(args.eval) / n / 'pages'} not found (--eval / ONEOCR_EVAL)")
    return names


def _tess_results(args, work: Path) -> Path:
    if args.tess_results:
        return Path(args.tess_results).expanduser()
    if os.environ.get("ONEOCR_BENCH_DIR"):
        return data_dir() / "results"
    return work / "tess-data" / "results"


def rerun_tess(args, names: list[str], work: Path) -> None:
    # tess writes into its data directory, and the frozen sets are read-only: give it a writable root that links to them.
    root = work / "tess-data"
    root.mkdir(parents=True, exist_ok=True)
    for n in names:
        link = root / n
        if not link.exists():
            link.symlink_to(Path(args.eval).resolve() / n)
    os.environ["ONEOCR_BENCH_DIR"] = str(root)
    for n in names:
        tess_mod.main(n, ["best", "fast"], args.psms, args.tess, None)


def lock_ok(args) -> str | None:
    """None when timing may go on. A smoke run of at most two pages is allowed while no lock file exists at all."""
    why = lock_held_by_this_job()
    if why and args.tess_timing and args.pages and args.pages <= 2 and not LOCK.exists():
        return None
    return why


def dry_run(args, names: list[str]) -> None:
    """List what a run would read; opens no page and runs no engine."""
    root = Path(args.eval)
    sums = root / "SHA256SUMS"
    print(f"eval root {root}{' (SEALED)' if args.sealed else ''}; SHA256SUMS {_sha(str(sums))[:12] + '…' if sums.exists() else 'absent'}")
    for n in names:
        pages = sorted(p for p in (root / n / "pages").iterdir() if p.suffix in (".png", ".jpg"))
        truth = sorted((root / n / "truth").glob("*.txt"))
        print(f"{n}: {len(pages)} pages, {len(truth)} truth files" + ("" if len(pages) == len(truth) else "  MISMATCH"))


def main(args) -> None:
    if args.sealed:
        args.eval = args.sealed
    work = Path(args.work).expanduser()
    out_dir = Path(args.out).expanduser() if args.out else work / "gate"
    args.work = str(work)
    args.eval = str(Path(args.eval).expanduser())
    args.tess = str(Path(args.tess).expanduser())
    tess_only = args.timing and args.tess_timing
    if args.tess_timing and not args.timing:
        raise SystemExit("--tess-timing is a timing mode: use it with --timing")
    if not tess_only and not args.model and not args.dry_run:
        raise SystemExit("--model is required")
    if args.timing:
        why = lock_ok(args)
        if why:
            print(f"gate --timing refused: {why}.\nClaim the lock inside the job (see `python -m oneocr_bench gate --help`) and retry.", file=sys.stderr)
            raise SystemExit(75)
    names = _eval_sets(args)
    if args.dry_run:
        dry_run(args, names)
        return
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.model = str(Path(args.model).expanduser()) if args.model else None
    args.detector = str(Path(args.detector).expanduser()) if args.detector else None
    if args.wordlm and args.no_wordlm:
        raise SystemExit("--wordlm and --no-wordlm exclude each other")
    if args.model and not args.wordlm and not args.no_wordlm:
        raise SystemExit("--wordlm FILE is required: the shipped pipeline decides word breaks with the character model (charlm-v1). "
                        "Pass --no-wordlm to measure without it.")
    args.wordlm = str(Path(args.wordlm).expanduser()) if args.wordlm else None
    if args.rerun_tess:
        rerun_tess(args, names, work)
    tess_results = _tess_results(args, work)
    comp = load_competitors(args.competitors)
    comp_by = {sc: competitors_for(comp, sc) for sc in SCORERS} if not tess_only else {}
    default_degree = min(4, os.cpu_count() or 1)
    if tess_only:
        time_tesseract(args, names, tess_results, default_degree, out_dir)
        render(out_dir)
        return
    if args.timing:
        tool, tool_info = build_tool(work, args.dotnet, args.isa)
        print(f"timing process: {tool_info['process']}, {tool_info['rid']}, instruction set {tool_info['instruction_set']}", flush=True)
    else:
        dll = Path(args.bench_dll) if args.bench_dll else build_bench(work, args.dotnet)
    sums = Path(args.eval) / "SHA256SUMS"
    meta = Path(args.eval) / "SEALED.json"
    # Reading order is information, not judged, where the tree's SEALED.json says so; with no SEALED.json the registry default applies.
    order_info = {n: d.get("order_metric") == "information" for n, d in json.loads(meta.read_text())["sets"].items()} if meta.exists() else {}

    if not args.timing:
        env, load_before, sets = _env_info(), _load(), {}
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        runtime = ""
        for n in names:
            truth = Path(args.eval) / n / "truth"
            hyp = work / "out" / n
            if not (args.reuse and (hyp / "timing.json").exists()):
                _, t = run_engine(args.dotnet, dll, args, n, hyp, 0, True)
            t = json.loads((hyp / "timing.json").read_text())
            runtime = t["runtime"]
            per = {}
            for sc in (args.scorer, "v2" if args.scorer == "v3" else "v3"):
                ss = score_dir(truth, hyp, "ink", sc)
                tbs = tess_baselines(n, truth, tess_results, sc)
                if tbs is None:
                    print(f"warning: no Tesseract results for {n} under {tess_results}; run with --rerun-tess", file=sys.stderr)
                per[sc] = (ss, tbs, quality(n, ss, tbs, comp_by[sc], order_info.get(n, n in ORDER_INFO_ONLY)))
            s, tb, qual = per[args.scorer]
            other = "v2" if args.scorer == "v3" else "v3"
            sets[n] = {"scorer": args.scorer,
                       "other_scorer": {"scorer": other, "scores": {k: per[other][0][k] for k in ("cer", "wer", "order_err", "cer_recall", "wer_recall")}, "quality": per[other][2]},
                       "pages": s["pages"], "missing": s["missing"], "scores": {k: s[k] for k in ("cer", "wer", "order_err", "cer_recall", "wer_recall")},
                       "quality": qual, "calibration": calibration(n, s),
                       "f1": {"ours": s["det"]["f1"] if s["det"] else None,
                              "tess_best": tb["models"]["best"]["f1_ink"] if tb and "best" in tb["models"] else None,
                              "tess_fast": tb["models"]["fast"]["f1_ink"] if tb and "fast" in tb["models"] else None},
                       "ece": s["ece"], "det": s["det"], "tess_source": tb["source"] if tb else None}
            print(f"{n}: CER {s['cer']:.4%} WER {s['wer']:.4%} order {s['order_err']:.4%} rCER {s['cer_recall']:.4%} "
                  f"quality {'PASS' if sets[n]['quality']['pass'] else 'FAIL'}", flush=True)
        (out_dir / "accuracy.json").write_text(json.dumps({
            "started": started, "env": env, "load_before": load_before, "load_after": _load(), "runtime": runtime,
            "model": args.model, "model_sha256": _sha(args.model), "detector": args.detector, "detector_sha256": _sha(args.detector),
            "models": model_files(args),
            "competitors": args.competitors, "competitors_sha256": _sha(str(Path(args.competitors).expanduser() / "scores.json")) if args.competitors else None,
            "eval": args.eval, "sealed": bool(args.sealed), "eval_sums_sha256": _sha(str(sums)) if sums.exists() else None, "scorer": args.scorer, "scorer_sha256": _sha(str(Path(__file__).with_name("score.py"))),
            "sets": sets}, indent=1) + "\n")
        render(out_dir)
        return

    # timing: separate processes per set and degree so peak RSS belongs to one configuration.
    load_before, during, sets = _load(), [], {}
    for n in names:
        entry: dict = {}
        for tag, deg in (("1", 1), ("default", 0)):
            ms, rss, proc, load_ms = [], [], [], []
            for i in range(args.repeats):
                why = lock_ok(args)
                if why:
                    raise SystemExit(f"lost the lock during timing: {why}")
                wall, t = run_tool(tool, args, n, work / "timing" / f"{n}-d{tag}-r{i}", deg)
                during.append(_load()[0])
                npages = len(t["pages"])
                ms.append(t["mean_page_ms"])
                rss.append(t["peak_rss_mb"])
                load_ms.append(t["model_load_ms"])
                proc.append(wall * 1000 / max(1, npages))
                print(f"{n} degree {tag} run {i}: {t['mean_page_ms']} ms/page, {t['peak_rss_mb']} MB, load1 {during[-1]}", flush=True)
            entry[tag] = {"ms_per_page_runs": ms, "mean_ms_per_page": statistics.fmean(ms), "peak_rss_mb_runs": rss,
                          "mean_peak_rss_mb": statistics.fmean(rss), "mean_process_ms_per_page": statistics.fmean(proc),
                          "mean_model_load_ms": statistics.fmean(load_ms)}
        sets[n] = entry
    (out_dir / "timing.json").write_text(json.dumps({
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": _env_info(), "repeats": args.repeats, "default_degree": default_degree,
        "engine": tool_info, "lock": LOCK.read_text().strip() if LOCK.exists() else "none (smoke)", "load": {"before": load_before, "during": during, "after": _load()},
        "model_sha256": _sha(args.model), "detector_sha256": _sha(args.detector), "models": model_files(args), "sets": sets}, indent=1) + "\n")
    render(out_dir)
