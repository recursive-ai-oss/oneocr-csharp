from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import assets, competitors, gate, realpdf, realweb, score, seal, sets, tess


def main() -> None:
    ap = argparse.ArgumentParser(prog="oneocr_bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    names = list(sets.SETS)
    all_names = list(sets.ALL_SETS)

    f = sub.add_parser("fetch", help="download and verify fonts and public-domain text")
    f.add_argument("--update", action="store_true")

    g = sub.add_parser("gen", help="render a set (or all)")
    g.add_argument("--set", default="clean300-v1", choices=[*names, "all"])
    g.add_argument("--count", type=int, default=None)
    g.add_argument("--check", action="store_true", help="compare with the committed manifest")
    g.add_argument("--seed", type=int, default=None, help="another seed: render a new page set for --set into --out (pinned manifests untouched)")
    g.add_argument("--out", default=None, help="with --seed: output directory")
    g.add_argument("--synthetic-receipts", action="store_true", help="with --seed: receipts-v1 from the receipt generator, not CORD")
    g.add_argument("--avoid-text", nargs="*", default=[], metavar="DIR", help="with --seed: eval roots (<set>/truth) whose passages are not reused")

    sl = sub.add_parser("seal", help="cut or verify a sealed page set (all six categories, seeds from a private file)")
    sl.add_argument("action", choices=["build", "verify", "add-real"])
    sl.add_argument("--web", help="add-real: directory of the real screenshot set (pages/, truth/)")
    sl.add_argument("--replace", action="store_true", help="add-real: swap sets that are already in the tree (real-sourced sets only)")
    sl.add_argument("--pdf", help="add-real: directory holding clean-real, layout-real, degraded-real")
    sl.add_argument("root", help="sealed eval root: <set>/pages, <set>/truth, SHA256SUMS")
    sl.add_argument("--seeds", help="build: JSON file {set: seed}, kept outside the repository")
    sl.add_argument("--avoid-text", nargs="*", default=[], metavar="DIR", help="eval roots whose passages (and page hashes) must not recur")
    sl.add_argument("--sets", nargs="+", choices=all_names, default=None)

    d = sub.add_parser("draw", help="overlay truth boxes on a page")
    d.add_argument("--set", default="clean300-v1", choices=all_names)
    d.add_argument("page")

    s = sub.add_parser("score", help="score hypotheses: <id>.txt (and optional <id>.json boxes) per page")
    s.add_argument("dirs", nargs="*", metavar="DIR", help="TRUTH_DIR OUTPUT_DIR: a set directory (or its truth/ directory) and a directory of <id>.txt")
    s.add_argument("--set", choices=names, help="truth set (default clean300-v1 unless --truth is given)")
    s.add_argument("--truth", default=None, help="explicit truth directory")
    s.add_argument("--hyp", help="directory of hypotheses for one set")
    s.add_argument("--hyp-root", help="directory with one hypothesis subdirectory per set name; scores every set found")
    s.add_argument("--json")
    s.add_argument("--boxes", choices=["ink", "layout"], default="ink", help="truth boxes for F1: tight to the ink (default; falls back to layout where a truth has no ink box) or the layout rectangles")

    t = sub.add_parser("tess", help="Tesseract baselines for one set")
    t.add_argument("--set", default="clean300-v1")
    t.add_argument("--root", default=None, help="eval root holding <set>/pages and <set>/truth (default: the benchmark data directory)")
    t.add_argument("--jobs", type=int, default=1, help="pages run in this many single-thread processes at once (accuracy is unchanged, times are not publishable)")
    t.add_argument("--no-timing-runs", action="store_true", help="skip the default-threading re-run that only serves timing")
    t.add_argument("--models", nargs="+", default=["best", "fast"])
    t.add_argument("--psms", nargs="+", type=int, default=[3, 4, 6, 11])
    t.add_argument("--tess", default=os.path.expanduser("~/tools/tesseract/tess"))
    t.add_argument("--results")

    c = sub.add_parser("competitors-report", help="one table over competitor run directories (Apple Vision, PaddleOCR, docTR)")
    c.add_argument("root")
    c.add_argument("--timing", default=None, help="take timing.json from this tree instead of ROOT")
    c.add_argument("--eval", default=None, help="directory with <set>/truth (default: the benchmark data directory)")

    gt = sub.add_parser("gate", help="the whole gate table (quality, calibration; speed and memory with --timing) in one run",
                        description=gate.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gt.add_argument("--model", default=None, help="recognizer container (not needed with --tess-timing)")
    gt.add_argument("--detector", default=None, help="detector container (omit for the classical line finder)")
    gt.add_argument("--wordlm", default=None, help="word character model (charlm-v1.bin) the engine uses for word breaks; required with --model, as the shipped pipeline has it")
    gt.add_argument("--wordfreq", default=None, help="word frequency table the engine uses to take back spaces inside words; pass the bundled one to measure the shipped pipeline")
    gt.add_argument("--no-wordlm", action="store_true", help="measure without the word character model (not the shipped pipeline)")
    gt.add_argument("--eval", default=os.environ.get("ONEOCR_EVAL", "~/oneocr/eval/bench-v2"),
                    help="frozen sets: <set>/pages and <set>/truth (env ONEOCR_EVAL)")
    gt.add_argument("--sealed", default=None, metavar="DIR", help="score the sealed set at DIR (same layout as --eval) and mark the report as sealed")
    gt.add_argument("--dry-run", action="store_true", help="list the eval files and exit without building or running anything")
    gt.add_argument("--sets", nargs="+", default=None, help="set names under --eval (default: every set found there)")
    gt.add_argument("--work", default=os.environ.get("ONEOCR_WORK", "~/oneocr/work/gate"), help="engine outputs and the build (env ONEOCR_WORK)")
    gt.add_argument("--out", default=None, help="where gate.md, gate.json, accuracy.json and timing.json go (default WORK/gate)")
    gt.add_argument("--tess-results", default=None, help="Tesseract results root holding <set>/tesseract.json (default $ONEOCR_BENCH_DIR/results)")
    gt.add_argument("--rerun-tess", action="store_true", help="run Tesseract again first (slow; hold the lock if its timing matters)")
    gt.add_argument("--tess", default=os.path.expanduser("~/tools/tesseract/tess"))
    gt.add_argument("--psms", nargs="+", type=int, default=[3, 4, 6, 11])
    gt.add_argument("--timing", action="store_true", help="speed and memory mode; refuses unless this job holds the machine lock")
    gt.add_argument("--competitors", default=None,
                    help="frozen competitor-results directory (scores.json from competitors-freeze): the baseline per metric becomes the best engine measured")
    gt.add_argument("--tess-timing", action="store_true",
                    help="with --timing: time Tesseract best and fast (one process per page, 1 and default-degree OpenMP threads) instead of our engine; "
                         "writes tess-timing.json, merged per set")
    gt.add_argument("--pages", type=int, default=None, help="Tesseract timing: first N pages of each set only (2 or fewer is a smoke run)")
    gt.add_argument("--repeats", type=int, default=3)
    gt.add_argument("--reuse", action="store_true", help="accuracy: score engine outputs already under WORK/out instead of running the engine")
    gt.add_argument("--dotnet", default="dotnet")
    gt.add_argument("--isa", default="auto", help="timing: IlcInstructionSet of the NativeAOT oneocr tool: auto (x86-64-v4 on AVX-512 hosts, else v3), default (the project's), or a value such as x86-64-v3")
    gt.add_argument("--bench-dll", default=None, help="use this OneOcr.Bench.dll instead of building")

    cf = sub.add_parser("competitors-freeze", help="copy competitor hypotheses into a frozen directory and score them (input of gate --competitors)")
    cf.add_argument("--src", action="append", required=True, metavar="[PREFIX=]ROOT", help="directory with one subdirectory per configuration; repeatable")
    cf.add_argument("--eval", default=os.environ.get("ONEOCR_EVAL", "~/oneocr/eval/bench-v2"))
    cf.add_argument("--out", required=True)
    cf.add_argument("--info-only", nargs="*", default=["fast"], metavar="WORD", help="configurations whose name contains one of these words do not set the baseline (default: fast)")

    rw = sub.add_parser("realweb", help="capture real web pages as screenshot eval pages (list file is private)")
    rw.add_argument("action", choices=["capture"])
    rw.add_argument("--list", required=True)
    rw.add_argument("--out", required=True)
    rw.add_argument("--target", type=int, default=40)
    rw.add_argument("--seconds", type=float, default=1e9, help="stop starting new pages after this many seconds")

    rp = sub.add_parser("realpdf", help="fetch real PDFs and cut verified pages from them (list file is private)")
    rp.add_argument("action", choices=["fetch", "cut", "hashdir"])
    rp.add_argument("--list")
    rp.add_argument("--out")
    rp.add_argument("--dir", help="hashdir: directory of PDFs")
    rp.add_argument("--pdfs", help="cut: directory written by fetch")
    rp.add_argument("--seeds")
    rp.add_argument("--pdftruth", help="cut: PdfTruth.dll")
    rp.add_argument("--exclude", help="cut: file of SHA-256 hashes of PDFs that may be in training data")
    rp.add_argument("--clean", type=int, default=30)
    rp.add_argument("--layout", type=int, default=40)
    rp.add_argument("--degraded", type=int, default=40)

    sub.add_parser("tess-report", help="merge per-set Tesseract results into one table")

    a = ap.parse_args()
    if a.cmd == "fetch":
        assets.fetch(a.update)
    elif a.cmd == "gen" and a.seed is not None:
        if a.set == "all" or not a.out or a.check:
            ap.error("--seed needs one --set, --out, and no --check")
        sets.run_gen_seeded(a.set, a.count, a.seed, Path(a.out).expanduser(), a.synthetic_receipts, [Path(p).expanduser() for p in a.avoid_text])
    elif a.cmd == "realweb":
        realweb.capture(a)
    elif a.cmd == "realpdf":
        {"fetch": realpdf.fetch, "cut": realpdf.cut, "hashdir": realpdf.hashdir}[a.action](a)
    elif a.cmd == "seal":
        seal.main(a)
    elif a.cmd == "gen":
        for n in names if a.set == "all" else [a.set]:
            sets.run_gen(n, a.count, a.check)
    elif a.cmd == "draw":
        sets.draw(a.set, a.page)
    elif a.cmd == "score":
        if a.dirs:
            if len(a.dirs) != 2 or a.hyp or a.hyp_root or a.truth or a.set:
                ap.error("score takes either TRUTH_DIR OUTPUT_DIR or --hyp/--hyp-root options")
            t = Path(a.dirs[0])
            score.main(str(t / "truth" if (t / "truth").is_dir() else t), a.dirs[1], a.json, a.boxes)
        elif a.hyp_root:
            score.main_root(a.hyp_root, a.json, a.boxes)
        else:
            if not a.hyp:
                ap.error("--hyp or --hyp-root is required")
            truth = a.truth or str(sets.set_dir(a.set or "clean300-v1") / "truth")
            score.main(truth, a.hyp, a.json, a.boxes)
    elif a.cmd == "tess":
        tess.main(a.set, a.models, a.psms, a.tess, a.results, a.root, a.jobs, not a.no_timing_runs)
    elif a.cmd == "competitors-report":
        print(competitors.report(a.root, a.eval, a.timing))
    elif a.cmd == "competitors-freeze":
        competitors.freeze(a.src, a.eval, a.out, a.info_only)
    elif a.cmd == "gate":
        gate.main(a)
    elif a.cmd == "tess-report":
        tess.report()


main()
