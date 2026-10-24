from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import assets, competitors, gate, score, sets, tess


def main() -> None:
    ap = argparse.ArgumentParser(prog="oneocr_bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    names = list(sets.SETS)

    f = sub.add_parser("fetch", help="download and verify fonts and public-domain text")
    f.add_argument("--update", action="store_true")

    g = sub.add_parser("gen", help="render a set (or all)")
    g.add_argument("--set", default="clean300-v1", choices=[*names, "all"])
    g.add_argument("--count", type=int, default=None)
    g.add_argument("--check", action="store_true", help="compare with the committed manifest")

    d = sub.add_parser("draw", help="overlay truth boxes on a page")
    d.add_argument("--set", default="clean300-v1", choices=names)
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
    t.add_argument("--set", default="clean300-v1", choices=names)
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
    gt.add_argument("--eval", default=os.environ.get("ONEOCR_EVAL", "~/oneocr/eval/bench-v2"),
                    help="frozen sets: <set>/pages and <set>/truth (env ONEOCR_EVAL)")
    gt.add_argument("--sets", nargs="+", choices=names, default=None)
    gt.add_argument("--work", default=os.environ.get("ONEOCR_WORK", "~/oneocr/work/gate"), help="engine outputs and the build (env ONEOCR_WORK)")
    gt.add_argument("--out", default=None, help="where gate.md, gate.json, accuracy.json and timing.json go (default WORK/gate)")
    gt.add_argument("--tess-results", default=None, help="Tesseract results root holding <set>/tesseract.json (default $ONEOCR_BENCH_DIR/results)")
    gt.add_argument("--rerun-tess", action="store_true", help="run Tesseract again first (slow; hold the lock if its timing matters)")
    gt.add_argument("--tess", default=os.path.expanduser("~/tools/tesseract/tess"))
    gt.add_argument("--psms", nargs="+", type=int, default=[3, 4, 6, 11])
    gt.add_argument("--timing", action="store_true", help="speed and memory mode; refuses unless this job holds the machine lock")
    gt.add_argument("--tess-timing", action="store_true",
                    help="with --timing: time Tesseract best and fast (one process per page, 1 and default-degree OpenMP threads) instead of our engine; "
                         "writes tess-timing.json, merged per set")
    gt.add_argument("--pages", type=int, default=None, help="Tesseract timing: first N pages of each set only (2 or fewer is a smoke run)")
    gt.add_argument("--repeats", type=int, default=3)
    gt.add_argument("--reuse", action="store_true", help="accuracy: score engine outputs already under WORK/out instead of running the engine")
    gt.add_argument("--dotnet", default="dotnet")
    gt.add_argument("--bench-dll", default=None, help="use this OneOcr.Bench.dll instead of building")

    sub.add_parser("tess-report", help="merge per-set Tesseract results into one table")

    a = ap.parse_args()
    if a.cmd == "fetch":
        assets.fetch(a.update)
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
        tess.main(a.set, a.models, a.psms, a.tess, a.results)
    elif a.cmd == "competitors-report":
        print(competitors.report(a.root, a.eval, a.timing))
    elif a.cmd == "gate":
        gate.main(a)
    elif a.cmd == "tess-report":
        tess.report()


main()
