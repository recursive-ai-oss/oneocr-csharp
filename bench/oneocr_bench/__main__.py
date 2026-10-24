from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import assets, competitors, score, sets, tess


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
    elif a.cmd == "tess-report":
        tess.report()


main()
