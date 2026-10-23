from __future__ import annotations

import argparse
import os

from . import assets, score, sets, tess


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
    s.add_argument("--set", choices=names, help="truth set (default clean300-v1 unless --truth is given)")
    s.add_argument("--truth", default=None, help="explicit truth directory")
    s.add_argument("--hyp", help="directory of hypotheses for one set")
    s.add_argument("--hyp-root", help="directory with one hypothesis subdirectory per set name; scores every set found")
    s.add_argument("--json")

    t = sub.add_parser("tess", help="Tesseract baselines for one set")
    t.add_argument("--set", default="clean300-v1", choices=names)
    t.add_argument("--models", nargs="+", default=["best", "fast"])
    t.add_argument("--psms", nargs="+", type=int, default=[3, 4, 6, 11])
    t.add_argument("--tess", default=os.path.expanduser("~/tools/tesseract/tess"))
    t.add_argument("--results")

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
        if a.hyp_root:
            score.main_root(a.hyp_root, a.json)
        else:
            if not a.hyp:
                ap.error("--hyp or --hyp-root is required")
            truth = a.truth or str(sets.set_dir(a.set or "clean300-v1") / "truth")
            score.main(truth, a.hyp, a.json)
    elif a.cmd == "tess":
        tess.main(a.set, a.models, a.psms, a.tess, a.results)
    elif a.cmd == "tess-report":
        tess.report()


main()
