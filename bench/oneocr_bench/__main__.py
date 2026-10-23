from __future__ import annotations

import argparse
import os

from . import assets, gen, score, tess


def main() -> None:
    ap = argparse.ArgumentParser(prog="oneocr_bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download and verify fonts and public-domain text")
    f.add_argument("--update", action="store_true")

    g = sub.add_parser("gen", help="render the page set")
    g.add_argument("--count", type=int, default=30)
    g.add_argument("--seed", type=int, default=20261023)
    g.add_argument("--check", action="store_true", help="compare with the committed manifest")

    d = sub.add_parser("draw", help="overlay truth boxes on a page")
    d.add_argument("page")

    s = sub.add_parser("score", help="score a directory of <page-id>.txt hypotheses")
    s.add_argument("--truth", default=None, help="default: the generated set's truth directory")
    s.add_argument("--hyp", required=True)
    s.add_argument("--json")

    t = sub.add_parser("tess", help="Tesseract baselines")
    t.add_argument("--models", nargs="+", default=["best", "fast"])
    t.add_argument("--psms", nargs="+", type=int, default=[3, 4, 6])
    t.add_argument("--tess", default=os.path.expanduser("~/tools/tesseract/tess"))
    t.add_argument("--results")

    a = ap.parse_args()
    if a.cmd == "fetch":
        assets.fetch(a.update)
    elif a.cmd == "gen":
        gen.gen(a.count, a.seed, a.check)
    elif a.cmd == "draw":
        gen.draw(a.page)
    elif a.cmd == "score":
        truth = a.truth or str(gen.data_dir() / gen.SET_NAME / "truth")
        score.main(truth, a.hyp, a.json)
    elif a.cmd == "tess":
        tess.main(a.models, a.psms, gen.SET_NAME, a.tess, a.results)


main()
