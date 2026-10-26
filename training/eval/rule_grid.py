# ruff: noqa: E501
"""Engine outputs of manifest-only detector variants over several page sets, for `eval.rule_select`.

For every variant (text_threshold:text_expand) the detector is re-exported with those post-processing values in its manifest (same
weights) and the C# bench runs the pages of every set. Output lands in OUT/t<thr>-e<exp>/<set>/ with word sidecars and page JSON.

    cd bench && uv run python ../training/eval/rule_grid.py --checkpoint CKPT --run-manifest RM --recognizer REC.onm \
        --set NAME=PAGES_DIR ... --variants 0.3:0.4,0.4:0.4 --out OUT [--detector-file MODEL.onm:NAME]

`--detector-file` adds an already exported container (the previous detector, say) as an extra variant NAME.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1]
BENCH = TRAIN.parent / "bench"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run-manifest", required=True)
    ap.add_argument("--recognizer", required=True)
    ap.add_argument("--set", action="append", required=True, help="NAME=PAGES_DIR")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="0.2.1")
    ap.add_argument("--degree", type=int, default=1)
    ap.add_argument("--dotnet", default="dotnet")
    ap.add_argument("--wordlm", help="character model file for word breaks, as the shipped engine loads it")
    ap.add_argument("--extra-post", default="{}", help="JSON merged into every variant's postprocess (the page-noise rule keys, say)")
    ap.add_argument("--detector-file", action="append", default=[], help="MODEL.onm:NAME")
    a = ap.parse_args()
    sys.path.insert(0, str(BENCH))
    from oneocr_bench.gate import build_bench  # noqa: PLC0415

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dll = build_bench(out, a.dotnet)
    models: dict[str, Path] = {}
    for pair in a.variants.split(","):
        thr, exp = pair.split(":")
        name = f"t{thr}-e{exp}"
        m = out / name / "model.onm"
        m.parent.mkdir(parents=True, exist_ok=True)
        post = json.dumps({"text_threshold": float(thr), "text_expand": float(exp), "layout_threshold": 0.6, "layout_expand": 0.85, **json.loads(a.extra_post)})
        subprocess.run(
            ["uv", "run", "--project", str(TRAIN), "python", "-m", "export.dbnet", "--checkpoint", a.checkpoint, "--run-manifest", a.run_manifest,
             "--version", a.version, "--exclude-classes", "form_field,checkbox", "--postprocess", post, "--out", str(m)],
            check=True, cwd=TRAIN, capture_output=True,
        )  # fmt: skip
        models[name] = m
    for spec in a.detector_file:
        f, name = spec.rsplit(":", 1)
        models[name] = Path(f)
    for name, m in models.items():
        for s in a.set:
            sname, pages = s.split("=", 1)
            dest = out / name / sname
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [a.dotnet, str(dll), "run", "--model", a.recognizer, "--detector", str(m), "--pages", pages, "--out", str(dest),
                 "--degree", str(a.degree), "--repeat", "1", "--words", "1", "--pagejson", "1", *(["--wordlm", a.wordlm] if a.wordlm else [])],
                check=True, stdout=subprocess.DEVNULL,
            )  # fmt: skip
        print("done", name, flush=True)


if __name__ == "__main__":
    main()
