"""Dump every scorer number that an older scorer also computes, so two scorer versions can be compared byte for byte.

    PYTHONPATH=OLD_TREE python scripts/scorer_regress.py dump --scorer v3.1 --truth ROOT --hyp ROOT --out old.json
    PYTHONPATH=NEW_TREE python scripts/scorer_regress.py dump --scorer v3.2 --truth ROOT --hyp ROOT --out new.json
    python scripts/scorer_regress.py diff old.json new.json

ROOT holds one directory per set: `--truth` ROOT/<set>/truth and `--hyp` ROOT/<set> (`<id>.txt` and sidecars). With `--hyp-config`,
`--hyp` is a directory of engine configurations and every ROOT/<config>/<set> is scored. OLD_TREE is a directory holding the
older `oneocr_bench` package (`git archive` of its tag). The dump keeps CER, WER, order, recall variants, per-page counts, detection and
ECE; the `amounts` member that scorer v3.2 adds is left out because the older scorer does not have it. `diff` exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KEYS = ("pages", "missing", "cer", "wer", "order_err", "cer_recall", "wer_recall", "totals", "per_page", "det", "ece")


def dump(a) -> None:
    from oneocr_bench import score

    truth_root, hyp_root = Path(a.truth), Path(a.hyp)
    out: dict = {"scorer_module": score.__file__}
    configs = sorted(p.name for p in hyp_root.iterdir() if p.is_dir()) if a.hyp_config else [""]
    for c in configs:
        base = hyp_root / c if c else hyp_root
        for s in sorted(p.name for p in truth_root.iterdir() if (p / "truth").is_dir()):
            if not (base / s).is_dir():
                continue
            r = score.score_dir(truth_root / s / "truth", base / s, "ink", a.scorer)
            out[f"{c}/{s}"] = {k: r[k] for k in KEYS}
    Path(a.out).write_text(json.dumps(out, sort_keys=True, indent=1) + "\n")
    print(f"{len(out) - 1} (configuration, set) pairs scored with {a.scorer} from {score.__file__}")


def diff(a) -> None:
    x = json.loads(Path(a.old).read_text())
    y = json.loads(Path(a.new).read_text())
    x.pop("scorer_module")
    y.pop("scorer_module")
    bad = 0
    for k in sorted(set(x) | set(y)):
        if x.get(k) != y.get(k):
            bad += 1
            print("DIFFERENT:", k)
    pages = sum(v["pages"] for v in x.values())
    print(f"{len(x)} pairs, {pages} pages: {bad} different")
    sys.exit(1 if bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--scorer", required=True)
    d.add_argument("--truth", required=True)
    d.add_argument("--hyp", required=True)
    d.add_argument("--hyp-config", action="store_true")
    d.add_argument("--out", required=True)
    f = sub.add_parser("diff")
    f.add_argument("old")
    f.add_argument("new")
    a = ap.parse_args()
    {"dump": dump, "diff": diff}[a.cmd](a)


if __name__ == "__main__":
    main()
