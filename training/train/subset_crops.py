"""A view of a rendered crop set (images/ + labels.jsonl) restricted by rows: small type, lines with spaces.

Used to give tiny screen text a larger share of a fine-tune batch without rendering again: the full set stays as it is, the view
symlinks the chosen crops and writes its own labels.jsonl, so the training loop reads it like any other crop set.

    uv run python -m train.subset_crops --src SET/train --out VIEW/train --max-ink-h 14 [--needs-space]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ink-h", type=float, default=14.0, help="device px of the ink box height (rows without ink_h are dropped)")
    ap.add_argument("--needs-space", action="store_true", help="keep lines with at least one word space")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    kept = []
    for ln in (src / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        if "ink_h" not in r or r["ink_h"] > a.max_ink_h or (a.needs_space and " " not in r["text"]):
            continue
        dst = out / "images" / f"{r['id']}.png"
        if not dst.exists():
            os.symlink((src / "images" / f"{r['id']}.png").resolve(), dst)
        kept.append(ln)
    (out / "labels.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(json.dumps({"src": str(src), "kept": len(kept)}))


if __name__ == "__main__":
    main()
