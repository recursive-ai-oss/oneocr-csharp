"""Compares the tensors of two detector containers (exact fp16 equality) and their post-processing and trained-class config.

    uv run python -m export.same_weights A.onm B.onm
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from export.dbnet import read_tensors


def main() -> None:
    (ma, ta), (mb, tb) = read_tensors(Path(sys.argv[1])), read_tensors(Path(sys.argv[2]))
    same = set(ta) == set(tb) and all(np.array_equal(ta[k], tb[k]) for k in ta)
    worst = max((float(np.abs(ta[k].astype(np.float32) - tb[k].astype(np.float32)).max()) for k in ta if k in tb), default=0.0)
    print(f"tensors identical: {same} (max abs diff {worst:.3g}, {len(ta)} vs {len(tb)} tensors)")
    for key in ("postprocess", "layout_trained"):
        print(key, "equal" if ma["config"].get(key) == mb["config"].get(key) else f"differ: {ma['config'].get(key)} vs {mb['config'].get(key)}")


if __name__ == "__main__":
    main()
