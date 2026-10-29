"""Offline check of cheaper angle-search policies against a forced-search trace: which angle each policy would pick compared with the
full search, and the recall-CER that follows when the picks agree.

    uv run python scripts/angle_policy.py --truth T --log TRACE.log --base DIR --forced DIR [--scorer v3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from angle_trace import MARGIN, misses, parse  # noqa: E402


def full_pick(p: dict) -> int:
    """The angle (degrees) the full search keeps: best coarse score above the margin, then the best finer step (all in the trace)."""
    best, thr = 0, p["first"] * MARGIN
    for ang, sc in p["scores"].items():
        if sc > thr:
            thr, best = sc, ang
    return best


def policy_pick(p: dict, coarse: tuple[int, ...]) -> int:
    best, thr = 0, p["first"] * MARGIN
    for ang in coarse:
        sc = p["scores"].get(ang)
        if sc is not None and sc > thr:
            thr, best = sc, ang
    if best:
        # fine: one step of 5 degrees either side, as the search does (they are in the trace for the full search's winner only)
        for ang in (best + 5, best - 5):
            sc = p["scores"].get(ang)
            if sc is not None and sc > thr and (ang < 0) == (best < 0):
                thr, best = sc, ang
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--forced", required=True)
    ap.add_argument("--scorer", default="v3")
    a = ap.parse_args()
    pages = parse(Path(a.log))
    policies = {"full (10,20,30)": (-30, -20, -10, 10, 20, 30), "10 only": (-10, 10), "10,20": (-20, -10, 10, 20)}
    for share_hi in (0.6, 0.8, 0.9):
        for name, coarse in policies.items():
            total = base_t = forced_t = pol_t = unknown = same = 0
            for stem, p in pages.items():
                b, n = misses(Path(a.truth), Path(a.base), stem, a.scorer)
                f, _ = misses(Path(a.truth), Path(a.forced), stem, a.scorer)
                total += n
                base_t += b
                forced_t += f
                if p["share"] >= share_hi:
                    pol_t += b
                    continue
                fp, pp = full_pick(p), policy_pick(p, coarse)
                if pp == fp:
                    pol_t += f
                    same += 1
                elif pp == 0:
                    pol_t += b
                else:
                    pol_t += f  # a different angle: assumed to read like the full search's (upper bound, counted as unknown)
                    unknown += 1
            print(f"share<{share_hi} policy {name:16s}: recall-CER {100 * pol_t / total:.3f} (base {100 * base_t / total:.3f}, full forced {100 * forced_t / total:.3f}); agree {same}, different {unknown}")


if __name__ == "__main__":
    main()
