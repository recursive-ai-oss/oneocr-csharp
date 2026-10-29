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
    # coarse: the angles probed (degrees); the finer step either side of a winner is used when the full search traced it
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
    # Share under 0.6 gets the full search (as before). Share from 0.6 up to D gets the light search (10 degrees either way, then the
    # finer step). Pages where the light search picks what the full one would are scored with the forced run; where it picks no turn
    # they are scored with the base run; where it picks another turn the number is a guess (the forced run's), counted as "different".
    for d in (0.7, 0.8, 0.85, 0.9, 0.95):
        total = base_t = forced_t = pol_t = same = diff = missed = band = 0
        for stem, p in pages.items():
            b, n = misses(Path(a.truth), Path(a.base), stem, a.scorer)
            f, _ = misses(Path(a.truth), Path(a.forced), stem, a.scorer)
            total += n
            base_t += b
            forced_t += f
            if p["share"] < 0.6:
                pol_t += f
            elif p["share"] < d:
                band += 1
                fp, pp = full_pick(p), policy_pick(p, (-10, 10))
                if pp == fp:
                    pol_t += f if fp else b
                    same += 1
                elif pp == 0:
                    pol_t += b
                    missed += 1
                else:
                    pol_t += f
                    diff += 1
            else:
                pol_t += b
        print(f"doubt<{d}: band pages {band}, light = full pick {same}, no turn where full turns {missed}, other turn {diff}; recall-CER {100 * pol_t / total:.3f} (base {100 * base_t / total:.3f}, forced {100 * forced_t / total:.3f})")


if __name__ == "__main__":
    main()
