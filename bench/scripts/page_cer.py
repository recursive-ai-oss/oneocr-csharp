"""Per-page character errors of our run against other runs on one set (scorer v3.2).

    uv run python scripts/page_cer.py --eval SET_DIR --run OURS [--ref NAME=DIR ...]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oneocr_bench.score import score_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref", action="append", default=[])
    a = ap.parse_args()
    t = Path(a.eval) / "truth"
    runs = [("ours", a.run)] + [tuple(r.split("=", 1)) for r in a.ref]
    res = {n: score_dir(t, Path(d), "ink", "v3.2") for n, d in runs}
    print("page chars " + " ".join(f"{n:>7}" for n, _ in runs))
    for p in sorted(res["ours"]["per_page"]):
        pr = res["ours"]["per_page"][p]
        print(p, f"{pr['ref_chars']:5d}", " ".join(f"{res[n]['per_page'][p]['char_errors']:7d}" for n, _ in runs))
    print("total", " ".join(f"{res[n]['totals'].get('char_errors', '')!s:>7}" for n, _ in runs))


if __name__ == "__main__":
    main()
