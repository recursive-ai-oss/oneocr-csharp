"""Side-by-side page-level table of recognizer runs written by eval/page_eval.sh.

    uv run python -m eval.page_table NAME=SCORES_DIR [NAME=SCORES_DIR ...]

One row per set: CER/WER/order (percent), amount exact match where the truth is tagged, recall-CER/WER on the receipts sets, ECE.
The first run is the reference; a '*' marks a metric that is worse than the reference by more than its noise floor
(0.02 pp for CER/WER/order, 0.3 pp for amount match).
"""

import json
import sys
from pathlib import Path


def row(r: dict) -> dict:
    a = r.get("amounts")
    e = r.get("ece")
    return {
        "cer": 100 * r["cer"], "wer": 100 * r["wer"], "order": 100 * r["order_err"],
        "rcer": 100 * r["cer_recall"], "rwer": 100 * r["wer_recall"],
        "amt": 100 * a["amount_exact"] if a else None,
        "ece": (e.get("ece_m") or e.get("ece")) if isinstance(e, dict) else None,
    }  # fmt: skip


def main() -> None:
    runs = [(a.split("=")[0], Path(a.split("=", 1)[1])) for a in sys.argv[1:]]
    data = {n: {p.stem: row(json.loads(p.read_text())) for p in d.glob("*.json")} for n, d in runs}
    ref = runs[0][0]
    print("| set | " + " | ".join(n for n, _ in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for s in sorted(data[ref]):
        cells = []
        for n, _ in runs:
            r = data[n].get(s)
            if not r:
                cells.append("-")
                continue
            b = data[ref][s]
            flag = lambda k, tol: "*" if n != ref and r[k] is not None and b[k] is not None and r[k] > b[k] + tol else ""  # noqa: E731
            if s.startswith("receipts-gen"):
                t = f"{r['rcer']:.2f}{flag('rcer', .02)}/{r['rwer']:.2f}{flag('rwer', .02)}"
            else:
                t = f"{r['cer']:.3f}{flag('cer', .02)}/{r['wer']:.3f}{flag('wer', .02)}/{r['order']:.2f}{flag('order', .02)}"
            if r["amt"] is not None:
                t += f" amt {r['amt']:.1f}" + ("*" if n != ref and b["amt"] is not None and r["amt"] < b["amt"] - 0.3 else "")
            if r["ece"] is not None:
                t += f" ece {r['ece']:.3f}"
            cells.append(t)
        print(f"| {s} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
