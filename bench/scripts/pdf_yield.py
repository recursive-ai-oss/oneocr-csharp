"""Per-file yield of the PDF acceptance rules on a few random pages of every fetched file; stages nothing.

    python scripts/pdf_yield.py SEEDS.json PAGES_PER_FILE SECONDS PDF_DIR PDFTRUTH_DLL
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from oneocr_bench import realpdf

pdfs = Path(sys.argv[4]).expanduser()
dll = os.path.expanduser(sys.argv[5])
seed = int(json.load(open(sys.argv[1]))["real-pdf"])
per = int(sys.argv[2])
deadline = time.time() + float(sys.argv[3])
tot = Counter()
reasons = Counter()
for r in json.load(open(pdfs / "FETCHED.json")):
    if not r.get("ok") or time.time() > deadline:
        continue
    path = pdfs / r["file"]
    rng = random.Random(f"{seed}:{r['file']}")
    pages = sorted(rng.sample(range(1, r["pages"] + 1), min(r["pages"], per)))
    tp = realpdf.dotnet_run(dll, path, pages)
    c = Counter()
    for p in pages:
        if p not in tp:
            c["PdfTruth returned no page"] += 1
            continue
        try:
            cand, why, info = realpdf.judge(path, p, tp[p])
        except Exception as e:  # noqa: BLE001
            cand, why, info = None, type(e).__name__, {}
        k = "ACCEPT " + info["class"] if cand else re.sub(r"\(.*?\)", "", why).strip()
        c[k] += 1
        reasons[k] += 1
    src = r["source"]
    tot[src + " pages"] += len(pages)
    for k, v in c.items():
        if k.startswith("ACCEPT"):
            tot[src + " " + k] += v
    print(r["file"], src, r["pages"], dict(c), flush=True)
print(json.dumps(dict(tot), indent=1))
print(json.dumps(dict(reasons.most_common()), indent=1))
