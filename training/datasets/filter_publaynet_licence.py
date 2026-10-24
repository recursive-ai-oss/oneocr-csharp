#!/usr/bin/env python3
"""Look up each PubLayNet article's licence and keep only CC0 and CC BY articles.

PubLayNet images are PubMed Central article pages, governed by the article's own
licence (see LICENCES.md). PMC publishes a per-article metadata object with a
`license_code` in its Open Access cloud bucket (https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/,
a public, anonymous HTTPS bucket that NCBI names as a supported way to retrieve
this data). Only articles whose code is exactly "CC0" or "CC BY" are kept;
a missing object, a missing code or any other code excludes the article.

    python3 filter_publaynet_licence.py [--root DIR]

Reads  <root>/publaynet/1.0.0/licence/pmcids_{train,val}.txt
Writes <root>/publaynet/1.0.0/licence/pmc_licences.json  (every article's code)
       <root>/publaynet/1.0.0/licence/allowed_{train,val}.txt  (kept articles)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
ALLOWED = {"CC0", "CC BY"}


def lookup(pmcid: str) -> str:
    """Licence code of the article's first version object; "" when the object is absent."""
    url = f"{BASE}/{pmcid}.1/{pmcid}.1.json"
    for _attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": c.USER_AGENT}), timeout=30) as r:
                return str(json.load(r).get("license_code") or "")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return ""
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
    raise SystemExit(f"{pmcid}: lookup kept failing")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    a = ap.parse_args()
    lic = c.data_root(a.root) / "publaynet" / "1.0.0" / "licence"
    ids = {s: (lic / f"pmcids_{s}.txt").read_text().split() for s in ("train", "val")}
    allids = sorted(set(ids["train"]) | set(ids["val"]))
    with ThreadPoolExecutor(32) as ex:
        codes = dict(zip(allids, ex.map(lookup, allids), strict=False))
    (lic / "pmc_licences.json").write_text(json.dumps(codes, indent=0, sort_keys=True))
    for s, lst in ids.items():
        keep = [p for p in lst if codes[p] in ALLOWED]
        (lic / f"allowed_{s}.txt").write_text("\n".join(keep) + "\n")
        c.log(f"{s}: {len(keep)} of {len(lst)} articles kept; codes {dict(Counter(codes[p] or '(none)' for p in lst).most_common())}")


if __name__ == "__main__":
    main()
