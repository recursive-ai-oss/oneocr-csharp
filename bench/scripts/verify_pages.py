#!/usr/bin/env python3
"""Check a directory of copied page images against the manifests (standard library only).

    python3 scripts/verify_pages.py PAGES [set ...]      # PAGES/<set>/<id>.png|jpg
"""
import hashlib
import json
import sys
from pathlib import Path

manifests = Path(__file__).resolve().parent.parent / "manifests"
root = Path(sys.argv[1])
names = sys.argv[2:] or ["clean300-v1", "degraded-v1", "screenshot-v1", "layout-v1", "phone-v1", "receipts-v1"]
bad = 0
for name in names:
    pages = json.loads((manifests / f"{name}.json").read_text())["pages"]
    ok = 0
    for p in pages:
        f = next((root / name / f"{p['id']}{e}" for e in (".png", ".jpg") if (root / name / f"{p['id']}{e}").exists()), None)
        if f is not None and hashlib.sha256(f.read_bytes()).hexdigest() == (p.get("image_sha256") or p["png_sha256"]):
            ok += 1
        else:
            bad += 1
            print(f"MISMATCH {name}/{p['id']}")
    print(f"{name}: {ok}/{len(pages)} pages match")
sys.exit(1 if bad else 0)
