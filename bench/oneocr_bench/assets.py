"""Fetch the held-out open fonts and the public-domain text, verify them against pinned hashes.

Fonts come from the Google Fonts repository at a pinned commit; every family is
licensed under the SIL Open Font License 1.1 and ships its own OFL.txt, which is
downloaded next to the font files and hashed into manifests/assets.json.
Text comes from Project Gutenberg plain-text files (public domain in the US).

    python -m oneocr_bench fetch            # verify against manifests/assets.json
    python -m oneocr_bench fetch --update   # (re)write the manifest after a deliberate change
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from .paths import MANIFESTS, data_dir

FONTS_COMMIT = "e7db6ca603fd4862010383ff17ffd6cc4ffac1c0"
FONTS_BASE = f"https://raw.githubusercontent.com/google/fonts/{FONTS_COMMIT}/ofl"

# family -> (directory, [(file, css weight range, style)]). Regular and bold are all the pages use.
FAMILIES: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "Gelasio": ("gelasio", [("Gelasio[wght].ttf", "400 700", "normal")]),
    "Carlito": ("carlito", [("Carlito-Regular.ttf", "400", "normal"), ("Carlito-Bold.ttf", "700", "normal")]),
    "Caladea": ("caladea", [("Caladea-Regular.ttf", "400", "normal"), ("Caladea-Bold.ttf", "700", "normal")]),
    "Source Serif 4": ("sourceserif4", [("SourceSerif4[opsz,wght].ttf", "200 900", "normal")]),
    "Source Sans 3": ("sourcesans3", [("SourceSans3[wght].ttf", "200 900", "normal")]),
    "IBM Plex Serif": (
        "ibmplexserif",
        [("IBMPlexSerif-Regular.ttf", "400", "normal"), ("IBMPlexSerif-Bold.ttf", "700", "normal")],
    ),
    "IBM Plex Sans": ("ibmplexsans", [("IBMPlexSans[wdth,wght].ttf", "100 700", "normal")]),
}

# Project Gutenberg ebook number -> title. Plain text, public domain in the United States.
WORKS = {1342: "Pride and Prejudice (Jane Austen, 1813)", 11: "Alice's Adventures in Wonderland (Lewis Carroll, 1865)"}
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"
GUTENBERG_TERMS = "https://www.gutenberg.org/policy/permission.html"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "oneocr-bench"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def gutenberg_body(raw: bytes) -> str:
    """Text between the START/END markers, which drops the licence boilerplate."""
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    s = re.search(r"^\*\*\* START OF .*?\*\*\*\s*$", text, re.M)
    e = re.search(r"^\*\*\* END OF .*?\*\*\*\s*$", text, re.M)
    if not s or not e:
        raise ValueError("Gutenberg start/end markers not found")
    return text[s.end() : e.start()].strip("\n") + "\n"


def fetch(update: bool = False) -> None:
    root = data_dir()
    manifest_path = MANIFESTS / "assets.json"
    pinned = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    out: dict = {"fonts_commit": FONTS_COMMIT, "fonts": {}, "works": {}}

    for fam, (d, files) in FAMILIES.items():
        entry = {"licence": "SIL Open Font License 1.1", "files": {}, "licence_file": None}
        for name in [*(f for f, _, _ in files), "OFL.txt"]:
            url = f"{FONTS_BASE}/{d}/{urllib.parse.quote(name)}"
            dest = root / "fonts" / d / name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(_get(url))
            h = sha256(dest.read_bytes())
            if name == "OFL.txt":
                head = dest.read_text(encoding="utf-8", errors="replace")
                if "SIL OPEN FONT LICENSE Version 1.1" not in head:
                    raise ValueError(f"{fam}: OFL.txt is not OFL 1.1")
                entry["licence_file"] = {"url": url, "sha256": h, "copyright": head.splitlines()[0].strip()}
            else:
                entry["files"][name] = {"url": url, "sha256": h}
        out["fonts"][fam] = entry

    for n, title in WORKS.items():
        dest = root / "text" / f"pg{n}.txt"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_get(GUTENBERG_URL.format(n=n)))
        body = gutenberg_body(dest.read_bytes())
        out["works"][str(n)] = {
            "title": title,
            "url": GUTENBERG_URL.format(n=n),
            "terms": GUTENBERG_TERMS,
            "body_sha256": sha256(body.encode()),
        }

    if update or not pinned:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {manifest_path}")
        return
    if pinned != out:
        raise SystemExit("fetched assets differ from manifests/assets.json (use --update after review)")
    print("assets match manifests/assets.json")


def font_path(family: str, name: str) -> Path:
    return data_dir() / "fonts" / FAMILIES[family][0] / name
