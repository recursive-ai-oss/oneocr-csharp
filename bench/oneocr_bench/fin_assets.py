"""Fonts of the financial sets: the seven held-out families of `assets.py` plus three more that no training pool uses.

IBM Plex Mono, DM Mono and Barlow come from the Google Fonts repository at the same pinned commit as the other
families, each under the SIL Open Font License 1.1 with its own OFL.txt (downloaded next to the font files and hashed
into `manifests/fin-assets.json`). They are used unmodified, so the licence's Reserved Font Names do not apply.

    python -m oneocr_bench fin fetch            # verify against manifests/fin-assets.json
    python -m oneocr_bench fin fetch --update   # (re)write the manifest after a deliberate change
"""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

from .assets import FAMILIES, FONTS_BASE, FONTS_COMMIT, _get, sha256
from .paths import MANIFESTS, data_dir

EXTRA: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "IBM Plex Mono": ("ibmplexmono", [("IBMPlexMono-Regular.ttf", "400", "normal"), ("IBMPlexMono-Bold.ttf", "700", "normal")]),
    "DM Mono": ("dmmono", [("DMMono-Regular.ttf", "400", "normal"), ("DMMono-Medium.ttf", "500 900", "normal")]),
    "Barlow": ("barlow", [("Barlow-Regular.ttf", "400", "normal"), ("Barlow-Bold.ttf", "700", "normal")]),
}
ALL = {**FAMILIES, **EXTRA}
MONO = ("IBM Plex Mono", "DM Mono")


def font_file(family: str, name: str) -> Path:
    return data_dir() / "fonts" / ALL[family][0] / name


def font_css(family: str) -> str:
    return "\n".join(
        f"@font-face{{font-family:'{family}';src:url('{font_file(family, name).as_uri()}');font-weight:{weight};font-style:{style};}}"
        for name, weight, style in ALL[family][1]
    )


def wait_fonts(page, family: str) -> None:
    page.evaluate("f => Promise.all([document.fonts.load(`1em '${f}'`), document.fonts.load(`bold 1em '${f}'`)])", family)
    page.evaluate("document.fonts.ready")
    loaded = page.evaluate("[...document.fonts].filter(f=>f.status==='loaded').map(f=>f.family)")
    if family not in [x.strip("'\"") for x in loaded]:
        raise RuntimeError(f"font {family} did not load ({loaded})")


def fetch(update: bool = False) -> None:
    root = data_dir()
    path = MANIFESTS / "fin-assets.json"
    out: dict = {"fonts_commit": FONTS_COMMIT, "fonts": {}}
    for fam, (d, files) in EXTRA.items():
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
    if update or not path.exists():
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {path}")
        return
    if json.loads(path.read_text()) != out:
        raise SystemExit("fetched fonts differ from manifests/fin-assets.json (use --update after review)")
    print("fonts match manifests/fin-assets.json")
