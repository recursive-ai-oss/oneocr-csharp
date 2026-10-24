"""Font pool for generated pages: the licensed families of data/fonts.yaml, grouped by role, with the glyph coverage of each.

Only families of the `train` split are used, and only for the `train` page split; the `heldout` families are reserved for the
validation page split. The families the page evaluation sets are set in (data/font_families.yaml `never`) are not in
fonts.yaml at all, so they cannot be drawn here. `assert_disjoint` checks this against the pinned lists.

Files are described once (weight, italic, variable axis, cmap) by fontTools and cached as JSON next to the output.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from fontTools.ttLib import TTFont

DATA = Path(__file__).resolve().parent.parent / "data"
SCRIPT_SAMPLES = {
    "latin": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()-/%€",
    "greek": "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω",
    "cyrillic": "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюя",
}
# fixed fallbacks for the roles; families are placed in a role by the section comments of font_families.yaml
ROLES = ("sans", "serif", "mono", "display")


def font_root() -> Path:
    return Path(os.environ["ONEOCR_FONT_ROOT"])


def family_roles() -> dict[str, str]:
    """family -> role (sans/serif/mono/display) from the section comments in font_families.yaml."""
    role, out, section = "sans", {}, None
    for line in (DATA / "font_families.yaml").read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("#"):
            low = s.lower()
            for r, key in (("sans", "# sans"), ("serif", "# serif"), ("mono", "# mono"), ("display", "# display")):
                if low.startswith(key):
                    role = r
        elif s.startswith("heldout:"):
            section = "heldout"
        elif s.startswith("train:"):
            section = "train"
        m = re.match(r"- (\w+)$", s)
        if m and section:
            out[m.group(1)] = role
    return out


@dataclass
class FontFile:
    path: str  # relative to font_root
    weight: str  # css font-weight value or range
    italic: bool
    cps: frozenset[int]


@dataclass
class Family:
    name: str  # directory name, also used as the css family
    role: str
    split: str
    files: list[FontFile]

    def covers(self, chars: set[str]) -> bool:
        reg = self.files[0].cps
        return all(ord(c) in reg or c.isspace() for c in chars)

    def css(self, root: Path) -> str:
        return "\n".join(
            f"@font-face{{font-family:'{self.name}';src:url('{(root / f.path).as_uri()}');font-weight:{f.weight};"
            f"font-style:{'italic' if f.italic else 'normal'};}}"
            for f in self.files
        )

    @property
    def has_bold(self) -> bool:
        return any(not f.italic and (" " in f.weight or int(f.weight) >= 600) for f in self.files)

    @property
    def has_italic(self) -> bool:
        return any(f.italic for f in self.files)


def _describe(root: Path, rel: str) -> dict:
    f = TTFont(root / rel, lazy=True)
    cps = sorted(f.getBestCmap().keys())
    wght = None
    if "fvar" in f:
        for ax in f["fvar"].axes:
            if ax.axisTag == "wght":
                wght = f"{int(ax.minValue)} {int(ax.maxValue)}"
    os2 = f["OS/2"]
    italic = bool(os2.fsSelection & 1) or bool(f["head"].macStyle & 2)
    return {"path": rel, "weight": wght or str(os2.usWeightClass), "italic": italic, "cps": cps}


def load_pool(cache: Path | None = None) -> dict[str, Family]:
    """All families of fonts.yaml with their files. The description is cached in `cache` (JSON) when given."""
    root = font_root()
    fonts = yaml.safe_load((DATA / "fonts.yaml").read_text(encoding="utf-8"))["fonts"]
    roles = family_roles()
    desc: dict[str, dict] = json.loads(cache.read_text()) if cache and cache.exists() else {}
    fams: dict[str, Family] = {}
    dirty = False
    for e in fonts:
        if e["file"] not in desc:
            desc[e["file"]] = _describe(root, e["file"])
            dirty = True
        d = desc[e["file"]]
        fam = fams.setdefault(e["family"], Family(e["family"], roles.get(e["family"], "sans"), e["split"], []))
        fam.files.append(FontFile(d["path"], d["weight"], d["italic"], frozenset(d["cps"])))
    if dirty and cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(desc))
    for fam in fams.values():
        # the upright file with the lowest weight first: it decides coverage
        fam.files.sort(key=lambda f: (f.italic, f.weight))
        # pick the regular-ish upright file for coverage checks
        up = [f for f in fam.files if not f.italic]
        if up:
            fam.files = up + [f for f in fam.files if f.italic]
    return fams


def pick(pool: dict[str, Family], rng: random.Random, split: str, roles: tuple[str, ...], chars: set[str]) -> Family:
    """A family of the split with a role in `roles` whose regular file covers every character of `chars`."""
    want = "train" if split == "train" else "heldout"
    cands = [f for f in pool.values() if f.split == want and f.role in roles and f.covers(chars)]
    if not cands and split != "train":  # the held-out split has few families; fall back is explicit and recorded by the caller
        return pick(pool, rng, "train", roles, chars)
    if not cands:
        raise LookupError(f"no {roles} family covers {sorted(chars)[:20]}")
    return rng.choice(sorted(cands, key=lambda f: f.name))


def assert_disjoint(pool: dict[str, Family]) -> None:
    never = set(yaml.safe_load((DATA / "font_families.yaml").read_text(encoding="utf-8"))["never"])
    bad = never & set(pool)
    if bad:
        raise AssertionError(f"evaluation page families in the training pool: {sorted(bad)}")
