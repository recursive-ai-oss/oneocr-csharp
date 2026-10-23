"""Plans synthetic text lines: what to say, in which font, how to draw and degrade it.

The plan is a JSONL file (one line spec per crop) written from a seed and a YAML config. The C#
`LineGen` project renders it with OneImage's text drawing, so crops are made by the same code the
library uses at inference time. Splitting plan from render keeps the corpus sampler (charset
aware, font-coverage aware) in Python and the rendering exact.

Provenance: every label is the string the generator itself drew (exact truth). No external text,
OCR output or model produces a label.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import yaml
from fontTools.ttLib import TTFont

from data import charset as cs
from data import fonts as fontlib

# Rough English letter frequencies (a-z). Only shapes the letter mix; accented letters are mixed in separately.
_EN_FREQ = np.array(
    [8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.15, 0.8, 4.0, 2.4,
     6.7, 7.5, 1.9, 0.1, 6.0, 6.3, 9.1, 2.8, 1.0, 2.4, 0.15, 2.0, 0.07]
)  # fmt: skip
_VOWELS = {
    "latin": set("aeiou"),
    "greek": set("αεηιουωάέήίόύώ"),
    "cyrillic": set("аеёиоуыэюяі"),
}
_PUNCT_END = ",.;:!?"
_CURRENCY = "€£$¥₽₴₺"
_SCRIPT_P = {"latin": 0.60, "greek": 0.17, "cyrillic": 0.23}
_WRAP = {"(": ")", "“": "”", "„": "“", "«": "»", "‘": "’", "'": "'", '"': '"'}


class Coverage:
    """Code points each font file can draw (cmap), so no line contains a glyph the font lacks."""

    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, frozenset[str]] = {}

    def of(self, file: str) -> frozenset[str]:
        if file not in self._cache:
            with TTFont(self.root / file, lazy=True) as f:
                cmap = f.getBestCmap()
            self._cache[file] = frozenset(chr(c) for c in cmap)
        return self._cache[file]


class Sampler:
    """Charset-aware text sampler. Pseudo-words alternate vowels and consonants; the aim is to cover
    every class with plausible shapes and spacing, not to imitate a language."""

    def __init__(self, charset: list[str], allowed: frozenset[str], rng: np.random.Generator):
        self.rng = rng
        self.allowed = {c for c in charset[1:] if c in allowed}
        self.scripts: dict[str, dict[str, list[str]]] = {}
        for name in ("latin", "greek", "cyrillic"):
            chars = [c for c in charset[1:] if c in self.allowed and cs.script_of(c) == name]
            self.scripts[name] = {
                "lower": [c for c in chars if c.islower()],
                "upper": [c for c in chars if c.isupper()],
            }
        self.digits = [c for c in "0123456789" if c in self.allowed]
        self.every = sorted(self.allowed)

    def usable(self, script: str) -> bool:
        s = self.scripts[script]
        return len(s["lower"]) >= 15 and len(s["upper"]) >= 15

    def _letter(self, script: str, vowel: bool) -> str:
        pool = self.scripts[script]["lower"]
        rng = self.rng
        want = _VOWELS[script]
        if script == "latin" and rng.random() < 0.8:
            letters = [c for c in "abcdefghijklmnopqrstuvwxyz" if c in self.allowed and (c in want) == vowel]
            w = np.array([_EN_FREQ[ord(c) - 97] for c in letters])
            return str(rng.choice(letters, p=w / w.sum()))
        sub = [c for c in pool if (c in want) == vowel] or pool
        return str(rng.choice(sub))

    def word(self, script: str) -> str:
        rng = self.rng
        n = int(min(14, 1 + rng.geometric(0.2)))
        vowel = bool(rng.random() < 0.4)
        out = []
        for _ in range(n):
            out.append(self._letter(script, vowel))
            vowel = not vowel if rng.random() < 0.85 else vowel
        w = "".join(out)
        r = rng.random()
        if r < 0.15:
            w = w[0].upper() + w[1:]
        elif r < 0.23:
            w = w.upper()
        return w

    def _digits(self, n: int) -> str:
        return "".join(str(self.rng.choice(self.digits)) for _ in range(n))

    def number(self) -> str:
        rng = self.rng
        d = self._digits
        k = int(rng.integers(0, 7))
        if k == 0:
            return d(int(rng.integers(1, 9)))
        if k == 1:
            return f"{d(int(rng.integers(1, 4)))}{rng.choice([',', '.', ' '])}{d(3)}{rng.choice([',', '.'])}{d(2)}"
        if k == 2:
            sep = rng.choice(["/", "-", "."])
            return f"{d(2)}{sep}{d(2)}{sep}{d(int(rng.choice([2, 4])))}"
        if k == 3:
            cur = str(rng.choice(list(_CURRENCY)))
            amt = f"{d(int(rng.integers(1, 5)))}{rng.choice(['.', ','])}{d(2)}"
            return f"{cur}{amt}" if rng.random() < 0.5 else f"{amt} {cur}"
        if k == 4:
            return f"{d(int(rng.integers(1, 3)))}{rng.choice(['%', '‰', '°'])}"
        if k == 5:
            return f"{rng.choice(['+', ''])}{d(2)} {d(3)} {d(3)} {d(int(rng.integers(2, 5)))}"
        return f"№ {d(int(rng.integers(1, 7)))}"

    def code(self) -> str:
        rng = self.rng
        up = self.scripts["latin"]["upper"]
        ascii_up = [c for c in up if c.isascii()] or up

        def seg() -> str:
            return "".join(str(rng.choice(ascii_up if rng.random() < 0.5 else self.digits)) for _ in range(int(rng.integers(2, 6))))

        return str(rng.choice(["-", "/", ".", "_", ""])).join(seg() for _ in range(int(rng.integers(1, 4))))

    def rare(self) -> str:
        """Uniform draw over every drawable class, so rare letters and symbols all get examples."""
        n = int(self.rng.integers(1, 13))
        return "".join(str(self.rng.choice(self.every)) for _ in range(n))

    def line(self) -> str:
        rng = self.rng
        kind = rng.random()
        scripts = [s for s in _SCRIPT_P if self.usable(s)]
        if kind < 0.70 and scripts:
            p = np.array([_SCRIPT_P[s] for s in scripts])
            script = str(rng.choice(scripts, p=p / p.sum()))
            toks: list[str] = []
            for _ in range(int(rng.integers(1, 9))):
                t = self.number() if rng.random() < 0.08 else self.word(script)
                r = rng.random()
                if r < 0.12:
                    t += str(rng.choice(list(_PUNCT_END)))
                elif r < 0.17:
                    o = str(rng.choice(list(_WRAP)))
                    t = o + t + _WRAP[o]
                toks.append(t)
                if sum(map(len, toks)) + len(toks) > 44:
                    break
            return " ".join(toks)
        if kind < 0.82:
            return " ".join(self.number() for _ in range(int(rng.integers(1, 4))))
        if kind < 0.92:
            return " ".join(self.code() for _ in range(int(rng.integers(1, 3))))
        return self.rare()

    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = "".join(c for c in text if c in self.allowed or c == " ")
        return " ".join(text.split())


def _rng_value(rng: np.random.Generator, ranges) -> float:
    if isinstance(ranges[0], (int, float)):
        ranges = [ranges]
    lo, hi = ranges[int(rng.integers(0, len(ranges)))]
    return float(lo + (hi - lo) * rng.random())


def plan_set(name: str, spec: dict, cfg: dict, charset: list[str], fonts: list[dict], cov: Coverage,
             set_index: int, banned: set[str]) -> list[dict]:  # fmt: skip
    rng = np.random.default_rng([cfg["seed"], set_index])
    deg = cfg["degrade"][spec["degrade"]]
    pool = [f for f in fonts if f["split"] == spec["fonts"]]
    texts_seen: set[str] = set()
    out = []
    while len(out) < spec["n"]:
        f = pool[int(rng.integers(0, len(pool)))]
        s = Sampler(charset, cov.of(f["file"]), rng)
        text = s.clean(s.line())
        if not text or text in banned or text in texts_seen:
            continue
        texts_seen.add(text)
        delta = int(rng.integers(deg["contrast"][0], deg["contrast"][1] + 1))
        bg = int(rng.integers(max(140, delta), 256))
        fg = max(0, bg - delta)
        if rng.random() < 0.04:
            fg, bg = bg, fg
        out.append({
            "id": f"{name}-{len(out):06d}",
            "set": name,
            "text": text,
            "font": f["file"],
            "em": round(_rng_value(rng, deg["em"]), 2),
            "fg": fg,
            "bg": bg,
            "pad": [round(float(rng.uniform(0.1, 0.9)), 3), round(float(rng.uniform(0.1, 0.9)), 3),
                    round(float(rng.uniform(0.05, 0.35)), 3), round(float(rng.uniform(0.05, 0.35)), 3)],
            "blur": round(_rng_value(rng, deg["blur"]), 3),
            "noise": round(_rng_value(rng, deg["noise"]), 3),
            "jpeg": int(_rng_value(rng, deg["jpeg_q"])) if rng.random() < deg["jpeg_prob"] else 0,
            "seed": int(rng.integers(0, 2**31 - 1)),
        })  # fmt: skip
    return out


def main() -> None:
    cfg_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base = Path(__file__).resolve().parents[1]
    charset = cs.load(base / cfg["charset"])
    fonts = yaml.safe_load((base / cfg["fonts"]).read_text(encoding="utf-8"))["fonts"]
    cov = Coverage(fontlib.font_root())
    out_dir.mkdir(parents=True, exist_ok=True)
    banned: set[str] = set()
    manifest = {
        "config": cfg,
        "charset_sha256": hashlib.sha256((base / cfg["charset"]).read_bytes()).hexdigest(),
        "fonts_sha256": hashlib.sha256((base / cfg["fonts"]).read_bytes()).hexdigest(),
        "sets": {},
    }
    # train is planned first so that no val/eval text duplicates a training text
    order = sorted(cfg["sets"], key=lambda n: (n != "train", n))
    for i, name in enumerate(order):
        recs = plan_set(name, cfg["sets"][name], cfg, charset, fonts, cov, i, banned)
        if name == "train":
            banned |= {r["text"] for r in recs}
        body = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in recs)
        (out_dir / f"{name}.plan.jsonl").write_text(body, encoding="utf-8")
        fonts_used = sorted({r["font"] for r in recs})
        manifest["sets"][name] = {
            "n": len(recs),
            "plan_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "fonts": fonts_used,
            "label_provenance": "exact-truth synthetic (generator text, seed above)",
        }
        print(f"{name}: {len(recs)} lines, {len(fonts_used)} font files")
    (out_dir / "plan.manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
