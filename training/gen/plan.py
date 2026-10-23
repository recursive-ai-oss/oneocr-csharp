"""Plans synthetic text lines: what to say, in which font, how to crop and degrade it.

The plan is JSONL (one line spec per crop) written from a seed and a YAML config. The C# `LineGen` project renders it with
OneImage's text drawing, so crops are made by the same code the library uses at inference time. Splitting plan from render
keeps the corpus sampler (charset aware, font-coverage aware) in Python and the rendering exact.

Text sources, all recorded per line in `src`:
  book:<id>      a window of consecutive words from a Project Gutenberg book (see data/texts.py)
  synthetic:*    numbers, dates, amounts, codes, pseudo-words and uniform draws over every class, made here
  Turkish and Vietnamese have no Gutenberg books, so a share of Latin book windows gets their letters swapped in
  (`book:<id>+tr`, `+vi`); the rendered string is the label, so the truth stays exact.

Needs ONEOCR_COVERAGE: the font coverage file written by `LineGen --coverage`.

Every label is the string the generator itself drew (exact truth). No OCR output or model produces a label.
Train and eval are disjoint by construction: eval and validation windows come from books the training set never reads,
and eval uses font families and degradation ranges that training never uses.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import yaml

from data import charset as cs
from data import texts as textlib

# Rough English letter frequencies (a-z) for pseudo-words.
_EN_FREQ = np.array(
    [8.2, 1.5, 2.8, 4.3, 12.7, 2.2, 2.0, 6.1, 7.0, 0.15, 0.8, 4.0, 2.4,
     6.7, 7.5, 1.9, 0.1, 6.0, 6.3, 9.1, 2.8, 1.0, 2.4, 0.15, 2.0, 0.07]
)  # fmt: skip
_VOWELS = {"latin": set("aeiou"), "greek": set("αεηιουωάέήίόύώ"), "cyrillic": set("аеёиоуыэюяі")}
_PUNCT_END = ",.;:!?"
_CURRENCY = "€£$¥₽₴₺"
_WRAP = {"(": ")", "“": "”", "„": "“", "«": "»", "‘": "’", "'": "'", '"': '"'}
# language weights for book windows
LANG_WEIGHT = {"en": 30, "fr": 10, "de": 9, "es": 7, "it": 6, "pt": 5, "nl": 4, "pl": 3, "sv": 2, "da": 2, "fi": 3, "hu": 3,
               "cs": 2, "ro": 1, "ca": 1, "el": 8, "ru": 6, "bg": 4, "sr": 3}  # fmt: skip
LATIN_LANGS = {"en", "fr", "de", "es", "it", "pt", "nl", "pl", "sv", "da", "fi", "hu", "cs", "ro", "ca"}
# letter -> (Turkish replacement, probability)
TURKISH = {"i": ("ı", 0.35), "s": ("ş", 0.3), "g": ("ğ", 0.3), "c": ("ç", 0.3), "o": ("ö", 0.25), "u": ("ü", 0.25), "I": ("İ", 0.4)}


class Coverage:
    """Charset entries each font file can draw, as reported by the renderer itself (`LineGen --coverage`), so a line never
    contains a glyph the renderer would leave empty or replace by a placeholder box."""

    def __init__(self, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._cache = {f: frozenset(s) for f, s in raw.items()}

    def of(self, file: str) -> frozenset[str]:
        return self._cache[file]


class Sampler:
    """Synthetic text: numbers, codes, pseudo-words, rare-class draws. Pseudo-words alternate vowels and consonants; the aim is
    to cover every class with plausible shapes and spacing, not to imitate a language."""

    def __init__(self, charset: list[str], allowed: frozenset[str], rng: np.random.Generator):
        self.rng = rng
        self.allowed = {c for c in charset[1:] if c in allowed}
        self.scripts: dict[str, dict[str, list[str]]] = {}
        for name in ("latin", "greek", "cyrillic"):
            chars = [c for c in charset[1:] if c in self.allowed and cs.script_of(c) == name]
            self.scripts[name] = {"lower": [c for c in chars if c.islower()], "upper": [c for c in chars if c.isupper()]}
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

    def pseudo_line(self) -> str:
        rng = self.rng
        scripts = [s for s in ("latin", "greek", "cyrillic") if self.usable(s)]
        script = str(rng.choice(scripts))
        toks: list[str] = []
        for _ in range(int(rng.integers(1, 10))):
            t = self.number() if rng.random() < 0.08 else self.word(script)
            r = rng.random()
            if r < 0.12:
                t += str(rng.choice(list(_PUNCT_END)))
            elif r < 0.17:
                o = str(rng.choice(list(_WRAP)))
                t = o + t + _WRAP[o]
            toks.append(t)
        return " ".join(toks)

    def numbers_line(self) -> str:
        return " ".join(self.number() for _ in range(int(self.rng.integers(1, 5))))

    def codes_line(self) -> str:
        return " ".join(self.code() for _ in range(int(self.rng.integers(1, 3))))


_VI_KEEP = {0x1A0, 0x1A1, 0x1AF, 0x1B0, 0x102, 0x103, 0xC2, 0xE2, 0xCA, 0xEA, 0xD4, 0xF4}


class Books:
    """Windows of consecutive words from the books of one split."""

    def __init__(self, split: str, charset: list[str]):
        self.by_lang = textlib.load_books(split)
        self.langs = [(lang, LANG_WEIGHT[lang]) for lang in self.by_lang if lang in LANG_WEIGHT]
        self.p = np.array([w for _, w in self.langs], dtype=float)
        self.p /= self.p.sum()
        # Vietnamese vowel variants available in the charset, keyed by their base letter (case kept)
        self.vi: dict[str, list[str]] = {}
        for c in charset[1:]:
            d = unicodedata.normalize("NFD", c)
            if d[0] in "aeiouyAEIOUY" and (0x1EA0 <= ord(c) <= 0x1EF9 or ord(c) in _VI_KEEP):
                self.vi.setdefault(d[0], []).append(c)

    def window(self, rng: np.random.Generator) -> tuple[str, str]:
        lang = self.langs[int(rng.choice(len(self.langs), p=self.p))][0]
        books = self.by_lang[lang]
        bid, paras = books[int(rng.integers(0, len(books)))]
        words = paras[int(rng.integers(0, len(paras)))]
        # line length: a quarter short fragments, the rest page-like lines (a justified line of narrow type holds 130 characters)
        target = int(rng.integers(1, 26)) if rng.random() < 0.25 else int(rng.integers(25, 136))
        start = int(rng.integers(0, len(words)))
        out: list[str] = []
        n = -1
        for w in words[start:]:
            if n + 1 + len(w) > target and out:
                break
            out.append(w)
            n += 1 + len(w)
        text = " ".join(out)
        tag = f"book:{bid}"
        r = rng.random()
        if r < 0.04:
            text = text.upper()
        elif lang in LATIN_LANGS and r < 0.065:
            text = "".join(TURKISH[c][0] if c in TURKISH and rng.random() < TURKISH[c][1] else c for c in text)
            tag += "+tr"
        elif lang in LATIN_LANGS and r < 0.10:
            text = "".join(
                str(rng.choice(self.vi[c])) if c in self.vi and rng.random() < 0.45 else ("đ" if c == "d" and rng.random() < 0.15 else c)
                for c in text
            )
            tag += "+vi"
        return text, tag


def _rng_value(rng: np.random.Generator, ranges) -> float:
    if isinstance(ranges[0], (int, float)):
        ranges = [ranges]
    lo, hi = ranges[int(rng.integers(0, len(ranges)))]
    return float(lo + (hi - lo) * rng.random())


# per-process state for the shard workers
_S: dict = {}


def _init(cfg, base):
    charset = cs.load(base / cfg["charset"])
    _S["cfg"] = cfg
    _S["charset"] = charset
    _S["fonts"] = yaml.safe_load((base / cfg["fonts"]).read_text(encoding="utf-8"))["fonts"]
    _S["cov"] = Coverage(Path(os.environ["ONEOCR_COVERAGE"]))
    _S["books"] = {}


def _books(split: str) -> Books:
    if split not in _S["books"]:
        _S["books"][split] = Books(split, _S["charset"])
    return _S["books"][split]


def _pick_font(rng, fams: dict[str, list[str]], need: set[str]) -> str | None:
    """Uniform over families, then over a family's files; only files that can draw every needed character."""
    cov = _S["cov"]
    names = list(fams)
    for _ in range(3):
        f = fams[names[int(rng.integers(0, len(names)))]]
        file = f[int(rng.integers(0, len(f)))]
        if need <= cov.of(file):
            return file
    ok = [n for n in names if any(need <= cov.of(x) for x in fams[n])]
    if not ok:
        return None
    files = [x for x in fams[ok[int(rng.integers(0, len(ok)))]] if need <= cov.of(x)]
    return files[int(rng.integers(0, len(files)))]


def plan_shard(job) -> list[dict]:
    name, spec, set_index, shard, count, banned = job
    cfg, charset = _S["cfg"], _S["charset"]
    rng = np.random.default_rng([cfg["seed"], set_index, shard])
    deg = cfg["degrade"][spec["degrade"]]
    mix = cfg["mix"]
    kinds = list(mix)
    kp = np.array([mix[k] for k in kinds], dtype=float)
    kp /= kp.sum()
    fams: dict[str, list[str]] = {}
    for f in _S["fonts"]:
        if f["split"] == spec["fonts"]:
            fams.setdefault(f["family"], []).append(f["file"])
    books = _books(spec["books"])
    cov = _S["cov"]
    out, seen = [], set()
    while len(out) < count:
        kind = kinds[int(rng.choice(len(kinds), p=kp))]
        if kind == "book":
            text, src = books.window(rng)
            file = _pick_font(rng, fams, set(text) - {" "})
        else:
            file = _pick_font(rng, fams, set())
            if file is None:
                continue
            s = Sampler(charset, cov.of(file), rng)
            text = {"numbers": s.numbers_line, "codes": s.codes_line, "rare": s.rare, "pseudo": s.pseudo_line}[kind]()
            src = f"synthetic:{kind}"
        text = unicodedata.normalize("NFC", " ".join(text.split()))
        # every character must be a charset class that the chosen font can draw (generated strings use literal symbols like "№")
        if not text or file is None or text in seen or text in banned or not set(text) - {" "} <= cov.of(file):
            continue
        seen.add(text)
        delta = int(rng.integers(deg["contrast"][0], deg["contrast"][1] + 1))
        bg = int(rng.integers(max(140, delta), 256))
        fg = max(0, bg - delta)
        if rng.random() < 0.04:
            fg, bg = bg, fg
        broad = rng.random() < 0.15  # a minority of crops with margins far from the finder's convention
        vm = [rng.uniform(0.0, 0.45) if broad else rng.uniform(0.18, 0.32) for _ in range(2)]
        hm = [rng.uniform(0.05, 1.0) if broad else rng.uniform(0.35, 0.65) for _ in range(2)]
        out.append({
            "id": f"{name}-{shard:03d}{len(out):05d}",
            "set": name,
            "text": text,
            "src": src,
            "font": file,
            "em": round(_rng_value(rng, deg["em"]), 2),
            "fg": fg,
            "bg": bg,
            "ref": round(float(rng.uniform(0.92, 1.12)), 3),
            "vm": [round(float(v), 3) for v in vm],
            "hm": [round(float(v), 3) for v in hm],
            "blur": round(_rng_value(rng, deg["blur"]), 3),
            "noise": round(_rng_value(rng, deg["noise"]), 3),
            "jpeg": int(_rng_value(rng, deg["jpeg_q"])) if rng.random() < deg["jpeg_prob"] else 0,
            "seed": int(rng.integers(0, 2**31 - 1)),
        })  # fmt: skip
    return out


SHARD = 5000


def main() -> None:
    cfg_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base = Path(__file__).resolve().parents[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": cfg,
        "charset_sha256": hashlib.sha256((base / cfg["charset"]).read_bytes()).hexdigest(),
        "fonts_sha256": hashlib.sha256((base / cfg["fonts"]).read_bytes()).hexdigest(),
        "texts_sha256": hashlib.sha256((base / "data/texts.yaml").read_bytes()).hexdigest(),
        "sets": {},
    }
    with mp.get_context("fork").Pool(min(24, mp.cpu_count()), initializer=_init, initargs=(cfg, base)) as pool:
        # validation and eval sets are planned first; their texts of 20+ characters are then kept out of the training set
        # (the same sentence can occur in two Project Gutenberg books, e.g. two editions of one work)
        banned: frozenset[str] = frozenset()
        order = sorted(enumerate(cfg["sets"].items()), key=lambda t: t[1][0] == "train")
        for i, (name, spec) in order:
            shards = [
                (name, spec, i, s, min(SHARD, spec["n"] - s * SHARD), banned if name == "train" else frozenset())
                for s in range((spec["n"] + SHARD - 1) // SHARD)
            ]
            recs = [r for part in pool.map(plan_shard, shards) for r in part]
            if name != "train":
                banned |= {r["text"] for r in recs if len(r["text"]) >= 20}
            body = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in recs)
            (out_dir / f"{name}.plan.jsonl").write_text(body, encoding="utf-8")
            srcs: dict[str, int] = {}
            for r in recs:
                key = "book" if r["src"].startswith("book") else r["src"]
                srcs[key] = srcs.get(key, 0) + 1
            manifest["sets"][name] = {
                "n": len(recs),
                "plan_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "fonts": sorted({r["font"] for r in recs}),
                "books": sorted({int(r["src"].split(":")[1].split("+")[0]) for r in recs if r["src"].startswith("book")}),
                "sources": srcs,
                "label_provenance": "exact-truth synthetic: the rendered string; text from Project Gutenberg books and generated strings",
            }
            print(f"{name}: {len(recs)} lines, {len(manifest['sets'][name]['fonts'])} font files, {srcs}", flush=True)
    (out_dir / "plan.manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
