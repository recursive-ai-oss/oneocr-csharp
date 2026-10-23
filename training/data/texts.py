"""Natural-language text from Project Gutenberg, for the line sampler.

  uv run python -m data.texts build   # choose books from the live catalog, download, write texts.yaml
  uv run python -m data.texts fetch   # download every book listed in texts.yaml into $ONEOCR_TEXT_ROOT, verifying SHA-256
  uv run python -m data.texts verify

Terms: https://www.gutenberg.org/policy/permission.html -- Project Gutenberg eBooks are public domain in the US; only the
body between the START and END markers is used (the licence header and footer are not). The manifest records, per book:
id, title, author, language, URL, SHA-256 of the downloaded file, and whether it belongs to the training books or to the
held-out books that only validation and evaluation lines are drawn from.

Excluded on purpose: every work by Lewis Carroll and every edition of Pride and Prejudice (the page evaluation set is
set in text from those books), and reference works (dictionaries, grammars) whose text is not running prose.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

import yaml

from data import charset as cs

HERE = Path(__file__).parent
MANIFEST = HERE / "texts.yaml"
CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
BOOK_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
SEED = 20261023

# books per language; Greek is over-selected because most Greek books are polytonic and fail the charset check
QUOTA = {"en": 36, "fr": 20, "de": 20, "es": 14, "it": 14, "pt": 12, "nl": 10, "pl": 10, "sv": 8, "da": 8, "fi": 8,
         "hu": 8, "cs": 8, "ro": 5, "ca": 5, "el": 30, "ru": 5, "bg": 6, "sr": 4}  # fmt: skip
EXCLUDE_TITLE = re.compile(
    r"alice|pride and prejudice|looking.glass|dictionary|grammar|lexicon|vocabulary|glossary|catalog|index|bibliograph|concordance",
    re.I,
)
EXCLUDE_AUTHOR = re.compile(r"carroll, lewis|dodgson", re.I)
MIN_USABLE_CHARS = 150_000
# Gutenberg has only a handful of short Cyrillic books, so they are accepted at a lower size
MIN_USABLE_CYRILLIC = 30_000
CYRILLIC = {"ru", "bg", "sr"}
MIN_COVERAGE = 0.93  # share of letters that are charset classes


def text_root() -> Path:
    return Path(os.environ.get("ONEOCR_TEXT_ROOT", "texts-cache"))


def http(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "oneocr-texts"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def body_of(raw: bytes) -> str:
    """The book text between the START and END markers, NFC-normalised."""
    text = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    m = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    e = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text)
    if not m or not e or e.start() < m.end():
        return ""
    return unicodedata.normalize("NFC", text[m.end() : e.start()])


_CLEAN = [
    (re.compile(r"\[(?:Illustration|Footnote|Note)[^\]]*\]", re.I), " "),
    (re.compile(r"\[[0-9]+\]"), " "),
    (re.compile(r"[_*]"), ""),  # italics / bold markers in the plain-text editions
    (re.compile(r"\t"), " "),
]


def paragraphs(text: str) -> list[str]:
    """Running prose: blank-line separated paragraphs with hard line breaks joined; page furniture dropped."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        for pat, rep in _CLEAN:
            para = pat.sub(rep, para)
        para = " ".join(para.split())
        if len(para) >= 40:
            out.append(para)
    return out


def usable(text: str, allowed: set[str]) -> tuple[int, float]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0, 0.0
    ok = sum(c in allowed for c in letters)
    return ok, ok / len(letters)


def build() -> None:
    charset = set(cs.load(HERE / "charset-v1.txt")[1:])
    rows = list(csv.DictReader(io.StringIO(http(CATALOG).decode("utf-8"))))
    rng = random.Random(SEED)
    root = text_root()
    root.mkdir(parents=True, exist_ok=True)
    books = []
    for lang, quota in QUOTA.items():
        cands = [
            r for r in rows
            if r["Type"] == "Text" and r["Language"] == lang and not EXCLUDE_TITLE.search(r["Title"])
            and not EXCLUDE_AUTHOR.search(r["Authors"]) and r["Text#"] not in ("11", "1342")
        ]  # fmt: skip
        rng.shuffle(cands)
        got = []
        for r in cands:
            if len(got) >= quota:
                break
            bid = r["Text#"]
            cached = root / f"pg{bid}.txt"
            try:
                raw = cached.read_bytes() if cached.exists() else http(BOOK_URL.format(id=bid))
            except Exception as ex:
                print(f"skip {bid}: {ex}", file=sys.stderr)
                continue
            n, frac = usable(" ".join(paragraphs(body_of(raw))), charset)
            if n < (MIN_USABLE_CYRILLIC if lang in CYRILLIC else MIN_USABLE_CHARS) or frac < MIN_COVERAGE:
                continue
            (root / f"pg{bid}.txt").write_bytes(raw)
            got.append({
                "id": int(bid), "title": r["Title"].replace("\n", " ")[:120], "author": r["Authors"][:80], "language": lang,
                "url": BOOK_URL.format(id=bid), "sha256": hashlib.sha256(raw).hexdigest(), "letters_in_charset": n,
                "charset_coverage": round(frac, 4),
            })  # fmt: skip
        # hold out about one book in eight (at least one when the language has three or more)
        k = max(1 if len(got) >= 3 else 0, round(len(got) / 8))
        for i, b in enumerate(got):
            b["split"] = "heldout" if i < k else "train"
        books += got
        print(f"{lang}: {len(got)}/{quota} books ({k} held out)", file=sys.stderr)
    head = (
        "# Generated by `python -m data.texts build`. Project Gutenberg eBooks (public domain in the US),\n"
        "# https://www.gutenberg.org/policy/permission.html ; only the text between the START and END markers is used.\n"
    )
    MANIFEST.write_text(head + yaml.safe_dump({"books": books}, sort_keys=False, allow_unicode=True, width=200), encoding="utf-8")
    print(f"{len(books)} books")


def fetch(verify_only: bool) -> None:
    root = text_root()
    root.mkdir(parents=True, exist_ok=True)
    books = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["books"]
    bad = 0
    for b in books:
        p = root / f"pg{b['id']}.txt"
        if not p.exists():
            if verify_only:
                print(f"MISSING {b['id']}")
                bad += 1
                continue
            p.write_bytes(http(b["url"]))
        if hashlib.sha256(p.read_bytes()).hexdigest() != b["sha256"]:
            print(f"MISMATCH {b['id']}")
            bad += 1
    print(f"{len(books)} books, {bad} problems")
    sys.exit(1 if bad else 0)


def load_books(split: str) -> dict[str, list[tuple[int, list[list[str]]]]]:
    """language -> [(book id, paragraphs as word lists)] for the given split; only paragraphs made of charset classes
    are kept, so windows cut from them never contain an unknown character."""
    charset = set(cs.load(HERE / "charset-v1.txt")[1:])
    out: dict[str, list] = {}
    for b in yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["books"]:
        if b["split"] != split:
            continue
        raw = (text_root() / f"pg{b['id']}.txt").read_bytes()
        paras = [p.split(" ") for p in paragraphs(body_of(raw)) if set(p) <= charset]
        if paras:
            out.setdefault(b["language"], []).append((b["id"], paras))
    return out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"build": build, "fetch": lambda: fetch(False), "verify": lambda: fetch(True)}[cmd]()


if __name__ == "__main__":
    main()
