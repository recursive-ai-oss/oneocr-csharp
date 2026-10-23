#!/usr/bin/env python3
"""Builds the character language model file (format: docs/spec/char-lm-v1.md) from Project Gutenberg text.

    python3 training/charlm/build_charlm.py --texts DIR --out charlm-v1.bin [--order 4] [--min-counts 1,3,4,6]

DIR holds pg<ID>.txt for the books listed in training/data/texts.yaml (the same files `python -m data.texts fetch` writes);
each file is checked against the SHA-256 in the manifest. Words are lowercased (simple case mapping, one code point to one) before they are counted. Only books whose split is `train` are used: the held-out books
the recognizer is validated on, and the two books the page evaluation sets are set in, never reach the model.

The model scores one word at a time. A word is a space-delimited token of a paragraph; it is dropped when it contains a digit,
a code symbol or a character outside the script, so the model never sees numbers or identifiers. Estimation is interpolated
Kneser-Ney (Kneser and Ney 1995; Chen and Goodman 1998, sec. 3: absolute discount D = n1 / (n1 + 2 n2) per order, lower orders
on continuation counts), written in back-off form (the conversion described for ARPA files: explicit entries keep their
interpolated probability, back-off weights renormalize what pruning removed). Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import struct
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

BOS, EOS = "\x00", "\x01"
STEP = 0.1
CODE_SYMBOLS = set("@#_/\\=%+*~^|<>[]{}")
SCRIPTS = {"latin": "LATIN", "greek": "GREEK", "cyrillic": "CYRILLIC"}
LANG_SCRIPT = {"el": "greek", "ru": "cyrillic", "bg": "cyrillic", "sr": "cyrillic"}  # every other language is Latin


def read_manifest(path: Path) -> list[dict]:
    """The books of texts.yaml (a flat list of `key: value` blocks), without a YAML library."""
    books: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip() or line.startswith("books:"):
            continue
        m = re.match(r"^(- |  )(\w+): ?(.*)$", line)
        if not m:
            continue
        if m.group(1) == "- ":
            books.append({})
        books[-1][m.group(2)] = m.group(3).strip()
    return books


def body_of(raw: bytes) -> str:
    text = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n")
    m = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    e = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text)
    if not m or not e or e.start() < m.end():
        return ""
    return unicodedata.normalize("NFC", text[m.end() : e.start()])


CLEAN = [
    (re.compile(r"\[(?:Illustration|Footnote|Note)[^\]]*\]", re.I), " "),
    (re.compile(r"\[[0-9]+\]"), " "),
    (re.compile(r"[_*]"), ""),
    (re.compile(r"\t"), " "),
]


def paragraphs(text: str):
    """Running prose: blank-line separated paragraphs with hard line breaks joined; page furniture dropped."""
    for para in re.split(r"\n\s*\n", text):
        for pat, rep in CLEAN:
            para = pat.sub(rep, para)
        para = " ".join(para.split())
        if len(para) >= 40 and sum(c.isalpha() for c in para) >= 0.6 * len(para):
            yield para


def script_of_char(ch: str) -> str | None:
    cat = unicodedata.category(ch)
    if cat[0] == "L":
        name = unicodedata.name(ch, "")
        for script, prefix in SCRIPTS.items():
            if name.startswith(prefix):
                return script
        return None
    if cat[0] in "PS" and ch not in CODE_SYMBOLS:
        return "common"
    return None


def load_charset(path: Path) -> list[str]:
    entries = path.read_text(encoding="utf-8").split("\n")
    if entries and entries[-1] == "":
        entries.pop()
    return entries[1:]


def fold(ch: str) -> str:
    """Simple lowercase form of one character (characters whose lowercase is not a single code point stay as they are)."""
    low = ch.lower()
    return low if len(low) == 1 else ch


def quant_prob(lnp: float) -> int:
    return max(0, min(254, round(-lnp / STEP)))


def quant_bow(lnb: float) -> int:
    return max(-127, min(127, round(lnb / STEP)))


class Estimator:
    """Interpolated Kneser-Ney with pruning, then back-off form."""

    def __init__(self, words: Counter, vocab: int, order: int, min_counts: list[int]):
        self.order = order
        self.vocab = vocab  # alphabet size + end-of-word
        raw = [None] + [defaultdict(int) for _ in range(order)]
        for word, c in words.items():
            seq = BOS + word + EOS
            for n in range(1, order + 1):
                for i in range(len(seq) - n + 1):
                    g = seq[i : i + n]
                    if n == 1 and g == BOS:
                        continue  # the start symbol is never predicted
                    raw[n][g] += c
        # Effective counts: raw at the top order and for grams that begin a word, continuation counts elsewhere.
        eff = [None] * (order + 1)
        eff[order] = raw[order]
        for n in range(order - 1, 0, -1):
            cont: dict[str, int] = defaultdict(int)
            for g in raw[n + 1]:
                cont[g[1:]] += 1
            e = {}
            for g, c in raw[n].items():
                e[g] = c if g[0] == BOS else cont.get(g, 0) or c
            eff[n] = e
        del raw
        self.eff = eff
        self.disc = [0.0] * (order + 1)
        self.tot: list[dict] = [None] * (order + 1)
        self.n1: list[dict] = [None] * (order + 1)
        for n in range(1, order + 1):
            coc = Counter(min(c, 3) for c in eff[n].values())
            d = coc[1] / (coc[1] + 2 * coc[2]) if coc[1] + 2 * coc[2] else 0.5
            self.disc[n] = min(0.95, max(0.1, d))
            tot: dict[str, int] = defaultdict(int)
            n1: dict[str, int] = defaultdict(int)
            for g, c in eff[n].items():
                tot[g[:-1]] += c
                n1[g[:-1]] += 1
            self.tot[n], self.n1[n] = tot, n1
        self.min_counts = min_counts

    def p_full(self, ctx: str, w: str) -> float:
        """Interpolated Kneser-Ney probability of w after ctx (any length below the order), unpruned."""
        n = len(ctx) + 1
        if n == 1:
            t = self.tot[1].get("", 0)
            c = self.eff[1].get(w, 0)
            d = self.disc[1]
            return (max(c - d, 0) + d * self.n1[1][""] / self.vocab) / t if t else 1.0 / self.vocab
        t = self.tot[n].get(ctx, 0)
        lower = self.p_full(ctx[1:], w)
        if not t:
            return lower
        c = self.eff[n].get(ctx + w, 0)
        d = self.disc[n]
        return (max(c - d, 0) + d * self.n1[n][ctx] * lower) / t

    def build(self):
        order = self.order
        kept: list[dict[str, float]] = [None] + [dict() for _ in range(order)]
        for n in range(1, order + 1):
            for g, c in self.eff[n].items():
                if c >= self.min_counts[n - 1] or n == 1:
                    kept[n][g] = 0.0
        # A kept gram needs its context kept, since the context carries the back-off weight.
        for n in range(order, 1, -1):
            for g in list(kept[n]):
                if len(g) > 1 and g[:-1] not in kept[n - 1]:
                    kept[n - 1][g[:-1]] = 0.0
        for n in range(1, order + 1):
            for g in kept[n]:
                kept[n][g] = self.p_full(g[:-1], g[-1])
        # The start symbol as a unigram carries only a back-off weight.
        kept[1].setdefault(BOS, 1e-12)
        bows: list[dict[str, float]] = [None] + [dict() for _ in range(order)]
        for n in range(1, order):
            by_ctx: dict[str, list[str]] = defaultdict(list)
            for g in kept[n + 1]:
                by_ctx[g[:-1]].append(g[-1])
            for ctx in kept[n]:
                words = by_ctx.get(ctx, [])
                num = 1.0 - sum(kept[n + 1][ctx + w] for w in words)
                den = 1.0 - sum(self.backoff_p(kept, bows, ctx[1:], w) for w in words)
                b = num / den if den > 1e-9 and num > 1e-9 else 1.0
                bows[n][ctx] = b
        return kept, bows

    def backoff_p(self, kept, bows, ctx: str, w: str) -> float:
        """Probability under the pruned model, from entries in tables below the one being completed."""
        back = 1.0
        for length in range(len(ctx), -1, -1):
            c = ctx[len(ctx) - length :]
            p = kept[length + 1].get(c + w)
            if p is not None:
                return back * p
            if length > 0:
                back *= bows[length].get(c, 1.0)
        return back * 1e-6


def log_prob(model, ctx: str, w: str) -> float:
    kept, bows, order = model
    ctx = ctx[-(order - 1) :] if order > 1 else ""
    back = 0.0
    for length in range(len(ctx), -1, -1):
        c = ctx[len(ctx) - length :]
        p = kept[length + 1].get(c + w)
        if p is not None:
            return back - STEP * quant_prob(math.log(max(p, 1e-300)))
        if length > 0:
            b = bows[length].get(c)
            if b is not None:
                back += STEP * quant_bow(math.log(b))
    return back - 14.0


def write_model(f, name: str, alphabet: list[str], order: int, mean: float, kept, bows) -> None:
    nm = name.encode("ascii")
    f.write(struct.pack("<B", len(nm)) + nm)
    f.write(struct.pack("<BB", order, len(alphabet)))
    for a in alphabet:
        b = a.encode("utf-8")
        f.write(struct.pack("<B", len(b)) + b)
    f.write(struct.pack("<f", mean))
    for n in range(1, order + 1):
        keys = sorted(kept[n])
        f.write(struct.pack("<I", len(keys)))
        for g in keys:
            f.write(bytes(ord(c) for c in g))
            f.write(struct.pack("<B", quant_prob(math.log(max(kept[n][g], 1e-300)))))
            if n < order:
                f.write(struct.pack("<b", quant_bow(math.log(max(bows[n].get(g, 1.0), 1e-300)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, type=Path, help="directory with pg<ID>.txt")
    ap.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "texts.yaml")
    ap.add_argument("--charset", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "charset-v1.txt")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--min-counts", default="1,3,4,6,8,8,8,8", help="smallest effective count kept per order")
    ap.add_argument("--min-symbol-count", type=int, default=4)
    args = ap.parse_args()
    min_counts = [int(x) for x in args.min_counts.split(",")][: args.order]
    charset = set(load_charset(args.charset))

    words: dict[str, Counter] = {s: Counter() for s in SCRIPTS}
    books = 0
    for b in read_manifest(args.manifest):
        if b.get("split") != "train":
            continue
        path = args.texts / f"pg{b['id']}.txt"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != b["sha256"]:
            sys.exit(f"SHA-256 mismatch for {path}")
        script = LANG_SCRIPT.get(b["language"], "latin")
        books += 1
        for para in paragraphs(body_of(raw)):
            for tok in para.split(" "):
                if tok:
                    words[script][tok] += 1
    print(f"{books} training books", file=sys.stderr)

    import io

    buf = io.BytesIO()
    models = 0
    for script in SCRIPTS:
        tokens = words[script]
        # Keep a word only if every character is a letter of the script or shared punctuation, and in the charset.
        # The model is case-folded: words are lowercased before counting, so "NASI" and "Nasi" are the word "nasi".
        sym = Counter()
        folded: Counter = Counter()
        for tok, c in tokens.items():
            if all(ch in charset and script_of_char(ch) in (script, "common") for ch in tok):
                low = "".join(fold(ch) for ch in tok)
                folded[low] += c
                for ch in low:
                    sym[ch] += c
        tokens = folded
        alphabet = [ch for ch, c in sorted(sym.items()) if c >= args.min_symbol_count]
        if len(alphabet) > 250:
            alphabet = sorted(ch for ch, _ in sym.most_common(250))
        ids = {ch: chr(i + 2) for i, ch in enumerate(alphabet)}
        kept_words = Counter()
        for tok, c in tokens.items():
            if all(ch in ids for ch in tok):
                kept_words["".join(ids[ch] for ch in tok)] += c
        total = sum(kept_words.values())
        print(f"{script}: {len(kept_words)} word types, {total} tokens, alphabet {len(alphabet)}", file=sys.stderr)
        if total < 1000:
            continue
        est = Estimator(kept_words, len(alphabet) + 1, args.order, min_counts)
        kept, bows = est.build()
        model = (kept, bows, args.order)
        lp_sum, n_sym = 0.0, 0
        for w, c in kept_words.items():
            ctx = BOS
            for ch in w + EOS:
                lp_sum += c * log_prob(model, ctx, ch)
                n_sym += c
                ctx += ch
        mean = lp_sum / n_sym
        sizes = [len(kept[n]) for n in range(1, args.order + 1)]
        print(f"{script}: entries per order {sizes}, mean log prob {mean:.3f} nats/symbol "
              f"(perplexity {math.exp(-mean):.2f}), discounts {[round(d, 2) for d in est.disc[1:]]}", file=sys.stderr)
        before = buf.tell()
        write_model(buf, script, alphabet, args.order, mean, kept, bows)
        print(f"{script}: {buf.tell() - before} bytes", file=sys.stderr)
        models += 1
    with open(args.out, "wb") as f:
        f.write(b"OCLM" + struct.pack("<HH", 1, models))
        f.write(buf.getvalue())
    print(f"wrote {args.out}: {args.out.stat().st_size} bytes, {models} script models", file=sys.stderr)


if __name__ == "__main__":
    main()
