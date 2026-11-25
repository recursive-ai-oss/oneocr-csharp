"""Artifact check and error breakdown for the amount exact match of the financial sets.

Two questions, answered with counts:
  1. Can the recognizer write every character the truth uses? (truth characters outside the charset, tagged values that contain one)
  2. For the values an engine misses, what exactly differs from the truth (substitution / deletion / insertion pairs), and is any miss
     explained by a character the charset lacks or the scorer's fold does not unify?

It reuses the benchmark scorer's own region and ownership rules by recording the token lists the scorer builds for each value.

    uv run python -m eval.fin_misses --eval DIR --run DIR [--bench PATH] [--out report.json]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein


def _sets(root: Path):
    return sorted(p for p in root.iterdir() if (p / "truth").is_dir())


def _ops(want: str, tokens: list[str]) -> tuple[str, list[tuple[str, str, str]]]:
    """Align `want` against the best window of the tokens' concatenation; return the window and the edit operations."""
    joined = "".join(tokens)
    if not joined:
        return "", [("del", c, "") for c in want]
    best = None
    n = len(want)
    for i in range(len(joined)):
        for ln in range(max(1, n - 3), n + 4):
            w = joined[i:i + ln]
            d = Levenshtein.distance(want, w)
            if best is None or d < best[0]:
                best = (d, w)
    w = best[1]
    out = []
    for op in Levenshtein.opcodes(want, w):
        if op.tag == "replace":
            out += [("sub", want[op.src_start + k], w[op.dest_start + k]) for k in range(op.src_end - op.src_start)]
        elif op.tag == "delete":
            out += [("del", want[k], "") for k in range(op.src_start, op.src_end)]
        elif op.tag == "insert":
            out += [("ins", "", w[k]) for k in range(op.dest_start, op.dest_end)]
    return w, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bench", default=str(Path(__file__).resolve().parents[2] / "bench"))
    ap.add_argument("--charset", default=str(Path(__file__).resolve().parents[1] / "data" / "charset-v1.txt"))
    ap.add_argument("--out")
    a = ap.parse_args()
    sys.path.insert(0, a.bench)
    from oneocr_bench import amounts
    from oneocr_bench.score import FOLD

    charset = set(Path(a.charset).read_text(encoding="utf8").splitlines())
    seen: list = []
    orig = amounts._spells

    def rec(tokens, want):
        r = orig(tokens, want)
        seen.append((list(tokens), want, r))
        return r

    amounts._spells = rec
    report = {}
    for sdir in _sets(Path(a.eval)):
        name = sdir.name
        chars = Counter()
        tagged_bad = Counter()
        n_val = n_miss = 0
        pairs = Counter()
        kinds = Counter()
        lacking_miss = 0
        for tf in sorted((sdir / "truth").glob("*.json")):
            t = json.loads(tf.read_text())
            for w in t["words"]:
                for c in w["text"]:
                    chars[c] += 1
            side = Path(a.run) / name / (tf.stem + ".json")
            if not side.exists() or not t.get("tags"):
                continue
            hyp = json.loads(side.read_text())["words"]
            seen.clear()
            tags = t["tags"]
            uniq = {}
            for g in tags:
                uniq.setdefault(tuple(g["words"]), len(uniq))
            amounts.read_values(t["words"], hyp, tags)
            for (key, k), (tokens, want, ok) in zip(uniq.items(), seen):
                raw = "".join(t["words"][i]["text"] for i in key)
                n_val += 1
                bad = {c for c in raw if c not in charset and not c.isspace()}
                for c in bad:
                    tagged_bad[c] += 1
                if ok:
                    continue
                n_miss += 1
                if bad:
                    lacking_miss += 1
                if not tokens:
                    kinds["nothing read in region"] += 1
                    continue
                win, ops = _ops(want, tokens)
                if len(ops) >= max(3, len(want) // 2):
                    kinds["unrelated or merged"] += 1
                    continue
                for op in ops:
                    pairs[op] += 1
                kinds["char errors"] += 1
        # the scorer folds these to ASCII; a truth character in this list is a check that the fold covers what the generator writes
        report[name] = {
            "values": n_val, "missed": n_miss, "miss_kinds": dict(kinds),
            "misses_with_truth_char_outside_charset": lacking_miss,
            "truth_chars_outside_charset": {c: v for c, v in chars.items() if c not in charset and not c.isspace()},
            "tagged_values_with_char_outside_charset": dict(tagged_bad),
            "non_ascii_truth_chars": {c: v for c, v in chars.items() if ord(c) > 127},
            "unfolded_nonascii_in_tags": sorted({c for c in tagged_bad if c not in FOLD}),
            "top_edit_pairs": [[list(k), v] for k, v in pairs.most_common(25)],
        }
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf8")
    for name, r in report.items():
        print(name, "values", r["values"], "missed", r["missed"], r["miss_kinds"], "outside-charset truth chars", r["truth_chars_outside_charset"],
              "misses-with-such-char", r["misses_with_truth_char_outside_charset"])


if __name__ == "__main__":
    main()
