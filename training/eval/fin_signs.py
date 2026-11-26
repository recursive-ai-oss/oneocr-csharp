"""Cause table for signs on amounts: where the minus, parentheses or CR/DR of a money value is lost.

usage: python -m eval.fin_signs --eval SETDIR --run RUNDIR [--rows out.jsonl]

RUNDIR is a bench run made with `--stages 1` (<id>.json and <id>.stages.json). For every money value whose truth carries a
sign (leading or trailing minus, parentheses, CR/DR), the words in the value's region are joined and compared. A value is
`ok` (read exactly), `sign-lost` (equal once signs are removed, hypothesis has no sign), `sign-other` (equal after removing
signs, but a different sign), `digits` (anything else) or `none` (no word in the region). For `sign-lost` the stage trace says
where the sign's side of the value was lost: no detected box reaches it, the detected box reaches it but the strip cut for the
recognizer does not, the strip covers it and the reading dropped it. A false sign is a hypothesis value with a sign where the
truth has none.
"""
import argparse, collections, glob, json, os, re

SIGNS = "-\u2212\u2013\u2014"


def sign_kind(t):
    if t[:1] in SIGNS: return "lead-minus"
    if t[-1:] in SIGNS: return "trail-minus"
    if t.startswith("(") and t.endswith(")"): return "parens"
    if re.search(r"(CR|DR|Cr|Dr)$", t): return "crdr"
    return None


def strip_sign(t):
    t = t.strip("()" + SIGNS)
    return re.sub(r"(CR|DR|Cr|Dr)$", "", t).strip(SIGNS + "()")


def quad_box(w):
    q = w.get("ink_quad") or w.get("quad")
    xs, ys = q[0::2], q[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--rows")
    a = ap.parse_args()
    out = collections.Counter(); cause = collections.Counter(); kinds = collections.Counter()
    falses = []; unsigned = 0; rows = []
    for tf in sorted(glob.glob(a.eval + "/truth/*.json")):
        pid = os.path.basename(tf)[:-5]
        t = json.load(open(tf))
        hp = f"{a.run}/{pid}.json"
        if not os.path.exists(hp): continue
        hyp = json.load(open(hp))["words"]
        sp = f"{a.run}/{pid}.stages.json"
        st = json.load(open(sp))["entries"] if os.path.exists(sp) else []
        for tag in t["tags"]:
            if tag["cls"] != "money": continue
            ws = [t["words"][i] for i in tag["words"]]
            text = "".join(w["text"] for w in ws)
            if not any(c.isdigit() for c in text): continue
            boxes = [quad_box(w) for w in ws]
            L = min(b[0] for b in boxes); T = min(b[1] for b in boxes); R = max(b[2] for b in boxes); B = max(b[3] for b in boxes)
            cx, cy, hw, hh = (L + R) / 2, (T + B) / 2, (R - L) * 0.65, (B - T) * 0.65
            reg = sorted((x for x in hyp if abs((x["box"][0] + x["box"][2]) / 2 - cx) <= hw and abs((x["box"][1] + x["box"][3]) / 2 - cy) <= hh),
                         key=lambda x: x["box"][0])
            got = "".join(x["text"] for x in reg)
            kind = sign_kind(text)
            if kind is None:
                unsigned += 1
                if got and sign_kind(got) in ("lead-minus", "trail-minus", "parens") and strip_sign(got) == strip_sign(text):
                    falses.append((pid, text, got))
                continue
            kinds[kind] += 1
            if got == text: res = "ok"
            elif not got: res = "none"
            elif strip_sign(got) == strip_sign(text):
                res = "sign-lost" if sign_kind(got) is None else "sign-other"
            else: res = "digits"
            out[(kind, res)] += 1
            if res == "sign-lost":
                # side of the value where the sign sits
                x_sign = L if kind in ("lead-minus", "parens") else R
                near = lambda e: e["box"][1] - 3 <= cy <= e["box"][3] + 3 and e["box"][0] - 4 <= x_sign <= e["box"][2] + 4
                det = [e for e in st if e["stage"] == "detected" and near(e)]
                strip = [e for e in st if e["stage"].startswith("strip") and near(e)]
                drop = [e for e in st if e["stage"].startswith(("junk-drop", "faint-drop")) and near(e)]
                if not det and not strip: c = "no detected box reaches the sign"
                elif not strip: c = "detected box reaches it, strip does not"
                elif drop: c = "sign mark dropped by a filter"
                else: c = "strip covers it, read without sign"
                cause[(kind, c)] += 1
                rows.append({"id": pid, "text": text, "got": got, "kind": kind, "cause": c})
    print("signed values by kind:", dict(kinds), " unsigned money values:", unsigned)
    for (k, r), n in sorted(out.items()): print(f"  {k:12s} {r:10s} {n}")
    print("sign-lost causes:")
    for (k, c), n in sorted(cause.items()): print(f"  {k:12s} {c:45s} {n}")
    print("false signs (unsigned truth value read with a sign):", len(falses))
    for f in falses[:10]: print("   ", *f)
    if a.rows:
        with open(a.rows, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
