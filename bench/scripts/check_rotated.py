"""Usage: check_rotated.py ROT_OUT [EXPECTED_JSON]

Compares the orientation and skew the engine reported (<id>.meta.json from `OneOcr.Bench run --meta 1`) with what
make_rotated.py applied (ROT_OUT/../rot/expected.json), per variant and set, and prints the share of pages whose skew is within
0.3 degrees and whose orientation is right.
"""
import json
import os
import sys

out = sys.argv[1]
expected = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(out.rstrip("/")), "rot", "expected.json")))
tot = skew_ok = orient_ok = 0
worst = []
for variant in sorted(os.listdir(out)):
    want = expected[variant]
    for s in sorted(os.listdir(os.path.join(out, variant))):
        d = os.path.join(out, variant, s)
        n = so = oo = 0
        for f in sorted(os.listdir(d)):
            if not f.endswith(".meta.json"):
                continue
            got = json.load(open(os.path.join(d, f)))
            n += 1
            err = abs(got["skew"] - want["skew"])
            so += err <= 0.3
            oo += got["orientation"] == want["orientation"]
            if err > 0.3 or got["orientation"] != want["orientation"]:
                worst.append((variant, s, f[:-10], got, round(err, 2)))
        tot += n
        skew_ok += so
        orient_ok += oo
        print(f"{variant:12s} {s:12s} pages {n:3d}  skew within 0.3: {so:3d}  orientation right: {oo:3d}")
print(f"TOTAL pages {tot}  skew within 0.3: {skew_ok} ({100*skew_ok/tot:.1f}%)  orientation right: {orient_ok} ({100*orient_ok/tot:.1f}%)")
for w in worst[:40]:
    print("miss", w)
