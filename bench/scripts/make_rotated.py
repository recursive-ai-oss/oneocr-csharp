"""Usage: make_rotated.py BENCH_DIR OUT

Writes rotated and tilted copies of the clean300-v1 and layout-v1 pages under OUT/<variant>/<set>/.

Variant names: s+3.0 (tilted clockwise by 3 degrees), r90 (content turned 90 degrees clockwise), r270s-4.0 (tilted 4 degrees counter-
clockwise, then turned 270 clockwise), ... ids stay p001.. so the original set's truth applies. expected.json lists, per variant,
the orientation and skew the engine should report (skew is measured in the upright frame, positive = lines descend to the right).
"""
import json
import os
import sys
from PIL import Image

SRC = sys.argv[1]
OUT = sys.argv[2]
SETS = ["clean300-v1", "layout-v1"]
VARIANTS = []  # (name, skew, turn)
for a in (-10, -6, -3, -1.5, -0.5, 0.5, 1.5, 3, 6, 10):
    VARIANTS.append((f"s{a:+.1f}", a, 0))
for r in (90, 180, 270):
    VARIANTS.append((f"r{r}", 0, r))
for r, a in ((90, 4), (180, -3), (270, -7)):
    VARIANTS.append((f"r{r}s{a:+.1f}", a, r))

TURN = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}
expected = {}
for name, skew, turn in VARIANTS:
    expected[name] = {"orientation": turn, "skew": skew}
    for s in SETS:
        d = os.path.join(OUT, name, s)
        os.makedirs(d, exist_ok=True)
        for f in sorted(os.listdir(os.path.join(SRC, s, "pages"))):
            if not f.endswith(".png"):
                continue
            im = Image.open(os.path.join(SRC, s, "pages", f)).convert("L")
            if skew:
                # PIL turns counter-clockwise for positive angles; a clockwise tilt of `skew` degrees is -skew.
                im = im.rotate(-skew, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
            if turn:
                im = im.transpose(TURN[turn])
            # A fixed, moderate compression level: some valid level-6 streams of large pages trip some PNG decoders.
            im.save(os.path.join(d, f), compress_level=3)
json.dump(expected, open(os.path.join(OUT, "expected.json"), "w"), indent=1)
print(len(VARIANTS), "variants")
