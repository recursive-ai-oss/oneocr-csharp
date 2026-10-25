"""Usage: gen_phone_tune.py VAL_DIR OUT [COUNT]

Phone-photo pages for choosing thresholds that must not be chosen on the frozen sets: the same camera simulation as phone-v1
(`degrade.phone_capture`: placement, perspective, desk, uneven light, blur, sensor noise, JPEG) applied to held-out generated pages
(the validation split of the page generator, whose templates the recognizer and detector never trained on) under a new seed.
Only single-column Latin-script prose pages without tables are used, so the page text in reading order is the truth
. Writes OUT/pages/<id>.jpg and
OUT/truth/<id>.txt. Run from bench/ with `uv run`.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from oneocr_bench import degrade

SEED = 20270301
DRAWS = 4
LATIN = {"en", "de", "fr", "es", "it", "nl", "pt", "sv", "da", "fi", "cs", "hu", "ro", "ca", "pl"}

val, out = Path(sys.argv[1]), Path(sys.argv[2])
count = int(sys.argv[3]) if len(sys.argv) > 3 else 40
(out / "pages").mkdir(parents=True, exist_ok=True)
(out / "truth").mkdir(parents=True, exist_ok=True)
made = 0
for f in sorted((val / "truth").glob("*.json")):
    t = json.loads(f.read_text())
    sy = t["synthetic"]
    if sy["kind"] != "prose" or sy["language"] not in LATIN or sy["template"].get("cols", 1) != 1:
        continue
    if t["pages"][0].get("tables"):
        continue
    heights = sorted(l["bounds"][3] - l["bounds"][1] for b in t["pages"][0]["blocks"] for l in b["lines"])
    if not heights or heights[len(heights) // 2] * 0.63 < 24:  # after the camera scale, body text must be at least ~24 px tall, as in phone-v1
        continue
    img = Image.open(val / sy["image"]["file"]).convert("L")
    for draw in range(DRAWS):  # several camera placements of one page: the placement, not the page, is what is being tuned on
        for attempt in range(40):
            r = random.Random(f"phone-tune:{SEED}:{made}:{draw}:{attempt}")
            n = np.random.default_rng([SEED, made, draw, attempt])
            try:
                data, _, params = degrade.phone_capture(img, [], r, n)
            except ValueError:
                continue
            break
        else:
            continue
        name = f"{f.stem}-{draw}"
        (out / "pages" / f"{name}.jpg").write_bytes(data)
        (out / "truth" / f"{name}.txt").write_text(" ".join(t["pages"][0]["text"].split()) + "\n")
    made += 1
    if made >= count:
        break
print(made, "pages x", DRAWS, "draws in", out)
