"""Low-resolution copies of the dot-leader tune sets: 300 DPI pages reduced to 150 and 200 DPI, blurred and saved as JPEG.

    python scripts/degrade_leader_sets.py SRC_ROOT OUT_ROOT

SRC_ROOT holds sets written by gen_leader_sets.py (`<set>/pages/*.png`, `<set>/truth/*.json|txt`) in the directories named by SETS.
For each resolution the pages of all of them go to `OUT_ROOT/leader-lr<dpi>/{pages,truth}`; the truth words' boxes are scaled by the
same factor, the text is unchanged. At these resolutions a full stop is one or two pixels wide and its middle is paler than the
page's ink threshold, which the 300 DPI sets do not show. The blur radius and JPEG quality vary per page from a fixed seed, so the
output is reproducible. Needs Pillow only.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageFilter

SETS = ("leader-tune", "leader-tune2", "leader-tune3")
# (dpi, scale from 300 DPI, blur radius, JPEG quality)
LEVELS = ((150, 0.5, 0.5, 60), (200, 2 / 3, 0.6, 65))
BOX_KEYS = ("box", "ink", "ink_quad", "quad")


def main(src: Path, out: Path) -> None:
    for dpi, scale, blur, quality in LEVELS:
        dest = out / f"leader-lr{dpi}"
        shutil.rmtree(dest, ignore_errors=True)
        (dest / "pages").mkdir(parents=True)
        (dest / "truth").mkdir()
        rng = random.Random(121 + dpi)
        for name in SETS:
            for page in sorted((src / name / "pages").glob("*.png")):
                pid = f"{name.removeprefix('leader-')}-{page.stem}"
                image = Image.open(page).convert("L")
                w, h = image.size
                image = image.resize((round(w * scale), round(h * scale)), Image.BOX)
                image = image.filter(ImageFilter.GaussianBlur(blur * (0.8 + 0.4 * rng.random())))
                image.save(dest / "pages" / f"{pid}.jpg", quality=quality - rng.randrange(0, 10), dpi=(dpi, dpi))
                truth = json.loads((src / name / "truth" / f"{page.stem}.json").read_text(encoding="utf-8"))
                for word in truth["words"]:
                    for key in BOX_KEYS:
                        if key in word:
                            word[key] = [round(v * scale, 2) for v in word[key]]
                truth["id"] = pid
                (dest / "truth" / f"{pid}.json").write_text(json.dumps(truth), encoding="utf-8")
                shutil.copy(src / name / "truth" / f"{page.stem}.txt", dest / "truth" / f"{pid}.txt")
        print(dest, len(list((dest / "pages").iterdir())))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
