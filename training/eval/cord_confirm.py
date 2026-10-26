"""CORD v2 confirm set for the detector: the calibration images, which no detector run trains on or selects with.

`datasets/cord_crops.py` holds out every 10th official-train image (SHA-256 order) as `calib`; the detector's loader drops exactly those
images, so they are unseen by it (the recognizer's confidence map used them, which does not matter for detection). This builds a page
directory with real copies of those images (a symlink makes the image reader reject the file) and `<id>.txt` / `<id>.json` truth in the
bench format, from the `calib` split `cord_crops.py` wrote.

    python -m datasets.cord_crops --out DIR     # from training/, first
    python eval/cord_confirm.py --crops DIR --out OUT/cord-conf
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="calib", help="calib (default) or val")
    a = ap.parse_args()
    src, out = Path(a.crops) / a.split, Path(a.out)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    n = 0
    for tf in sorted((src / "truth").glob("*.json")):
        t = json.loads(tf.read_text())
        img = next((src / "pages").glob(tf.stem + ".*"))
        shutil.copyfile(img.resolve(), out / "pages" / img.name)
        words = [{"text": w["text"], "box": w["box"]} for w in t["words"]]
        (out / "truth" / f"{tf.stem}.json").write_text(json.dumps({"words": words}), encoding="utf-8")
        (out / "truth" / f"{tf.stem}.txt").write_text(" ".join(w["text"] for w in words), encoding="utf-8")
        n += 1
    print("cord-conf", n, "receipts")


if __name__ == "__main__":
    main()
