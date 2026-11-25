"""Writes testdata/fax/std-g3.tif: a few lines of statement text as a standard-mode fax (CCITT Group 3, 1-bit, 200 x 100 dots per inch).

    python tools/make-fax-fixture.py testdata/fax/std-g3.tif

The page is drawn at 200 x 200 dpi with Pillow's built-in font, then every second row is dropped by box averaging (what a fax machine in
standard mode does) and thresholded. Nothing is random, so the file is the same on every run with the same Pillow version.
The truth is the list of lines below.
"""

import io
import sys

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1728
LINES = [
    "Opening balance 1,234.56",
    "03/14/2026 Deposit 2,500.00",
    "03/15/2026 Check 1042 -312.75",
    "03/17/2026 Transfer -1,000.00",
    "Closing balance 2,421.81",
]


def main(out: str) -> None:
    height = 120 + 90 * len(LINES)
    page = Image.new("L", (WIDTH, height), 255)
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=38)
    for i, text in enumerate(LINES):
        draw.text((90, 60 + 90 * i), text, fill=0, font=font)
    std = page.resize((WIDTH, height // 2), Image.Resampling.BOX)
    bits = std.point(lambda v: 255 if v > 150 else 0).convert("1", dither=Image.Dither.NONE)
    buf = io.BytesIO()
    bits.save(buf, format="TIFF", compression="group3", dpi=(200, 100))
    with open(out, "wb") as f:
        f.write(buf.getvalue())
    print(len(buf.getvalue()), "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
