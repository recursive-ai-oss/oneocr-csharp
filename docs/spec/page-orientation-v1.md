# Page orientation and deskew, version 1

What the local reader does to a page image before it looks for text lines, so that scans fed in sideways, upside down or crooked
are read like upright ones. Every step uses integers or the IEEE basic operations (`+ − × ÷ √`), so the result is the same on every
machine, kernel path and degree of parallelism. A port reproduces the numbers below.

The steps run in this order, each on the 8-bit gray page:

1. **Orientation** (one seam, two sources): how many quarter turns clockwise (0–3) make the text upright. With a page detector
   loaded, its orientation head decides and the classical decision below is not run; without one, the classical decision runs.
2. **Quarter turn**: the page is turned by exactly that many quarter turns (pixel moves, no resampling). Both sources use the same
   rotation step (`(x, y) → (h − 1 − y, x)` per clockwise quarter).
3. **Skew**: the tilt of the text lines is estimated on the turned page.
4. **Deskew**: when the tilt is large enough to act on, the page is turned by it on a larger canvas, with bilinear sampling.
5. The turned and straightened page is read as usual; every box is then carried back to the input image (below).
6. **Angle search** (only when the first reading is weak, below): the page is read again at other whole-page angles and the best reading is kept.

`OcrOptions.DetectOrientation`, `OcrOptions.Deskew` and `OcrOptions.SearchAngle` (all default true) switch steps 1–2, 3–4 and 6 off (with a detector, `DetectOrientation = false` also stops the head from turning the page). A page that needs neither
(no turn, tilt under 0.3°) is read from the very same pixels as with both switched off, so its text is the same.

## Ink grid

The estimators work on a coarse count of ink pixels. Ink is the minority class of Otsu's threshold (the same binarization the line
finder uses; a page with no contrast has no ink). With `f = ceil(max(width, height) / 1200)`, cell `(cx, cy)` holds the number of ink
pixels in the `f × f` block at `(cx·f, cy·f)`. A grid shrunk by 4 adds up 4 × 4 cells. A quarter turn clockwise of a grid maps cell
`(x, y)` to `(height − 1 − y, x)`.

## Skew (projection profile)

After Postl (1986) and Baird (1987). The ink of cells that are not solid (`0 < count ≤ max(1, ⌊0.6 · f²⌋)`) and lie outside a 2%
border on every side is projected on the axis perpendicular to a candidate line direction. For a slope `k / 1024` (lines descend by
`k` per 1024 across), the cell `(x, y)` goes into bin `⌊(1024·y − k·x) / 1024⌋` (arithmetic shift) and adds its count. The
sharpness of a candidate is the sum of squares of the bins; text lines pile ink into few bins, a wrong direction smears it.

- **Coarse search**: on the grid shrunk by 4, `k = −280 … 280` in steps of 2 (±15.3°); the best sharpness wins, ties go to the
  smaller `|k|`.
- **Fine search**: on the full grid, `k` from `coarse − 6` to `coarse + 6` in steps of 1 (and 0, to report the gain); same tie rule.
- **Gain** = sharpness at the best `k` ÷ sharpness at 0. **Contrast** = `best · bins ÷ (Σ counts)²`, where `bins` is the grid height.
  Contrast is 1 for evenly spread ink and large for ruled lines; it is comparable between a page and the same page turned a quarter.
- The tilt is acted on only when `|k| ≥ 5` (0.28°) and gain ≥ 1.03, and only when the lines run across the page: the contrast of
  the grid turned a quarter must not exceed the grid's (otherwise the slope is noise from vertical text).

The tilt in degrees is `atan(k / 1024) · 180/π`, computed with the series `x − x³/3 + x⁵/5 − …` (16 terms, in this order, doubles).
It is reported as `OcrPage.SkewDegrees`: positive when the lines descend to the right (clockwise in y-down coordinates).

## Deskew

With `t = k / 1024`: `cos = 1 / √(1 + t²)`, `sin = t · cos`. The turned page `Wq × Hq` is put on a canvas of
`⌈Wq·cos + Hq·|sin|⌉ × ⌈Wq·|sin| + Hq·cos⌉`; canvas pixel `(i, j)` samples the page at

```
dx = i + 0.5 − Wd/2           dy = j + 0.5 − Hd/2
qx = dx·cos − dy·sin + Wq/2   qy = dx·sin + dy·cos + Hq/2
```

with bilinear interpolation between the four pixel centers around `(qx − 0.5, qy − 0.5)` (the page edge is clamped), the result
rounded half up to 8 bits. Points outside the page take the page's most common gray value (lowest value on ties). If the canvas
would exceed `OcrLimits.MaxImagePixels` the page is read as it is and a diagnostic says so.

## Orientation: the classical decision

`IOrientationDetector.Detect` returns 0, 90, 180 or 270 (degrees clockwise to turn the page upright). A model-provided
detector can replace the classical one behind this interface. The classical decision:

1. **Line direction.** Contrast of the grid and of the grid turned a quarter clockwise. If the turned grid's contrast is at least 3×
   the grid's, text runs vertically: the page is a quarter turn off (candidate axis 1). If the grid's is at least 3× the turned
   grid's, the lines are horizontal (axis 0). Otherwise the profiles do not decide.
2. **Which way up.** If the lines are horizontal by the 3× rule, the page is not tilted enough to act on, and its strips (found
   with the page finder, ruling lines left in place) have clearly more ink above their body band than below it, the page is upright
   and nothing is read. Per strip at least 10 pixels tall and 3× as wide as tall, the body band is the rows with at least half the
   ink of the densest row; over all such strips `above` and `below` are the ink above and below their bands. The test passes when
   `above + below ≥ total/20` and `10·above ≥ 14·below` (ascenders and capitals against descenders; digits, capitals only and
   upside-down text fail it and are read). Otherwise (on the axis chosen, turned and straightened by that axis's skew estimate, when
   that is acted on): the page's strips are found with the page finder (ruling lines left in place), the 24 biggest by area whose width is at least 3 times their height are taken,
   those with a height between 0.6× and 1.7× of the median height of those are kept, and up to 8 are read twice by the recognizer:
   as cropped and with the crop turned half way round (the crop's pixel array reversed). The weight of a line is the longer reading's
   length. The page scores are the weighted means of the two confidences, and *share* is the weighted share of lines that read better
   turned. The first 3 lines are read first and the rest skipped when all 3 read better upright and the upright mean is at least
   0.85. The page is turned half way round when `turned > upright + 0.05`, `turned ≥ 0.75` and `share ≥ 0.6`.
3. **Undecided profiles** (neither contrast is 3× the other): axis 0 is read first; axis 1 is read too and wins only if its better
   reading is at least 0.75 and at least 0.15 above axis 0's. Photographs, receipts and forms end up here and are kept upright
   unless the recognizer clearly reads them better the other way.

The result is `quarter · 90` plus 180 when turned half way round, modulo 360. `OcrPage.Orientation` reports how far the content was
turned clockwise from upright: 0, 90, 180 or 270 (a page whose text tops point to the right has 90, and is turned 270° clockwise,
that is 90° counter-clockwise, to read).

## Orientation from a page detector

With a detector loaded (`OcrEngine.WithDetector`), step 1 is the detector's orientation head: the page is detected as given; when
the head's winner is at least 0.9 confident the page is turned clockwise by the winning quarter count (the shared rotation step) and
detected again on the turned page, otherwise the first detection is the one used. The classical decision is not run.

How the turns compose: quarter turn (head) → skew estimate and deskew on the turned page → detection on the turned, straightened page.
The detector's quarter turn is reported as it always was: the page model is in the *turned* frame (`Width`/`Height` are the turned
page's), `Orientation` stays 0 and a diagnostic names the turn and the head's confidence. Deskew, when it acts, is carried back to
that turned frame by the mapping below (so boxes get quads there) and reported in `SkewDegrees`. A page with neither a confident
turn nor a tilt is detected once, exactly as without the orientation step.

## Angle search

Photographs taken with the camera turned by 10 to 30 degrees are beyond the skew estimate (±15°) and the per-line tilt reading; the
page reads as garbage. The search costs nothing on pages that read well.

- **Trigger.** After the first reading (steps 1–5, mapped to the frame the caller gets), let *confident* words be those with
  calibrated confidence ≥ 0.9. Let the *share* be the characters in confident words over all characters of all words. The search
  runs when the share is under **0.9** (a page without words has share 0 and is searched only when at least 0.2% of its pixels are
  ink, so a blank page is not). It is the **full search** when the share is under **0.6** and the **light search** from 0.6 up to 0.9
  (a photograph turned by a few degrees reads mostly right and still loses lines). Pages with share ≥ 0.9 are never read again.
- **Score** of a reading: the sum, over its confident words in reading order, of `confidence · (number of UTF-16 code units of the
  text)`, accumulated in double precision.
- **Candidates.** The page as it was before step 4 (turned upright, not straightened) is turned by a slope `k / 1024` exactly as in
  Deskew, with `k` taken from the table for 5°, 10°, … 35°: `90, 181, 274, 373, 477, 591, 717` (round(1024·tan a)). The search is coarse to fine:
  1. 10°, then −10°, and in the full search 20°, −20°, 30°, −30° (positive slopes first), each read like the first reading
     (detection, lines, recognition, reading order, the same options) and mapped back through the frame of the candidate;
  2. if one of these won, 5° either side of the winner (the smaller magnitude first).
  A candidate whose canvas would exceed `OcrLimits.MaxImagePixels` is skipped.
- **Winner.** A candidate replaces the best so far only if its score is strictly greater, and the first candidate must beat the first
  reading's score times **1.1**; ties therefore keep the earlier (smaller-angle, positive-sign) reading. If none wins, the first
  reading is the result unchanged.
- **Result.** The winning page's boxes and quads are in the input frame (the turn is part of the frame, so quads are produced by the
  mapping below), `SkewDegrees` is the candidate's angle (`atan(k/1024)`), and a diagnostic says the page was read at several angles.
  The full search costs at most 8 more readings, the light search at most 4 (10°, −10°, and the two 5° steps around a winner).

Every reading is an ordinary page reading, so the search is bit-identical on every path and degree like the rest.

## Mapping back

Without a detector the page model is in the input image's coordinates: `Width`/`Height` are the input's. A point `(u, v)` of the straightened frame
maps to the turned frame by `R(s)` about the canvas and page centers (the sampling formula above, `qx, qy`), then to the input by the
inverse of the quarter turn: for 1 quarter `(x, y) = (qy, H − qx)`, for 2 `(W − qx, H − qy)`, for 3 `(W − qy, qx)` (`W`, `H` the input's
size). Every box becomes the corner quad of its four mapped corners, in the order top-left, top-right, bottom-right, bottom-left of
the text's own frame, plus the axis-aligned box around that quad. Words, characters, lines, blocks, regions and tables carry the
quad (`quad` in the JSON); table cells and key–value pairs keep only the bounding box. A page that needed no turn has no quads.
