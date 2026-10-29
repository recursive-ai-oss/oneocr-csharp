# PDF bridge (`RecursiveAI.OneOcr.Pdf`)

How a PDF page becomes an `oneocr.page` page. All PDF parsing is OnePDF's and all drawing is the OneImage PDF bridge's;
this package only converts, decides what to trust, and calls the engine's page entry point (`OcrEngine.Ocr`) on rendered regions.

## Versions

`RecursiveAI.OnePdf` 1.10.1 and `RecursiveAI.OneImage.Pdf` 1.10.1 (the versions in `src/OneOcr.Pdf/OneOcr.Pdf.csproj`). OneImage.Pdf pins
the exact `RecursiveAI.OneImage` version, so 1.10.1 is the one that matches the core package's OneImage 1.10.1 floor.

## Coordinates

A PDF page is `OcrPage` with `Width`/`Height` in points of the viewed page (box, `/Rotate` and `/UserUnit` applied; the crop box
is used), `UnitsPerInch` 72, origin top-left, y down. OnePDF's extraction is y up in the same upright space, so a text-layer box
`[l, b, r, t]` becomes `[l, H − t, r, H − b]`. `ToPdf` maps a page-unit point back to user space (`x' = a x + c y + e`,
`y' = b x + d y + f`), for the crop box `[l b r t]`, width `w`, height `h`, user unit `u` and `k = 1/u`:

| `/Rotate` | `[a b c d e f]` |
|---|---|
| 0 | `[k 0 0 −k l b+h]` |
| 90 | `[0 k k 0 l b]` |
| 180 | `[−k 0 0 k l+w b]` |
| 270 | `[0 −k −k 0 l+w b+h]` |

Text-layer glyph boxes are OnePDF's: 0.9 em above the baseline to 0.2 em below, from the glyph origin to its advance. Mapped through
`ToPdf` a word's left edge lands on the glyph origin and its edges on the baseline ∓ those distances; the tests hold this to
**0.6 pt** in user space for upright, rotated, cropped and `/UserUnit` pages. Recognized words come from a render at `RenderDpi` and
carry the recognizer's ink boxes, so their left edge is only held to within 4 pt (CTC alignment).

## What is read how

1. The text layer is extracted in reading order with glyphs retained. Words, lines, blocks, roles (header/footer), tables and cell
   starts carry over; words and characters have `Source = TextLayer`, confidence 1 (the file says so; it is not a calibrated probability).
2. **Invisible text** (render mode 3 or fully transparent) is somebody else's OCR layer. By default it is dropped and every image it
   sat on is read by the recognizer. `TrustExistingOcrLayer` keeps it as `TextLayer` and the images under it are not read.
3. **Garbled text**: a line with a word that contains U+FFFD, a private-use code point or a control character is dropped and its
   box (padded by 15% of its height) is rendered and read.
4. **Cross-check**: up to `CrossCheckLines` (3) evenly spaced, visible, left-to-right lines are rendered, read, and compared with the
   text layer on letters and digits (case-folded, Levenshtein distance more than 40% of the longer string = disagreement). If at
   least half the checked lines disagree, the text layer is dropped and the whole page is read from pixels. This catches a
   `ToUnicode` map that is wrong but plausible.
5. **Images**: every image the page draws (from the interpreter's device calls, not inside Type 3 glyphs) of at least
   `MinImagePoints` (24) in both sizes, whose area carries less than 10% text-layer word boxes and fewer than three trusted invisible words, is read.
   A page with no text at all therefore reads its images, and a scan drawn full-page is one region.
6. A region is the crop of the page rendered at `RenderDpi` (300), passed to `OcrEngine.Ocr` with that DPI, and its blocks, tables
   and regions are scaled back to points. Recognized words that overlap an accepted word by at least 40% of the smaller box (text
   drawn over a picture, or two regions that overlap) are dropped, text layer first; tables of a region that lost words are not kept.
7. **Order**: text-layer blocks keep OnePDF's reading order; each recognized region goes before the first text-layer block that starts
   below its top. Blocks, lines and words keep their own `Source`.

Failures (a page that cannot be rendered, an image that is too large, a region over an OCR limit) leave a diagnostic on the page
and the rest of the page is still read; `Strict` throws instead. Damaged files raise `OcrFormatException`, files over a limit
`OcrLimitException`; nothing else comes out of a public method except argument errors, I/O errors and cancellation.

Not done: outlined text (glyphs drawn as paths), annotations (they are neither read nor rendered), vector text-as-art.
