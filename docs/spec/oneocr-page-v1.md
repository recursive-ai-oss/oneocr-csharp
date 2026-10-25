# oneocr.page, version 1

The JSON form of an OCR result: pages of blocks, lines, words and characters, with the layout elements a layout model can add
(regions, tables, key–value pairs). It is written by `OcrPageJson.ToJson`/`Write`, read by `OcrPageJson.Read`/`Parse`, and is
deterministic: the same pages give the same bytes on every platform and path.

It is built on the OnePDF `onepdf.extraction` JSON (member names, order, rounding, text joining). Every member that has a
counterpart there keeps its name, its place and its meaning; what OCR needs beyond that is added at the end of each object.
Three things differ on purpose, and a converter must handle them:

1. **y runs down.** Origin at the top-left of the page, x to the right, y down. `onepdf.extraction` is y up.
2. **Box order is `[left, top, right, bottom]`** (OnePDF: `[left, bottom, right, top]`). With y down, `top <= bottom`.
3. A word has no `glyphs` list: OCR has no glyph indices. It has `chars` instead (below).

## Format

- `format` is `"oneocr.page"` and `version` is `1`. A reader must check both. New members keep the version; renaming, removing
  or changing the meaning of a member raises it.
- UTF-8, no byte-order mark (a reader rejects one). Strings are escaped the way OnePDF's writer does (the relaxed .NET JSON
  encoder: text in the Basic Multilingual Plane is written as it is, characters beyond it as `\uXXXX` surrogate-pair escapes).
  Unpaired surrogates are written as U+FFFD.
- Numbers: coordinates, sizes and `unitsPerInch` are rounded to **three decimals**, confidences to **four**, `toPdf` entries to
  **six**; ties to even; invariant form, no exponent, no trailing zeros; negative zero and non-finite values are `0`. Integers
  are plain.
- Member order is the order below. Whitespace is insignificant.
- Duplicate member names are an error. Unknown members are ignored by a reader (they may be additive members of a later minor
  revision).

## Coordinates

Page units: pixels for images, points for PDF pages. `unitsPerInch` says how many units make an inch (the image's DPI, or an
estimate, with a diagnostic, when the caller gave none; 72 for PDFs). `toPdf`, present only for pages that came from a PDF, is
the matrix `[a, b, c, d, e, f]` taking a point `(x, y)` of this page to PDF user space: `x' = a x + c y + e`,
`y' = b x + d y + f`.

## Shape

```
{
  "format": "oneocr.page", "version": 1,
  "pageCount": <int>,                 must equal the length of "pages"
  "pages": [ Page, ... ]
}

Page = {
  "index": <int>, "width": <number>, "height": <number>,
  "unitsPerInch": <number>,
  "toPdf": [a, b, c, d, e, f],        optional
  "orientation": 90 | 180 | 270,      optional, written only when not 0
  "skew": <number>,                   optional, written only when not 0
  "diagnostics": [ <string>, ... ],
  "text": <string>,                   blocks joined by "\n\n", lines by "\n", cells by "\t"; derived, ignored on read
  "blocks": [ Block, ... ],           reading order
  "regions": [ Region, ... ],
  "tables": [ Table, ... ],
  "keyValues": [ KeyValue, ... ]
}

Block = {
  "role": "body" | "header" | "footer",
  "table": <bool>,
  "bounds": [l, t, r, b],
  "lines": [ Line, ... ],
  "source": "ocr" | "textLayer" | "llm",
  "confidence": <number 0..1>,
  "quad": [x0, y0, x1, y1, x2, y2, x3, y3]       optional
}

Line = {
  "direction": "ltr" | "rtl", "vertical": <bool>,
  "bounds": [l, t, r, b],
  "cells": [ <int>, ... ],            indices into "words" where a cell starts; first is 0; empty only without words
  "text": <string>,                   words joined by " ", cells by "\t"
  "words": [ Word, ... ],
  "source", "confidence", "quad"      as in Block
}

Word = {
  "text": <string>,                   logical order, never empty
  "bounds": [l, t, r, b],             ink extent for recognized words
  "visible": <bool>,
  "source": ..., 
  "confidence": <number>,             calibrated probability the word is exactly right
  "rawConfidence": <number>,          uncalibrated (mean of the characters' largest softmax probability)
  "chars": [ Char, ... ],             optional
  "quad": [...]                       optional
}

Char = {
  "text": <string>, "bounds": [l, t, r, b], "confidence": <number>, "source": ...,
  "approximate": <bool>,              true when the horizontal edges come from CTC alignment, not the glyph outline
  "quad": [...]                       optional
}

Region = { "role": "text"|"title"|"list"|"table"|"figure"|"header"|"footer"|"formField"|"checkbox",
           "bounds", "confidence", "source", "blocks": [ <block index>, ... ], "quad" optional }

Table = { "bounds", "rows": <int>, "columns": <int>, "cells": [ Cell, ... ], "confidence", "source",
          "block": <block index>, "quad" optional }
Cell  = { "row", "column", "rowSpan", "columnSpan", "bounds", "words": [ <word index>, ... ] }

KeyValue = { "key": [ <word index>, ... ], "value": [ <word index>, ... ], "bounds", "confidence", "source" }
```

**Orientation and skew.** `orientation` is how far the page content was turned clockwise from upright when the image was read:
90 means the tops of the lines point to the right. `skew` is the tilt, in degrees (three decimals), of the text lines that was
removed before reading, positive when the lines descend to the right; it is within 45. Both are additive members of version 1, written
only when they are not 0 (so a page that needed neither has the same bytes as before) and read as 0 when absent. A reader rejects an
`orientation` other than 0, 90, 180 or 270 and a `skew` beyond 45 with `OcrFormatException`. All coordinates stay in the image as
given, whatever the turn; text that was turned or straightened carries a `quad` (corners in the text's own frame: top-left, top-right,
bottom-right, bottom-left), and its `bounds` is the box around the quad. How the turn and tilt are found and mapped back is in
`page-orientation-v1.md`. Pages from a PDF have neither member.

**Lines read along their own axis.** A line whose own lean differs from the page's (after any page-wide deskew) carries a `quad` on its
words, characters and the line even when the page itself was not turned or straightened: the quad is the line's tilted rectangle, and
`bounds` is the box around it. Straight lines carry none, as before. See `detector-v1.md`, "Line axis".

**Word index.** Words of a page are numbered in reading order: blocks in order, lines in order, words in order, from 0.
`words` of a cell and `key`/`value` of a key–value pair refer to that numbering. Block indices count from 0 in `blocks`.

**Line confidence** is the words' calibrated confidences averaged with each weighted by its character count; **block
confidence** averages its lines weighted by their word counts; both are 0 without words.

## What the reader checks

Everything is checked before the structure it guards is built. Errors are `OcrFormatException` (shape, types, ranges, indices,
unknown names for `role`/`source`/`direction`, the format and version members, a byte-order mark, invalid JSON, duplicate
members, `pageCount` disagreeing with `pages`, non-finite numbers, a box with `right < left` or `bottom < top`, probabilities
outside 0..1, `cells` not starting at 0 and strictly increasing, indices out of range, spans leaving the table) and
`OcrLimitException` (document bytes, `MaxJsonDepth`, pages per call, blocks, lines, words, characters and other elements per
page, string length; see `OcrLimits`). Nothing else leaves the reader.

## Reading order

Local OCR finds text lines as ink strips, then arranges them with the algorithm of OnePDF's text layout (recursive X–Y cut
over the strips, Nagy & Seth 1984; whitespace columns; line merging; grid detection for tables), applied to OCR words instead
of PDF fragments:

- Strips are found by a recursive X–Y cut of the ink mask: a vertical gutter at least 1.25 line heights wide that runs through
  at least 2.5 lines splits columns; rows inside a column split at gaps of half a line. Ruling lines are removed from the mask
  first. A page with no gutter and no rule is read as one column exactly as a single-column page always was.
- Strips on one row from different columns join into one line when their vertical centres agree within 0.3 em.
- A horizontal cut is refused when more than one column lies in the region and a column has text on both sides of the gap
  (columns that share paragraph gaps are not interleaved).
- A run of rows whose cells align in columns is a table block: `table` is true, each line is a row, and a `Table` entry carries
  its cells. Table `confidence` is the share of rows with more than one cell and is not calibrated.
  Tables drawn with ruling lines, and tables in a detected `table` region, get spans, empty cells (`words: []`) and multi-line
  cells; see `table-structure-v1.md`. The members are the same.

Word boxes are the ink extent inside the horizontal range the recognizer aligned the word to. Character boxes tile that
range by the alignment of the network's output and carry `approximate: true`.

## Compatibility

A reader of version 1 accepts documents with additional members. Documents written by a future version 2 are rejected with
the version in the message.
