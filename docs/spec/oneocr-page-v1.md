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
- Sections of columns are cut as a whole. When the horizontal cut would run through a page with several columns, every empty band
  that crosses all of them (paragraphs that end at the same height) would cut the columns into slices read left, right, left,
  right. Consecutive bands that share an empty vertical stripe of at least 1.25 em are therefore merged back into one section,
  which is then cut into its columns. A section that would be a table is not merged.
- Narrow gutters (`LayoutParameters.Soft*`). Print gutters are about one em of the font, a little over the measured text height, so
  a gutter of 1.25 text heights is missed by a pixel or two on tightly set pages and more on blurred scans, and the page was read
  row by row across its columns (the cells of one line joined by tabs). When a region has no stripe of 1.25 text heights and no
  fragment of any of its lines crosses a stripe of at least 0.5 text heights, and the region has at least 10 distinct lines, it is
  cut there into columns if every column has at least 5 distinct lines and the widest quarter of its fragments is at least 12
  text heights wide (full lines of running text; a column of bullets, labels or table cells is not). These columns skip the
  label, table, sidebar and tied-row rules and are read one after the other, top to bottom. Regions that already had a stripe of
  1.25 text heights are cut exactly as before.
- Gutters with a few words in them (`LayoutParameters.Robust*`, `ReadingOrder.Columns.cs`). On scans a stray word, a speck read as a
  word, or a justified line that runs a few pixels into the next column keeps every stripe from being empty. When a region of at
  least 30 distinct lines still has no cut after the narrow-gutter rule, the words of the region are swept left to right counting
  how many cover each position. A stripe is a run, inside the text (not at its margins) and at least 0.2 text heights wide, where
  at most a tenth of the peak count (at least one word) lies, and where the mean count over the 3 text heights on each side is at
  least a quarter of the peak (text stands up to it on both sides; the ragged right edge of long lines, with a few longer lines
  reaching beyond, is not a gutter). Its position is the middle of the longest run at the lowest count. Words are assigned to
  columns by their centres, and a fragment that runs across a stripe is cut between its words. The columns must pass the same
  running-text test as above (5 lines, widest quarter at least 12 text heights) and be at most 1.6 times as wide as each other
  (a wider group has a gutter missing in it); if not, the stripe with the most words in it is dropped and the rest tried again,
  until none is left. When more than a tenth of the lines (at least 5) have words on both sides of a stripe, the region is not cut
  into columns here at all: a table or figure across the columns is cut off by the horizontal cut first. Regions that had a cut by
  the earlier rules are cut as before.
  *Header and footer.* Before the stripes are sought, a band of at most 5 distinct lines at the top (bottom) of the region, set
  off from the rest by an empty band of one text height (fragments of any region, not only of one detector region), is read
  before (after) the columns when it spans at least 60% of the region's width and its fragments fill at most 40% of that width on
  each line (a footer of a few labels, not the first lines of two columns). A line with words on both sides of a stripe is also a
  header (footer) when at most 5 lines lie above (below) it, or a tenth of the lines when an empty band of half a text height sets
  it off from the lines below (above); every fragment from there outwards, within half a text height of the line, goes with it.
  The header and footer groups are cut as ordinary regions.
- Specks between columns. A group of at most 2 fragments, and at most 3% of the fragments, that the 1.25 em cut leaves between
  two others in a region of at least 30 fragments (a mark in the margin, a noise word) is not a column when it lies inside the
  vertical extent of the nearer neighbour, at least 2 text heights from its top and bottom: it joins that neighbour. Without
  this it looks like a column embedded between parts of another and the page is read in bands. A menu item at the top of a screen
  is not inside anything and stays a group.
- Screens and forms (`ReadingOrder.Sections.cs`). When a region has empty vertical stripes and is not a table, three rules
  decide whether the columns are the structure, in this order:
  1. *Header row*: the topmost band is read first, whole, when it is at most 2.5 em tall, has at least two fragments, spans at
     least 60% of the region's width and is separated from the rest by at least 2.5 em (a menu bar whose last item lines up with
     the body's right column). The label row of a field grid has a smaller gap and is not taken.
  1b. *Sidebar*: the first column is read before the rest when it has at least three single-fragment lines at an even pitch (each
     gap within 30% of the median), is at most 35% of the region's width, the body starts within 25% of the width after it, and
     at most half of its lines (and half of the body's lines) share a line with the other side. A label column with its values
     has partners on both sides, or far off, and is not a sidebar. The menu bar, the sidebar and the cells of a row of fields
     are marked standalone: no detected table region merges them into a table.
  2. *Tied rows*: the region is cut at empty bands of at least 1.5 em, bands being joined across a gap of at most 0.8 of both
     neighbouring gaps (the label above its value inside one cell). If the smaller column is not the first one, has at least
     three items in at least half as many bands, and every band holding one of them also holds other text, the bands are read
     in order and each is cut into its columns: a setting then its value, a field's label and value then the next field. A narrow
     first column (a sidebar) is read before the body instead.
  3. *Embedded column*: if one column lies wholly between two parts of another (text at least 2 em above and below it), the
     columns are not the structure and the bands of rule 2 (or, without them, of half an em) are used (cards between a title
     and a list).
- Fields. A label above its value is a cell; the cells of a row of fields (equal numbers of fragments in a label line and the
  line under it, each value starting within 2 em of its label) are read one cell at a time, left to right, not all labels and
  then all values. This is applied to the last two lines of a region (under at most one single-fragment heading) and to pairs
  of bands whose gap is at most 0.8 of the gap after the pair (before it for the last pair). Labels that end in a colon
  ("Total: 45.00") are list rows, not fields.
- Rows of `label: value`. A column whose fragments mostly (60%) end in a colon is a label column. With one label column and
  one or more value columns, rows are read across even when there are only two, and uneven spacing does not make them fields.
  With several label columns (two blocks side by side) each label column and the columns up to the next form a block; the
  blocks are read one after the other, each across its rows. A run of bands is not merged into a section of columns when one
  of its columns is made of cells (one or two lines at a time, with gaps of at least 3 em between them).
- A table needs more than most of its rows having several fragments: its columns must also be filled in 60% of the rows on
  average (short blocks side by side that share a few rows are not columns), no row of fragments may be 1.6 times taller than the
  median, nor set in a text size 1.6 times that of another row (a card's large number), and the distance between consecutive lines must not alternate short and long by a factor of
  1.25 or more (a label above its value). These tests only remove tables that the earlier tests accepted.
- A run of rows whose cells align in columns is a table block: `table` is true, each line is a row, and a `Table` entry carries
  its cells. Table `confidence` is the share of rows with more than one cell and is not calibrated.
  Tables drawn with ruling lines, and tables in a detected `table` region, get spans, empty cells (`words: []`) and multi-line
  cells; see `table-structure-v1.md`. The members are the same.

Word boxes are the ink extent inside the horizontal range the recognizer aligned the word to. Character boxes tile that
range by the alignment of the network's output and carry `approximate: true`.

## Word segmentation

The recognizer emits a space class where it believes there is a gap, and it is wrong in both directions: it cuts words
(`immedi ately`, `$40, 181.65`, `2014- 02-15.`) where the ink is continuous, and it fuses words (`Iagree`) across an empty stretch.
After a line is read and before its words get boxes, the reader re-decides the spaces from the line image. Only spaces change:
every character, its probability and its extent stay as the network gave them, so numbers, amounts, dates and addresses keep
exactly the characters that were read. A word whose characters do not change keeps its edges and confidence; a word made by
joining or cutting gets text from the characters, edges halfway across the neighbouring gaps (the line's ends where the network
first and last fired), and a raw confidence that is the mean of its characters' probabilities (calibrated as any word).

**Evidence.** The scaled line image (the recognizer's input, 32 rows) gives the ink columns: the line's background is the median
pixel value; `top` is the 99.5th percentile of the distance from it; a pixel is ink when its distance exceeds `(top + 1) / 2`, and
a column is ink when any pixel in it is. A line whose `top` is under `MinContrast` (64) is left as read. An *empty stretch* is a
maximal run of non-ink columns with ink on both sides (the margins are not stretches). A stretch belongs to a pair of neighbouring
characters when it touches the span between the two characters' centres (a character's run of steps is not always inside its
glyph); `widest[p]` is the widest stretch touching pair `p`.

**Taking a space out.** A space the network emitted after character `p` is removed when `widest[p]` is small and one of:

1. *Number or date* (`widest[p] <= NumberGap` = 6): the two words are the halves of one number. The left word ends in a number
   (its longest suffix of digits, `,` `.` `/` `:` `-`, from its first digit) and the right word begins with a digit or a date
   separator, and either (a) the left number ends in `,` after `1-3 digits (,3 digits)*` and the right word begins with exactly
   three digits (`$40,` `181.65`); (b) the left number's last comma group has one or two digits and the right word begins with
   exactly the digits that complete three (`16,0` `80.65`); (c) the left number is only digits and `/ - : .` with at least two
   digits and ends in `/` or `-`, and the right word begins with a digit (`2014-` `02-15.`); (d) the left number holds a `/` or
   `-` and the right word begins with `/` or `-` followed by a digit (`24/07` `/2010.`).
2. *Letters* (only when the engine has a word model): the left word ends and the right word begins with a letter; with their
   outer non-letters set aside, neither piece has an upper-case letter other than the first letter of the left piece (capitals
   inside mark codes, abbreviations and headings, which the model cannot judge); when the line has at least `MinSpacesForGap` (4)
   network spaces, `widest[p]` is at most `ModelRatio` (0.7) times the median stretch at those spaces; with `Score` the word
   score of the character model below, `gain = Score(joined) - (Score(left) + Score(right))`, and the space goes when either
   - `widest[p] <= ModelGap` (5) and `gain >= ModelMargin` (2.0 nats): plain evidence; or
   - `widest[p] <= ModelGapWeak` (6) and `gain >= ModelMarginWeak` (1.0 nat) and the lower of `Score(left)` and `Score(right)` is at
     most `WeakPieceMax` (-2.0): weak evidence, enough when one piece is clearly no word (`Janu` + `ary`, `Fieldn` + `ote`). A
     piece that is a rare real word (`tax` scores about -1.7) is not bad enough, and neither are two pieces that both score as
     words: they need the plain evidence.

   A piece the model has no score for (digits, code symbols, letters outside the script) never joins: the code-word rule.
3. *Word frequencies* (only when the engine has a word frequency table, `word-lm-v1.md`): the same piece rules as 2 (letters at
   the cut, outer marks set aside, no capital inside a piece other than the first letter of the left piece) and, in addition, no
   digit and no code symbol (`@ # _ / \ = % + * ~ ^ | < > [ ] { }`) anywhere in either word, the marks around the letters
   included. Both pieces are lowercased by the table's folding, and `a`, `b` and `ab` are looked up (`q` is the stored quarter-bits
   of a count; a missing word has `q_missing`). The space goes when `widest[p] <= FrequencyGap` (9), the stretch is at most
   `FrequencyRatio` (0.85) times the line's median word gap (on lines with at least `MinSpacesForGap` network spaces), `ab` is in
   the table with `q(ab) >= FrequencyMinQuarter` (10), and

       q(ab) + q_total - q(a) - q(b)  >=  FrequencyMargin (28) + 24 if a piece is one letter + 16 if both pieces have q >= 24

   The extras keep out pairs of real words that happen to make a word (`dare` `say`, `a` `gain`). Integer arithmetic only.
   Rules 1 to 3 are tried in that order and any one of them takes the space out.

**Putting a space in.** A stretch that no network space accounts for (no pair it touches has a network space) becomes a space at
the pair it overlaps most when its width is at least `max(SplitGap, SplitRatio x letterGap)` (`SplitGap` = 9 pixels,
`SplitRatio` = 3.5, `letterGap` the median `widest` over the line's pairs inside words, so letter-spaced print is not cut) and
the two characters may be separated: both sides are letters, digits or marks that can end or start a word (a closing mark `. , ; :
! ? ) ] } % ' " » …` and its quotes may end a word, a currency sign or an opening bracket or quote may start one),
not two digits, not a point or comma that follows a digit, and not inside a word that contains `@` or `//`.

**Word score.** `Score(letters)` (letters only) is the sum, over each letter and the word end, of the character model's
`LogProb(context, symbol) - MeanLogProb` floored at the decoder's floor (-1), with the case-folded symbols of the script that
covers the most of the word's letters (the first on ties; no score when it does not cover all of them) and the word start as the
first context symbol, as in `decoding.md`, in `double` in that order. A typical word scores about zero. The word model is the
model file that ships with the models package (`charlm-v1`); an engine without one skips rule 2 above and nothing else.

The frequency rule's parameters were chosen on the held-out tune sets (thirteen sets of generated and real-web pages) and checked on
five newly generated sets that were not looked at while choosing; no set got worse in CER, WER or reading order.

The first parameters were chosen on pages that are not part of the benchmark sets (a degraded-scan set and a screenshot set
rendered with new seeds). The weak-evidence values were chosen on the held-out tune sets, by counting joins that the truth text
confirms against joins of two real words over every candidate space the engine met there; the benchmark sets were counted in
the same tables as information and agree. `OcrOptions.FixWordSpacing = false` turns the whole step off. Pages from a PDF text layer or an LLM are not
re-segmented.

## Digits and the letter O

After word breaks are settled, the one step that changes a character: the recognizer reads `0` as `O` in identifiers and dates
(`ACC-71O3O5`, `2020-04-O7`), and the neighbours of the letter settle which it is. For each word:

1. A word with fewer than `MinDigits` (3) ASCII digits is left as read (`O2`, `No.7`, `COVID`).
2. The word is cut into runs of ASCII letters and digits (the pieces between punctuation). A run that holds only digits and the
   capital letter `O`, with at least one digit and at least one `O`, has each `O` replaced by `0`. A run with any other letter
   (`ACC`, `CO2`, `COVID19`) or without a digit (`OO`) is left alone, and so is a run that is not separated from letters by
   punctuation (`ACC71O3O5`).

Characters keep their extents and probabilities; the word keeps its confidence and box. Only the text of the word and of the
changed characters differ. The rule is deliberately narrow: over about 100,000 truth words of the benchmark and tune sets
(receipts included, with their many `0`s) it changes none, and on the engine's output it changes five words, all
corrections. It was written from the three `O`-for-`0` errors of two benchmark pages and checked on the held-out tune sets, which hold two more.
`OcrOptions.FixDigitLookalikes = false` turns it off. Beam decoding with the character model keeps words that contain digits
exactly as greedy decoding reads them (`decoding.md`, "code-word rule"); this step runs afterwards on either result, so the two
do not conflict: the code-word rule decides what the decoder may change, this step is a separate post-recognition fix. The reverse
(`l` or `I` for `1`) is not handled: the sets hold no such error.

## Separators

The recognizer often reads the middle dot, the bullet and the vertical bar between menu items, breadcrumbs and bylines as a hyphen
or another small mark (`By Ana Costa - 3/2/2021 - 4 min read`). After word breaks are settled, each one-character word between two
other words of a line is judged again from the pixels of the scaled line image. Like the digit step, this changes characters only;
the word keeps its box, confidence and per-character extents, and nothing is added or removed. All arithmetic is on integers, so the
result is the same on every path and at every degree.

**Candidates.** A word of exactly one character, neither first nor last in its line, read as one of
`- – — . , ' * ° ~ _ : ; ^ + = ¨ ˙ ‧ ∙ ⋅ ● ○ ▪ ◦ ¤ º ª` or the grave accent (a dot),
`·` or `•` (a dot the network already decided), or `I l 1 ! ¦ / \ ( ) [ ]` (a bar). Other letters and digits are never candidates for a
dot, and a word of two or more characters (`-x`, `--`, `a-b`) is never touched, so hyphens and dashes inside words stay.

**The glyph.** Ink pixels are those the word segmenter uses (distance from the line's median grey above half the line's strongest
contrast; lines fainter than `MinContrast` are left as read). The glyph is the run of ink columns that contains the first ink column
of the word's extent, grown over every adjacent ink column, so an extent that cuts the glyph short does not matter. A run wider than
the line image is high is not a separator. In it, the largest 8-connected piece must hold at least 80% of the ink; its bounding box
is `w` by `h` with `area` ink pixels. The reference is the line's other ink columns, leaving out the columns of all candidates (a row
of bars is no reference for itself); at least 8 are needed. `L` is the median over them of the column's ink height (top to bottom
ink row), `mid2` the median of top + bottom, and `tallest` the largest ink height.

**A dot or bullet** (the candidate is not a bar mark) when all hold:

- `w` and `h` are at least 1; `45 <= 100 w / h <= 160`; or, for a blob at most 25% of `L` wide, `w <= 2 h`. A hyphen or dash the network read
  with confidence 0.8 or more needs `100 w / h <= 130`: a short hyphen and a dot look alike, and the network is usually right
  about a hyphen it is sure of.
- `area >= 0.55 w h`.
- The blob's centre (`2 * top + h - 1`, twice the row) is within 25% of `L` of `mid2`.
- The diameter `(w + h) / 2` is between 10% and 90% of `L`.

The result is `•` when the diameter is at least 38% of `L`, else `·`. A word the network read as `·` becomes `•` only from
45%; one read as `•` becomes `·` only up to 28%; otherwise the network's own mark stays.

**A bar** (`|`) when `w <= max(2, 0.15 h)`, `area >= 0.8 w h`, `h >= 1.12 *` the tallest other ink column and `h` is at least half the
line image's height. A capital `I` or a `1` is not as tall as a vertical bar, which reaches from above the ascenders to below the
baseline, and is thicker.

The constants are in `SeparatorGlyphs` and were set on held-out screenshot, layout and generated interface-text pages; they are
not tuned on the benchmark. `OcrOptions.FixSeparators = false` turns the step off.

## Noise filter

Before lines are arranged, the reader drops detections that are noise read as text: specks, scan grain and edges that were cut as
tiny lines and read as a few unsure characters, and lone dots and ticks. Two rules work on the recognized strips (one strip is
one detected line, its height is the height of its ink box).

*Small unsure words:*

1. A strip is a reference line when the mean calibrated confidence of its words is at least `AnchorConfidence`. The reference
   height is the median ink height of the reference lines. With fewer than `MinAnchorLines` reference lines the page is left
   alone: there is no text to be small against.
2. A strip is small when its height is under `min(RelativeHeight x reference, MaxHeight)` pixels. Small is relative to the
   page's own text, so a page set entirely in small type keeps everything; the pixel limit keeps proportionally small text on
   high-resolution pages.
3. In a small strip every word whose calibrated confidence is under `MaxConfidence` is dropped. In a tiny strip (height under
   `min(RelativeHeight x reference, TinyHeight)`: specks and grain a few pixels high) the bar is `TinyConfidence` instead,
   because such strips hold no real text but the recognizer reads a lone punctuation mark in them with middling confidence. A strip
   left with no words is dropped. Words in other strips are never touched by this rule, whatever their confidence.

*Dots and ticks* (after the first rule, in strips of any height): a word made only of the characters . , : ; ' " ` ´ ‘ ’ “ ” (a
speck read as a mark) is dropped when its calibrated confidence is under `SpeckConfidence`, but only on a page that has at least
`SpeckPageMin` such words: dust comes in numbers, while a clean page has none or a stray one, and the stray one may be a real mark.
Dashes, bullets, middle dots and other separators are words of their own on real pages and are never dropped by this rule.

*Controls* (any strip height): radio buttons and checkboxes are drawn rings and squares that the detector finds as lines of their
own and the recognizer reads as a letter or a letter and a bracket. A word whose text is one character from
`O Q L D C [` or such a character followed by one from `] ) [ l I |` is dropped when its calibrated confidence is under
`ControlConfidence` and it is alone in its strip, or first or last in it with a gap to its neighbour of at least half its own
height (a control sits apart from its label; a letter in running text is a word space away). A letter read with confidence (a
grade, a column label) stays, as does any word with a letter outside this set (`Do`, `OK`) and a digit.

The parameters are `MaxConfidence` = 0.5, `RelativeHeight` = 0.6, `MaxHeight` = 12 px, `TinyHeight` = 8 px, `TinyConfidence` = 0.99,
`AnchorConfidence` = 0.9, `MinAnchorLines` = 3, `SpeckConfidence` = 0.9, `SpeckPageMin` = 3, `ControlConfidence` = 0.9. The control rule's glyph set and bar were read off the unmatched words of two benchmark sets (form and
dialog screenshots, form pages) and then checked on held-out screenshot and layout sets rendered with other seeds, where the same
glyphs appear. The other values were chosen on pages that are
not part of the benchmark sets: a degraded-scan set and a screenshot set rendered with new seeds, and the validation split of the
generated page set. The dropped words never reach reading order, tables, `text` or the JSON; the page gets one diagnostic stating
how many words were dropped. `OcrOptions.DropNoise = false` turns the filter off. Pages from a PDF text layer or an LLM are not
filtered.

## Compatibility

A reader of version 1 accepts documents with additional members. Documents written by a future version 2 are rejected with
the version in the message.
