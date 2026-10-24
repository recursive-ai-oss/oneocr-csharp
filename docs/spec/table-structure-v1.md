# Table structure v1

How the local reader turns a table on an image page into a `Table` of the page model: cells with row and column spans, empty cells,
and the words in each cell. No member of `oneocr.page` v1 changes; this document says what the existing members hold for a table
the reader recognized. It is deterministic: integer pixel arithmetic and fixed tie rules, independent of kernel path, degree and
operating system.

## Ruled tables

A table drawn with ruling lines is read from the lines, before any text finding, on the page's own binarization (Otsu; ink is the
smaller class).

1. **Runs.** One pass over the page collects horizontal and vertical runs of ink at least 24 pixels long. More than 2^17 runs of
   either direction means the page is noise, and no table is looked for.
2. **Lines.** Runs in neighbouring rows (or columns) that overlap by at least 60 % of the shorter one join into a line. A line is a
   rule when it is at most 12 pixels thick and at least 10 times longer than thick (the strokes of large letters are not slender
   enough). At most 4096 lines per direction are considered.
3. **Grids.** Rules that touch (the end of one within `max(3, 2 × thickness)` pixels of the other) form a component. Rule positions
   closer together than `max(3, 2 × thickness)` are one boundary; a boundary's position is the mean of its rules' centres. A
   component is a table when it has at least two rows and two columns of at most 2^16 cells in all, every row and column is at
   least 16 pixels, at least two rules each way run 60 % of the table's width (height), and at most 40 % of the pixels in its box
   are ink (a light panel read as ink is not a table).
4. **Cells.** The grid has one elementary cell per row and column. An edge between two neighbouring elementary cells is *drawn*
   when rules at its boundary cover at least 60 % of the edge, ignoring `thickness + 1` pixels at each end. Elementary cells whose
   shared edge is not drawn are one cell; a group that is not a rectangle falls apart into elementary cells. That gives the cell's
   `row`, `column`, `rowSpan` and `columnSpan`. A cell's `bounds` are the rectangle between its boundaries.
5. **Reading.** The rules are painted out of the page (each rule pixel and its fringe takes the paper value next to it), text lines
   that cross a drawn vertical rule are cut at the rule, and every piece is read on its own, so a word is never read across a
   column. Text belongs to the cell that holds the centre of its fragment.

The table's block holds one line per row that has text; a line's cells are the table cells with text in that row, left to right
(`Line.Cells` marks where each cell's words start); a cell with several lines of text lists them top to bottom, joined by a space in
the line's text; a cell with a span is listed once, in its first row. `Table.Rows` and `Columns` are the grid's; `Cells` lists every
cell in row-major order, empty ones with `words: []`. `Table.Box` is the grid's box; `Confidence` is 0.95 (not calibrated);
`Source` is `ocr`.

## Tables without a rule grid

For a table block the reading order found (rows with several aligned fragments), and for the blocks inside a detector region of
class `table` (merged into one table whose rows are the page's text lines, so that columns found as separate blocks read row by
row):

- **Columns** are the stretches of text between separators. A separator is a stretch of at least half a text height that is free of
  text in all but a quarter of the rows (a spanning header crosses it). A table needs two columns; with fewer the block is left as
  the reading order made it. A learned source of separators can replace this analysis (`ITableSeparators`, internal); none ships.
- **Rows** are the text lines. A cell that wraps onto a second line is two rows.
- **Cells.** A fragment belongs to a column when it covers at least 30 % of the smaller of itself and the column's text; a fragment
  that belongs to several columns is a cell with a column span. Fragments that reach the same column in one row are one cell. A
  column with nothing in a row is an empty cell (`words: []`) over the column's extent.
- `Confidence` is the share of rows with more than one cell, as before. The text of the page does not change.

A word that the recognizer read across a column gap stays one word and goes to one cell. Words are never split to fit a grid.

## Order and downstream

Reading order inside a table is row by row, cell by cell, header first. The table block takes the place in the page's block order
of the first block below the table that overlaps it horizontally. `OcrPage.ToText` joins the cells of a line with a tab; Markdown
writes the `Table` entry as a pipe table (spans leave the covered cells empty).
