# Local Markdown, version 1

`OcrMarkdown.ToMarkdown(page, options)` (and `image.ToMarkdown()`, and the same on PDF pages and documents) writes a page of the
page model as Markdown without a model or a network. The text is a function of the page and the options only: no clock, no random
choice, no culture. It is written for CommonMark with pipe tables (GFM).

## Elements

The blocks of the page are visited in reading order and each gives zero or more elements, separated by one blank line. The output
ends with one line feed, or is empty when nothing was written. Pages of a sequence are separated by a thematic break (`---`)
between pages that have content.

Roles come from the page: `Block.Role` (header, footer), the first region that lists the block in `BlockIndices` (title, text,
list, table, figure, header, footer, …), and table entries. A page from the local reader has none of them yet; everything below
works without them, using sizes and positions.

1. **Furniture is dropped.** A block with the header or footer role, or in a header or footer region, is left out unless
   `IncludeHeadersAndFooters`. With `DropPageNumbers` (default), a block of one line that is only a page number (`7`, `Page 7`,
   `7 of 12`, `7 / 12`, `- 7 -`, one to five digits) and lies in the top or bottom 8% of the page is left out too. Not done on pages
   that were turned or straightened.
2. **Tables.** A table entry whose `BlockIndex` names the block is written as a pipe table in the block's place; tables without a
   block go before the first element that starts below them. Cell text is the cell's words (by `WordIndices`) joined by a space.
   The first row is the header row. **Spans:** a cell's text goes into its top-left position; the other positions it covers stay
   empty. Columns in which every cell is empty are left out. `|` in cell text is written `\|`. A block flagged as a table that has no
   table entry becomes a table of its lines, cut into cells at `Line.Cells`.
3. **Headings.** A block in a title region is a heading. Without a region, a block of at most two lines, at most 160 characters, whose
   median line height is at least 1.25 times the page's body line height, that does not end with `,` or `;` and does not start with
   a list marker, is a heading (`HeadingsFromSize`; blocks that sit in any other region never are). The body line height is the
   median over the lines of body blocks that are not tables, titles, headers or footers. **Levels** follow size: the heading heights of
   the page are grouped (a height within 8% below the group's largest is in it); the largest group is `#`, the next `##`, down to
   `######`. Line height is the length of the quad's left edge for turned text, otherwise the box height.
4. **Lists.** A line that starts with a bullet (`•`, `●`, `○`, `◦`, `▪`, `▫`, `■`, `□`, `‣`, `⁃`, `∙`, `·`, which may be followed
   directly by the text), or `-`, `*`, `+`, `–` followed by a space, or one to three digits followed by `.` or `)` and a space (or
   the same in parentheses), starts a list item. Bullets are written `- `; numbers are kept as printed and written `N. `. A line
   without a marker that is indented by at least 0.3 line heights past the marker's left edge, and no more than 1.5 body line heights
   below the item, continues the item (joined with a space). **Nesting:** an item whose left edge is more than 0.8 line heights to
   the right of the current level's starts a new level; one more than 0.8 to the left returns to a level that fits. Each level
   indents by four spaces. Pages with turned text have one level. Lettered and Roman markers are not recognised.
5. **Paragraphs.** The remaining lines of a block make paragraphs: lines are joined with one space (cells of a line, which are
   separated by a tab, by one space as well), and a vertical gap of more than 0.75 body line heights between two lines starts a new
   paragraph. A hyphen that ends a line and follows a letter is removed, without a space, when the next line starts with a lower-case
   letter (a hyphenation break); otherwise lines join with a space.
6. **Escaping.** `\ ` * _ [ ] < >` are escaped with a backslash wherever they occur. At the start of a paragraph, `# + - = ~` are
   escaped too, and the `.` or `)` of a leading `123. ` / `123) `. Headings and list item text escape the first set only.

## What it does not do

No emphasis (bold, italics), links, images, code blocks, footnotes, math, definition lists or lettered lists. Figures contribute
only the text read inside them, as paragraphs. Table cells hold one paragraph of text (no breaks). Multi-column text is whatever
reading order gave: one column after the other.
