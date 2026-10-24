using System.Text;

namespace OneOcr;

/// <summary>Where an element's content came from.</summary>
public enum OcrSource
{
    /// <summary>Read from pixels by the local recognizer.</summary>
    Ocr = 0,

    /// <summary>Taken from the text layer of a PDF.</summary>
    TextLayer = 1,

    /// <summary>Read by a language model.</summary>
    Llm = 2,
}

/// <summary>
/// An axis-aligned rectangle in page units: origin at the top-left of the page, x to the right, y down.
/// </summary>
/// <param name="Left">Smallest x.</param>
/// <param name="Top">Smallest y.</param>
/// <param name="Right">Largest x.</param>
/// <param name="Bottom">Largest y.</param>
public readonly record struct OcrBox(double Left, double Top, double Right, double Bottom)
{
    /// <summary>Right minus left.</summary>
    public double Width => Right - Left;

    /// <summary>Bottom minus top.</summary>
    public double Height => Bottom - Top;

    /// <summary>The smallest box that holds both boxes.</summary>
    /// <param name="a">One box.</param>
    /// <param name="b">The other box.</param>
    /// <returns>The union.</returns>
    public static OcrBox Union(OcrBox a, OcrBox b) =>
        new(Math.Min(a.Left, b.Left), Math.Min(a.Top, b.Top), Math.Max(a.Right, b.Right), Math.Max(a.Bottom, b.Bottom));
}

/// <summary>A point in page units (top-left origin, y down).</summary>
/// <param name="X">Horizontal position.</param>
/// <param name="Y">Vertical position, growing downward.</param>
public readonly record struct OcrPoint(double X, double Y);

/// <summary>A quadrilateral for rotated or skewed text, corners in the order top-left, top-right, bottom-right, bottom-left of the text's own frame.</summary>
/// <param name="P0">Top-left corner of the text's own frame.</param>
/// <param name="P1">Top-right corner.</param>
/// <param name="P2">Bottom-right corner.</param>
/// <param name="P3">Bottom-left corner.</param>
public readonly record struct OcrQuad(OcrPoint P0, OcrPoint P1, OcrPoint P2, OcrPoint P3);

/// <summary>
/// The affine map from page units (top-left origin, y down) back to PDF user space: <c>x' = A*x + C*y + E</c>,
/// <c>y' = B*x + D*y + F</c>. Present only on pages that came from a PDF.
/// </summary>
/// <param name="A">Matrix entry a.</param>
/// <param name="B">Matrix entry b.</param>
/// <param name="C">Matrix entry c.</param>
/// <param name="D">Matrix entry d.</param>
/// <param name="E">Matrix entry e.</param>
/// <param name="F">Matrix entry f.</param>
public readonly record struct OcrMatrix(double A, double B, double C, double D, double E, double F);

/// <summary>The part a block plays in the page furniture.</summary>
public enum OcrBlockRole
{
    /// <summary>Main content.</summary>
    Body = 0,

    /// <summary>Running header.</summary>
    Header = 1,

    /// <summary>Running footer.</summary>
    Footer = 2,
}

/// <summary>What a layout region is.</summary>
public enum OcrRegionRole
{
    /// <summary>Running text.</summary>
    Text = 0,

    /// <summary>A title or heading.</summary>
    Title = 1,

    /// <summary>A list.</summary>
    List = 2,

    /// <summary>A table.</summary>
    Table = 3,

    /// <summary>A figure or picture.</summary>
    Figure = 4,

    /// <summary>A running header.</summary>
    Header = 5,

    /// <summary>A running footer.</summary>
    Footer = 6,

    /// <summary>A fillable form field.</summary>
    FormField = 7,

    /// <summary>A check box.</summary>
    Checkbox = 8,
}

/// <summary>One recognized character. Boxes of recognized characters are approximate.</summary>
/// <param name="Text">The character's text (one charset entry, usually one code point).</param>
/// <param name="Box">Where the character is, in page units.</param>
/// <param name="Confidence">The recognizer's probability for this character, 0 to 1.</param>
/// <param name="Source">Where the character came from.</param>
/// <param name="BoxIsApproximate">
/// True when <paramref name="Box"/> comes from the alignment of the network's output rather than from the glyph's outline:
/// the horizontal edges are where the network fires, the vertical edges those of the word.
/// </param>
/// <param name="Quad">The corners, for rotated text; null for upright text.</param>
public sealed record OcrChar(
    string Text, OcrBox Box, float Confidence, OcrSource Source = OcrSource.Ocr, bool BoxIsApproximate = true, OcrQuad? Quad = null);

/// <summary>One word.</summary>
/// <param name="Text">The word's text in logical order; never empty.</param>
/// <param name="Box">The ink extent of the word in page units for recognized words.</param>
/// <param name="Confidence">
/// Calibrated probability in 0..1 that the word is exactly right. Equal to <paramref name="RawConfidence"/> when the model
/// carries no calibration or the word did not come from a recognizer.
/// </param>
/// <param name="RawConfidence">Mean over the word's characters of the largest softmax probability seen while that character was the winning class.</param>
/// <param name="Source">Where the word came from.</param>
/// <param name="Chars">The word's characters left to right; empty when they were not kept.</param>
/// <param name="Quad">The corners, for rotated text; null for upright text.</param>
public sealed record OcrWord(
    string Text, OcrBox Box, float Confidence, float RawConfidence, OcrSource Source, IReadOnlyList<OcrChar> Chars, OcrQuad? Quad = null);

/// <summary>One line of text. A line may consist of several cells: runs set apart by a wide gap, for example the cells of a table row.</summary>
/// <param name="Text">The words joined by one space, cells joined by a tab.</param>
/// <param name="Words">The words in reading order.</param>
/// <param name="Cells">Indices into <paramref name="Words"/> where each cell starts; the first is 0. Empty for a line with no words.</param>
/// <param name="Box">The line's box.</param>
/// <param name="Confidence">The calibrated confidences of the words, averaged with each word weighted by its character count; 0 without words.</param>
/// <param name="Source">Where the line came from.</param>
/// <param name="IsRightToLeft">True when the base direction is right to left. Always false for lines read by the local recognizer.</param>
/// <param name="IsVertical">True for vertical writing. Always false for lines read by the local recognizer.</param>
/// <param name="Quad">The corners, for rotated text; null for upright text.</param>
public sealed record OcrLine(
    string Text, IReadOnlyList<OcrWord> Words, IReadOnlyList<int> Cells, OcrBox Box, float Confidence, OcrSource Source,
    bool IsRightToLeft = false, bool IsVertical = false, OcrQuad? Quad = null);

/// <summary>A run of lines that read together: a paragraph, a column of text or a table.</summary>
/// <param name="Role">Body, header or footer.</param>
/// <param name="IsTable">True for a grid of short cells, whose lines are rows.</param>
/// <param name="Box">The block's box.</param>
/// <param name="Lines">The lines in reading order.</param>
/// <param name="Confidence">The lines' confidences averaged with each line weighted by its word count; 0 without words.</param>
/// <param name="Source">Where the block came from.</param>
/// <param name="Quad">The corners, for rotated text; null for upright text.</param>
public sealed record OcrBlock(
    OcrBlockRole Role, bool IsTable, OcrBox Box, IReadOnlyList<OcrLine> Lines, float Confidence, OcrSource Source = OcrSource.Ocr, OcrQuad? Quad = null);

/// <summary>A layout region with a role, as found by a layout model. The local pipeline does not fill regions yet.</summary>
/// <param name="Role">What the region is.</param>
/// <param name="Box">The region's box.</param>
/// <param name="Confidence">Probability that the role is right, 0 to 1.</param>
/// <param name="Source">Where the region came from.</param>
/// <param name="BlockIndices">Indices into the page's blocks that lie in the region.</param>
/// <param name="Quad">The corners, for rotated regions; null otherwise.</param>
public sealed record OcrRegion(
    OcrRegionRole Role, OcrBox Box, float Confidence, OcrSource Source, IReadOnlyList<int> BlockIndices, OcrQuad? Quad = null);

/// <summary>A table cell.</summary>
/// <param name="Row">Zero-based row of the cell's top edge.</param>
/// <param name="Column">Zero-based column of the cell's left edge.</param>
/// <param name="RowSpan">Rows the cell covers, at least 1.</param>
/// <param name="ColumnSpan">Columns the cell covers, at least 1.</param>
/// <param name="Box">The cell's box.</param>
/// <param name="WordIndices">Indices into the page's reading-order word sequence (<see cref="OcrPage.Words"/>), left to right.</param>
public sealed record OcrTableCell(int Row, int Column, int RowSpan, int ColumnSpan, OcrBox Box, IReadOnlyList<int> WordIndices);

/// <summary>A table: cells with row and column spans that refer to the page's words.</summary>
/// <param name="Box">The table's box.</param>
/// <param name="Rows">Number of rows.</param>
/// <param name="Columns">Number of columns.</param>
/// <param name="Cells">The cells in row-major order.</param>
/// <param name="Confidence">Probability that this is a table, 0 to 1.</param>
/// <param name="Source">Where the table structure came from.</param>
/// <param name="BlockIndex">Index of the page block that holds the table's text, or -1 when there is none.</param>
/// <param name="Quad">The corners, for a rotated table; null otherwise.</param>
public sealed record OcrTable(
    OcrBox Box, int Rows, int Columns, IReadOnlyList<OcrTableCell> Cells, float Confidence, OcrSource Source, int BlockIndex = -1, OcrQuad? Quad = null);

/// <summary>A key and the value it labels, both as words of the page.</summary>
/// <param name="KeyWordIndices">Indices into <see cref="OcrPage.Words"/> of the key's words.</param>
/// <param name="ValueWordIndices">Indices into <see cref="OcrPage.Words"/> of the value's words.</param>
/// <param name="Box">The box around key and value.</param>
/// <param name="Confidence">Probability that the pairing is right, 0 to 1.</param>
/// <param name="Source">Where the pairing came from.</param>
public sealed record OcrKeyValue(
    IReadOnlyList<int> KeyWordIndices, IReadOnlyList<int> ValueWordIndices, OcrBox Box, float Confidence, OcrSource Source);

/// <summary>
/// One page as structure: blocks in reading order, each with lines, words and characters, plus layout regions, tables and
/// key–value pairs. Coordinates are page units with the origin at the top-left and y growing downward: pixels for images,
/// points for PDF pages. Immutable once built; the JSON form is described in <c>docs/spec/oneocr-page-v1.md</c>.
/// </summary>
public sealed class OcrPage
{
    private OcrWord[]? _words;

    /// <summary>Builds a page.</summary>
    /// <param name="index">Zero-based index of the page in its document.</param>
    /// <param name="width">Page width in page units.</param>
    /// <param name="height">Page height in page units.</param>
    /// <param name="unitsPerInch">Page units per inch: the image's resolution, or 72 for PDF pages.</param>
    /// <param name="blocks">Blocks in reading order.</param>
    /// <param name="regions">Layout regions; null for none.</param>
    /// <param name="tables">Tables; null for none.</param>
    /// <param name="keyValues">Key–value pairs; null for none.</param>
    /// <param name="diagnostics">Notes about the page; null for none.</param>
    /// <param name="toPdf">For pages from a PDF, the map back to PDF user space; otherwise null.</param>
    /// <param name="orientation">How far the content is turned clockwise from upright: 0, 90, 180 or 270.</param>
    /// <param name="skewDegrees">The tilt of the text lines in degrees, positive when they descend to the right; within 45.</param>
    /// <exception cref="ArgumentException"><paramref name="orientation"/> is not a quarter turn or <paramref name="skewDegrees"/> is out of range.</exception>
    public OcrPage(
        int index, double width, double height, double unitsPerInch, IReadOnlyList<OcrBlock> blocks,
        IReadOnlyList<OcrRegion>? regions = null, IReadOnlyList<OcrTable>? tables = null, IReadOnlyList<OcrKeyValue>? keyValues = null,
        IReadOnlyList<string>? diagnostics = null, OcrMatrix? toPdf = null, int orientation = 0, double skewDegrees = 0)
    {
        ArgumentNullException.ThrowIfNull(blocks);
        if (orientation is not (0 or 90 or 180 or 270))
        {
            throw new ArgumentException("Orientation must be 0, 90, 180 or 270.", nameof(orientation));
        }

        if (!(Math.Abs(skewDegrees) <= 45))
        {
            throw new ArgumentException("SkewDegrees must be within 45 degrees.", nameof(skewDegrees));
        }

        Index = index;
        Width = width;
        Height = height;
        UnitsPerInch = unitsPerInch;
        Blocks = blocks;
        Regions = regions ?? [];
        Tables = tables ?? [];
        KeyValues = keyValues ?? [];
        Diagnostics = diagnostics ?? [];
        ToPdf = toPdf;
        Orientation = orientation;
        SkewDegrees = skewDegrees;
    }

    /// <summary>Zero-based index of the page in its document.</summary>
    public int Index { get; }

    /// <summary>Page width in page units.</summary>
    public double Width { get; }

    /// <summary>Page height in page units.</summary>
    public double Height { get; }

    /// <summary>
    /// Page units per inch. For an image, the resolution the caller gave or, when none was given, an estimate from the size of
    /// the text (a diagnostic says so); 72 for PDF pages.
    /// </summary>
    public double UnitsPerInch { get; }

    /// <summary>The map from page units back to PDF user space; null unless the page came from a PDF.</summary>
    public OcrMatrix? ToPdf { get; }

    /// <summary>
    /// How far the page content is turned clockwise from upright when the image was read: 0, 90, 180 or 270 (90 means the tops of
    /// the lines point to the right). Coordinates stay in the image as given; rotated elements carry a quad. 0 for PDF pages.
    /// </summary>
    public int Orientation { get; }

    /// <summary>
    /// The tilt in degrees of the text lines that was removed before reading, positive when lines descend to the right (clockwise
    /// in y-down coordinates). 0 when the page was not tilted or the tilt was too small to act on.
    /// </summary>
    public double SkewDegrees { get; }

    /// <summary>Blocks in reading order.</summary>
    public IReadOnlyList<OcrBlock> Blocks { get; }

    /// <summary>Layout regions with roles; empty until a layout model fills them.</summary>
    public IReadOnlyList<OcrRegion> Regions { get; }

    /// <summary>Tables found on the page.</summary>
    public IReadOnlyList<OcrTable> Tables { get; }

    /// <summary>Key–value pairs found on the page; empty until a model fills them.</summary>
    public IReadOnlyList<OcrKeyValue> KeyValues { get; }

    /// <summary>Notes about this page, for example that the resolution was estimated.</summary>
    public IReadOnlyList<string> Diagnostics { get; }

    /// <summary>
    /// Every word of the page in reading order (block by block, line by line, left to right). Table cells, regions and
    /// key–value pairs refer to words by their index in this sequence.
    /// </summary>
    public IReadOnlyList<OcrWord> Words => _words ??= CollectWords();

    /// <summary>
    /// The page's text in reading order: lines joined by <c>\n</c>, blocks by <c>\n\n</c>, table cells (and other cells) by <c>\t</c>.
    /// </summary>
    public string Text
    {
        get
        {
            var sb = new StringBuilder();
            bool firstBlock = true;
            foreach (OcrBlock block in Blocks)
            {
                if (block.Lines.Count == 0)
                {
                    continue;
                }

                if (!firstBlock)
                {
                    sb.Append("\n\n");
                }

                firstBlock = false;
                for (int i = 0; i < block.Lines.Count; i++)
                {
                    if (i > 0)
                    {
                        sb.Append('\n');
                    }

                    sb.Append(block.Lines[i].Text);
                }
            }

            return sb.ToString();
        }
    }

    /// <summary>The page's lines in reading order, joined by <c>\n</c> (blocks are not set apart; cells are joined by a tab).</summary>
    /// <returns>The text; empty when the page has no lines.</returns>
    public string ToText()
    {
        var sb = new StringBuilder();
        foreach (OcrBlock block in Blocks)
        {
            foreach (OcrLine line in block.Lines)
            {
                if (sb.Length > 0)
                {
                    sb.Append('\n');
                }

                sb.Append(line.Text);
            }
        }

        return sb.ToString();
    }

    private OcrWord[] CollectWords()
    {
        var list = new List<OcrWord>();
        foreach (OcrBlock block in Blocks)
        {
            foreach (OcrLine line in block.Lines)
            {
                list.AddRange(line.Words);
            }
        }

        return [.. list];
    }
}
