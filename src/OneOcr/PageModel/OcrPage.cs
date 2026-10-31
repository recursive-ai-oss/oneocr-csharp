using System.Diagnostics.CodeAnalysis;
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

/// <summary>How <see cref="OcrPage.ToText(OcrTextLayout)"/> joins the blocks of a page.</summary>
public enum OcrTextLayout
{
    /// <summary>Every line on its own line, blocks run together.</summary>
    Lines = 0,

    /// <summary>A blank line between blocks.</summary>
    Paragraphs = 1,
}

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

/// <summary>One recognized character. Boxes of recognized characters are approximate. Records compare by value (collections element by element).</summary>
public sealed record OcrChar
{
    /// <summary>Creates an empty character; set the required members.</summary>
    public OcrChar()
    {
    }

    [SetsRequiredMembers]
    internal OcrChar(string text, OcrBox box, float confidence, OcrSource source = OcrSource.Ocr, bool boxIsApproximate = true, OcrQuad? quad = null)
    {
        Text = text;
        Box = box;
        Confidence = confidence;
        Source = source;
        BoxIsApproximate = boxIsApproximate;
        Quad = quad;
    }

    /// <summary>The character's text (one charset entry, usually one code point).</summary>
    public required string Text { get; init; }

    /// <summary>Where the character is, in page units.</summary>
    public required OcrBox Box { get; init; }

    /// <summary>The recognizer's probability for this character, 0 to 1.</summary>
    public required float Confidence { get; init; }

    /// <summary>Where the character came from. Default <see cref="OcrSource.Ocr"/>.</summary>
    public OcrSource Source { get; init; } = OcrSource.Ocr;

    /// <summary>
    /// True (the default) when <see cref="Box"/> comes from the alignment of the network's output rather than from the glyph's outline:
    /// the horizontal edges are where the network fires, the vertical edges those of the word.
    /// </summary>
    public bool BoxIsApproximate { get; init; } = true;

    /// <summary>The corners, for rotated text; null for upright text.</summary>
    public OcrQuad? Quad { get; init; }
}

/// <summary>One word. Records compare by value (collections element by element).</summary>
public sealed record OcrWord
{
    /// <summary>Creates an empty word; set the required members.</summary>
    public OcrWord()
    {
    }

    [SetsRequiredMembers]
    internal OcrWord(string text, OcrBox box, float confidence, float rawConfidence, OcrSource source, IReadOnlyList<OcrChar> chars, OcrQuad? quad = null)
    {
        Text = text;
        Box = box;
        Confidence = confidence;
        RawConfidence = rawConfidence;
        Source = source;
        Chars = chars;
        Quad = quad;
    }

    /// <summary>The word's text in logical order; never empty.</summary>
    public required string Text { get; init; }

    /// <summary>The ink extent of the word in page units for recognized words.</summary>
    public required OcrBox Box { get; init; }

    /// <summary>
    /// Calibrated probability in 0..1 that the word is exactly right. Equal to <see cref="RawConfidence"/> when the model
    /// carries no calibration or the word did not come from a recognizer.
    /// </summary>
    public required float Confidence { get; init; }

    /// <summary>Mean over the word's characters of the largest softmax probability seen while that character was the winning class.</summary>
    public required float RawConfidence { get; init; }

    /// <summary>Where the word came from.</summary>
    public required OcrSource Source { get; init; }

    /// <summary>The word's characters left to right; empty when they were not kept.</summary>
    public required IReadOnlyList<OcrChar> Chars { get; init; }

    /// <summary>The corners, for rotated text; null for upright text.</summary>
    public OcrQuad? Quad { get; init; }

    /// <inheritdoc />
    public bool Equals(OcrWord? other) =>
        other is not null && Text == other.Text && Box == other.Box && Confidence.Equals(other.Confidence) && RawConfidence.Equals(other.RawConfidence)
        && Source == other.Source && Quad == other.Quad && ModelEquality.Same(Chars, other.Chars);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Text, Box, Confidence, RawConfidence, Source, Quad, Chars.Count);
}

/// <summary>One line of text. A line may consist of several cells: runs set apart by a wide gap, for example the cells of a table row. Records compare by value (collections element by element).</summary>
public sealed record OcrLine
{
    /// <summary>Creates an empty line; set the required members.</summary>
    public OcrLine()
    {
    }

    [SetsRequiredMembers]
    internal OcrLine(
        string text, IReadOnlyList<OcrWord> words, IReadOnlyList<int> cells, OcrBox box, float confidence, OcrSource source,
        bool isRightToLeft = false, bool isVertical = false, OcrQuad? quad = null)
    {
        Text = text;
        Words = words;
        Cells = cells;
        Box = box;
        Confidence = confidence;
        Source = source;
        IsRightToLeft = isRightToLeft;
        IsVertical = isVertical;
        Quad = quad;
    }

    /// <summary>The words joined by one space, cells joined by a tab.</summary>
    public required string Text { get; init; }

    /// <summary>The words in reading order.</summary>
    public required IReadOnlyList<OcrWord> Words { get; init; }

    /// <summary>Indices into <see cref="Words"/> where each cell starts; the first is 0. Empty for a line with no words.</summary>
    public required IReadOnlyList<int> Cells { get; init; }

    /// <summary>The line's box.</summary>
    public required OcrBox Box { get; init; }

    /// <summary>The calibrated confidences of the words, averaged with each word weighted by its character count; 0 without words.</summary>
    public required float Confidence { get; init; }

    /// <summary>Where the line came from.</summary>
    public required OcrSource Source { get; init; }

    /// <summary>True when the base direction is right to left. Always false for lines read by the local recognizer.</summary>
    public bool IsRightToLeft { get; init; }

    /// <summary>True for vertical writing. Always false for lines read by the local recognizer.</summary>
    public bool IsVertical { get; init; }

    /// <summary>The corners, for rotated text; null for upright text.</summary>
    public OcrQuad? Quad { get; init; }

    /// <inheritdoc />
    public bool Equals(OcrLine? other) =>
        other is not null && Text == other.Text && Box == other.Box && Confidence.Equals(other.Confidence) && Source == other.Source
        && IsRightToLeft == other.IsRightToLeft && IsVertical == other.IsVertical && Quad == other.Quad
        && ModelEquality.Same(Words, other.Words) && ModelEquality.Same(Cells, other.Cells);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Text, Box, Confidence, Source, Quad, Words.Count);
}

/// <summary>A run of lines that read together: a paragraph, a column of text or a table. Records compare by value (collections element by element).</summary>
public sealed record OcrBlock
{
    /// <summary>Creates an empty block; set the required members.</summary>
    public OcrBlock()
    {
    }

    [SetsRequiredMembers]
    internal OcrBlock(OcrBlockRole role, bool isTable, OcrBox box, IReadOnlyList<OcrLine> lines, float confidence, OcrSource source = OcrSource.Ocr, OcrQuad? quad = null)
    {
        Role = role;
        IsTable = isTable;
        Box = box;
        Lines = lines;
        Confidence = confidence;
        Source = source;
        Quad = quad;
    }

    /// <summary>Body, header or footer.</summary>
    public required OcrBlockRole Role { get; init; }

    /// <summary>True for a grid of short cells, whose lines are rows.</summary>
    public required bool IsTable { get; init; }

    /// <summary>The block's box.</summary>
    public required OcrBox Box { get; init; }

    /// <summary>The lines in reading order.</summary>
    public required IReadOnlyList<OcrLine> Lines { get; init; }

    /// <summary>The lines' confidences averaged with each line weighted by its word count; 0 without words.</summary>
    public required float Confidence { get; init; }

    /// <summary>Where the block came from. Default <see cref="OcrSource.Ocr"/>.</summary>
    public OcrSource Source { get; init; } = OcrSource.Ocr;

    /// <summary>The corners, for rotated text; null for upright text.</summary>
    public OcrQuad? Quad { get; init; }

    /// <inheritdoc />
    public bool Equals(OcrBlock? other) =>
        other is not null && Role == other.Role && IsTable == other.IsTable && Box == other.Box && Confidence.Equals(other.Confidence)
        && Source == other.Source && Quad == other.Quad && ModelEquality.Same(Lines, other.Lines);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Role, IsTable, Box, Confidence, Source, Quad, Lines.Count);
}

/// <summary>A layout region with a role, as found by a layout model. The local pipeline fills regions where the page detector reports them; without a detector the list is empty. Records compare by value (collections element by element).</summary>
public sealed record OcrRegion
{
    /// <summary>Creates an empty region; set the required members.</summary>
    public OcrRegion()
    {
    }

    [SetsRequiredMembers]
    internal OcrRegion(OcrRegionRole role, OcrBox box, float confidence, OcrSource source, IReadOnlyList<int> blockIndices, OcrQuad? quad = null)
    {
        Role = role;
        Box = box;
        Confidence = confidence;
        Source = source;
        BlockIndices = blockIndices;
        Quad = quad;
    }

    /// <summary>What the region is.</summary>
    public required OcrRegionRole Role { get; init; }

    /// <summary>The region's box.</summary>
    public required OcrBox Box { get; init; }

    /// <summary>Probability that the role is right, 0 to 1.</summary>
    public required float Confidence { get; init; }

    /// <summary>Where the region came from.</summary>
    public required OcrSource Source { get; init; }

    /// <summary>Indices into the page's blocks that lie in the region.</summary>
    public required IReadOnlyList<int> BlockIndices { get; init; }

    /// <summary>The corners, for rotated regions; null otherwise.</summary>
    public OcrQuad? Quad { get; init; }

    /// <inheritdoc />
    public bool Equals(OcrRegion? other) =>
        other is not null && Role == other.Role && Box == other.Box && Confidence.Equals(other.Confidence) && Source == other.Source
        && Quad == other.Quad && ModelEquality.Same(BlockIndices, other.BlockIndices);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Role, Box, Confidence, Source, Quad, BlockIndices.Count);
}

/// <summary>A table cell. Records compare by value (collections element by element).</summary>
public sealed record OcrTableCell
{
    /// <summary>Creates an empty cell; set the required members.</summary>
    public OcrTableCell()
    {
    }

    [SetsRequiredMembers]
    internal OcrTableCell(int row, int column, int rowSpan, int columnSpan, OcrBox box, IReadOnlyList<int> wordIndices)
    {
        Row = row;
        Column = column;
        RowSpan = rowSpan;
        ColumnSpan = columnSpan;
        Box = box;
        WordIndices = wordIndices;
    }

    /// <summary>Zero-based row of the cell's top edge.</summary>
    public required int Row { get; init; }

    /// <summary>Zero-based column of the cell's left edge.</summary>
    public required int Column { get; init; }

    /// <summary>Rows the cell covers, at least 1.</summary>
    public required int RowSpan { get; init; }

    /// <summary>Columns the cell covers, at least 1.</summary>
    public required int ColumnSpan { get; init; }

    /// <summary>The cell's box.</summary>
    public required OcrBox Box { get; init; }

    /// <summary>Indices into the page's reading-order word sequence (<see cref="OcrPage.Words"/>), left to right.</summary>
    public required IReadOnlyList<int> WordIndices { get; init; }

    /// <inheritdoc />
    public bool Equals(OcrTableCell? other) =>
        other is not null && Row == other.Row && Column == other.Column && RowSpan == other.RowSpan && ColumnSpan == other.ColumnSpan
        && Box == other.Box && ModelEquality.Same(WordIndices, other.WordIndices);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Row, Column, RowSpan, ColumnSpan, Box, WordIndices.Count);
}

/// <summary>A table: cells with row and column spans that refer to the page's words. Records compare by value (collections element by element).</summary>
public sealed record OcrTable
{
    /// <summary>Creates an empty table; set the required members.</summary>
    public OcrTable()
    {
    }

    [SetsRequiredMembers]
    internal OcrTable(OcrBox box, int rows, int columns, IReadOnlyList<OcrTableCell> cells, float confidence, OcrSource source, int blockIndex = -1, OcrQuad? quad = null)
    {
        Box = box;
        Rows = rows;
        Columns = columns;
        Cells = cells;
        Confidence = confidence;
        Source = source;
        BlockIndex = blockIndex;
        Quad = quad;
    }

    /// <summary>The table's box.</summary>
    public required OcrBox Box { get; init; }

    /// <summary>Number of rows.</summary>
    public required int Rows { get; init; }

    /// <summary>Number of columns.</summary>
    public required int Columns { get; init; }

    /// <summary>The cells in row-major order.</summary>
    public required IReadOnlyList<OcrTableCell> Cells { get; init; }

    /// <summary>Probability that this is a table, 0 to 1.</summary>
    public required float Confidence { get; init; }

    /// <summary>Where the table structure came from.</summary>
    public required OcrSource Source { get; init; }

    /// <summary>Index of the page block that holds the table's text; -1 (the default) when there is none.</summary>
    public int BlockIndex { get; init; } = -1;

    /// <summary>The corners, for a rotated table; null otherwise.</summary>
    public OcrQuad? Quad { get; init; }

    /// <inheritdoc />
    public bool Equals(OcrTable? other) =>
        other is not null && Box == other.Box && Rows == other.Rows && Columns == other.Columns && Confidence.Equals(other.Confidence)
        && Source == other.Source && BlockIndex == other.BlockIndex && Quad == other.Quad && ModelEquality.Same(Cells, other.Cells);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Box, Rows, Columns, Confidence, Source, BlockIndex, Cells.Count);
}

/// <summary>A key and the value it labels, both as words of the page. Records compare by value (collections element by element).</summary>
public sealed record OcrKeyValue
{
    /// <summary>Creates an empty pair; set the required members.</summary>
    public OcrKeyValue()
    {
    }

    [SetsRequiredMembers]
    internal OcrKeyValue(IReadOnlyList<int> keyWordIndices, IReadOnlyList<int> valueWordIndices, OcrBox box, float confidence, OcrSource source)
    {
        KeyWordIndices = keyWordIndices;
        ValueWordIndices = valueWordIndices;
        Box = box;
        Confidence = confidence;
        Source = source;
    }

    /// <summary>Indices into <see cref="OcrPage.Words"/> of the key's words.</summary>
    public required IReadOnlyList<int> KeyWordIndices { get; init; }

    /// <summary>Indices into <see cref="OcrPage.Words"/> of the value's words.</summary>
    public required IReadOnlyList<int> ValueWordIndices { get; init; }

    /// <summary>The box around key and value.</summary>
    public required OcrBox Box { get; init; }

    /// <summary>Probability that the pairing is right, 0 to 1.</summary>
    public required float Confidence { get; init; }

    /// <summary>Where the pairing came from.</summary>
    public required OcrSource Source { get; init; }

    /// <inheritdoc />
    public bool Equals(OcrKeyValue? other) =>
        other is not null && Box == other.Box && Confidence.Equals(other.Confidence) && Source == other.Source
        && ModelEquality.Same(KeyWordIndices, other.KeyWordIndices) && ModelEquality.Same(ValueWordIndices, other.ValueWordIndices);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Box, Confidence, Source, KeyWordIndices.Count, ValueWordIndices.Count);
}

/// <summary>Element-wise comparison of the collections inside the page model's records.</summary>
internal static class ModelEquality
{
    public static bool Same<T>(IReadOnlyList<T> a, IReadOnlyList<T> b)
    {
        if (ReferenceEquals(a, b))
        {
            return true;
        }

        if (a.Count != b.Count)
        {
            return false;
        }

        EqualityComparer<T> comparer = EqualityComparer<T>.Default;
        for (int i = 0; i < a.Count; i++)
        {
            if (!comparer.Equals(a[i], b[i]))
            {
                return false;
            }
        }

        return true;
    }
}

/// <summary>
/// One page as structure: blocks in reading order, each with lines, words and characters, plus layout regions, tables and
/// key–value pairs. Coordinates are page units with the origin at the top-left and y growing downward: pixels for images,
/// points for PDF pages. Immutable once built; the JSON form is described in <c>docs/spec/oneocr-page-v1.md</c>.
/// </summary>
public sealed class OcrPage
{
    private OcrWord[]? _words;

    /// <summary>Creates an empty page; set the required members (<see cref="Index"/>, <see cref="Width"/>, <see cref="Height"/>, <see cref="UnitsPerInch"/> and <see cref="Blocks"/>).</summary>
    public OcrPage()
    {
    }

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
    [SetsRequiredMembers]
    internal OcrPage(
        int index, double width, double height, double unitsPerInch, IReadOnlyList<OcrBlock> blocks,
        IReadOnlyList<OcrRegion>? regions = null, IReadOnlyList<OcrTable>? tables = null, IReadOnlyList<OcrKeyValue>? keyValues = null,
        IReadOnlyList<OcrDiagnostic>? diagnostics = null, OcrMatrix? toPdf = null, int orientation = 0, double skewDegrees = 0)
    {
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
    public required int Index { get; init; }

    /// <summary>Page width in page units.</summary>
    public required double Width { get; init; }

    /// <summary>Page height in page units.</summary>
    public required double Height { get; init; }

    /// <summary>
    /// Page units per inch. For an image, the resolution the caller gave or, when none was given, an estimate from the size of
    /// the text (a diagnostic says so); 72 for PDF pages.
    /// </summary>
    public required double UnitsPerInch { get; init; }

    /// <summary>The map from page units back to PDF user space; null unless the page came from a PDF.</summary>
    public OcrMatrix? ToPdf { get; init; }

    /// <summary>
    /// How far the page content is turned clockwise from upright when the image was read: 0 (the default), 90, 180 or 270 (90 means the
    /// tops of the lines point to the right). Coordinates stay in the image as given, so boxes can be drawn on the image you passed in;
    /// rotated elements carry a quad whose corners follow the text's own frame. 0 for PDF pages.
    /// </summary>
    /// <exception cref="ArgumentException">The value is not 0, 90, 180 or 270.</exception>
    public int Orientation
    {
        get;
        init => field = value is 0 or 90 or 180 or 270 ? value : throw new ArgumentException("Orientation must be 0, 90, 180 or 270.", nameof(value));
    }

    /// <summary>
    /// The tilt in degrees of the text lines that was removed before reading, positive when lines descend to the right (clockwise
    /// in y-down coordinates). 0 (the default) when the page was not tilted or the tilt was too small to act on; within 45.
    /// </summary>
    /// <exception cref="ArgumentException">The value is not within 45 degrees.</exception>
    public double SkewDegrees
    {
        get;
        init => field = Math.Abs(value) <= 45 ? value : throw new ArgumentException("SkewDegrees must be within 45 degrees.", nameof(value));
    }

    /// <summary>Blocks in reading order.</summary>
    public required IReadOnlyList<OcrBlock> Blocks
    {
        get;
        init => field = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>Layout regions with roles; empty until a layout model fills them.</summary>
    public IReadOnlyList<OcrRegion> Regions { get; init => field = value ?? []; } = [];

    /// <summary>Tables found on the page.</summary>
    public IReadOnlyList<OcrTable> Tables { get; init => field = value ?? []; } = [];

    /// <summary>Key–value pairs found on the page; empty until a model fills them.</summary>
    public IReadOnlyList<OcrKeyValue> KeyValues { get; init => field = value ?? []; } = [];

    /// <summary>
    /// Notes about this page: recoverable conditions the library handled instead of throwing, for example that the resolution was
    /// estimated or that the page was turned. Each has a stable <see cref="OcrDiagnostic.Code"/>. Pages read back from JSON carry the
    /// stored messages with the code <see cref="OcrDiagnosticCodes.Unspecified"/>.
    /// </summary>
    public IReadOnlyList<OcrDiagnostic> Diagnostics { get; init => field = value ?? []; } = [];

    /// <summary>
    /// Every word of the page in reading order (block by block, line by line, left to right). Table cells, regions and
    /// key–value pairs refer to words by their index in this sequence.
    /// </summary>
    public IReadOnlyList<OcrWord> Words => _words ??= CollectWords();

    /// <summary>
    /// The page's text with blocks set apart: lines joined by <c>\n</c>, blocks by <c>\n\n</c>, table cells (and other cells) by <c>\t</c>.
    /// This is the form written to the <c>text</c> member of the page JSON; it equals <c>ToText(OcrTextLayout.Paragraphs)</c>. Where blocks
    /// split depends on segmentation, so for plain text prefer <see cref="ToText()"/>, which is what <c>image.ToText()</c> returns.
    /// </summary>
    public string Text => ToText(OcrTextLayout.Paragraphs);

    /// <summary>
    /// The page's lines in reading order, joined by <c>\n</c> with no blank lines between blocks (cells are joined by a tab). This is the text
    /// that <c>image.ToText()</c> returns for the same page.
    /// </summary>
    /// <returns>The text; empty when the page has no lines.</returns>
    public string ToText() => ToText(OcrTextLayout.Lines);

    /// <summary>The page's text in reading order, with blocks either run together or set apart by a blank line.</summary>
    /// <param name="layout">How blocks are joined.</param>
    /// <returns>The text; empty when the page has no lines.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="layout"/> is not a defined value.</exception>
    public string ToText(OcrTextLayout layout)
    {
        if (layout is not (OcrTextLayout.Lines or OcrTextLayout.Paragraphs))
        {
            throw new ArgumentOutOfRangeException(nameof(layout), layout, "Not a defined text layout.");
        }

        var sb = new StringBuilder();
        bool first = true;
        foreach (OcrBlock block in Blocks)
        {
            if (block.Lines.Count == 0)
            {
                continue;
            }

            bool firstInBlock = true;
            foreach (OcrLine line in block.Lines)
            {
                if (!first)
                {
                    sb.Append(firstInBlock && layout == OcrTextLayout.Paragraphs ? "\n\n" : "\n");
                }

                first = false;
                firstInBlock = false;
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
