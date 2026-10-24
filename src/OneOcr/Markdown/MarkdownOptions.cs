namespace OneOcr;

/// <summary>Options of <see cref="OcrMarkdown"/>. Everything has a default.</summary>
public sealed class MarkdownOptions
{
    /// <summary>
    /// Whether running headers and footers (blocks with a header or footer role, regions with those roles) are written.
    /// Default false: they are dropped.
    /// </summary>
    public bool IncludeHeadersAndFooters { get; init; }

    /// <summary>
    /// Whether a lone page number ("7", "Page 7", "7 of 12", "- 7 -") in the top or bottom 8% of a page is dropped as furniture
    /// even when no layout role says so. Default true.
    /// </summary>
    public bool DropPageNumbers { get; init; } = true;

    /// <summary>
    /// Whether short blocks set in clearly larger type than the body become headings when no layout region says what they are.
    /// Default true. Blocks in a title region are always headings.
    /// </summary>
    public bool HeadingsFromSize { get; init; } = true;

    /// <summary>Whether lines that start with a bullet or a number become list items. Default true.</summary>
    public bool Lists { get; init; } = true;
}
