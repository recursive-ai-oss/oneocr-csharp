namespace OneOcr.Pdf;

/// <summary>
/// Options for reading PDF pages into the page model. Immutable; every member has a default, so <c>new PdfOcrOptions()</c> is a
/// sensible setup: trust a page's own text only when it is visible and agrees with what the page looks like, read scanned
/// regions at 300 DPI, and report problems as diagnostics on the page.
/// </summary>
public sealed class PdfOcrOptions
{
    /// <summary>
    /// When true, text that is drawn invisibly (render mode 3, as scanners and OCR programs write it over a page image) is kept as it
    /// is, with <see cref="OcrSource.TextLayer"/>, and the image under it is not read again. When false (the default) invisible
    /// text is dropped and the image regions it covered are read by the local recognizer: third-party OCR layers are often wrong,
    /// and ours is deterministic.
    /// </summary>
    public bool TrustExistingOcrLayer { get; init; }

    /// <summary>
    /// When true (the default), image regions that the page's text does not already cover, and text that is garbled, are rendered
    /// and read. When false the page model holds only the text layer (no rendering at all).
    /// </summary>
    public bool ReadImages { get; init; } = true;

    /// <summary>Resolution in DPI at which regions are rendered for the recognizer, 72 to 1200. Default 300.</summary>
    public double RenderDpi { get; init; } = 300;

    /// <summary>
    /// How many lines of a page's visible text are rendered and read to check that the text layer says what the page shows (a font whose
    /// ToUnicode table maps to the wrong letters gives text that looks fine and is wrong). When at least half of the checked lines
    /// disagree, the whole page is read from pixels instead. 0 turns the check off. Default 3. The check costs one small render and one
    /// line of recognition per sampled line.
    /// </summary>
    public int CrossCheckLines { get; init; } = 3;

    /// <summary>Images whose width or height is smaller than this many points are not read (logos, bullets, rules). Default 24.</summary>
    public double MinImagePoints { get; init; } = 24;

    /// <summary>When true (the default) words carry their characters. Text-layer characters have exact boxes; recognized ones approximate boxes.</summary>
    public bool IncludeChars { get; init; } = true;

    /// <summary>Most threads to use; 0 means min(4, cores). Whole documents read that many pages at once. The result does not depend on it.</summary>
    public int MaxDegreeOfParallelism { get; init; }

    /// <summary>Zero-based index of the first page to read in a document call. Default 0.</summary>
    public int FirstPage { get; init; }

    /// <summary>How many pages to read in a document call, counted from <see cref="FirstPage"/>; fewer if the document ends sooner. Default: all.</summary>
    public int PageCount { get; init; } = int.MaxValue;

    /// <summary>The password of an encrypted document; null for the empty user password.</summary>
    public string? Password { get; init; }

    /// <summary>
    /// When false (the default) a page that cannot be fully read (its content is damaged, an image is too large to render, a region
    /// exceeds an OCR limit) gives what could be read plus a diagnostic. When true that situation throws instead, and so does a document whose page tree yields no pages at all (lenient gives one empty page with a diagnostic).
    /// </summary>
    public bool Strict { get; init; }

    /// <summary>The largest PDF in bytes that is read from a path or stream. Default 256 MiB. Checked before the bytes are read.</summary>
    public long MaxFileBytes { get; init; } = 256L * 1024 * 1024;

    internal void Validate()
    {
        if (!double.IsFinite(RenderDpi) || RenderDpi < 72 || RenderDpi > 1200)
        {
            throw new ArgumentException("RenderDpi must be between 72 and 1200.", nameof(RenderDpi));
        }

        if (CrossCheckLines < 0 || CrossCheckLines > 64)
        {
            throw new ArgumentException("CrossCheckLines must be between 0 and 64.", nameof(CrossCheckLines));
        }

        if (!double.IsFinite(MinImagePoints) || MinImagePoints < 0)
        {
            throw new ArgumentException("MinImagePoints must be finite and not negative.", nameof(MinImagePoints));
        }

        if (MaxDegreeOfParallelism < 0)
        {
            throw new ArgumentException("MaxDegreeOfParallelism must not be negative.", nameof(MaxDegreeOfParallelism));
        }

        if (FirstPage < 0)
        {
            throw new ArgumentException("FirstPage must not be negative.", nameof(FirstPage));
        }

        if (PageCount < 0)
        {
            throw new ArgumentException("PageCount must not be negative.", nameof(PageCount));
        }

        if (MaxFileBytes <= 0)
        {
            throw new ArgumentException("MaxFileBytes must be positive.", nameof(MaxFileBytes));
        }
    }
}
