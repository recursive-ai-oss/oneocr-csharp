namespace OneOcr;

/// <summary>Per-call options of <see cref="OcrEngine.Ocr"/>. Everything has a default.</summary>
public sealed class OcrOptions
{
    /// <summary>
    /// Resolution of the image in pixels per inch, written to <see cref="OcrPage.UnitsPerInch"/>. Null (the default) estimates it from
    /// the size of the text, assuming body text of about 11 points, and says so in the page's diagnostics. Must be positive and finite.
    /// </summary>
    public double? Dpi { get; init; }

    /// <summary>Most threads to use; 0 (the default) means min(4, cores). The result does not depend on it.</summary>
    public int MaxDegreeOfParallelism { get; init; }

    /// <summary>Whether words carry their characters. Default true; turn it off to save memory when only words are needed.</summary>
    public bool IncludeChars { get; init; } = true;

    /// <summary>The index written to <see cref="OcrPage.Index"/>, for pages of a document. Default 0.</summary>
    public int PageIndex { get; init; }

    /// <summary>
    /// Whether to find out how far the page is turned (0, 90, 180 or 270 degrees) and read it upright. Default true. The result is
    /// reported in <see cref="OcrPage.Orientation"/>; boxes stay in the input image's coordinates and rotated elements carry a quad.
    /// A page whose text has no clear direction is read as it is.
    /// </summary>
    public bool DetectOrientation { get; init; } = true;

    /// <summary>
    /// Whether to straighten text lines tilted by up to about 15 degrees before reading. Default true. The tilt is reported in
    /// <see cref="OcrPage.SkewDegrees"/>; tilts under about 0.3 degrees are left alone, so an upright page reads exactly as before.
    /// </summary>
    public bool Deskew { get; init; } = true;

    internal void Validate()
    {
        if (Dpi is double d && (!double.IsFinite(d) || d <= 0))
        {
            throw new ArgumentException("Dpi must be positive and finite.", nameof(Dpi));
        }

        if (MaxDegreeOfParallelism < 0)
        {
            throw new ArgumentException("MaxDegreeOfParallelism must not be negative.", nameof(MaxDegreeOfParallelism));
        }

        if (PageIndex < 0)
        {
            throw new ArgumentException("PageIndex must not be negative.", nameof(PageIndex));
        }
    }
}
