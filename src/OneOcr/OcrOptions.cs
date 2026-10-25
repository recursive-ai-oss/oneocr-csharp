namespace OneOcr;

/// <summary>Per-call options of <see cref="OcrEngine.Ocr(OneImage.Image, OcrOptions?, CancellationToken)"/>. Everything has a default.</summary>
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

    /// <summary>
    /// Whether to drop words that are noise read as text (specks, scan grain and edges). Default true. Two kinds go: unsure words
    /// that sit in lines much smaller than the page's own text (calibrated confidence under 0.5, and a line both under 60% of the
    /// height of the page's confidently read lines and under 12 pixels; in lines under 8 pixels, unless it is read with 0.99; pages
    /// with fewer than three confidently read lines are left alone), and words made only of dots, commas, colons, semicolons and
    /// quotes that are read with confidence under 0.9. Dashes, bullets and middle dots are never dropped. The number dropped is
    /// reported in the page's diagnostics.
    /// </summary>
    public bool DropNoise { get; init; } = true;

    /// <summary>
    /// Whether to decide the spaces between words from the line's pixels as well as from the network. Default true. A space the
    /// network put inside a number or date, or between two letters where the ink is continuous, is taken out when the pieces
    /// only make sense joined (for the letters: when the engine has a word model, see <see cref="OcrEngine.WithWordModel"/>);
    /// a space is added where the line has an empty stretch far wider than its letter gaps. Only spaces change, never characters.
    /// </summary>
    public bool FixWordSpacing { get; init; } = true;

    /// <summary>
    /// Whether to read the letter <c>O</c> as the digit <c>0</c> inside identifiers and dates. Default true. In a word of at least
    /// three digits, a run of letters and digits (the pieces between punctuation) that holds only digits and capital <c>O</c>s,
    /// with at least one of each, has its <c>O</c>s read as <c>0</c>: <c>ACC-71O3O5</c> becomes <c>ACC-710305</c> and
    /// <c>2020-04-O7</c> becomes <c>2020-04-07</c>. Words with other letters inside the run (<c>CO2</c>, <c>COVID19</c>) and
    /// short strings (<c>O2</c>) are left as read. This is the only step after recognition that changes a character; set it to
    /// false to keep exactly what the network read.
    /// </summary>
    public bool FixDigitLookalikes { get; init; } = true;

    /// <summary>Replaces the noise filter's thresholds; used to tune them. Null uses the defaults.</summary>
    internal global::OneOcr.Layout.JunkParameters? JunkTuning { get; init; }

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
