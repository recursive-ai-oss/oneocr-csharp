namespace OneOcr;

/// <summary>
/// Options of <see cref="OcrEngine.Ocr(OneImage.Image, OcrOptions?, CancellationToken)"/>, <see cref="OcrEngine.ToText(OneImage.Image, OcrOptions?, CancellationToken)"/>
/// and their relatives. Everything has a default. Set engine-wide defaults with <see cref="OcrEngine.WithOptions"/>; options passed to a call
/// replace them as a whole. Invalid values make the call throw <see cref="ArgumentOutOfRangeException"/> for the parameter <c>options</c>.
/// </summary>
public sealed class OcrOptions
{
    /// <summary>
    /// Resolution of the image in pixels per inch, written to <see cref="OcrPage.UnitsPerInch"/>. Null (the default) takes the resolution
    /// the file declares when the image is read with <c>OcrPages</c> from a path, a seekable stream or bytes (PNG, JPEG and TIFF are
    /// understood); otherwise it estimates it from the size of the text, assuming body text of about 11 points, and says so in the page's
    /// diagnostics (<see cref="OcrDiagnosticCodes.DpiEstimated"/>). Must be positive and finite.
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
    /// Whether to look for a better whole-page angle when the first reading is weak. Default true. Photographs taken with the camera
    /// turned by 10 to 30 degrees are beyond <see cref="Deskew"/>; when under 60% of a page's text is read confidently, the page is read
    /// again turned by -30 to +30 degrees in steps of 10, then 5 degrees either side of the best, and the angle with the most
    /// confidently read text is kept if it clearly beats the first reading. The angle is reported in <see cref="OcrPage.SkewDegrees"/>
    /// and a diagnostic says so. Pages that read well are not read twice; a weak page costs up to eight more readings.
    /// </summary>
    public bool SearchAngle { get; init; } = true;

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
    /// short strings (<c>O2</c>) are left as read. Together with <see cref="FixSeparators"/> and <see cref="FixStrayAccents"/> it is a step after recognition that changes a character; set it to
    /// false to keep exactly what the network read.
    /// </summary>
    public bool FixDigitLookalikes { get; init; } = true;

    /// <summary>
    /// Whether to read the middle dot, the bullet and the vertical bar between words from their pixels. Default true. A
    /// one-character word between two words that the network read as a hyphen or another small mark, and whose ink is a small round
    /// blob at the middle height of the line's letters, becomes <c>·</c> (small) or <c>•</c> (large); one read as
    /// <c>I</c>, <c>l</c>, <c>1</c> or <c>!</c> whose ink is a thin bar taller than every other mark of the line becomes <c>|</c>.
    /// Box and confidence are kept, and words with more than one character, words at the ends of a line and hyphens or dashes
    /// inside words are never touched. Set it to false to keep exactly what the network read.
    /// </summary>
    public bool FixSeparators { get; init; } = true;

    /// <summary>
    /// Whether to take stray accents and dots off letters. Default true. On a degraded scan a speck over or under a letter is read as
    /// part of it (<c>thé</c> for <c>the</c>, <c>dọwn</c> for <c>down</c>). A word is changed only when all of these hold: it is read with
    /// a calibrated confidence under 0.97, it consists of Latin letters (and marks at its ends), its plain spelling is in the word
    /// frequency table and at least eight times as frequent as the accented one, and no more than 12% of the page's words have
    /// accents (a page in French or Vietnamese keeps them). Box and confidence are kept. It needs the engine's word frequencies
    /// (<see cref="OcrEngine.WithWordFrequencies"/>; the bundled engine has them) and is part of the noise cleanup: it does nothing when
    /// <see cref="DropNoise"/> is false. Set it to false to keep exactly what the network read.
    /// </summary>
    public bool FixStrayAccents { get; init; } = true;

    /// <summary>Replaces the noise filter's thresholds; used to tune them. Null uses the defaults.</summary>
    internal global::OneOcr.Layout.JunkParameters? JunkTuning { get; init; }

    /// <summary>These options with <see cref="Dpi"/> set to the resolution a file declares, unless the caller set one.</summary>
    internal OcrOptions WithDeclaredDpi(double? declared) =>
        Dpi is not null || declared is null
            ? this
            : new OcrOptions
            {
                Dpi = declared, MaxDegreeOfParallelism = MaxDegreeOfParallelism, IncludeChars = IncludeChars, PageIndex = PageIndex, DetectOrientation = DetectOrientation,
                Deskew = Deskew, SearchAngle = SearchAngle, DropNoise = DropNoise, FixWordSpacing = FixWordSpacing, FixDigitLookalikes = FixDigitLookalikes,
                FixSeparators = FixSeparators, FixStrayAccents = FixStrayAccents, JunkTuning = JunkTuning,
            };

    internal void Validate()
    {
        if (Dpi is double d && (!double.IsFinite(d) || d <= 0))
        {
            throw new ArgumentOutOfRangeException("options", d, "Dpi must be positive and finite.");
        }

        if (MaxDegreeOfParallelism < 0)
        {
            throw new ArgumentOutOfRangeException("options", MaxDegreeOfParallelism, "MaxDegreeOfParallelism must not be negative.");
        }

        if (PageIndex < 0)
        {
            throw new ArgumentOutOfRangeException("options", PageIndex, "PageIndex must not be negative.");
        }
    }
}
