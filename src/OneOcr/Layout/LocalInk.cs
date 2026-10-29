namespace OneOcr.Layout;

/// <summary>
/// A binarization chosen for one line: gray values at or below <paramref name="Threshold"/> are ink, or above it when
/// <paramref name="InkIsBright"/>.
/// </summary>
/// <remarks>
/// The page-wide binarization (one Otsu threshold, ink is the smaller class) fails on a photo whose light falls off across the page.
/// Two things go wrong. When most of the page is dark (shadow, or the table the sheet lies on), the bright class is the smaller one
/// and is taken for ink, so in the shadowed part the "ink" is the odd bright speck and the line's real ink is not found. And when
/// the page is mostly bright, the threshold can lie inside the paper of a shadowed line, so the whole line counts as ink. In both
/// cases the text is still darker than its own paper, and Otsu's threshold over the line's own pixels finds it.
/// </remarks>
internal readonly record struct LocalInk(int Threshold, bool InkIsBright)
{
    /// <summary>A line with more than this share of ink under the page's dark-ink threshold is not text on paper: the threshold is inside its paper.</summary>
    internal const double PaperShare = 0.5;

    /// <summary>The smallest share of a line that its own ink may cover for the line's own threshold to be believed.</summary>
    internal const double MinInkShare = 0.015;

    /// <summary>The smallest difference of mean gray between a line's paper and its ink for the line's own threshold to be believed.</summary>
    internal const double MinContrast = 40;

    /// <summary>
    /// The least page noise (<see cref="Detection.PageDetection.Sigma"/>) of a camera photo. Local binarization is only for photos: a
    /// screenshot or a clean render has flat colours, one threshold fits all its lines, and its lines keep the page's reading.
    /// </summary>
    internal const double MinPageNoise = 0.1;

    /// <summary>
    /// Whether a page that reads ink as bright has dark text on lighter paper after all: true when at least half of its lines are
    /// read that way by their own Otsu threshold. A dark-mode screen has bright text on dark in most of its lines and keeps the
    /// page's reading, including for a light button or card on it.
    /// </summary>
    /// <param name="page">Gray pixels, row-major.</param>
    /// <param name="width">Row length of <paramref name="page"/>.</param>
    /// <param name="boxes">Pixel rectangles (x0, y0, x1, y1; right and bottom exclusive) of the lines.</param>
    public static bool PageIsDarkInk(ReadOnlySpan<byte> page, int width, IReadOnlyList<(int X0, int Y0, int X1, int Y1)> boxes)
    {
        int dark = 0;
        foreach ((int x0, int y0, int x1, int y1) in boxes)
        {
            if (Otsu(page, width, x0, x1, y0, y1) is { InkIsBright: false })
            {
                dark++;
            }
        }

        return boxes.Count > 0 && dark * 2 >= boxes.Count;
    }

    /// <summary>
    /// The binarization that fits a rectangle of the page better than the page-wide one, or null when the page's stands (the usual case).
    /// </summary>
    /// <param name="page">Gray pixels, row-major.</param>
    /// <param name="width">Row length of <paramref name="page"/>.</param>
    /// <param name="x0">Left edge, inclusive.</param>
    /// <param name="x1">Right edge, exclusive.</param>
    /// <param name="y0">Top edge, inclusive.</param>
    /// <param name="y1">Bottom edge, exclusive.</param>
    /// <param name="threshold">The page-wide threshold.</param>
    /// <param name="inkIsBright">Whether the page-wide ink is the bright class.</param>
    /// <param name="pageIsDarkInk">The result of <see cref="PageIsDarkInk"/> for a page whose ink is bright.</param>
    public static LocalInk? Choose(ReadOnlySpan<byte> page, int width, int x0, int x1, int y0, int y1, int threshold, bool inkIsBright, bool pageIsDarkInk)
    {
        if (inkIsBright && !pageIsDarkInk)
        {
            return null;
        }

        var histogram = new long[256];
        long total = Fill(page, width, x0, x1, y0, y1, histogram);
        if (total == 0)
        {
            return null;
        }

        long below = 0;
        for (int v = 0; v <= threshold; v++)
        {
            below += histogram[v];
        }

        // The share of the line that the page's reading calls ink.
        double share = (double)(inkIsBright ? total - below : below) / total;
        if (!inkIsBright && share <= PaperShare)
        {
            return null;
        }

        (int t, bool bright) = LineFinder.Binarize(histogram, total);
        // Light-on-dark bars on a light page also read as "mostly ink"; their own reading is bright ink and stays the page's.
        if (t < 0 || bright || (!inkIsBright && t == threshold))
        {
            return null;
        }

        // Real text is a visible share of its line and clearly darker than the paper; a speck or two in grain is neither.
        long darkCount = 0;
        double darkSum = 0, brightSum = 0;
        for (int v = 0; v < 256; v++)
        {
            if (v <= t)
            {
                darkCount += histogram[v];
                darkSum += (double)v * histogram[v];
            }
            else
            {
                brightSum += (double)v * histogram[v];
            }
        }

        long brightCount = total - darkCount;
        bool visible = darkCount >= total * MinInkShare && brightCount > 0 && (brightSum / brightCount) - (darkSum / Math.Max(1, darkCount)) >= MinContrast;
        return visible ? new LocalInk(t, false) : null;
    }

    private static LocalInk? Otsu(ReadOnlySpan<byte> page, int width, int x0, int x1, int y0, int y1)
    {
        var histogram = new long[256];
        long total = Fill(page, width, x0, x1, y0, y1, histogram);
        if (total == 0)
        {
            return null;
        }

        (int t, bool bright) = LineFinder.Binarize(histogram, total);
        return t < 0 ? null : new LocalInk(t, bright);
    }

    private static long Fill(ReadOnlySpan<byte> page, int width, int x0, int x1, int y0, int y1, long[] histogram)
    {
        long total = 0;
        for (int y = y0; y < y1; y++)
        {
            LineFinder.AddHistogram(page.Slice((y * width) + x0, x1 - x0), histogram);
            total += x1 - x0;
        }

        return total;
    }
}
