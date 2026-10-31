using System.Globalization;

namespace OneOcr.Layout;

/// <summary>
/// The page angle search: a page whose first reading is weak (photographs taken with the camera turned by 10 to 30 degrees, beyond
/// what the skew estimate and the per-line tilt reading handle) is read again at a few whole-page angles, and the angle whose reading
/// holds the most confidently read text wins. Pages that read well never get here, so they pay nothing.
/// </summary>
internal static class PageAngleSearch
{
    // Slopes (tan of the angle times 1024, rounded) of the angles the search tries, in steps of 5 degrees: index i is (i + 1) * 5 degrees.
    // A table, so that no trigonometry is evaluated at run time.
    private static readonly int[] Slopes = [90, 181, 274, 373, 477, 591, 717];

    /// <summary>A word counts as confidently read at or above this calibrated confidence.</summary>
    internal static float ConfidentWord = 0.9f;

    /// <summary>The first reading is weak (full search) when the share of its characters in confidently read words is under this.</summary>
    internal static double WeakShare = 0.6;

    /// <summary>
    /// A reading whose confident share is under this but not under <see cref="WeakShare"/> is a doubtful reading (a photograph turned by a
    /// few degrees reads mostly right and still loses lines). It gets the light search: only the angles of 10 degrees either way, and a
    /// finer step around the one that wins.
    /// </summary>
    internal static double DoubtShare = 0.9;

    /// <summary>A reading at another angle replaces the first one only when its score is at least this many times the first one's.</summary>
    internal static double Margin = 1.1;

    /// <summary>Largest angle tried, in steps of 5 degrees (6 = 30 degrees).</summary>
    internal static int MaxStep = 6;

    /// <summary>Coarse step in units of 5 degrees (2 = 10 degrees).</summary>
    internal static int CoarseStep = 2;

    /// <summary>Receives one line per reading made by the search (used to tune it).</summary>
    internal static Action<string>? Trace;

    /// <summary>When true, every page is searched (used to tune the trigger).</summary>
    internal static bool Always;

    /// <summary>
    /// The score of a reading: the sum over its confidently read words (calibrated confidence at or above <see cref="ConfidentWord"/>)
    /// of confidence times character count, accumulated in reading order in double precision.
    /// </summary>
    public static double Score(OcrPage page)
    {
        double sum = 0;
        foreach (OcrWord w in page.Words)
        {
            if (w.Confidence >= ConfidentWord)
            {
                sum += (double)w.Confidence * w.Text.Length;
            }
        }

        return sum;
    }

    /// <summary>Whether the reading is weak enough to look for a better angle: little of its text is confidently read.</summary>
    public static bool IsWeak(OcrPage page, out double share)
    {
        long all = 0, sure = 0;
        foreach (OcrWord w in page.Words)
        {
            all += w.Text.Length;
            if (w.Confidence >= ConfidentWord)
            {
                sure += w.Text.Length;
            }
        }

        share = all == 0 ? 0 : (double)sure / all;
        Trace?.Invoke(string.Create(CultureInfo.InvariantCulture, $"weak? share {share:0.####} chars {all} words {page.Words.Count}"));
        return Always || share < DoubtShare;
    }

    /// <summary>Whether a reading with this confident share gets the light search instead of the full one.</summary>
    public static bool IsLight(double share) => !Always && share >= WeakShare;

    /// <summary>
    /// Whether a page that read as no words at all has enough ink for text to be there: at least 0.2% of its pixels on the ink side of
    /// the page's binarization. A blank or flat page is not searched, since turning it cannot give it words.
    /// </summary>
    public static bool HasInk(SearchSource source)
    {
        ReadOnlySpan<byte> bytes = Kernels.PixelSpans.AsBytes(source.Pixels.AsSpan(0, source.Width * source.Height));
        (int threshold, bool inkIsBright) = LineFinder.Binarize(bytes);
        if (threshold < 0)
        {
            return false;
        }

        long ink = 0;
        foreach (byte b in bytes)
        {
            if (inkIsBright ? b > threshold : b <= threshold)
            {
                ink++;
            }
        }

        return ink * 500 >= bytes.Length;
    }

    /// <summary>
    /// Searches coarse to fine. <paramref name="readAt"/> reads the page turned by a slope (as <see cref="PageFrame"/> uses it) and
    /// returns the page in the input frame, or null when that angle cannot be read (its canvas would exceed the pixel limit).
    /// </summary>
    /// <returns>The winning page and its angle in degrees (null for the first reading when it stays).</returns>
    public static (OcrPage Page, int? AngleDegrees) Search(OcrPage first, Func<int, OcrPage?> readAt, CancellationToken cancellationToken, bool light = false)
    {
        int maxStep = light ? CoarseStep : MaxStep;
        double threshold = Score(first) * Margin;
        Trace?.Invoke(string.Create(CultureInfo.InvariantCulture, $"first {Score(first):0.##}"));
        OcrPage best = first;
        int bestStep = 0;
        // Ties go to the smaller angle and, between the two signs, to the positive one: candidates are tried in that order and replace the best only when strictly better.
        for (int step = CoarseStep; step <= maxStep; step += CoarseStep)
        {
            for (int sign = 1; sign >= -1; sign -= 2)
            {
                cancellationToken.ThrowIfCancellationRequested();
                OcrPage? page = readAt(sign * Slopes[step - 1]);
                Trace?.Invoke(string.Create(CultureInfo.InvariantCulture, $"{sign * step * 5} {(page is null ? -1 : Score(page)):0.##}"));
                if (page is not null && Score(page) > threshold)
                {
                    threshold = Score(page);
                    best = page;
                    bestStep = sign * step;
                }
            }
        }

        if (bestStep != 0)
        {
            // One finer step either side of the coarse winner.
            int center = bestStep;
            foreach (int next in new[] { center + 1, center - 1 })
            {
                int mag = Math.Abs(next);
                if (mag < 1 || mag > maxStep + 1 || mag > Slopes.Length || (next < 0) != (center < 0))
                {
                    continue;
                }

                cancellationToken.ThrowIfCancellationRequested();
                OcrPage? page = readAt(Math.Sign(next) * Slopes[mag - 1]);
                Trace?.Invoke(string.Create(CultureInfo.InvariantCulture, $"{next * 5} {(page is null ? -1 : Score(page)):0.##}"));
                if (page is not null && Score(page) > threshold)
                {
                    threshold = Score(page);
                    best = page;
                    bestStep = next;
                }
            }
        }

        return (best, bestStep == 0 ? null : bestStep * 5);
    }

    public static OcrDiagnostic Note(int degrees) =>
        Diag.Info(OcrDiagnosticCodes.AngleSearched, string.Create(CultureInfo.InvariantCulture, $"The first reading was weak, so the page was read at several angles; it was turned by {degrees} degrees to read it."));
}
