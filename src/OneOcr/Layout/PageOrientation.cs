using OneImage;
using OneOcr.Kernels;
using OneOcr.Recognition;

namespace OneOcr.Layout;

/// <summary>The page handed to an orientation detector.</summary>
/// <param name="Pixels">8-bit grayscale, row-major; the first <c>Width * Height</c> entries.</param>
/// <param name="Width">Width in pixels.</param>
/// <param name="Height">Height in pixels.</param>
/// <param name="Degree">Most threads the detector may use.</param>
internal readonly record struct OrientationInput(Gray8[] Pixels, int Width, int Height, int Degree);

/// <summary>
/// Decides how far a page must be turned clockwise to stand upright. This is the seam where a learned orientation head can replace
/// the classical decision: it only has to return 0, 90, 180 or 270.
/// </summary>
internal interface IOrientationDetector
{
    /// <summary>Clockwise degrees (0, 90, 180 or 270) to turn the page so its text reads upright; 0 when there is no evidence.</summary>
    int Detect(OrientationInput page, CancellationToken cancellationToken);
}

/// <summary>
/// The orientation decision without a learned model. First, which way the lines run: the ink is projected as for skew estimation
/// on the page and on the page turned a quarter, and the direction with the sharper profile (lines of text) is horizontal; this
/// holds at any tilt up to 15 degrees. Second, which way is up: up to eight of the longest lines are read as they are and turned
/// half way round, and the reading the recognizer is more sure of wins (upside-down text reads as noise with low confidence). If
/// neither reading is sure, the other line direction is tried too and the most confident of the four wins.
/// </summary>
internal sealed class ClassicalOrientation : IOrientationDetector
{
    // The reading turned half way round must beat the upright one by this much mean confidence to win.
    private const double FlipMargin = 0.05;

    // One profile this many times as sharp as the other decides the line direction without a reading.
    private const double DecisiveRatio = 3.0;

    // A reading turned half way round must itself be at least this confident to win.
    private const double SureConfidence = 0.75;

    // When the profiles do not decide, the other direction wins only by reading this much more confidently.
    private const double AxisMargin = 0.15;

    // Turning half way round needs most of the sample lines (by length) to read better that way.
    private const double TurnedShareNeeded = 0.6;

    private const int SampleLines = 8;

    // Lines shorter than this carry no usable ascender profile; above must be at least this many tenths of below.
    private const int MinBiasHeight = 10;
    private const int AscenderRatioTenths = 14;

    // Early exit: when the first QuickLines lines all read better upright, and read at least QuickSure confidently, the page is upright.
    private const int QuickLines = 3;
    private const double QuickSure = 0.85;

    private readonly LineRecognizer _recognizer;
    private readonly BufferPool<Gray8> _pixels;
    private readonly BufferPool<byte> _masks;
    private readonly ResizerPool _resizers;
    private readonly OcrLimits _limits;

    public ClassicalOrientation(LineRecognizer recognizer, BufferPool<Gray8> pixels, BufferPool<byte> masks, ResizerPool resizers, OcrLimits limits)
    {
        _recognizer = recognizer;
        _pixels = pixels;
        _masks = masks;
        _resizers = resizers;
        _limits = limits;
    }

    public int Detect(OrientationInput page, CancellationToken cancellationToken) => Detect(page, null, cancellationToken);

    /// <summary>As <see cref="Detect(OrientationInput, CancellationToken)"/>, noting the evidence in <paramref name="trace"/>.</summary>
    internal int Detect(OrientationInput page, List<string>? trace, CancellationToken cancellationToken)
    {
        int count = checked(page.Width * page.Height);
        ReadOnlySpan<byte> bytes = PixelSpans.AsBytes(page.Pixels.AsSpan(0, count));
        (int threshold, bool inkIsBright) = LineFinder.Binarize(bytes);
        if (threshold < 0)
        {
            return 0;
        }

        using InkGrid upright = InkGrid.Build(bytes, page.Width, page.Height, threshold, inkIsBright);
        SkewEstimate flat = SkewEstimator.Estimate(upright);
        SkewEstimate sideways;
        using (InkGrid turnedGrid = upright.TurnClockwise())
        {
            sideways = SkewEstimator.Estimate(turnedGrid);
        }

        if (flat.Sharpness == 0 && sideways.Sharpness == 0)
        {
            return 0;
        }

        // Lines run along the direction with the much sharper profile. When the profiles are close (photographs, receipts, forms), the
        // upright direction is assumed and the other one is believed only if the recognizer reads it far better.
        double ratio = sideways.Contrast / flat.Contrast;
        int first = ratio >= DecisiveRatio ? 1 : 0;
        if (ratio <= 1 / DecisiveRatio && PagePreparation.ActedSlope(flat) == 0 && AscendersAbove(page, cancellationToken))
        {
            // Ordinary upright prose: no need to read anything.
            trace?.Add($"axis 0: ascenders above (contrast flat {flat.Contrast:F2} sideways {sideways.Contrast:F2})");
            return 0;
        }

        Reading firstReading = Read(page, first, PagePreparation.ActedSlope(first == 1 ? sideways : flat), cancellationToken);
        trace?.Add($"axis {first}: upright {firstReading.Upright:F3} turned {firstReading.Turned:F3} share {firstReading.TurnedShare:F2} (contrast flat {flat.Contrast:F2} sideways {sideways.Contrast:F2})");
        if (ratio >= DecisiveRatio || ratio <= 1 / DecisiveRatio)
        {
            return firstReading.Degrees(first);
        }

        Reading otherReading = Read(page, 1, PagePreparation.ActedSlope(sideways), cancellationToken);
        trace?.Add($"axis 1: upright {otherReading.Upright:F3} turned {otherReading.Turned:F3} share {otherReading.TurnedShare:F2}");
        return otherReading.Best >= SureConfidence && otherReading.Best > firstReading.Best + AxisMargin ? otherReading.Degrees(1) : firstReading.Degrees(0);
    }

    // Latin, Greek and Cyrillic text has more ink above its x-height band (ascenders, capitals, accents) than below it (descenders).
    // When the page's lines show that by a clear margin, the page stands upright and the recognizer need not read it turned
    // half way round. Anything else (digits, capitals only, scripts without the asymmetry, upside-down text) returns false and is
    // decided by reading.
    private bool AscendersAbove(OrientationInput page, CancellationToken cancellationToken)
    {
        List<StripLine> strips = StripFinder.Find(
            page.Pixels, page.Width, page.Height, _limits.MaxLinesPerPage, _masks, page.Degree, out _, out int threshold, out bool inkIsBright, eraseRules: false);
        if (threshold < 0)
        {
            return false;
        }

        long above = 0, below = 0, total = 0;
        int[] rows = [];
        foreach (StripLine strip in strips)
        {
            cancellationToken.ThrowIfCancellationRequested();
            LineBox b = strip.Box;
            if (b.Height < MinBiasHeight || b.Width < 3 * b.Height)
            {
                continue;
            }

            if (rows.Length < b.Height)
            {
                rows = new int[b.Height];
            }

            int max = 0;
            for (int y = 0; y < b.Height; y++)
            {
                ReadOnlySpan<byte> row = PixelSpans.AsBytes(page.Pixels.AsSpan(((b.Y + y) * page.Width) + b.X, b.Width));
                int n = 0;
                foreach (byte v in row)
                {
                    if (inkIsBright ? v > threshold : v <= threshold)
                    {
                        n++;
                    }
                }

                rows[y] = n;
                max = Math.Max(max, n);
            }

            // The body band: the rows holding at least half as much ink as the densest row.
            int top = 0, bottom = b.Height - 1;
            while (rows[top] * 2 < max)
            {
                top++;
            }

            while (rows[bottom] * 2 < max)
            {
                bottom--;
            }

            for (int y = 0; y < b.Height; y++)
            {
                total += rows[y];
                if (y < top)
                {
                    above += rows[y];
                }
                else if (y > bottom)
                {
                    below += rows[y];
                }
            }
        }

        // Enough ink must lie outside the body bands to mean something, and above must clearly outweigh below.
        return above + below >= total / 20 && above * 10 >= below * AscenderRatioTenths;
    }

    // Mean confidence of the sample lines read as they are and turned half way round.
    private readonly record struct Reading(double Upright, double Turned, double TurnedShare)
    {
        public double Best => Math.Max(Upright, Turned);

        public int Degrees(int quarter) => Turned > Upright + FlipMargin && Turned >= SureConfidence && TurnedShare >= TurnedShareNeeded ? ((quarter * 90) + 180) % 360 : quarter * 90;
    }

    // Turns the page by `quarter` quarter turns and straightens it by `slope`, then reads sample lines upright and half way round.
    private Reading Read(OrientationInput page, int quarter, int slope, CancellationToken cancellationToken)
    {
        var frame = new PageFrame(page.Width, page.Height, quarter, slope);
        Gray8[]? turned = null, straight = null;
        try
        {
            Gray8[] pixels = page.Pixels;
            int width = page.Width, height = page.Height;
            if (quarter != 0)
            {
                turned = _pixels.Rent((long)width * height);
                PageRotation.QuarterTurn(page.Pixels, width, height, quarter, turned, page.Degree, cancellationToken);
                pixels = turned;
                (width, height) = (frame.TurnedWidth, frame.TurnedHeight);
            }

            if (slope != 0)
            {
                straight = _pixels.Rent((long)frame.DeskewedWidth * frame.DeskewedHeight);
                byte background = PageRotation.Background(PixelSpans.AsBytes(pixels.AsSpan(0, width * height)));
                PageRotation.Deskew(pixels, width, height, frame.Sin, frame.Cos, background, straight, frame.DeskewedWidth, frame.DeskewedHeight, page.Degree, cancellationToken);
                pixels = straight;
                (width, height) = (frame.DeskewedWidth, frame.DeskewedHeight);
            }

            return Sample(pixels, width, height, page.Degree, cancellationToken);
        }
        finally
        {
            _pixels.Return(turned);
            _pixels.Return(straight);
        }
    }

    // Mean character-weighted confidence of up to SampleLines of the longest lines, as they are and turned half way round.
    private Reading Sample(Gray8[] pixels, int width, int height, int degree, CancellationToken cancellationToken)
    {
        int count = width * height;
        // The page finder's own strips: ruling lines and columns are handled as they are when the page is read, so a table does not
        // turn into a few long junk lines.
        List<StripLine> strips = StripFinder.Find(pixels, width, height, _limits.MaxLinesPerPage, _masks, degree, out _, out _, out _, eraseRules: false);
        var lines = new List<LineBox>(strips.Count);
        foreach (StripLine strip in strips)
        {
            lines.Add(strip.Box);
        }

        // The biggest strips are text lines; thin fragments of ruling lines (common after a page was turned and resampled) are many but small.
        // Their median height is the scale: the sample keeps strips of about that height.
        var candidates = new List<LineBox>();
        foreach (LineBox l in lines)
        {
            // Lines too wide for the model once scaled are no evidence.
            if (l.Width >= 3 * l.Height && Math.Ceiling((double)l.Width * _recognizer.InputHeight / Math.Max(1, l.Height)) <= _limits.MaxLineWidth / 2)
            {
                candidates.Add(l);
            }
        }

        // Biggest first; ties keep page order.
        LineBox[] bySize = [.. candidates.Select((l, i) => (l, i)).OrderByDescending(t => (long)t.l.Width * t.l.Height).ThenBy(t => t.i).Select(t => t.l).Take(3 * SampleLines)];
        int typical = LinePreprocessor.TypicalHeight(bySize);
        LineBox[] picked = [.. bySize.Where(l => (10 * l.Height >= 6 * typical) && (10 * l.Height <= 17 * typical)).Take(SampleLines)];
        int modelHeight = _recognizer.InputHeight;
        var weights = new double[picked.Length];
        var up = new double[picked.Length];
        var down = new double[picked.Length];
        var votes = new double[picked.Length];
        // A few lines first: when every one of them reads better upright, and well, the rest would only repeat that.
        int first = Math.Min(picked.Length, QuickLines);
        ReadLines(0, first);
        Reading quick = Summarize(first);
        if (picked.Length > first && quick.Upright >= QuickSure && quick.TurnedShare == 0)
        {
            return quick;
        }

        ReadLines(first, picked.Length);
        return Summarize(picked.Length);

        void ReadLines(int from, int to)
        {
            ParallelRun.For(to - from, degree, k =>
            {
                int n = from + k;
                LineBox crop = LinePreprocessor.CropBox(picked[n], typical, width, height);
                Gray8[] pixelsOfLine = LinePreprocessor.Prepare(PixelSpans.AsBytes(pixels.AsSpan(0, count)), width, crop, modelHeight, _pixels, _resizers, out int cropWidth);
                try
                {
                    int size = cropWidth * modelHeight;
                    byte[] a = PixelSpans.AsBytes(pixelsOfLine.AsSpan(0, size)).ToArray();
                                RecognizedLine readUp = _recognizer.Recognize(a, cropWidth, 1, cancellationToken);
                    Array.Reverse(a);
                    RecognizedLine readDown = _recognizer.Recognize(a, cropWidth, 1, cancellationToken);
                    // Weighted by the longer reading, so a line that reads as nothing either way adds no evidence.
                    double w = Math.Max(readUp.Text.Length, readDown.Text.Length);
                    weights[n] = w;
                    up[n] = readUp.Confidence * w;
                    down[n] = readDown.Confidence * w;
                    votes[n] = readDown.Confidence > readUp.Confidence ? w : 0;
                }
                finally
                {
                    _pixels.Return(pixelsOfLine);
                }
            }, cancellationToken);
        }

        Reading Summarize(int lines)
        {
            double total = 0, sumUp = 0, sumDown = 0, sumVotes = 0;
            for (int n = 0; n < lines; n++)
            {
                total += weights[n];
                sumUp += up[n];
                sumDown += down[n];
                sumVotes += votes[n];
            }

            return total == 0 ? new Reading(0, 0, 0) : new Reading(sumUp / total, sumDown / total, sumVotes / total);
        }
    }
}
