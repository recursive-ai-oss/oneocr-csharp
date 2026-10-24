using System.Globalization;
using OneImage;
using OneOcr.Detection;
using OneOcr.Kernels;

namespace OneOcr.Layout;

/// <summary>A page turned upright and straightened for reading, with the frame that maps results back.</summary>
internal sealed class PreparedPage
{
    private readonly BufferPool<Gray8> _pool;
    private readonly List<Gray8[]> _owned;

    public PreparedPage(Gray8[] pixels, int width, int height, PageFrame frame, List<string> notes, List<Gray8[]> owned, BufferPool<Gray8> pool)
    {
        Pixels = pixels;
        Width = width;
        Height = height;
        Frame = frame;
        Notes = notes;
        _owned = owned;
        _pool = pool;
    }

    /// <summary>The pixels to read: the input array itself when nothing was turned, otherwise a pooled array.</summary>
    public Gray8[] Pixels { get; }

    public int Width { get; }

    public int Height { get; }

    public PageFrame Frame { get; }

    public List<string> Notes { get; }

    /// <summary>The page detector's reading of <see cref="Pixels"/> when preparing the page already made it, so the reader need not detect again.</summary>
    public PageDetection? Detection { get; init; }

    /// <summary>The binarization (threshold, whether ink is bright) of <see cref="Pixels"/> when preparing the page already made it on those pixels.</summary>
    public (int Level, bool InkIsBright)? Level { get; init; }

    /// <summary>
    /// The page as the caller gets it: carried back through the frame when the page was straightened or its quarter turn is reported in
    /// the input frame, otherwise as read, plus the notes about what was done.
    /// </summary>
    public OcrPage Finish(OcrPage page)
    {
        if (!Frame.IsIdentity)
        {
            return Frame.MapPage(page, Notes);
        }

        if (Notes.Count == 0)
        {
            return page;
        }

        var diagnostics = new List<string>(page.Diagnostics);
        diagnostics.AddRange(Notes);
        return new OcrPage(
            page.Index, page.Width, page.Height, page.UnitsPerInch, page.Blocks, page.Regions, page.Tables, page.KeyValues, diagnostics, page.ToPdf, page.Orientation, page.SkewDegrees);
    }

    /// <summary>Gives the arrays made here back to the pool.</summary>
    public void Release()
    {
        foreach (Gray8[] a in _owned)
        {
            _pool.Return(a);
        }

        _owned.Clear();
    }
}

/// <summary>Orientation and deskew: the step between the page's pixels and line finding.</summary>
internal static class PagePreparation
{
    // Smaller tilts are left alone: they change nothing the recognizer can see, and an upright page then reads exactly as before.
    private const int MinSlope = 5;

    // The sharpest profile must beat the flat one by this much to count as evidence of a tilt.
    private const double MinGain = 1.03;

    /// <summary>The slope a page is straightened by: the estimate when it is a clear tilt, otherwise 0.</summary>
    public static int ActedSlope(SkewEstimate estimate) => Math.Abs(estimate.Slope) >= MinSlope && estimate.Gain >= MinGain ? estimate.Slope : 0;

    // The slope the page's text lines are straightened by (0 when the page is not clearly tilted).
    private static int SearchSlope(Gray8[] pixels, int w, int h, out (int Level, bool InkIsBright) binarization)
    {
        long clock = StageClock.Start();
        int slope = SearchSlopeUntimed(pixels, w, h, out binarization);
        StageClock.Stop("4 deskew search", clock);
        return slope;
    }

    private static int SearchSlopeUntimed(Gray8[] pixels, int w, int h, out (int Level, bool InkIsBright) binarization)
    {
        ReadOnlySpan<byte> bytes = PixelSpans.AsBytes(pixels.AsSpan(0, w * h));
        (int threshold, bool inkIsBright) = LineFinder.Binarize(bytes);
        binarization = (threshold, inkIsBright);
        if (threshold < 0)
        {
            return 0;
        }

        using InkGrid grid = InkGrid.Build(bytes, w, h, threshold, inkIsBright);
        SkewEstimate estimate = SkewEstimator.Estimate(grid);
        int slope = ActedSlope(estimate);
        // A tilt is only a tilt when the lines run across the page: with the lines running up and down (an orientation decision that
        // left a sideways page as it is) the "slope" is noise.
        return slope != 0 && SkewEstimator.EstimateTurned(grid).Contrast > estimate.Contrast ? 0 : slope;
    }

    // Straightens an upright-tilted page and reads the head on the straightened page. Null when the page cannot be prepared this way
    // (the straightened page would exceed the pixel limit, or the head asks for a turn); nothing is left behind then.
    private static PreparedPage? TryStraightFirst(
        HeadOrientation head, Gray8[] gray, int width, int height, int slope, OcrLimits limits, BufferPool<Gray8> pool, int degree, List<Gray8[]> owned,
        List<string> notes, CancellationToken cancellationToken)
    {
        var frame = new PageFrame(width, height, 0, slope);
        long canvas = (long)frame.DeskewedWidth * frame.DeskewedHeight;
        if (canvas > limits.MaxImagePixels)
        {
            return null;
        }

        Gray8[] straight = pool.Rent(canvas);
        bool kept = false;
        try
        {
            long clock = StageClock.Start();
            byte background = PageRotation.Background(PixelSpans.AsBytes(gray.AsSpan(0, width * height)));
            PageRotation.Deskew(gray, width, height, frame.Sin, frame.Cos, background, straight, frame.DeskewedWidth, frame.DeskewedHeight, degree, cancellationToken);
            StageClock.Stop("5 deskew warp", clock);
            PageDetection detection = head.Look(straight, frame.DeskewedWidth, frame.DeskewedHeight, degree, cancellationToken);
            if (PageOrientation.QuarterTurnsClockwise(detection) != 0)
            {
                return null;
            }

            owned.Add(straight);
            kept = true;
            notes.Add(string.Create(CultureInfo.InvariantCulture, $"The text lines were tilted by {frame.SkewDegrees:0.0} degrees and were straightened."));
            return new PreparedPage(straight, frame.DeskewedWidth, frame.DeskewedHeight, frame, notes, owned, pool) { Detection = detection };
        }
        finally
        {
            if (!kept)
            {
                pool.Return(straight);
            }
        }
    }

    public static PreparedPage Prepare(
        Gray8[] gray, int width, int height, bool orient, bool deskew, IOrientationDetector detector, OcrLimits limits,
        BufferPool<Gray8> pool, int degree, OrientationEvidence evidence, bool quarterInInputFrame, CancellationToken cancellationToken)
    {
        var owned = new List<Gray8[]>();
        var notes = new List<string>();
        try
        {
            // Straightening first. On a tilted page whose lines run across it the orientation head would be read twice: once on the
            // tilted page (to learn the quarter turn) and once more on the straightened page the reader needs anyway. The head is
            // trained on flat pages, so it is read on the straightened page only; if it then asks for a turn the page is prepared
            // the usual way (turn first, then straighten), which is the one place a tilted page costs an extra detection.
            int? upright = null;
            (int Level, bool InkIsBright) uprightLevel = default;
            if (orient && deskew && detector is HeadOrientation head)
            {
                int tilt = SearchSlope(gray, width, height, out uprightLevel);
                upright = tilt;
                if (tilt != 0)
                {
                    PreparedPage? early = TryStraightFirst(head, gray, width, height, tilt, limits, pool, degree, owned, notes, cancellationToken);
                    if (early is not null)
                    {
                        return early;
                    }
                }
            }

            int quarter = orient ? (detector.Detect(new OrientationInput(gray, width, height, degree, evidence), cancellationToken) / 90) & 3 : 0;
            Gray8[] pixels = gray;
            int w = width, h = height;
            if (quarter != 0)
            {
                // The one rotation step both orientation sources share.
                cancellationToken.ThrowIfCancellationRequested();
                Gray8[] turned = PageOrientation.Rotate(gray.AsSpan(0, width * height), width, height, quarter, pool, out w, out h);
                owned.Add(turned);
                pixels = turned;
                notes.Add(
                    quarterInInputFrame
                        ? string.Create(CultureInfo.InvariantCulture, $"The page was turned {quarter * 90} degrees clockwise to read upright.")
                        : DetectorLines.TurnedNote(quarter, evidence.Detection?.OrientationConfidence ?? 1f));
            }

            int slope = 0;
            (int Level, bool InkIsBright)? level = null;
            if (deskew)
            {
                // The search above looked at these very pixels when no turn was made.
                // The classical check (run when the head asked for a turn) erases ruling lines from the pixels, so a search made before it
                // is stale.
                if (quarter == 0 && upright is int known && !evidence.PixelsMayHaveChanged)
                {
                    slope = known;
                    level = uprightLevel;
                }
                else
                {
                    slope = SearchSlope(pixels, w, h, out (int, bool) searched);
                    level = searched;
                }
            }

            // With the quarter turn reported in the input frame the frame carries it back; otherwise the turned page is the page and
            // the frame covers the deskew turn alone.
            var frame = quarterInInputFrame ? new PageFrame(width, height, quarter, slope) : new PageFrame(w, h, 0, slope);
            if (slope != 0)
            {
                long canvas = (long)frame.DeskewedWidth * frame.DeskewedHeight;
                if (canvas > limits.MaxImagePixels)
                {
                    notes.Add("The text lines are tilted but the straightened page would exceed the pixel limit, so it was read as it is.");
                    frame = new PageFrame(width, height, quarter, 0);
                }
                else
                {
                    Gray8[] straight = pool.Rent(canvas);
                    owned.Add(straight);
                    byte background = PageRotation.Background(PixelSpans.AsBytes(pixels.AsSpan(0, w * h)));
                    long clock = StageClock.Start();
                    PageRotation.Deskew(pixels, w, h, frame.Sin, frame.Cos, background, straight, frame.DeskewedWidth, frame.DeskewedHeight, degree, cancellationToken);
                    StageClock.Stop("5 deskew warp", clock);
                    pixels = straight;
                    level = null;
                    (w, h) = (frame.DeskewedWidth, frame.DeskewedHeight);
                    notes.Add(string.Create(CultureInfo.InvariantCulture, $"The text lines were tilted by {frame.SkewDegrees:0.0} degrees and were straightened."));
                }
            }

            return new PreparedPage(pixels, w, h, frame, notes, owned, pool) { Level = level };
        }
        catch
        {
            foreach (Gray8[] a in owned)
            {
                pool.Return(a);
            }

            throw;
        }
    }
}
