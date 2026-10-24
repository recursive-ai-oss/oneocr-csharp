using System.Globalization;
using OneImage;
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

    public static PreparedPage Prepare(
        Gray8[] gray, int width, int height, bool orient, bool deskew, IOrientationDetector detector, OcrLimits limits,
        BufferPool<Gray8> pool, int degree, OrientationEvidence evidence, bool quarterInInputFrame, CancellationToken cancellationToken)
    {
        var owned = new List<Gray8[]>();
        var notes = new List<string>();
        try
        {
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
            if (deskew)
            {
                ReadOnlySpan<byte> bytes = PixelSpans.AsBytes(pixels.AsSpan(0, w * h));
                (int threshold, bool inkIsBright) = LineFinder.Binarize(bytes);
                if (threshold >= 0)
                {
                    using InkGrid grid = InkGrid.Build(bytes, w, h, threshold, inkIsBright);
                    SkewEstimate estimate = SkewEstimator.Estimate(grid);
                    slope = ActedSlope(estimate);
                    if (slope != 0)
                    {
                        // A tilt is only a tilt when the lines run across the page: with the lines running up and down (an orientation
                        // decision that left a sideways page as it is) the "slope" is noise.
                        if (SkewEstimator.EstimateTurned(grid).Contrast > estimate.Contrast)
                        {
                            slope = 0;
                        }
                    }
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
                    PageRotation.Deskew(pixels, w, h, frame.Sin, frame.Cos, background, straight, frame.DeskewedWidth, frame.DeskewedHeight, degree, cancellationToken);
                    pixels = straight;
                    (w, h) = (frame.DeskewedWidth, frame.DeskewedHeight);
                    notes.Add(string.Create(CultureInfo.InvariantCulture, $"The text lines were tilted by {frame.SkewDegrees:0.0} degrees and were straightened."));
                }
            }

            return new PreparedPage(pixels, w, h, frame, notes, owned, pool);
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
