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
        BufferPool<Gray8> pool, int degree, CancellationToken cancellationToken)
    {
        var owned = new List<Gray8[]>();
        var notes = new List<string>();
        try
        {
            int quarter = orient ? (detector.Detect(new OrientationInput(gray, width, height, degree), cancellationToken) / 90) & 3 : 0;
            Gray8[] pixels = gray;
            int w = width, h = height;
            if (quarter != 0)
            {
                Gray8[] turned = pool.Rent((long)width * height);
                owned.Add(turned);
                PageRotation.QuarterTurn(gray, width, height, quarter, turned, degree, cancellationToken);
                pixels = turned;
                if ((quarter & 1) != 0)
                {
                    (w, h) = (height, width);
                }

                notes.Add(string.Create(CultureInfo.InvariantCulture, $"The page was turned {quarter * 90} degrees clockwise to read upright."));
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
                }
            }

            var frame = new PageFrame(width, height, quarter, slope);
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
