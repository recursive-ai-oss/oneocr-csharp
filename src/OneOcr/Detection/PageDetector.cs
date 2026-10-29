using OneImage;
using OneOcr.Kernels;
using OneOcr.Layout;
using OneOcr.Recognition;

namespace OneOcr.Detection;

/// <summary>A text line found by the detector, in page pixels.</summary>
/// <param name="Left">Left edge.</param>
/// <param name="Top">Top edge.</param>
/// <param name="Right">Right edge.</param>
/// <param name="Bottom">Bottom edge.</param>
/// <param name="Score">Mean probability over the component's cells.</param>
/// <param name="Axis">The principal axis of the line's text-map component, when it has a reliable one; null otherwise.</param>
/// <param name="MinConfidence">0 for an ordinary line; for a line recovered from faint text, the confidence a word read from it needs to be kept.</param>
internal readonly record struct DetectedLine(double Left, double Top, double Right, double Bottom, float Score, LineAxis? Axis = null, float MinConfidence = 0);

/// <summary>
/// The direction and extent of a detected line along its principal axis (docs/spec/detector-v1.md, "Line axis"). The line frame has its
/// origin at the component's centroid, <c>u</c> along the text and <c>v</c> across it, pointing down: the page point of (u, v) is
/// <c>(CenterX + u*Cos - v*Sin, CenterY + u*Sin + v*Cos)</c>.
/// </summary>
/// <param name="CenterX">Centroid x in page pixels.</param>
/// <param name="CenterY">Centroid y in page pixels.</param>
/// <param name="Cos">Cosine of the axis angle (never negative).</param>
/// <param name="Sin">Sine of the axis angle; positive when the text descends to the right.</param>
/// <param name="U0">Start of the line along the axis, page pixels, grown like the box.</param>
/// <param name="U1">End of the line along the axis.</param>
/// <param name="V0">Top of the line across the axis.</param>
/// <param name="V1">Bottom of the line across the axis.</param>
/// <param name="Cells">Number of text-map cells in the component.</param>
/// <param name="Elongation">Ratio of the component's variance along the axis to the variance across it (plus one cell's own).</param>
internal readonly record struct LineAxis(
    double CenterX, double CenterY, double Cos, double Sin, double U0, double U1, double V0, double V1, int Cells, double Elongation);

/// <summary>A layout region found by the detector, in page pixels.</summary>
/// <param name="Class">Index of the layout class.</param>
/// <param name="Name">Name of the layout class (<c>text</c>, <c>title</c>, ...).</param>
/// <param name="Left">Left edge.</param>
/// <param name="Top">Top edge.</param>
/// <param name="Right">Right edge.</param>
/// <param name="Bottom">Bottom edge.</param>
/// <param name="Score">Mean probability over the component's cells.</param>
internal readonly record struct DetectedRegion(int Class, string Name, double Left, double Top, double Right, double Bottom, float Score);

/// <summary>What the detector found on one page.</summary>
internal sealed class PageDetection
{
    public required List<DetectedLine> Lines { get; init; }

    public required List<DetectedRegion> Regions { get; init; }

    /// <summary>Counter-clockwise quarter turns (0 to 3) the page content is rotated from upright, as the orientation head sees it.</summary>
    public required int Quarter { get; init; }

    /// <summary>The softmax probability of <see cref="Quarter"/>.</summary>
    public required float OrientationConfidence { get; init; }

    /// <summary>The page noise estimate, when the model carries the page-conditional threshold rule.</summary>
    public double? Sigma { get; init; }

    /// <summary>The speck density of the text map, when the model carries the rule.</summary>
    public double? Junk { get; init; }

    /// <summary>Whether the rule chose its own text threshold for this page.</summary>
    public bool RuleApplied { get; init; }
}

/// <summary>Something that finds text lines, layout regions and the orientation of a gray page.</summary>
internal interface IPageDetector
{
    /// <summary>Detects on an 8-bit gray page; boxes are in page pixels.</summary>
    PageDetection Detect(
        ReadOnlySpan<Gray8> page, int width, int height, BufferPool<Gray8> pixels, ResizerPool resizers, int degree, CancellationToken cancellationToken);
}

/// <summary>
/// Runs a <c>dbnet-lite-v0</c> detector on a gray page: scales the long side to 1024 pixels, pads to a multiple of 32 with white,
/// normalizes, runs the network and decodes the maps (docs/spec/detector-v0.md).
/// </summary>
internal sealed class PageDetector : IPageDetector
{
    /// <summary>Pixels of the long side the page is scaled to before it enters the network.</summary>
    public const int LongSide = 1024;

    private readonly DbNetModel _model;
    private readonly KernelPath _path;
    private readonly OcrLimits _limits;
    private readonly BufferPool<float> _floats;

    /// <param name="model">The detector network.</param>
    /// <param name="limits">Bounds to enforce.</param>
    /// <param name="path">The kernel path.</param>
    /// <param name="floats">Scratch pool to use; an engine passes the recognizer's, so the detector's scratch arrays, free once detection is over, serve the line reads instead of a second set being allocated beside them. A private pool when null.</param>
    public PageDetector(DbNetModel model, OcrLimits limits, KernelPath path, BufferPool<float>? floats = null)
    {
        _model = model;
        _limits = limits;
        _path = Dispatch.Require(path);
        _floats = floats ?? new BufferPool<float>(limits.MaxPooledBufferBytes, sizeof(float));
    }

    public DbNetModel Model => _model;

    /// <summary>The scale factor from page pixels to network pixels, <c>1024 / max(width, height)</c>.</summary>
    public static double Scale(int width, int height) => (double)LongSide / Math.Max(width, height);

    /// <summary>Detects lines, regions and orientation on an 8-bit gray page.</summary>
    /// <param name="page">The page pixels, row-major.</param>
    /// <param name="width">Page width in pixels.</param>
    /// <param name="height">Page height in pixels.</param>
    /// <param name="pixels">Pool for the scaled page.</param>
    /// <param name="resizers">Resizers for the scaling.</param>
    /// <param name="degree">Most threads to use; the result does not depend on it.</param>
    /// <param name="cancellationToken">Cancels between layers.</param>
    /// <returns>The detection in page pixels.</returns>
    public PageDetection Detect(
        ReadOnlySpan<Gray8> page, int width, int height, BufferPool<Gray8> pixels, ResizerPool resizers, int degree, CancellationToken cancellationToken)
    {
        if (width < 1 || height < 1 || page.Length < (long)width * height)
        {
            throw new ArgumentException("The page is smaller than width times height.", nameof(page));
        }

        double scale = Scale(width, height);
        int nw = Math.Max(1, (int)Math.Round(width * scale, MidpointRounding.ToEven));
        int nh = Math.Max(1, (int)Math.Round(height * scale, MidpointRounding.ToEven));
        int paddedW = (nw + 31) / 32 * 32, paddedH = (nh + 31) / 32 * 32;
        DbNetForward.CheckInput(_model, paddedH, paddedW, _limits);

        Gray8[]? scaled = null;
        float[]? canvas = null;
        try
        {
            ReadOnlySpan<Gray8> source = page[..(width * height)];
            long clock = StageClock.Start();
            if (nw != width || nh != height)
            {
                scaled = pixels.Rent((long)nw * nh);
                PixelResizer resizer = resizers.Rent();
                try
                {
                    // Gamma-space resampling (linear: false) because the model was trained on gamma-space bilinear resizes of the page;
                    // averaging in linear light would thin dark strokes relative to what it saw.
                    resizer.Resize<Gray8>(source, width, height, scaled.AsSpan(0, nw * nh), nw, nh, ResamplingFilter.Auto, linear: false, cancellationToken: cancellationToken);
                }
                finally
                {
                    resizers.Return(resizer);
                }

                source = scaled.AsSpan(0, nw * nh);
            }

            StageClock.Stop("n1 detector resize", clock);
            clock = StageClock.Start();
            double? sigma = _model.Postprocess.HasRule ? PageNoise.Sigma(source, nw, nh) : null;
            StageClock.Stop("n2 page noise", clock);
            clock = StageClock.Start();
            canvas = _floats.Rent((long)paddedW * paddedH);
            Normalize(source, nw, nh, paddedW, paddedH, _model.Input, canvas);
            StageClock.Stop("n3 normalize", clock);
            clock = StageClock.Start();
            DetectorMaps maps = DbNetForward.Run(_model, _path, degree, _floats, canvas, paddedH, paddedW, _limits, null, cancellationToken, pooledMaps: true);
            StageClock.Stop("n4 network", clock);
            try
            {
                clock = StageClock.Start();
                PageDetection found = DetectorPostprocess.Decode(maps, _model, scale, width, height, _limits, sigma, source, nw, nh);
                StageClock.Stop("n5 map decode", clock);
                return found;
            }
            finally
            {
                maps.Return(_floats);
            }
        }
        finally
        {
            if (scaled is not null)
            {
                pixels.Return(scaled);
            }

            if (canvas is not null)
            {
                _floats.Return(canvas);
            }
        }
    }

    /// <summary>Normalizes <paramref name="source"/> into the top-left of a padded canvas; the padding is white.</summary>
    internal static void Normalize(ReadOnlySpan<Gray8> source, int width, int height, int paddedW, int paddedH, ModelInputView input, float[] canvas)
    {
        Span<float> table = stackalloc float[256];
        for (int p = 0; p < 256; p++)
        {
            float x = p / 255f;
            if (input.Invert)
            {
                x = 1f - x;
            }

            table[p] = (x - input.Mean) / input.Std;
        }

        float white = table[255];
        for (int y = 0; y < paddedH; y++)
        {
            Span<float> row = canvas.AsSpan(y * paddedW, paddedW);
            if (y >= height)
            {
                row.Fill(white);
                continue;
            }

            ReadOnlySpan<Gray8> from = source.Slice(y * width, width);
            for (int x = 0; x < width; x++)
            {
                row[x] = table[from[x].V];
            }

            row[width..].Fill(white);
        }
    }
}
