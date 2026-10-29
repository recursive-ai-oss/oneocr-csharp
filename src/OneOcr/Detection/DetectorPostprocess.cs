using OneImage;
using OneOcr.Kernels;

namespace OneOcr.Detection;

/// <summary>
/// Turns the detector's logit maps into boxes (docs/spec/detector-v0.md, "Post-processing"): threshold, 4-connected components, the
/// component's cell extent as the shrunk core, grown back by the inverse of the training shrink. Safe code over validated maps.
/// </summary>
internal static class DetectorPostprocess
{
    /// <summary>Stride of the text map relative to the network input.</summary>
    public const int TextStride = 2;

    /// <summary>Stride of the layout maps relative to the network input.</summary>
    public const int LayoutStride = 8;

    /// <summary>A text cell is on when its logit exceeds ln(0.3 / 0.7), that is, its probability exceeds 0.3.</summary>
    public const float TextLogitThreshold = -0.847297860387203f;

    /// <summary>A layout cell is on when its logit exceeds ln(0.6 / 0.4), that is, its probability exceeds 0.6.</summary>
    public const float LayoutLogitThreshold = 0.405465108108164f;

    /// <summary>Components of fewer cells than this are not text lines.</summary>
    public const int MinTextCells = 2;

    /// <summary>The training shrank each text line by this fraction of its short side on each side.</summary>
    public const double TextShrink = 0.3;

    /// <summary>Regions whose core is at most this many network pixels grow by a third of it; larger ones by <see cref="LayoutGrowCap"/>.</summary>
    public const double LayoutGrowSmall = 60;

    /// <summary>Largest growth of a layout core in network pixels.</summary>
    public const double LayoutGrowCap = 20;

    /// <summary>How far, in network pixels, a region snapped to its text lines may leave the region's own box.</summary>
    public const double SnapTolerance = 8;

    /// <summary>A page whose orientation head is less sure than this is left as it is.</summary>
    public const float OrientationConfidenceMinimum = 0.9f;

    /// <summary>The post-processing values a model may carry in <c>config.postprocess</c>; the defaults are the values of the first model.</summary>
    /// <param name="TextThreshold">Probability above which a text cell is on.</param>
    /// <param name="TextExpand">Factor on the growth of a text line core.</param>
    /// <param name="LayoutThreshold">Probability above which a layout cell is on.</param>
    /// <param name="LayoutExpand">Factor on the growth of a layout region core.</param>
    /// <param name="RuleSigmaMin">Page noise above which the page-conditional text threshold may apply; null (with the other two) turns the rule off.</param>
    /// <param name="RuleJunkMax">Speck density (per megapixel) below which the rule applies.</param>
    /// <param name="RuleTextThreshold">Probability above which a text cell is on when the rule applies.</param>
    /// <param name="Faint">The faint-text recovery on dark backgrounds; null turns it off.</param>
    internal readonly record struct Options(
        double TextThreshold, double TextExpand, double LayoutThreshold, double LayoutExpand,
        double? RuleSigmaMin = null, double? RuleJunkMax = null, double? RuleTextThreshold = null, FaintRule? Faint = null)
    {
        /// <summary>Whether the page-conditional text threshold is configured.</summary>
        public bool HasRule => RuleSigmaMin is not null && RuleJunkMax is not null && RuleTextThreshold is not null;

        /// <summary>Text 0.3 / 1.0, layout 0.6 / 0.85.</summary>
        public static Options Default { get; } = new(0.3, 1.0, 0.6, 0.85);

        /// <summary>The threshold as a logit, compared in float32; the defaults use the exact constants.</summary>
        internal static float Logit(double p, double dflt, float dfltLogit) =>
            p == dflt ? dfltLogit : (float)Math.Log(p / (1 - p));
    }

    /// <summary>
    /// The faint-text recovery (docs/spec/detector-v1.md, "Faint text on dark backgrounds"): on pages the noise rule does not select, text-map
    /// components that only appear at a lower threshold are kept as lines when they sit on a dark background; the recognizer must then read
    /// them with at least <see cref="MinConfidence"/>.
    /// </summary>
    /// <param name="TextThreshold">Probability above which a cell is on for the second pass (below the page's own text threshold).</param>
    /// <param name="BackgroundMax">Largest median gray value (0 to 255) of a candidate's box for it to count as a dark background.</param>
    /// <param name="MinCells">Fewest text-map cells a candidate has.</param>
    /// <param name="MinAspect">Smallest width over height of a candidate's cell box.</param>
    /// <param name="MinConfidence">Smallest calibrated word confidence a word read from a recovered line needs to be kept.</param>
    internal readonly record struct FaintRule(double TextThreshold, int BackgroundMax, int MinCells, double MinAspect, double MinConfidence);

    /// <summary>Components with fewer cells than this get no principal axis: a handful of cells has no reliable direction.</summary>
    public const int MinAxisCells = 12;

    // The principal axis of a component in cell units (cell centres at x + 0.5), and its extent along and across it.
    private readonly record struct CellAxis(double Cx, double Cy, double Cos, double Sin, double U0, double U1, double V0, double V1, double Elongation, int Cells);

    private readonly record struct Component(int X0, int Y0, int X1, int Y1, int Count, double ScoreSum, CellAxis? Axis = null)
    {
        public float Score => (float)(ScoreSum / Count);
    }

    /// <summary>Decodes the maps of one page.</summary>
    /// <param name="maps">The network outputs.</param>
    /// <param name="model">The detector, for the class names and which classes are trained.</param>
    /// <param name="scale">Network pixels per page pixel.</param>
    /// <param name="pageWidth">Page width in pixels.</param>
    /// <param name="pageHeight">Page height in pixels.</param>
    /// <param name="limits">Bounds on the number of lines and regions.</param>
    /// <param name="sigma">Noise estimate of the page at detector scale (<see cref="PageNoise.Sigma"/>), when the model's rule needs it.</param>
    /// <returns>Lines, regions and orientation in page pixels.</returns>
    /// <param name="gray">The gray page at detector scale (<paramref name="grayWidth"/> by <paramref name="grayHeight"/>), when the model's faint-text recovery needs it.</param>
    /// <param name="grayWidth">Width of <paramref name="gray"/>.</param>
    /// <param name="grayHeight">Height of <paramref name="gray"/>.</param>
    public static PageDetection Decode(
        DetectorMaps maps, DbNetModel model, double scale, int pageWidth, int pageHeight, OcrLimits limits, double? sigma = null,
        ReadOnlySpan<Gray8> gray = default, int grayWidth = 0, int grayHeight = 0) =>
        Decode(
            maps.Prob, maps.ProbWidth, maps.ProbHeight, maps.Layout, maps.LayoutWidth, maps.LayoutHeight, maps.LayoutStride,
            model.LayoutClasses, model.LayoutTrained, maps.Orientation, scale, pageWidth, pageHeight, limits, model.Postprocess, sigma, gray, grayWidth, grayHeight);

    /// <summary>Decodes maps given as plain arrays: text logits <c>[h][w]</c> and layout logits <c>[h][w][stride]</c>.</summary>
    internal static PageDetection Decode(
        float[] prob, int probWidth, int probHeight, float[] layout, int layoutWidth, int layoutHeight, int layoutStride,
        IReadOnlyList<string> classNames, IReadOnlyList<bool> trained, float[] orientation, double scale, int pageWidth, int pageHeight, OcrLimits limits, Options? options = null, double? sigma = null,
        ReadOnlySpan<Gray8> gray = default, int grayWidth = 0, int grayHeight = 0)
    {
        Options o = options ?? Options.Default;
        float textThreshold = Options.Logit(o.TextThreshold, 0.3, TextLogitThreshold);
        double? junk = null;
        bool ruleApplied = false;
        if (o.HasRule && sigma is not null)
        {
            junk = PageNoise.JunkDensity(prob, probWidth, probHeight);
            if (sigma > o.RuleSigmaMin && junk < o.RuleJunkMax)
            {
                textThreshold = Options.Logit(o.RuleTextThreshold!.Value, 0.3, TextLogitThreshold);
                ruleApplied = true;
            }
        }

        float layoutThreshold = Options.Logit(o.LayoutThreshold, 0.6, LayoutLogitThreshold);
        // Lines, in network pixels first: regions snap to them before everything is scaled back to the page.
        List<(double X0, double Y0, double X1, double Y1, float Score, CellAxis? Axis)> lines = [];
        const double k = TextShrink / (1 - (2 * TextShrink));
        List<Component> found = Label(prob, probWidth, probHeight, 1, 0, textThreshold, MinTextCells, limits.MaxLinesPerPage, nameof(OcrLimits.MaxLinesPerPage), withAxis: true);
        foreach (Component c in found)
        {
            double x0 = c.X0 * TextStride, y0 = c.Y0 * TextStride, x1 = (c.X1 + 1) * TextStride, y1 = (c.Y1 + 1) * TextStride;
            double d = k * Math.Min(x1 - x0, y1 - y0) * o.TextExpand;
            lines.Add((x0 - d, y0 - d, x1 + d, y1 + d, c.Score, c.Axis is { } a ? GrowAxis(a, k * o.TextExpand) : null));
        }

        int table = IndexOf(classNames, "table"), figure = IndexOf(classNames, "figure");
        List<(int Class, double X0, double Y0, double X1, double Y1, float Score)> regions = [];
        int total = 0;
        for (int cls = 0; cls < classNames.Count; cls++)
        {
            if (!trained[cls])
            {
                continue;
            }

            foreach (Component c in Label(layout, layoutWidth, layoutHeight, layoutStride, cls, layoutThreshold, 1, limits.MaxPageElements - total, nameof(OcrLimits.MaxPageElements)))
            {
                total++;
                double x0 = c.X0 * LayoutStride, y0 = c.Y0 * LayoutStride, x1 = (c.X1 + 1) * LayoutStride, y1 = (c.Y1 + 1) * LayoutStride;
                double core = Math.Min(x1 - x0, y1 - y0);
                double d = (core <= LayoutGrowSmall ? core / 3.0 : LayoutGrowCap) * o.LayoutExpand;
                var box = (X0: x0 - d, Y0: y0 - d, X1: x1 + d, Y1: y1 + d);
                if (cls != table && cls != figure)
                {
                    box = Snap(box, lines);
                }

                regions.Add((cls, box.X0, box.Y0, box.X1, box.Y1, c.Score));
            }
        }

        // Faint lines come after everything else and take no part in the regions' snapping, so a page without any is decoded as before.
        List<(double X0, double Y0, double X1, double Y1, float Score, CellAxis? Axis)> faintLines = [];
        if (o.Faint is { } faint && !ruleApplied && !gray.IsEmpty)
        {
            FindFaint(faintLines, found, lines, prob, probWidth, probHeight, textThreshold, faint, o.TextExpand, gray, grayWidth, grayHeight, limits.MaxLinesPerPage - found.Count);
        }

        var outLines = new List<DetectedLine>(lines.Count + faintLines.Count);
        foreach (var l in lines)
        {
            if (Clip(l.X0 / scale, l.Y0 / scale, l.X1 / scale, l.Y1 / scale, pageWidth, pageHeight) is { } b)
            {
                outLines.Add(new DetectedLine(b.L, b.T, b.R, b.B, l.Score, l.Axis is { } ax ? ToPage(ax, scale) : null));
            }
        }

        foreach (var l in faintLines)
        {
            if (Clip(l.X0 / scale, l.Y0 / scale, l.X1 / scale, l.Y1 / scale, pageWidth, pageHeight) is { } b)
            {
                outLines.Add(new DetectedLine(b.L, b.T, b.R, b.B, l.Score, l.Axis is { } ax ? ToPage(ax, scale) : null, (float)o.Faint!.Value.MinConfidence));
            }
        }

        var outRegions = new List<DetectedRegion>(regions.Count);
        foreach (var r in regions)
        {
            if (Clip(r.X0 / scale, r.Y0 / scale, r.X1 / scale, r.Y1 / scale, pageWidth, pageHeight) is { } b)
            {
                outRegions.Add(new DetectedRegion(r.Class, classNames[r.Class], b.L, b.T, b.R, b.B, r.Score));
            }
        }

        (int quarter, float confidence) = Orientation(orientation);
        return new PageDetection { Lines = outLines, Regions = outRegions, Quarter = quarter, OrientationConfidence = confidence, Sigma = sigma, Junk = junk, RuleApplied = ruleApplied };
    }

    // Text-map components that exist only below the page's text threshold, away from every ordinary line, on a dark background. Cells
    // next to an ordinary line (its fringe) are masked first, so a line's own halo is never a candidate; a candidate that still overlaps
    // an ordinary line's cell box by more than a third of the smaller box is dropped.
    private static void FindFaint(
        List<(double X0, double Y0, double X1, double Y1, float Score, CellAxis? Axis)> into, List<Component> main,
        List<(double X0, double Y0, double X1, double Y1, float Score, CellAxis? Axis)> mainLines, float[] prob, int width, int height, float mainThreshold, FaintRule rule, double expand, ReadOnlySpan<Gray8> gray, int grayWidth, int grayHeight, int room)
    {
        int cells = width * height;
        if (room <= 0 || cells == 0 || (long)grayWidth * grayHeight > gray.Length)
        {
            return;
        }

        float faintThreshold = (float)Math.Log(rule.TextThreshold / (1 - rule.TextThreshold));
        float[] masked = System.Buffers.ArrayPool<float>.Shared.Rent(cells);
        try
        {
            Array.Copy(prob, masked, cells);
            foreach (Component c in main)
            {
                // The ordinary component's own cells and their eight neighbours are not candidates; a component's box bounds them.
                for (int y = Math.Max(0, c.Y0 - 1); y <= Math.Min(height - 1, c.Y1 + 1); y++)
                {
                    for (int x = Math.Max(0, c.X0 - 1); x <= Math.Min(width - 1, c.X1 + 1); x++)
                    {
                        if (prob[(y * width) + x] > mainThreshold || NearOn(prob, width, height, x, y, mainThreshold))
                        {
                            masked[(y * width) + x] = float.NegativeInfinity;
                        }
                    }
                }
            }

            const double k = TextShrink / (1 - (2 * TextShrink));
            Span<int> histogram = stackalloc int[256];
            foreach (Component c in Label(masked, width, height, 1, 0, faintThreshold, Math.Max(MinTextCells, rule.MinCells), room, nameof(OcrLimits.MaxLinesPerPage), withAxis: true))
            {
                int w = c.X1 - c.X0 + 1, h = c.Y1 - c.Y0 + 1;
                if (w < rule.MinAspect * h || OverlapsMain(main, c))
                {
                    continue;
                }

                // The median gray of the cell box, in detector pixels (one cell is TextStride pixels).
                histogram.Clear();
                int count = 0;
                for (int y = c.Y0 * TextStride; y < Math.Min(grayHeight, (c.Y1 + 1) * TextStride); y++)
                {
                    for (int x = c.X0 * TextStride; x < Math.Min(grayWidth, (c.X1 + 1) * TextStride); x++)
                    {
                        histogram[gray[(y * grayWidth) + x].V]++;
                        count++;
                    }
                }

                int seen = 0, median = 0;
                while (median < 255 && seen + histogram[median] < (count + 1) / 2)
                {
                    seen += histogram[median++];
                }

                if (count == 0 || median > rule.BackgroundMax)
                {
                    continue;
                }

                double x0 = c.X0 * TextStride, y0 = c.Y0 * TextStride, x1 = (c.X1 + 1) * TextStride, y1 = (c.Y1 + 1) * TextStride;
                (x0, x1) = ExtendAlongLine(gray, grayWidth, grayHeight, (int)x0, (int)y0, (int)x1, (int)y1, median, rule.BackgroundMax, mainLines);
                double d = k * Math.Min(x1 - x0, y1 - y0) * expand;
                (double X0, double Y0, double X1, double Y1) box = (x0 - d, y0 - d, x1 + d, y1 + d);
                if (OverlapsLine(mainLines, box))
                {
                    continue;
                }

                into.Add((box.X0, box.Y0, box.X1, box.Y1, c.Score, c.Axis is { } a ? GrowAxis(a, k * expand) : null));
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(masked);
        }
    }

    /// <summary>A pixel this far above the background's median gray counts as ink when a faint line is followed past its detected cells.</summary>
    public const int FaintInkStep = 5;

    // The text map holds only the most visible stretch of a faint line, so the line is followed along its rows: left and right of the
    // detected cells for as long as columns of the band keep ink (a pixel at least FaintInkStep above the median), over gaps of up to
    // one band height. It stops where the band turns brighter than the background limit, and before any ordinary line that shares its rows.
    private static (double X0, double X1) ExtendAlongLine(
        ReadOnlySpan<Gray8> gray, int width, int height, int x0, int y0, int x1, int y1, int median, int backgroundMax,
        List<(double X0, double Y0, double X1, double Y1, float Score, CellAxis? Axis)> mainLines)
    {
        y1 = Math.Min(y1, height);
        x1 = Math.Min(x1, width);
        if (y1 <= y0 || x1 <= x0)
        {
            return (x0, x1);
        }

        int band = y1 - y0, gap = Math.Max(6, band);
        int leftLimit = 0, rightLimit = width;
        foreach (var l in mainLines)
        {
            double shared = Math.Min(y1, l.Y1) - Math.Max(y0, l.Y0);
            if (shared * 2 < band)
            {
                continue;
            }

            if (l.X1 <= x0)
            {
                leftLimit = Math.Max(leftLimit, (int)Math.Ceiling(l.X1));
            }
            else if (l.X0 >= x1)
            {
                rightLimit = Math.Min(rightLimit, (int)Math.Floor(l.X0));
            }
        }

        int newLeft = x0, empty = 0;
        for (int x = x0 - 1; x >= leftLimit && empty <= gap; x--)
        {
            if (ColumnHasInk(gray, width, x, y0, y1, median, backgroundMax))
            {
                newLeft = x;
                empty = 0;
            }
            else
            {
                empty++;
            }
        }

        int newRight = x1;
        empty = 0;
        for (int x = x1; x < rightLimit && empty <= gap; x++)
        {
            if (ColumnHasInk(gray, width, x, y0, y1, median, backgroundMax))
            {
                newRight = x + 1;
                empty = 0;
            }
            else
            {
                empty++;
            }
        }

        return (newLeft, newRight);
    }

    private static bool ColumnHasInk(ReadOnlySpan<Gray8> gray, int width, int x, int y0, int y1, int median, int backgroundMax)
    {
        int sum = 0, top = median + FaintInkStep;
        bool ink = false;
        for (int y = y0; y < y1; y++)
        {
            int v = gray[(y * width) + x].V;
            sum += v;
            ink |= v >= top;
        }

        // A column of the band that is bright on average is another background, not a faint line's gap.
        return sum <= backgroundMax * (y1 - y0) && ink;
    }

    private static bool NearOn(float[] prob, int width, int height, int x, int y, float threshold)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            int yy = y + dy;
            if (yy < 0 || yy >= height)
            {
                continue;
            }

            for (int dx = -1; dx <= 1; dx++)
            {
                int xx = x + dx;
                if (xx >= 0 && xx < width && prob[(yy * width) + xx] > threshold)
                {
                    return true;
                }
            }
        }

        return false;
    }

    // Whether the grown box covers more than a quarter of the smaller of itself and an ordinary line's grown box: it would read text that is read already.
    private static bool OverlapsLine(List<(double X0, double Y0, double X1, double Y1, float Score, CellAxis? Axis)> lines, (double X0, double Y0, double X1, double Y1) box)
    {
        double area = (box.X1 - box.X0) * (box.Y1 - box.Y0);
        foreach (var l in lines)
        {
            double w = Math.Min(box.X1, l.X1) - Math.Max(box.X0, l.X0), h = Math.Min(box.Y1, l.Y1) - Math.Max(box.Y0, l.Y0);
            if (w <= 0 || h <= 0)
            {
                continue;
            }

            if (4 * w * h > Math.Min(area, (l.X1 - l.X0) * (l.Y1 - l.Y0)))
            {
                return true;
            }
        }

        return false;
    }

    private static bool OverlapsMain(List<Component> main, Component c)
    {
        long area = (long)(c.X1 - c.X0 + 1) * (c.Y1 - c.Y0 + 1);
        foreach (Component m in main)
        {
            int w = Math.Min(c.X1, m.X1) - Math.Max(c.X0, m.X0) + 1, h = Math.Min(c.Y1, m.Y1) - Math.Max(c.Y0, m.Y0) + 1;
            if (w <= 0 || h <= 0)
            {
                continue;
            }

            long smaller = Math.Min(area, (long)(m.X1 - m.X0 + 1) * (m.Y1 - m.Y0 + 1));
            if (3L * w * h > smaller)
            {
                return true;
            }
        }

        return false;
    }

    // The axis extent grown like the axis-aligned box: by k * expand times the shorter side, on each side, in network pixels.
    private static CellAxis GrowAxis(CellAxis a, double factor)
    {
        double length = (a.U1 - a.U0) * TextStride, thickness = (a.V1 - a.V0) * TextStride;
        double d = factor * Math.Min(length, thickness) / TextStride;
        return a with { U0 = a.U0 - d, U1 = a.U1 + d, V0 = a.V0 - d, V1 = a.V1 + d };
    }

    // Cell units to page pixels: one cell is TextStride network pixels, one network pixel 1 / scale page pixels.
    private static LineAxis ToPage(CellAxis a, double scale)
    {
        double f = TextStride / scale;
        return new LineAxis(a.Cx * f, a.Cy * f, a.Cos, a.Sin, a.U0 * f, a.U1 * f, a.V0 * f, a.V1 * f, a.Cells, a.Elongation);
    }

    /// <summary>Softmax over the orientation logits with the library's own exp; the winner (lowest index on a tie) and its probability.</summary>
    internal static (int Quarter, float Confidence) Orientation(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }

        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            sum += FastMath.Exp(logits[i] - logits[best]);
        }

        return (best, 1f / sum);
    }

    private static int IndexOf(IReadOnlyList<string> names, string name)
    {
        for (int i = 0; i < names.Count; i++)
        {
            if (names[i] == name)
            {
                return i;
            }
        }

        return -1;
    }

    // The union of the lines whose centres lie in the box, limited to the box grown by the tolerance; the box itself when no line lies
    // inside or the union would be empty.
    private static (double X0, double Y0, double X1, double Y1) Snap(
        (double X0, double Y0, double X1, double Y1) box, List<(double X0, double Y0, double X1, double Y1, float Score, CellAxis? Axis)> lines)
    {
        double u0 = double.MaxValue, v0 = double.MaxValue, u1 = double.MinValue, v1 = double.MinValue;
        bool any = false;
        foreach (var l in lines)
        {
            double cx = (l.X0 + l.X1) / 2, cy = (l.Y0 + l.Y1) / 2;
            if (cx >= box.X0 && cx <= box.X1 && cy >= box.Y0 && cy <= box.Y1)
            {
                any = true;
                u0 = Math.Min(u0, l.X0);
                v0 = Math.Min(v0, l.Y0);
                u1 = Math.Max(u1, l.X1);
                v1 = Math.Max(v1, l.Y1);
            }
        }

        if (!any)
        {
            return box;
        }

        var snapped = (X0: Math.Max(u0, box.X0 - SnapTolerance), Y0: Math.Max(v0, box.Y0 - SnapTolerance),
            X1: Math.Min(u1, box.X1 + SnapTolerance), Y1: Math.Min(v1, box.Y1 + SnapTolerance));
        return snapped.X1 > snapped.X0 && snapped.Y1 > snapped.Y0 ? snapped : box;
    }

    private static (double L, double T, double R, double B)? Clip(double x0, double y0, double x1, double y1, int width, int height)
    {
        double l = Math.Clamp(x0, 0, width), t = Math.Clamp(y0, 0, height), r = Math.Clamp(x1, 0, width), b = Math.Clamp(y1, 0, height);
        return r > l && b > t ? (l, t, r, b) : null;
    }

    // 4-connected components of the cells whose logit exceeds the threshold, in the order of their first cell in row-major order.
    // Iterative (an explicit stack), so no input makes the depth grow. Components with fewer than minCells cells are dropped; more than
    // maxCount kept components is a limit error, raised as soon as the count passes it.
    private static List<Component> Label(
        float[] logits, int width, int height, int stride, int channel, float threshold, int minCells, int maxCount, string limitName, bool withAxis = false)
    {
        var result = new List<Component>();
        if (width < 1 || height < 1)
        {
            return result;
        }

        int cells = checked(width * height);
        if (logits.Length < ((long)(cells - 1) * stride) + channel + 1)
        {
            throw new ArgumentException("The map is smaller than its declared shape.", nameof(logits));
        }

        // Scratch from the shared pools: a page's maps are large, and fresh arrays per page would pile up as garbage.
        bool[] seen = System.Buffers.ArrayPool<bool>.Shared.Rent(cells);
        int[] stack = System.Buffers.ArrayPool<int>.Shared.Rent(cells);
        int[]? order = withAxis ? System.Buffers.ArrayPool<int>.Shared.Rent(cells) : null;
        try
        {
            Array.Clear(seen, 0, cells);
            LabelInto(result, seen, stack, order, logits, width, height, stride, channel, threshold, minCells, maxCount, limitName);
        }
        finally
        {
            if (order is not null)
            {
                System.Buffers.ArrayPool<int>.Shared.Return(order);
            }

            System.Buffers.ArrayPool<bool>.Shared.Return(seen);
            System.Buffers.ArrayPool<int>.Shared.Return(stack);
        }

        return result;
    }

    private static void LabelInto(
        List<Component> result, bool[] seen, int[] stack, int[]? order, float[] logits, int width, int height, int stride, int channel, float threshold,
        int minCells, int maxCount, string limitName)
    {
        int cells = width * height;
        int ordered = 0;
        for (int start = 0; start < cells; start++)
        {
            if (seen[start] || !(logits[((long)start * stride) + channel] > threshold))
            {
                continue;
            }

            int top = 0;
            stack[top++] = start;
            seen[start] = true;
            int x0 = int.MaxValue, y0 = int.MaxValue, x1 = -1, y1 = -1, count = 0, firstOrdered = ordered;
            double sum = 0, sumX = 0, sumY = 0;
            while (top > 0)
            {
                int cell = stack[--top];
                int y = cell / width, x = cell - (y * width);
                if (order is not null)
                {
                    order[ordered++] = cell;
                    sumX += x;
                    sumY += y;
                }

                count++;
                sum += FastMath.Sigmoid(logits[((long)cell * stride) + channel]);
                x0 = Math.Min(x0, x);
                x1 = Math.Max(x1, x);
                y0 = Math.Min(y0, y);
                y1 = Math.Max(y1, y);
                if (x > 0 && !seen[cell - 1] && logits[((long)(cell - 1) * stride) + channel] > threshold)
                {
                    seen[cell - 1] = true;
                    stack[top++] = cell - 1;
                }

                if (x + 1 < width && !seen[cell + 1] && logits[((long)(cell + 1) * stride) + channel] > threshold)
                {
                    seen[cell + 1] = true;
                    stack[top++] = cell + 1;
                }

                if (y > 0 && !seen[cell - width] && logits[((long)(cell - width) * stride) + channel] > threshold)
                {
                    seen[cell - width] = true;
                    stack[top++] = cell - width;
                }

                if (y + 1 < height && !seen[cell + width] && logits[((long)(cell + width) * stride) + channel] > threshold)
                {
                    seen[cell + width] = true;
                    stack[top++] = cell + width;
                }
            }

            if (count >= minCells)
            {
                if (result.Count >= maxCount)
                {
                    throw new OcrLimitException(limitName, maxCount, (long)result.Count + 1);
                }

                CellAxis? axis = order is not null && count >= MinAxisCells ? AxisOf(order.AsSpan(firstOrdered, count), width, sumX / count, sumY / count) : null;
                result.Add(new Component(x0, y0, x1, y1, count, sum, axis));
            }
        }
    }

    // The principal axis of a component's cells (the eigenvector of their 2x2 covariance with the larger eigenvalue), its centroid, and
    // the extent of the cell centres along and across it, each side widened by half a cell. Only +, -, *, / and square root, which
    // every platform rounds the same way; the cells are visited in the order the flood fill found them, so the sums are the same
    // on every run. Null when the cells have no direction (a round blob).
    private static CellAxis? AxisOf(ReadOnlySpan<int> cells, int width, double meanX, double meanY)
    {
        double sxx = 0, sxy = 0, syy = 0;
        foreach (int cell in cells)
        {
            int y = cell / width, x = cell - (y * width);
            double dx = x - meanX, dy = y - meanY;
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }

        double n = cells.Length;
        double a = sxx / n, b = sxy / n, c = syy / n;
        double half = (a - c) / 2;
        double disc = Math.Sqrt((half * half) + (b * b));
        if (!(disc > 1e-9))
        {
            return null;
        }

        double major = ((a + c) / 2) + disc, minor = Math.Max(0, ((a + c) / 2) - disc);
        // Double angle: cos 2t = (a - c) / (2 disc), sin 2t = b / disc; the axis points to the right (cos t >= 0).
        double cos2 = half / disc;
        double cosT = Math.Sqrt(Math.Max(0, (1 + cos2) / 2));
        double sinT = Math.Sqrt(Math.Max(0, (1 - cos2) / 2));
        if (b < 0)
        {
            sinT = -sinT;
        }

        double cx = meanX + 0.5, cy = meanY + 0.5;
        double u0 = double.MaxValue, u1 = double.MinValue, v0 = double.MaxValue, v1 = double.MinValue;
        foreach (int cell in cells)
        {
            int y = cell / width, x = cell - (y * width);
            double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
            double u = (dx * cosT) + (dy * sinT), v = (dy * cosT) - (dx * sinT);
            u0 = Math.Min(u0, u);
            u1 = Math.Max(u1, u);
            v0 = Math.Min(v0, v);
            v1 = Math.Max(v1, v);
        }

        // One cell's own variance (1/12) keeps a one-cell-thick line from an infinite ratio.
        return new CellAxis(cx, cy, cosT, sinT, u0 - 0.5, u1 + 0.5, v0 - 0.5, v1 + 0.5, major / (minor + (1.0 / 12)), cells.Length);
    }
}
