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

    /// <summary>A layout cell is on when its logit exceeds ln(0.4 / 0.6), that is, its probability exceeds 0.4.</summary>
    public const float LayoutLogitThreshold = -0.405465108108164f;

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

    private readonly record struct Component(int X0, int Y0, int X1, int Y1, int Count, double ScoreSum)
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
    /// <returns>Lines, regions and orientation in page pixels.</returns>
    public static PageDetection Decode(DetectorMaps maps, DbNetModel model, double scale, int pageWidth, int pageHeight, OcrLimits limits) =>
        Decode(
            maps.Prob, maps.ProbWidth, maps.ProbHeight, maps.Layout, maps.LayoutWidth, maps.LayoutHeight, maps.LayoutStride,
            model.LayoutClasses, model.LayoutTrained, maps.Orientation, scale, pageWidth, pageHeight, limits);

    /// <summary>Decodes maps given as plain arrays: text logits <c>[h][w]</c> and layout logits <c>[h][w][stride]</c>.</summary>
    internal static PageDetection Decode(
        float[] prob, int probWidth, int probHeight, float[] layout, int layoutWidth, int layoutHeight, int layoutStride,
        IReadOnlyList<string> classNames, IReadOnlyList<bool> trained, float[] orientation, double scale, int pageWidth, int pageHeight, OcrLimits limits)
    {
        // Lines, in network pixels first: regions snap to them before everything is scaled back to the page.
        List<(double X0, double Y0, double X1, double Y1, float Score)> lines = [];
        const double k = TextShrink / (1 - (2 * TextShrink));
        foreach (Component c in Label(prob, probWidth, probHeight, 1, 0, TextLogitThreshold, MinTextCells, limits.MaxLinesPerPage, nameof(OcrLimits.MaxLinesPerPage)))
        {
            double x0 = c.X0 * TextStride, y0 = c.Y0 * TextStride, x1 = (c.X1 + 1) * TextStride, y1 = (c.Y1 + 1) * TextStride;
            double d = k * Math.Min(x1 - x0, y1 - y0);
            lines.Add((x0 - d, y0 - d, x1 + d, y1 + d, c.Score));
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

            foreach (Component c in Label(layout, layoutWidth, layoutHeight, layoutStride, cls, LayoutLogitThreshold, 1, limits.MaxPageElements - total, nameof(OcrLimits.MaxPageElements)))
            {
                total++;
                double x0 = c.X0 * LayoutStride, y0 = c.Y0 * LayoutStride, x1 = (c.X1 + 1) * LayoutStride, y1 = (c.Y1 + 1) * LayoutStride;
                double core = Math.Min(x1 - x0, y1 - y0);
                double d = core <= LayoutGrowSmall ? core / 3.0 : LayoutGrowCap;
                var box = (X0: x0 - d, Y0: y0 - d, X1: x1 + d, Y1: y1 + d);
                if (cls != table && cls != figure)
                {
                    box = Snap(box, lines);
                }

                regions.Add((cls, box.X0, box.Y0, box.X1, box.Y1, c.Score));
            }
        }

        var outLines = new List<DetectedLine>(lines.Count);
        foreach (var l in lines)
        {
            if (Clip(l.X0 / scale, l.Y0 / scale, l.X1 / scale, l.Y1 / scale, pageWidth, pageHeight) is { } b)
            {
                outLines.Add(new DetectedLine(b.L, b.T, b.R, b.B, l.Score));
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
        return new PageDetection { Lines = outLines, Regions = outRegions, Quarter = quarter, OrientationConfidence = confidence };
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
        (double X0, double Y0, double X1, double Y1) box, List<(double X0, double Y0, double X1, double Y1, float Score)> lines)
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
        float[] logits, int width, int height, int stride, int channel, float threshold, int minCells, int maxCount, string limitName)
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

        var seen = new bool[cells];
        int[] stack = new int[cells];
        for (int start = 0; start < cells; start++)
        {
            if (seen[start] || !(logits[((long)start * stride) + channel] > threshold))
            {
                continue;
            }

            int top = 0;
            stack[top++] = start;
            seen[start] = true;
            int x0 = int.MaxValue, y0 = int.MaxValue, x1 = -1, y1 = -1, count = 0;
            double sum = 0;
            while (top > 0)
            {
                int cell = stack[--top];
                int y = cell / width, x = cell - (y * width);
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

                result.Add(new Component(x0, y0, x1, y1, count, sum));
            }
        }

        return result;
    }
}
