using OneImage;
using OneOcr.Kernels;

namespace OneOcr.Layout;

/// <summary>
/// Pixel rotations of a page: exact quarter turns, and a small turn by bilinear sampling. Every output pixel is computed from
/// the input alone with IEEE basic operations (add, multiply, divide, square root), so the result is the same on every machine,
/// kernel path and degree of parallelism.
/// </summary>
internal static class PageRotation
{
    /// <summary>Turns the page clockwise by <paramref name="quarterTurns"/> quarter turns (0 to 3) into <paramref name="to"/> (width and height swap for odd counts).</summary>
    public static void QuarterTurn(Gray8[] from, int width, int height, int quarterTurns, Gray8[] to, int degree, CancellationToken cancellationToken)
    {
        int count = checked(width * height);
        switch (quarterTurns & 3)
        {
            case 0:
                System.Array.Copy(from, to, count);
                break;
            case 1:
                // (x, y) -> (height - 1 - y, x); the new page is `height` wide.
                ParallelRun.For(width, degree, v =>
                {
                    ReadOnlySpan<byte> src = PixelSpans.AsBytes(from.AsSpan(0, count));
                    for (int u = 0; u < height; u++)
                    {
                        to[(v * height) + u] = new Gray8(src[((height - 1 - u) * width) + v]);
                    }
                }, cancellationToken);
                break;
            case 2:
                ParallelRun.For(height, degree, y =>
                {
                    ReadOnlySpan<byte> src = PixelSpans.AsBytes(from.AsSpan(0, count));
                    for (int x = 0; x < width; x++)
                    {
                        to[(y * width) + x] = new Gray8(src[count - 1 - ((y * width) + x)]);
                    }
                }, cancellationToken);
                break;
            default:
                // (x, y) -> (y, width - 1 - x).
                ParallelRun.For(width, degree, v =>
                {
                    ReadOnlySpan<byte> src = PixelSpans.AsBytes(from.AsSpan(0, count));
                    for (int u = 0; u < height; u++)
                    {
                        to[(v * height) + u] = new Gray8(src[(u * width) + (width - 1 - v)]);
                    }
                }, cancellationToken);
                break;
        }
    }

    /// <summary>
    /// The size of the canvas that holds a <paramref name="width"/> by <paramref name="height"/> page turned by an angle whose
    /// sine and cosine are given (cosine positive).
    /// </summary>
    public static (int Width, int Height) CanvasFor(int width, int height, double sin, double cos)
    {
        double s = Math.Abs(sin);
        double w = Math.Ceiling((width * cos) + (height * s));
        double h = Math.Ceiling((width * s) + (height * cos));
        return ((int)w, (int)h);
    }

    /// <summary>
    /// Fills <paramref name="to"/> (<paramref name="toWidth"/> by <paramref name="toHeight"/>) with the page turned so that text
    /// whose lines run at the angle with the given sine and cosine become horizontal. A destination point <c>d</c> samples the source
    /// at <c>R(s) (d - centre of the canvas) + centre of the page</c>, where R(s) turns by the line angle s (y down, so a
    /// positive angle is clockwise); points outside the page take <paramref name="background"/>.
    /// </summary>
    public static void Deskew(
        Gray8[] from, int width, int height, double sin, double cos, byte background, Gray8[] to, int toWidth, int toHeight,
        int degree, CancellationToken cancellationToken)
    {
        int count = checked(width * height);
        double dcx = toWidth / 2.0, dcy = toHeight / 2.0, qcx = width / 2.0, qcy = height / 2.0;
        ParallelRun.For(toHeight, degree, j =>
        {
            ReadOnlySpan<byte> src = PixelSpans.AsBytes(from.AsSpan(0, count));
            double dy = (j + 0.5) - dcy;
            for (int i = 0; i < toWidth; i++)
            {
                double dx = (i + 0.5) - dcx;
                double qx = ((dx * cos) - (dy * sin)) + qcx;
                double qy = ((dx * sin) + (dy * cos)) + qcy;
                to[(j * toWidth) + i] = new Gray8(Sample(src, width, height, qx - 0.5, qy - 0.5, background));
            }
        }, cancellationToken);
    }

    private static byte Sample(ReadOnlySpan<byte> src, int width, int height, double sx, double sy, byte background)
    {
        if (sx < -0.5 || sy < -0.5 || sx > width - 0.5 || sy > height - 0.5)
        {
            return background;
        }

        int x0 = (int)Math.Floor(sx), y0 = (int)Math.Floor(sy);
        double fx = sx - x0, fy = sy - y0;
        int x1 = Math.Min(x0 + 1, width - 1), y1 = Math.Min(y0 + 1, height - 1);
        x0 = Math.Max(x0, 0);
        y0 = Math.Max(y0, 0);
        double top = (src[(y0 * width) + x0] * (1 - fx)) + (src[(y0 * width) + x1] * fx);
        double bottom = (src[(y1 * width) + x0] * (1 - fx)) + (src[(y1 * width) + x1] * fx);
        double v = (top * (1 - fy)) + (bottom * fy);
        return (byte)(int)(v + 0.5);
    }

    /// <summary>The most common gray value of the page: its background for filling new corners.</summary>
    public static byte Background(ReadOnlySpan<byte> gray)
    {
        var histogram = new int[256];
        foreach (byte b in gray)
        {
            histogram[b]++;
        }

        int best = 0;
        for (int v = 1; v < 256; v++)
        {
            if (histogram[v] > histogram[best])
            {
                best = v;
            }
        }

        return (byte)best;
    }
}
