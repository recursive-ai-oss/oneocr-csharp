using OneImage;
using OneOcr.Kernels;

namespace OneOcr.Detection;

/// <summary>
/// The two page measures behind the page-conditional text threshold (docs/spec/detector-v1.md, "Page-conditional text threshold"):
/// a robust noise estimate of the gray page and the density of speck-sized components in the detector's text map. Safe code.
/// </summary>
internal static class PageNoise
{
    /// <summary>Divisor that turns the median of |I * N| into a noise deviation: 0.6745 * 6 (Immerkaer 1996).</summary>
    public const double SigmaDivisor = 4.047;

    // Centre 255 with neighbours at 0 gives +2040; centre 0 with edges at 255 gives -2040.
    private const int MaxResponse = 2040;

    /// <summary>
    /// Noise deviation of a gray page by Immerkaer's method ("Fast noise variance estimation", 1996) with a median in place of the
    /// mean: the 3x3 kernel [[1,-2,1],[-2,4,-2],[1,-2,1]] is applied to every interior pixel, and the lower median of the absolute
    /// responses, from an exact integer histogram, over 4.047. Text edges raise only the tail of the responses, so a page that is mostly
    /// paper reads the paper. Pages under 3 pixels in either direction measure 0.
    /// </summary>
    /// <param name="page">Gray pixels, row-major, <paramref name="width"/> by <paramref name="height"/> (the page scaled to the detector's size, without padding).</param>
    /// <param name="width">Width in pixels.</param>
    /// <param name="height">Height in pixels.</param>
    /// <returns>The estimate; the median is an integer, so the result is exactly reproducible.</returns>
    public static double Sigma(ReadOnlySpan<Gray8> page, int width, int height)
    {
        if (width < 3 || height < 3)
        {
            return 0;
        }

        if (page.Length < (long)width * height)
        {
            throw new ArgumentException("The page is smaller than its declared shape.", nameof(page));
        }

        Span<int> histogram = stackalloc int[MaxResponse + 1];
        histogram.Clear();
        for (int y = 1; y < height - 1; y++)
        {
            ReadOnlySpan<Gray8> up = page.Slice((y - 1) * width, width);
            ReadOnlySpan<Gray8> mid = page.Slice(y * width, width);
            ReadOnlySpan<Gray8> down = page.Slice((y + 1) * width, width);
            for (int x = 1; x < width - 1; x++)
            {
                int corners = up[x - 1].V + up[x + 1].V + down[x - 1].V + down[x + 1].V;
                int edges = up[x].V + down[x].V + mid[x - 1].V + mid[x + 1].V;
                int r = corners - (2 * edges) + (4 * mid[x].V);
                histogram[r < 0 ? -r : r]++;
            }
        }

        long total = (long)(width - 2) * (height - 2);
        long half = (total + 1) / 2;
        long seen = 0;
        for (int v = 0; v <= MaxResponse; v++)
        {
            seen += histogram[v];
            if (seen >= half)
            {
                return v / SigmaDivisor;
            }
        }

        return MaxResponse / SigmaDivisor;
    }

    /// <summary>
    /// Speck components of the text map per megapixel of detector input: 4-connected components of cells above the default text
    /// threshold (probability 0.3) with 2 to 12 cells and a mean probability under 0.5, the blobs that become one-glyph lines.
    /// </summary>
    /// <param name="logits">Text logits, <paramref name="width"/> by <paramref name="height"/>, one per cell.</param>
    /// <param name="width">Map width in cells.</param>
    /// <param name="height">Map height in cells.</param>
    /// <returns>Components per megapixel; the map has 4 input pixels per cell (stride 2) and includes the padding.</returns>
    public static double JunkDensity(float[] logits, int width, int height)
    {
        if (width < 1 || height < 1)
        {
            return 0;
        }

        int cells = checked(width * height);
        if (logits.Length < cells)
        {
            throw new ArgumentException("The map is smaller than its declared shape.", nameof(logits));
        }

        bool[] seen = System.Buffers.ArrayPool<bool>.Shared.Rent(cells);
        int[] stack = System.Buffers.ArrayPool<int>.Shared.Rent(cells);
        int junk = 0;
        try
        {
            Array.Clear(seen, 0, cells);
            for (int start = 0; start < cells; start++)
            {
                if (seen[start] || !(logits[start] > DetectorPostprocess.TextLogitThreshold))
                {
                    continue;
                }

                int top = 0, count = 0;
                double sum = 0;
                stack[top++] = start;
                seen[start] = true;
                while (top > 0)
                {
                    int cell = stack[--top];
                    int y = cell / width, x = cell - (y * width);
                    count++;
                    sum += FastMath.Sigmoid(logits[cell]);
                    if (x > 0)
                    {
                        Push(cell - 1);
                    }

                    if (x < width - 1)
                    {
                        Push(cell + 1);
                    }

                    if (y > 0)
                    {
                        Push(cell - width);
                    }

                    if (y < height - 1)
                    {
                        Push(cell + width);
                    }
                }

                if (count is >= 2 and <= 12 && sum / count < 0.5)
                {
                    junk++;
                }

                void Push(int next)
                {
                    if (!seen[next] && logits[next] > DetectorPostprocess.TextLogitThreshold)
                    {
                        seen[next] = true;
                        stack[top++] = next;
                    }
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<bool>.Shared.Return(seen);
            System.Buffers.ArrayPool<int>.Shared.Return(stack);
        }

        return junk / ((double)cells * 4 / 1e6);
    }
}
