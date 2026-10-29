using System.Runtime.CompilerServices;
using OneImage;
using OneOcr.Kernels;

namespace OneOcr.Layout;

/// <summary>An axis-aligned pixel rectangle of ink: left, top, width, height.</summary>
internal readonly record struct LineBox(int X, int Y, int Width, int Height)
{
    public int Right => X + Width;

    public int Bottom => Y + Height;
}

/// <summary>
/// Finds text lines on a clean, upright, single-column page with classical image analysis: global Otsu binarization, then a
/// horizontal projection profile cut into bands of ink rows. This is the projection-profile approach to line segmentation
/// (Nagy and Seth, "Hierarchical representation of optically scanned documents", 1984; surveyed in O'Gorman and Kasturi,
/// "Document Image Analysis", 1995, ch. 2). Bands that are several line heights tall because ascenders touch descenders are
/// cut again at the deepest valley of the profile. Lines come back top to bottom.
/// Not handled: skewed or multi-column pages, tables and figures, lines of only punctuation. Those belong to the learned
/// layout stage.
/// </summary>
internal static class LineFinder
{
    // A band taller than this many typical line heights is a candidate for splitting.
    private const double SplitHeightFactor = 1.7;

    // A split point must lie at least this many typical heights from both ends of the band.
    private const double SplitMargin = 0.6;

    // The profile at a split point must be below this fraction of the lower of the two peaks it separates.
    private const double ValleyDepth = 0.2;

    // Bands separated by gaps up to this fraction of the typical height are one line (the dot of an i, a stray hairline).
    private const double MergeGap = 0.06;

    /// <summary>Finds the lines of an 8-bit grayscale page (row-major, <paramref name="width"/> x <paramref name="height"/>).</summary>
    /// <param name="gray">Pixels; length must be width times height.</param>
    /// <param name="width">Page width in pixels.</param>
    /// <param name="height">Page height in pixels.</param>
    /// <param name="maxLines">Most lines to return; more throws <see cref="OcrLimitException"/>.</param>
    /// <returns>Line boxes in reading order. Empty for a blank page.</returns>
    public static List<LineBox> Find(ReadOnlySpan<byte> gray, int width, int height, int maxLines)
    {
        if (width < 1 || height < 1 || gray.Length != (long)width * height)
        {
            throw new ArgumentException("The pixel count must equal width times height, both positive.", nameof(gray));
        }

        (int threshold, bool inkIsBright) = Binarize(gray);
        return Find(gray, width, height, maxLines, threshold, inkIsBright);
    }

    /// <summary>Like <see cref="Find(ReadOnlySpan{byte}, int, int, int)"/> with the binarization already done (a negative threshold means no contrast).</summary>
    internal static List<LineBox> Find(ReadOnlySpan<byte> gray, int width, int height, int maxLines, int threshold, bool inkIsBright)
    {
        var lines = new List<LineBox>();
        if (threshold < 0)
        {
            return lines;
        }

        // Row profile and the horizontal extent of the ink in each row.
        var rowInk = new int[height];
        var rowFirst = new int[height];
        var rowLast = new int[height];
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<byte> row = gray.Slice(y * width, width);
            int count = 0, first = -1, last = -1;
            for (int x = 0; x < row.Length; x++)
            {
                bool ink = inkIsBright ? row[x] > threshold : row[x] <= threshold;
                if (ink)
                {
                    count++;
                    if (first < 0)
                    {
                        first = x;
                    }

                    last = x;
                }
            }

            rowInk[y] = count;
            rowFirst[y] = first;
            rowLast[y] = last;
        }

        return FromProfiles(rowInk, rowFirst, rowLast, width, height, maxLines);
    }

    /// <summary>Finds the lines of a region of the ink mask (ink is value 1; ruling lines are not ink). The boxes are in the mask's coordinates.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static List<LineBox> FindInMask(PageMask mask, LineBox region, int maxLines)
    {
        int[] rowInk = System.Buffers.ArrayPool<int>.Shared.Rent(region.Height);
        int[] rowFirst = System.Buffers.ArrayPool<int>.Shared.Rent(region.Height);
        int[] rowLast = System.Buffers.ArrayPool<int>.Shared.Rent(region.Height);
        byte[] scratch = System.Buffers.ArrayPool<byte>.Shared.Rent(region.Width);
        try
        {
            for (int y = 0; y < region.Height; y++)
            {
                mask.ReadRow(region.Y + y, region.X, region.Width, scratch);
                // Only the ink value counts (ruling lines are 2); the span searches are vectorized.
                ReadOnlySpan<byte> row = scratch.AsSpan(0, region.Width);
                int count = row.Count(PageMask.Ink);
                rowInk[y] = count;
                rowFirst[y] = count == 0 ? -1 : row.IndexOf(PageMask.Ink);
                rowLast[y] = count == 0 ? -1 : row.LastIndexOf(PageMask.Ink);
            }

            List<LineBox> found = FromProfiles(rowInk, rowFirst, rowLast, region.Width, region.Height, maxLines);
            for (int i = 0; i < found.Count; i++)
            {
                LineBox b = found[i];
                found[i] = new LineBox(b.X + region.X, b.Y + region.Y, b.Width, b.Height);
            }

            return found;
        }
        finally
        {
            System.Buffers.ArrayPool<int>.Shared.Return(rowInk);
            System.Buffers.ArrayPool<int>.Shared.Return(rowFirst);
            System.Buffers.ArrayPool<int>.Shared.Return(rowLast);
            System.Buffers.ArrayPool<byte>.Shared.Return(scratch);
        }
    }

    private static List<LineBox> FromProfiles(int[] rowInk, int[] rowFirst, int[] rowLast, int width, int height, int maxLines)
    {
        var lines = new List<LineBox>();
        // Rows with only a few ink pixels are specks, not text.
        int noise = Math.Max(1, width / 1000);
        List<(int Y0, int Y1)> bands = Runs(rowInk, noise, height);
        if (bands.Count == 0)
        {
            return lines;
        }

        int typical = Median(bands);
        bands = Merge(bands, (int)(MergeGap * typical));
        typical = Median(bands);

        // Cut tall bands. A stack bounds the work by the number of rows, whatever the input looks like.
        var finalBands = new List<(int Y0, int Y1)>();
        var pending = new Stack<(int Y0, int Y1)>();
        for (int i = bands.Count - 1; i >= 0; i--)
        {
            pending.Push(bands[i]);
        }

        while (pending.Count > 0)
        {
            var (y0, y1) = pending.Pop();
            int split = y1 - y0 > SplitHeightFactor * typical ? FindValley(rowInk, y0, y1, typical) : -1;
            if (split < 0)
            {
                finalBands.Add((y0, y1));
            }
            else
            {
                pending.Push((split + 1, y1));
                pending.Push((y0, split - 1));
            }
        }

        foreach ((int y0, int y1) in finalBands)
        {
            int top = y0, bottom = y1;
            while (top <= bottom && rowInk[top] <= noise)
            {
                top++;
            }

            while (bottom >= top && rowInk[bottom] <= noise)
            {
                bottom--;
            }

            if (bottom < top)
            {
                continue;
            }

            int left = int.MaxValue, right = -1;
            for (int y = top; y <= bottom; y++)
            {
                if (rowFirst[y] >= 0)
                {
                    left = Math.Min(left, rowFirst[y]);
                    right = Math.Max(right, rowLast[y]);
                }
            }

            int h = bottom - top + 1;
            if (h < Math.Max(3, typical / 4))
            {
                continue;
            }

            if (lines.Count >= maxLines)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxLinesPerPage), maxLines, maxLines + 1L);
            }

            lines.Add(new LineBox(left, top, right - left + 1, h));
        }

        return lines;
    }

    /// <summary>
    /// Otsu's method (Otsu 1979, "A threshold selection method from gray-level histograms"): the threshold that maximizes the
    /// between-class variance. Ink is the smaller class. Returns a negative threshold when the page has no contrast.
    /// </summary>
    internal static (int Threshold, bool InkIsBright) Binarize(ReadOnlySpan<byte> gray)
    {
        var histogram = new long[256];
        AddHistogram(gray, histogram);
        return Binarize(histogram, gray.Length);
    }

    /// <summary>The same for a page in an array, counting its pixels on up to <paramref name="degree"/> threads (the counts are exact, so the threshold does not depend on it).</summary>
    internal static (int Threshold, bool InkIsBright) Binarize(Gray8[] gray, int count, int degree)
    {
        int chunks = degree <= 1 ? 1 : Math.Clamp(count / (1 << 20), 1, degree * 2);
        var histograms = new long[chunks][];
        int per = (count + chunks - 1) / chunks;
        ParallelRun.For(chunks, degree, i =>
        {
            int from = i * per, to = Math.Min(count, from + per);
            histograms[i] = new long[256];
            AddHistogram(PixelSpans.AsBytes(gray.AsSpan(from, to - from)), histograms[i]);
        }, CancellationToken.None);
        for (int i = 1; i < chunks; i++)
        {
            for (int v = 0; v < 256; v++)
            {
                histograms[0][v] += histograms[i][v];
            }
        }

        return Binarize(histograms[0], count);
    }

    // Four interleaved tables: consecutive pixels are often equal, and one table would make each increment wait for the last.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AddHistogram(ReadOnlySpan<byte> gray, long[] histogram)
    {
        Span<int> h = stackalloc int[4 * 256];
        h.Clear();
        int i = 0;
        for (; i + 4 <= gray.Length; i += 4)
        {
            h[gray[i]]++;
            h[256 + gray[i + 1]]++;
            h[512 + gray[i + 2]]++;
            h[768 + gray[i + 3]]++;
        }

        for (; i < gray.Length; i++)
        {
            h[gray[i]]++;
        }

        for (int v = 0; v < 256; v++)
        {
            histogram[v] += (long)h[v] + h[256 + v] + h[512 + v] + h[768 + v];
        }
    }

    internal static (int Threshold, bool InkIsBright) Binarize(long[] histogram, long total)
    {
        int min = 0, max = 255;
        while (min < 255 && histogram[min] == 0)
        {
            min++;
        }

        while (max > 0 && histogram[max] == 0)
        {
            max--;
        }

        if (max - min < 24)
        {
            return (-1, false);
        }

        double sumAll = 0;
        for (int i = 0; i < 256; i++)
        {
            sumAll += (double)i * histogram[i];
        }

        double sumBackground = 0, bestVariance = -1;
        long weightBackground = 0;
        int best = min;
        for (int t = min; t < max; t++)
        {
            weightBackground += histogram[t];
            if (weightBackground == 0)
            {
                continue;
            }

            long weightForeground = total - weightBackground;
            if (weightForeground == 0)
            {
                break;
            }

            sumBackground += (double)t * histogram[t];
            double meanB = sumBackground / weightBackground;
            double meanF = (sumAll - sumBackground) / weightForeground;
            double variance = (double)weightBackground * weightForeground * (meanB - meanF) * (meanB - meanF);
            if (variance > bestVariance)
            {
                bestVariance = variance;
                best = t;
            }
        }

        // Pixels at or below the threshold are the dark class; ink is whichever class has fewer pixels.
        long dark = 0;
        for (int i = 0; i <= best; i++)
        {
            dark += histogram[i];
        }

        return (best, dark > total - dark);
    }

    private static List<(int Y0, int Y1)> Runs(int[] profile, int noise, int length)
    {
        var runs = new List<(int, int)>();
        int start = -1;
        for (int y = 0; y < length; y++)
        {
            if (profile[y] > noise)
            {
                if (start < 0)
                {
                    start = y;
                }
            }
            else if (start >= 0)
            {
                runs.Add((start, y - 1));
                start = -1;
            }
        }

        if (start >= 0)
        {
            runs.Add((start, length - 1));
        }

        return runs;
    }

    private static int Median(List<(int Y0, int Y1)> bands)
    {
        var heights = new int[bands.Count];
        for (int i = 0; i < heights.Length; i++)
        {
            heights[i] = bands[i].Y1 - bands[i].Y0 + 1;
        }

        Array.Sort(heights);
        return heights[heights.Length / 2];
    }

    private static List<(int Y0, int Y1)> Merge(List<(int Y0, int Y1)> bands, int maxGap)
    {
        var merged = new List<(int Y0, int Y1)>(bands.Count) { bands[0] };
        for (int i = 1; i < bands.Count; i++)
        {
            var last = merged[^1];
            if (bands[i].Y0 - last.Y1 - 1 <= maxGap)
            {
                merged[^1] = (last.Y0, bands[i].Y1);
            }
            else
            {
                merged.Add(bands[i]);
            }
        }

        return merged;
    }

    /// <summary>
    /// The row to cut a tall band at, or -1. The profile is smoothed over three rows; the candidate is its minimum inside the
    /// middle of the band, and it counts only if it is deep compared with the strongest ink above and below it, so a large
    /// heading (whose profile has no deep valley) stays whole.
    /// </summary>
    private static int FindValley(int[] profile, int y0, int y1, int typical)
    {
        int margin = (int)(SplitMargin * typical);
        int from = y0 + margin, to = y1 - margin;
        if (to <= from)
        {
            return -1;
        }

        int valleyRow = -1;
        double valley = double.MaxValue;
        for (int y = from; y <= to; y++)
        {
            double v = Smooth(profile, y, y0, y1);
            if (v < valley)
            {
                valley = v;
                valleyRow = y;
            }
        }

        double above = 0, below = 0;
        for (int y = y0; y < valleyRow; y++)
        {
            above = Math.Max(above, Smooth(profile, y, y0, y1));
        }

        for (int y = valleyRow + 1; y <= y1; y++)
        {
            below = Math.Max(below, Smooth(profile, y, y0, y1));
        }

        return valley < ValleyDepth * Math.Min(above, below) ? valleyRow : -1;
    }

    private static double Smooth(int[] profile, int y, int y0, int y1)
    {
        int a = Math.Max(y0, y - 1), b = Math.Min(y1, y + 1);
        long sum = 0;
        for (int i = a; i <= b; i++)
        {
            sum += profile[i];
        }

        return (double)sum / (b - a + 1);
    }
}
