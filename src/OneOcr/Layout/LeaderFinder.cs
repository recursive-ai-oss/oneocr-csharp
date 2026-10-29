using System.Buffers;

namespace OneOcr.Layout;

/// <summary>
/// A run of evenly spaced full stops on a line's baseline: the dot leader of a table of contents or a list of figures.
/// Coordinates are page pixels.
/// </summary>
/// <param name="From">The left edge of the first dot.</param>
/// <param name="To">The right edge (exclusive) of the last dot.</param>
/// <param name="Count">The number of dots in the run, counting dots that are too faint to see between two that are seen.</param>
/// <param name="Pitch">The mean distance between the centres of neighbouring dots.</param>
/// <param name="Centers">The centre of every dot, <see cref="Count"/> of them; a dot that is missing from the pixels is placed on the pitch.</param>
/// <param name="Top">The top row of the dots.</param>
/// <param name="Bottom">The bottom row (exclusive) of the dots.</param>
/// <param name="Left">The box around the ink left of the run, or null when there is none.</param>
/// <param name="Right">The box around the ink right of the run (the page number), up to the first wide gap, or null when there is none.</param>
internal sealed record Leader(int From, int To, int Count, double Pitch, int[] Centers, int Top, int Bottom, LineBox? Left, LineBox? Right);

/// <summary>
/// Finds a dot leader in a line from its pixels (<c>docs/spec/oneocr-page-v1.md</c>, "Dot leaders"). The recognizer reads text, and a
/// run of fifty full stops in the middle of a line makes it lose the words in front of the run and invent letters; so the page
/// reader cuts a line at its leader and reads what is left and right of it. A leader is recognised by its shape alone: at least
/// <see cref="MinDots"/> small, roundish, equal blobs on one baseline at an even pitch, with nothing else between them.
/// </summary>
internal static class LeaderFinder
{
    /// <summary>A run needs at least this many dots; an ellipsis or a row of points in running text is shorter.</summary>
    internal const int MinDots = 6;

    /// <summary>A run needs at least this many dots that were seen (the rest of its count may be dots that are too faint to see).</summary>
    internal const int MinSeenDots = 6;

    /// <summary>Lines narrower than this many line heights cannot hold a leader and a word on each side of it, and are not searched.</summary>
    internal const double MinWidthInHeights = 3.0;

    /// <summary>A dot is at most this fraction of the line's height wide and tall.</summary>
    internal const double MaxDotFraction = 0.3;

    /// <summary>The distance between the centres of neighbouring dots is at most this fraction of the line's height.</summary>
    internal const double MaxPitchFraction = 0.6;

    /// <summary>The distance to the next dot may differ from the run's pitch by this share (jitter of the rounding to pixels).</summary>
    internal const double PitchTolerance = 0.3;

    /// <summary>The search window grows by this many line heights each time a run reaches its right edge.</summary>
    internal const double WindowGrowthInHeights = 4.0;

    /// <summary>Ink right of the run belongs to the line (the page number) up to the first gap wider than this many line heights.</summary>
    internal const double ContinuationGapInHeights = 1.5;

    /// <summary>More components than this in one line means noise or a photographed texture; the line is not searched.</summary>
    internal const int MaxComponents = 20000;

    private readonly record struct Component(int MinX, int MaxX, int MinY, int MaxY, int Area)
    {
        public int Width => MaxX - MinX + 1;

        public int Height => MaxY - MinY + 1;

        public double CenterX => (MinX + MaxX) / 2.0;
    }

    /// <summary>
    /// Looks for a leader in the line <paramref name="box"/> of an 8-bit grayscale page.
    /// </summary>
    /// <param name="page">The page, row-major.</param>
    /// <param name="pageWidth">Its width.</param>
    /// <param name="pageHeight">Its height.</param>
    /// <param name="box">The line's ink box.</param>
    /// <param name="threshold">The page's binarization threshold: a pixel is ink when it is at most this (or above it when <paramref name="inkIsBright"/>).</param>
    /// <param name="inkIsBright">Whether ink is the bright side of the threshold.</param>
    /// <param name="typicalHeight">The page's typical line height; a box that holds only the dots is shorter than a line, and dots are judged against the line.</param>
    /// <returns>The longest leader in the line, or null when there is none.</returns>
    public static Leader? Find(ReadOnlySpan<byte> page, int pageWidth, int pageHeight, LineBox box, int threshold, bool inkIsBright, int typicalHeight)
    {
        int x0 = Math.Max(0, box.X), y0 = Math.Max(0, box.Y);
        int x1 = Math.Min(pageWidth, box.Right), y1 = Math.Min(pageHeight, box.Bottom);
        int h = y1 - y0, lineHeight = Math.Max(h, typicalHeight);
        if (h < 3 || lineHeight < 8 || x1 - x0 < MinWidthInHeights * lineHeight || page.Length < (long)pageWidth * pageHeight)
        {
            return null;
        }

        // A leader that runs up to the box's right edge goes on beyond it (the detector may cut a long run of dots into pieces): the
        // window grows to the right until the run ends inside it.
        Leader? found = FindIn(page, pageWidth, x0, y0, x1 - x0, h, lineHeight, threshold, inkIsBright);
        while (found is not null && found.To >= x1 - (2 * found.Pitch) && x1 < pageWidth)
        {
            x1 = Math.Min(pageWidth, x1 + (int)(WindowGrowthInHeights * lineHeight));
            found = FindIn(page, pageWidth, x0, y0, x1 - x0, h, lineHeight, threshold, inkIsBright);
        }

        return found;
    }

    // Searches the window of the page at (x0, y0), w by h pixels.
    private static Leader? FindIn(ReadOnlySpan<byte> page, int pageWidth, int x0, int y0, int w, int h, int lineHeight, int threshold, bool inkIsBright)
    {
        List<Component>? comps = Label(page, pageWidth, x0, y0, w, h, threshold, inkIsBright);
        if (comps is null || comps.Count < MinDots)
        {
            return null;
        }

        comps.Sort((a, b) => a.MinX != b.MinX ? a.MinX.CompareTo(b.MinX) : a.MinY.CompareTo(b.MinY));
        int n = comps.Count;
        var isDot = new bool[n];
        for (int i = 0; i < n; i++)
        {
            Component c = comps[i];
            double maxSize = MaxDotFraction * lineHeight;
            double aspect = (double)c.Width / c.Height;
            isDot[i] = c.Width >= 2 && c.Height >= 2 && c.Width <= maxSize && c.Height <= maxSize && aspect >= 0.5 && aspect <= 2.0
                && (h < 0.6 * lineHeight || (c.MinY + c.MaxY) / 2.0 >= h / 2.0);
        }

        // Runs of neighbouring dots (neighbours in left-to-right order, so a letter between two dots ends a run) with the same size and
        // baseline and an even pitch.
        var runs = new List<(int First, int Last, int Steps)>();
        for (int i = 0; i < n; i++)
        {
            if (!isDot[i])
            {
                continue;
            }

            int j = i;
            while (j + 1 < n && isDot[j + 1] && Continues(comps, i, j, j + 1, lineHeight))
            {
                j++;
            }

            if (j - i + 1 >= 3)
            {
                runs.Add((i, j, j - i));
            }

            i = j;
        }

        if (runs.Count == 0)
        {
            return null;
        }

        // A dot that is too faint to be seen leaves a gap of two or three pitches; neighbouring runs of the same pitch join across it.
        var merged = new List<(int First, int Last, int Steps)>();
        foreach ((int first, int last, int steps) in runs)
        {
            if (merged.Count > 0)
            {
                (int pf, int pl, int stepsA) = merged[^1];
                double pitchA = Pitch(comps, pf, pl, stepsA), pitchB = Pitch(comps, first, last, steps);
                double gap = comps[first].CenterX - comps[pl].CenterX;
                int k = (int)Math.Round(gap / pitchA);
                if (first == pl + 1 && k is >= 2 and <= 3 && Math.Abs(gap - (k * pitchA)) <= PitchTolerance * pitchA
                    && Math.Abs(pitchB - pitchA) <= PitchTolerance * pitchA && SameBaseline(comps, pf, first))
                {
                    merged[^1] = (pf, last, stepsA + k + steps);
                    continue;
                }
            }

            merged.Add((first, last, steps));
        }

        (int bestFirst, int bestLast) = (-1, -1);
        int bestCount = 0, bestSteps = 0;
        foreach ((int first, int last, int steps) in merged)
        {
            int count = steps + 1;
            if (count > bestCount && last - first + 1 >= MinSeenDots && count >= MinDots)
            {
                (bestFirst, bestLast, bestCount) = (first, last, count);
                bestSteps = steps;
            }
        }

        if (bestCount == 0)
        {
            return null;
        }

        // A full stop that ends the text before the leader sits right behind the last letter, while the first dot of a leader follows a
        // space; when the run's first dot touches what is before it, that dot belongs to the text.
        if (bestFirst > 0 && bestCount > MinDots && comps[bestFirst].MinX - comps[bestFirst - 1].MaxX < 0.5 * Pitch(comps, bestFirst, bestLast, bestSteps))
        {
            bestFirst++;
            bestSteps--;
            bestCount--;
        }

        double p = Pitch(comps, bestFirst, bestLast, bestSteps);
        double meanWidth = 0;
        for (int i = bestFirst; i <= bestLast; i++)
        {
            meanWidth += comps[i].Width;
        }

        meanWidth /= bestLast - bestFirst + 1;
        if (p > MaxPitchFraction * lineHeight || p < 1.3 * meanWidth)
        {
            return null;
        }

        var centers = new int[bestCount];
        double start = comps[bestFirst].CenterX;
        for (int i = 0; i < bestCount; i++)
        {
            centers[i] = x0 + (int)Math.Round(start + (i * p));
        }

        int top = int.MaxValue, bottom = int.MinValue;
        for (int i = bestFirst; i <= bestLast; i++)
        {
            top = Math.Min(top, comps[i].MinY);
            bottom = Math.Max(bottom, comps[i].MaxY + 1);
        }

        int from = comps[bestFirst].MinX, to = comps[bestLast].MaxX + 1;
        int leftFirst = int.MaxValue, leftLast = -1, leftTop = int.MaxValue, leftBottom = -1;
        int rightFirst = int.MaxValue, rightLast = -1, rightTop = int.MaxValue, rightBottom = -1;
        for (int i = 0; i < n; i++)
        {
            if (i >= bestFirst && i <= bestLast)
            {
                continue;
            }

            Component c = comps[i];
            if (c.MaxX < from)
            {
                leftFirst = Math.Min(leftFirst, c.MinX);
                leftLast = Math.Max(leftLast, c.MaxX);
                leftTop = Math.Min(leftTop, c.MinY);
                leftBottom = Math.Max(leftBottom, c.MaxY);
            }
            else if (c.MinX >= to && (rightLast < 0 || c.MinX - rightLast <= ContinuationGapInHeights * lineHeight))
            {
                rightFirst = Math.Min(rightFirst, c.MinX);
                rightLast = Math.Max(rightLast, c.MaxX);
                rightTop = Math.Min(rightTop, c.MinY);
                rightBottom = Math.Max(rightBottom, c.MaxY);
            }
        }

        return new Leader(
            x0 + from, x0 + to, bestCount, p, centers, y0 + top, y0 + bottom,
            leftLast < 0 ? null : new LineBox(x0 + leftFirst, y0 + leftTop, leftLast - leftFirst + 1, leftBottom - leftTop + 1),
            rightLast < 0 ? null : new LineBox(x0 + rightFirst, y0 + rightTop, rightLast - rightFirst + 1, rightBottom - rightTop + 1));
    }

    // Whether component next continues the run that starts at first and ends at last: same baseline and size as its first dot, and a
    // distance from the last dot that is the run's pitch (or any distance in the dot's own size range, for the second dot).
    private static bool Continues(List<Component> comps, int first, int last, int next, int lineHeight)
    {
        Component a = comps[first], b = comps[next];
        if (!SameBaseline(comps, first, next))
        {
            return false;
        }

        double delta = b.CenterX - comps[last].CenterX;
        if (last == first)
        {
            return delta >= 1.3 * Math.Max(a.Width, b.Width) && delta <= MaxPitchFraction * lineHeight;
        }

        double pitch = Pitch(comps, first, last, last - first);
        return Math.Abs(delta - pitch) <= PitchTolerance * pitch + 1;
    }

    private static bool SameBaseline(List<Component> comps, int a, int b)
    {
        Component p = comps[a], q = comps[b];
        double sizeP = Math.Max(p.Width, p.Height), sizeQ = Math.Max(q.Width, q.Height);
        double ratio = sizeQ / sizeP;
        return Math.Abs(q.MaxY - p.MaxY) <= Math.Max(2, 0.3 * sizeP) && ratio >= 0.6 && ratio <= 1.7;
    }

    // The distance between the centres of neighbouring dots of a run from its first to its last dot over the given number of steps.
    private static double Pitch(List<Component> comps, int first, int last, int steps) =>
        steps <= 0 ? 1 : (comps[last].CenterX - comps[first].CenterX) / steps;

    // 8-connected components of the ink in the line, by union-find over two rows of labels. Null when there are too many.
    private static List<Component>? Label(ReadOnlySpan<byte> page, int pageWidth, int x0, int y0, int w, int h, int threshold, bool inkIsBright)
    {
        var parent = new List<int> { 0 };
        var comps = new List<Component> { default };
        int[] prev = ArrayPool<int>.Shared.Rent(w + 2), cur = ArrayPool<int>.Shared.Rent(w + 2);
        try
        {
            Array.Clear(prev, 0, w + 2);
            Array.Clear(cur, 0, w + 2);
            for (int y = 0; y < h; y++)
            {
                ReadOnlySpan<byte> row = page.Slice(((y0 + y) * pageWidth) + x0, w);
                cur[0] = 0;
                for (int x = 0; x < w; x++)
                {
                    bool ink = inkIsBright ? row[x] > threshold : row[x] <= threshold;
                    if (!ink)
                    {
                        cur[x + 1] = 0;
                        continue;
                    }

                    int label = 0;
                    // Neighbours: west, north-west, north, north-east (offset by one for the border).
                    Span<int> around = [cur[x], prev[x], prev[x + 1], prev[x + 2]];
                    foreach (int l in around)
                    {
                        if (l == 0)
                        {
                            continue;
                        }

                        int root = Find(parent, l);
                        if (label == 0)
                        {
                            label = root;
                        }
                        else if (root != label)
                        {
                            Component a = comps[label], b = comps[root];
                            comps[label] = new Component(Math.Min(a.MinX, b.MinX), Math.Max(a.MaxX, b.MaxX), Math.Min(a.MinY, b.MinY), Math.Max(a.MaxY, b.MaxY), a.Area + b.Area);
                            parent[root] = label;
                        }
                    }

                    if (label == 0)
                    {
                        if (parent.Count > MaxComponents)
                        {
                            return null;
                        }

                        label = parent.Count;
                        parent.Add(label);
                        comps.Add(new Component(x, x, y, y, 1));
                    }
                    else
                    {
                        Component c = comps[label];
                        comps[label] = new Component(Math.Min(c.MinX, x), Math.Max(c.MaxX, x), c.MinY, y, c.Area + 1);
                    }

                    cur[x + 1] = label;
                }

                (prev, cur) = (cur, prev);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(prev);
            ArrayPool<int>.Shared.Return(cur);
        }

        var result = new List<Component>();
        for (int i = 1; i < parent.Count; i++)
        {
            if (parent[i] == i)
            {
                result.Add(comps[i]);
            }
        }

        return result;
    }

    private static int Find(List<int> parent, int i)
    {
        while (parent[i] != i)
        {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }

        return i;
    }
}
