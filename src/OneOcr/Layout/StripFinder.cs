using OneImage;
using OneOcr.Kernels;

namespace OneOcr.Layout;

/// <summary>A text line cut out of one leaf region of the page, with the leaf's number (lines of one leaf are never one line of the page).</summary>
internal readonly record struct StripLine(LineBox Box, int Leaf);

/// <summary>
/// Finds the text lines of a page that may have columns and ruled tables. It cuts the ink recursively into regions that have no
/// column gutter left, in the manner of recursive X–Y cuts (Nagy and Seth, "Hierarchical representation of optically scanned
/// documents", 1984): a gutter is an empty vertical stripe at least 1.25 line heights wide that runs through at least two lines
/// of a region; blocks of rows are cut at empty horizontal bands of half a line height. The leaves are then cut into lines with
/// <see cref="LineFinder"/>. Ruling lines (very long thin runs of ink, horizontal or vertical) are not text: they are left out of
/// the ink before any profile is taken and erased from the grayscale page, so cell borders do not hide gutters or end up in a
/// line crop.
/// A page with neither a gutter nor a ruling line takes exactly the path of <see cref="LineFinder.Find(ReadOnlySpan{byte}, int, int, int)"/>
/// on the unchanged page, so single-column pages read as they always did.
/// </summary>
internal static class StripFinder
{
    private const double GutterLines = 1.25;
    private const double BlockGapLines = 0.5;
    private const double MinRegionLines = 2.5;

    private const byte Ink = 1;
    private const byte Rule = 2;

    private readonly record struct Rect(int X, int Y, int W, int H);

    /// <summary>Finds the lines. When the page has ruling lines, they are erased from <paramref name="gray"/> (in place).</summary>
    /// <param name="gray">The page as 8-bit gray, row-major, <paramref name="width"/> by <paramref name="height"/>.</param>
    /// <param name="width">Page width.</param>
    /// <param name="height">Page height.</param>
    /// <param name="maxLines">Most lines to return; more throws <see cref="OcrLimitException"/>.</param>
    /// <param name="masks">Scratch for the ink mask.</param>
    /// <param name="usedRegions">True when the page was cut into regions; false when the single-column finder alone was used.</param>
    /// <param name="threshold">The page's binarization threshold (negative when the page has no contrast).</param>
    /// <param name="inkIsBright">Whether ink is the bright side of the threshold.</param>
    /// <returns>The lines, leaf by leaf.</returns>
    public static List<StripLine> Find(
        Gray8[] gray, int width, int height, int maxLines, BufferPool<byte> masks, out bool usedRegions, out int threshold, out bool inkIsBright)
    {
        usedRegions = false;
        int count = checked(width * height);
        var result = new List<StripLine>();
        (threshold, inkIsBright) = LineFinder.Binarize(PixelSpans.AsBytes(gray.AsSpan(0, count)));
        if (threshold < 0)
        {
            return result;
        }

        byte[] mask = masks.Rent(count);
        try
        {
            bool rules = BuildMask(PixelSpans.AsBytes(gray.AsSpan(0, count)), width, height, threshold, inkIsBright, mask);
            int lineHeight = TypicalHeight(mask, width, height);
            bool cut = false;
            List<Rect> leaves = lineHeight == 0 ? [] : CutIntoLeaves(mask, width, height, lineHeight, maxLines, out cut);
            if (!rules && !cut)
            {
                foreach (LineBox box in LineFinder.Find(PixelSpans.AsBytes(gray.AsSpan(0, count)), width, height, maxLines, threshold, inkIsBright))
                {
                    result.Add(new StripLine(box, 0));
                }

                return result;
            }

            usedRegions = true;
            if (rules)
            {
                EraseRules(gray, mask, width, height, inkIsBright);
            }

            for (int leaf = 0; leaf < leaves.Count; leaf++)
            {
                Rect r = leaves[leaf];
                foreach (LineBox box in LineFinder.FindInMask(mask, width, new LineBox(r.X, r.Y, r.W, r.H), maxLines))
                {
                    if (result.Count >= maxLines)
                    {
                        throw new OcrLimitException(nameof(OcrLimits.MaxLinesPerPage), maxLines, maxLines + 1L);
                    }

                    result.Add(new StripLine(box, leaf));
                }
            }

            return result;
        }
        finally
        {
            masks.Return(mask);
        }
    }

    // Ink where the gray value is on the ink side of the threshold, except ruling lines: runs of ink at least `min(width, height) / 16`
    // pixels (at least 48) long along a row or a column. Those pixels are marked Rule. Returns whether there were any.
    private static bool BuildMask(ReadOnlySpan<byte> gray, int width, int height, int threshold, bool inkIsBright, byte[] mask)
    {
        int longRun = Math.Max(48, Math.Min(width, height) / 16);
        var vertical = new int[width];
        bool rules = false;
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<byte> row = gray.Slice(y * width, width);
            int rowBase = y * width;
            int runStart = -1;
            for (int x = 0; x < row.Length; x++)
            {
                bool ink = inkIsBright ? row[x] > threshold : row[x] <= threshold;
                mask[rowBase + x] = ink ? Ink : (byte)0;
                if (ink)
                {
                    if (runStart < 0)
                    {
                        runStart = x;
                    }

                    vertical[x]++;
                    continue;
                }

                if (runStart >= 0)
                {
                    if (x - runStart >= longRun)
                    {
                        mask.AsSpan(rowBase + runStart, x - runStart).Fill(Rule);
                        rules = true;
                    }

                    runStart = -1;
                }

                if (vertical[x] > 0)
                {
                    if (vertical[x] >= longRun)
                    {
                        MarkColumn(mask, width, x, y - vertical[x], y);
                        rules = true;
                    }

                    vertical[x] = 0;
                }
            }

            if (runStart >= 0 && width - runStart >= longRun)
            {
                mask.AsSpan(rowBase + runStart, width - runStart).Fill(Rule);
                rules = true;
            }
        }

        for (int x = 0; x < width; x++)
        {
            if (vertical[x] >= longRun)
            {
                MarkColumn(mask, width, x, height - vertical[x], height);
                rules = true;
            }
        }

        return rules;
    }

    private static void MarkColumn(byte[] mask, int width, int x, int fromRow, int toRowExclusive)
    {
        for (int y = fromRow; y < toRowExclusive; y++)
        {
            mask[(y * width) + x] = Rule;
        }
    }

    // Rule pixels and their direct neighbours (the anti-aliased edge of a thin line) become background, except pixels of text ink.
    private static void EraseRules(Gray8[] gray, byte[] mask, int width, int height, bool inkIsBright)
    {
        var background = new Gray8(inkIsBright ? (byte)0 : (byte)255);
        for (int y = 0; y < height; y++)
        {
            int rowBase = y * width;
            for (int x = 0; x < width; x++)
            {
                if (mask[rowBase + x] != Rule)
                {
                    continue;
                }

                gray[rowBase + x] = background;
                if (x > 0 && mask[rowBase + x - 1] != Ink)
                {
                    gray[rowBase + x - 1] = background;
                }

                if (x + 1 < width && mask[rowBase + x + 1] != Ink)
                {
                    gray[rowBase + x + 1] = background;
                }

                if (y > 0 && mask[rowBase - width + x] != Ink)
                {
                    gray[rowBase - width + x] = background;
                }

                if (y + 1 < height && mask[rowBase + width + x] != Ink)
                {
                    gray[rowBase + width + x] = background;
                }
            }
        }
    }

    // The page's typical line height: the median height of the lines found in vertical bands a quarter of the page wide, so that
    // staggered columns do not run into each other. 0 when no line is found.
    private static int TypicalHeight(byte[] mask, int width, int height)
    {
        int bandWidth = Math.Max(64, width / 4);
        int step = Math.Max(1, bandWidth / 2);
        var heights = new List<int>();
        for (int x = 0; x < width; x += step)
        {
            int w = Math.Min(bandWidth, width - x);
            if (w < Math.Min(32, width))
            {
                break;
            }

            foreach (LineBox line in LineFinder.FindInMask(mask, width, new LineBox(x, 0, w, height), int.MaxValue))
            {
                if (line.Height >= 4)
                {
                    heights.Add(line.Height);
                }
            }

            if (x + bandWidth >= width)
            {
                break;
            }
        }

        if (heights.Count == 0)
        {
            return 0;
        }

        heights.Sort();
        return heights[heights.Count / 2];
    }

    private static List<Rect> CutIntoLeaves(byte[] mask, int width, int height, int line, int maxLeaves, out bool cut)
    {
        cut = false;
        var leaves = new List<Rect>();
        int gutter = Math.Max(3, (int)Math.Ceiling(GutterLines * line));
        int blockGap = Math.Max(2, (int)(BlockGapLines * line));
        int minRows = (int)(MinRegionLines * line);
        var rows = new int[height];
        var cols = new int[width];
        var colsScratch = new int[width];
        var parts = new List<(int Start, int End)>();
        var stack = new Stack<Rect>();
        stack.Push(new Rect(0, 0, width, height));
        while (stack.Count > 0)
        {
            Rect r = stack.Pop();
            Profile(mask, width, r, rows, cols);
            if (!Trim(r, rows, cols, out Rect t))
            {
                continue;
            }

            int dx = t.X - r.X, dy = t.Y - r.Y;
            if (t.H >= minRows)
            {
                Split(cols, dx, dx + t.W - 1, Math.Max(1, t.H / 1000), gutter, parts);
                if (parts.Count > 1)
                {
                    cut = true;
                    for (int i = parts.Count - 1; i >= 0; i--)
                    {
                        stack.Push(new Rect(r.X + parts[i].Start, t.Y, parts[i].End - parts[i].Start + 1, t.H));
                    }

                    continue;
                }
            }

            Split(rows, dy, dy + t.H - 1, Math.Max(1, t.W / 1000), blockGap, parts);
            if (parts.Count <= 1)
            {
                AddLeaf(t);
                continue;
            }

            // Slices that have a column gutter of their own are cut further; the stretches between them are one leaf each, so a
            // paragraph is not chopped into slices. Consecutive slices that share a gutter are one region: the columns run across
            // their paragraph gaps, and a cut between them would read the columns' paragraphs in turn instead of column by column.
            var slices = parts.ToArray();
            var gutters = new List<(int Start, int End)>[slices.Length];
            var empties = new List<(int Start, int End)>[slices.Length];
            var inkFrom = new int[slices.Length];
            var inkTo = new int[slices.Length];
            for (int i = 0; i < slices.Length; i++)
            {
                (gutters[i], empties[i], inkFrom[i], inkTo[i]) = Gutters(mask, width, new Rect(t.X, r.Y + slices[i].Start, t.W, slices[i].End - slices[i].Start + 1), colsScratch, gutter, minRows);
            }

            int pendingStart = -1, pendingEnd = -1;
            for (int i = 0; i < slices.Length; i++)
            {
                if (gutters[i].Count == 0)
                {
                    if (pendingStart < 0)
                    {
                        pendingStart = r.Y + slices[i].Start;
                    }

                    pendingEnd = r.Y + slices[i].End;
                    continue;
                }

                if (pendingStart >= 0)
                {
                    AddLeaf(new Rect(t.X, pendingStart, t.W, pendingEnd - pendingStart + 1));
                    pendingStart = -1;
                }

                var common = gutters[i];
                int last = i;
                // The next slice continues the columns when no ink of it lies in a stripe the columns share: it may have a gutter of
                // its own, or hold only one column (the other has ended).
                while (last + 1 < slices.Length)
                {
                    List<(int Start, int End)> shared = Intersect(common, empties[last + 1], gutter);

                    // A short slice with ink on both sides of a stripe is a row across it (a table row, a heading), not a column.
                    int sliceRows = slices[last + 1].End - slices[last + 1].Start + 1;
                    if (sliceRows < minRows)
                    {
                        int from = inkFrom[last + 1], to = inkTo[last + 1];
                        shared.RemoveAll(stripe => from < stripe.Start && to > stripe.End);
                    }

                    if (shared.Count == 0)
                    {
                        break;
                    }

                    common = shared;
                    last++;
                }

                var region = new Rect(t.X, r.Y + slices[i].Start, t.W, slices[last].End - slices[i].Start + 1);
                if (region.H >= t.H)
                {
                    // The columns run through everything but the whole-region cut did not see them (its speck threshold
                    // differs): no progress is possible, so the region is a leaf.
                    AddLeaf(t);
                    pendingStart = -1;
                    break;
                }

                stack.Push(region);
                i = last;
            }

            if (pendingStart >= 0)
            {
                AddLeaf(new Rect(t.X, pendingStart, t.W, pendingEnd - pendingStart + 1));
            }
        }

        return leaves;

        void AddLeaf(Rect leaf)
        {
            if (leaves.Count >= maxLeaves)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxLinesPerPage), maxLeaves, maxLeaves + 1L);
            }

            leaves.Add(leaf);
        }
    }

    private static void Profile(byte[] mask, int stride, Rect r, int[] rows, int[] cols)
    {
        Array.Clear(rows, 0, r.H);
        Array.Clear(cols, 0, r.W);
        for (int y = 0; y < r.H; y++)
        {
            ReadOnlySpan<byte> row = mask.AsSpan(((r.Y + y) * stride) + r.X, r.W);
            int n = 0;
            for (int x = 0; x < row.Length; x++)
            {
                if (row[x] == Ink)
                {
                    n++;
                    cols[x]++;
                }
            }

            rows[y] = n;
        }
    }

    // The rectangle shrunk to the rows and columns that hold more than a speck of ink.
    private static bool Trim(Rect r, int[] rows, int[] cols, out Rect trimmed)
    {
        int rowNoise = Math.Max(1, r.W / 1000), colNoise = Math.Max(1, r.H / 1000);
        int y0 = 0, y1 = r.H - 1, x0 = 0, x1 = r.W - 1;
        while (y0 <= y1 && rows[y0] <= rowNoise)
        {
            y0++;
        }

        while (y1 >= y0 && rows[y1] <= rowNoise)
        {
            y1--;
        }

        while (x0 <= x1 && cols[x0] <= colNoise)
        {
            x0++;
        }

        while (x1 >= x0 && cols[x1] <= colNoise)
        {
            x1--;
        }

        trimmed = y0 > y1 || x0 > x1 ? default : new Rect(r.X + x0, r.Y + y0, x1 - x0 + 1, y1 - y0 + 1);
        return y0 <= y1 && x0 <= x1;
    }

    // Cuts [from, to] of a profile at every run of at least minGap entries not above noise; the parts are the stretches between them.
    private static void Split(int[] profile, int from, int to, int noise, int minGap, List<(int Start, int End)> parts)
    {
        parts.Clear();
        int start = -1, lastInk = -1;
        for (int i = from; i <= to; i++)
        {
            if (profile[i] > noise)
            {
                if (start < 0)
                {
                    start = i;
                }
                else if (i - lastInk - 1 >= minGap)
                {
                    parts.Add((start, lastInk));
                    start = i;
                }

                lastInk = i;
            }
        }

        if (start >= 0)
        {
            parts.Add((start, lastInk));
        }
    }

    // The empty vertical stripes, at least `gutter` wide, of a region (x ranges in page coordinates): `Between` those between its ink
    // columns, `Free` also those at its sides. `Between` is empty when the region is shorter than `minRows`.
    private static (List<(int Start, int End)> Between, List<(int Start, int End)> Free, int FirstInk, int LastInk) Gutters(
        byte[] mask, int stride, Rect slice, int[] cols, int gutter, int minRows)
    {
        var between = new List<(int, int)>();
        var free = new List<(int, int)>();
        Array.Clear(cols, 0, slice.W);
        for (int y = 0; y < slice.H; y++)
        {
            ReadOnlySpan<byte> row = mask.AsSpan(((slice.Y + y) * stride) + slice.X, slice.W);
            for (int x = 0; x < row.Length; x++)
            {
                if (row[x] == Ink)
                {
                    cols[x]++;
                }
            }
        }

        int noise = Math.Max(1, slice.H / 1000);
        int first = 0, last = slice.W - 1;
        while (first <= last && cols[first] <= noise)
        {
            first++;
        }

        while (last >= first && cols[last] <= noise)
        {
            last--;
        }

        int runStart = -1;
        for (int x = 0; x <= slice.W; x++)
        {
            bool empty = x < slice.W && cols[x] <= noise;
            if (empty && runStart < 0)
            {
                runStart = x;
            }
            else if (!empty && runStart >= 0)
            {
                if (x - runStart >= gutter)
                {
                    free.Add((slice.X + runStart, slice.X + x - 1));
                    if (runStart > first && x - 1 < last)
                    {
                        between.Add((slice.X + runStart, slice.X + x - 1));
                    }
                }

                runStart = -1;
            }
        }

        if (slice.H < minRows)
        {
            between.Clear();
        }

        return (between, free, slice.X + first, slice.X + last);
    }

    // The stretches that lie in a stripe of both lists and are at least `gutter` wide.
    private static List<(int Start, int End)> Intersect(List<(int Start, int End)> a, List<(int Start, int End)> b, int gutter)
    {
        var result = new List<(int, int)>();
        foreach ((int s1, int e1) in a)
        {
            foreach ((int s2, int e2) in b)
            {
                int s = Math.Max(s1, s2), e = Math.Min(e1, e2);
                if (e - s + 1 >= gutter)
                {
                    result.Add((s, e));
                }
            }
        }

        return result;
    }
}
