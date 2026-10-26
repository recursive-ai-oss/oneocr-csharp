using System.Runtime.CompilerServices;
using System.Buffers;
using System.Numerics;
using OneImage;
using OneOcr.Kernels;

namespace OneOcr.Layout;

/// <summary>
/// A text line cut out of one leaf region of the page, with the leaf's number (lines of one leaf are never one line of the page).
/// <paramref name="Tilt"/> is set for a line read along its own tilted axis (see <see cref="TiltedLines"/>); its <paramref name="Box"/> is then
/// the straightened line's ink rectangle placed, unrotated, at the centre of the tilted one, which is what reading order compares.
/// <paramref name="Local"/> is set when the page-wide binarization does not suit this line (see <see cref="LocalInk"/>); null means the page's.
/// </summary>
internal readonly record struct StripLine(LineBox Box, int Leaf, TiltedLine? Tilt = null, LocalInk? Local = null);

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

    private const byte Ink = PageMask.Ink;

    private readonly record struct Rect(int X, int Y, int W, int H);

    /// <summary>Finds the lines. When the page has ruling lines, they are erased from <paramref name="gray"/> (in place) unless <paramref name="eraseRules"/> is false.</summary>
    /// <param name="gray">The page as 8-bit gray, row-major, <paramref name="width"/> by <paramref name="height"/>.</param>
    /// <param name="width">Page width.</param>
    /// <param name="height">Page height.</param>
    /// <param name="maxLines">Most lines to return; more throws <see cref="OcrLimitException"/>.</param>
    /// <param name="masks">Scratch for the ink mask.</param>
    /// <param name="degree">Most threads to use for the binarization and the mask; the result does not depend on it.</param>
    /// <param name="usedRegions">True when the page was cut into regions; false when the single-column finder alone was used.</param>
    /// <param name="threshold">The page's binarization threshold (negative when the page has no contrast).</param>
    /// <param name="inkIsBright">Whether ink is the bright side of the threshold.</param>
    /// <param name="eraseRules">Whether ruling lines are erased from <paramref name="gray"/>; the lines found are the same either way.</param>
    /// <returns>The lines, leaf by leaf.</returns>
    public static List<StripLine> Find(
        Gray8[] gray, int width, int height, int maxLines, BufferPool<byte> masks, int degree, out bool usedRegions, out int threshold, out bool inkIsBright, bool eraseRules = true)
    {
        usedRegions = false;
        int count = checked(width * height);
        var result = new List<StripLine>();
        (threshold, inkIsBright) = LineFinder.Binarize(gray, count, degree);
        if (threshold < 0)
        {
            return result;
        }

        byte[] storage = masks.Rent(PageMask.BytesFor(width, height));
        try
        {
            var mask = new PageMask(storage, width, height);
            bool rules = BuildMask(gray, width, height, threshold, inkIsBright, mask, degree);
            int lineHeight = TypicalHeight(mask, degree);
            bool cut = false;
            List<Rect> leaves = lineHeight == 0 ? [] : CutIntoLeaves(mask, lineHeight, maxLines, out cut);
            if (!rules && !cut)
            {
                // Without rules the mask holds exactly the pixels on the ink side of the threshold, so its rows give the profiles
                // LineFinder.Find would compute from the page.
                foreach (LineBox box in LineFinder.FindInMask(mask, new LineBox(0, 0, width, height), maxLines))
                {
                    result.Add(new StripLine(box, 0));
                }

                    return result;
            }

            usedRegions = true;
            if (rules && eraseRules)
            {
                EraseRules(gray, mask, inkIsBright);
            }

            for (int leaf = 0; leaf < leaves.Count; leaf++)
            {
                Rect r = leaves[leaf];
                foreach (LineBox box in LineFinder.FindInMask(mask, new LineBox(r.X, r.Y, r.W, r.H), maxLines))
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
            masks.Return(storage);
        }
    }

    // Ink where the gray value is on the ink side of the threshold, except ruling lines: runs of ink at least `min(width, height) / 16`
    // pixels (at least 48) long along a row or a column. Those pixels are marked Rule. Returns whether there were any.
    // The page is cut into bands of rows that are filled in parallel. A horizontal run lies in one row. A vertical run that crosses
    // a band edge is found when the bands are joined, which gives the marks a sequential scan would make.
    internal static bool BuildMask(Gray8[] gray, int width, int height, int threshold, bool inkIsBright, PageMask mask, int degree)
    {
        int longRun = Math.Max(48, Math.Min(width, height) / 16);
        int bandCount = degree <= 1 ? 1 : Math.Clamp(height / 64, 1, degree * 2);
        int bandRows = (height + bandCount - 1) / bandCount;
        bandCount = (height + bandRows - 1) / bandRows;
        var bands = new Band[bandCount];
        ParallelRun.For(bandCount, degree, i =>
        {
            int y0 = i * bandRows;
            bands[i] = FillBand(gray, width, y0, Math.Min(height, y0 + bandRows), threshold, inkIsBright, mask, longRun);
        }, CancellationToken.None);

        bool rules = false;
        int[] carry = ArrayPool<int>.Shared.Rent(width);
        Array.Clear(carry, 0, width);
        for (int i = 0; i < bandCount; i++)
        {
            Band band = bands[i];
            rules |= band.Rules;
            int top = i * bandRows, rows = Math.Min(height, top + bandRows) - top;
            for (int x = 0; x < width; x++)
            {
                int first = band.FirstRun[x];
                if (first == rows)
                {
                    // Ink all the way down this band: the run goes on.
                    carry[x] += rows;
                    continue;
                }

                if (carry[x] > 0 || first > 0)
                {
                    int length = carry[x] + first;
                    if (length >= longRun)
                    {
                        MarkColumn(mask, x, top + first - length, top + first);
                        rules = true;
                    }
                }

                carry[x] = band.LastRun[x];
            }
        }

        for (int x = 0; x < width; x++)
        {
            if (carry[x] >= longRun)
            {
                MarkColumn(mask, x, height - carry[x], height);
                rules = true;
            }
        }

        ArrayPool<int>.Shared.Return(carry);
        foreach (Band band in bands)
        {
            ArrayPool<int>.Shared.Return(band.FirstRun);
            ArrayPool<int>.Shared.Return(band.LastRun);
        }

        return rules;
    }

    // What a band tells its neighbours: the length of the vertical ink run at its top and at its bottom for each column (the whole
    // band's height for a column that is ink throughout), and whether it holds a ruling line.
    private sealed record Band(int[] FirstRun, int[] LastRun, bool Rules);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static Band FillBand(Gray8[] gray, int width, int y0, int y1, int threshold, bool inkIsBright, PageMask mask, int longRun)
    {
        ReadOnlySpan<byte> pixels = PixelSpans.AsBytes(gray.AsSpan(y0 * width, (y1 - y0) * width));
        int[] first = ArrayPool<int>.Shared.Rent(width);
        int[] last = ArrayPool<int>.Shared.Rent(width);
        int[] vertical = ArrayPool<int>.Shared.Rent(width);
        Array.Clear(first, 0, width);
        Array.Clear(last, 0, width);
        Array.Clear(vertical, 0, width);
        // Ink runs of the row just done and of the row being done, as start/end pairs.
        int[] previous = ArrayPool<int>.Shared.Rent(width + 2);
        int[] current = ArrayPool<int>.Shared.Rent(width + 2);
        int previousRuns = 0;
        byte[] ink = ArrayPool<byte>.Shared.Rent(width);
        bool rules = false;
        try
        {
            for (int y = y0; y < y1; y++)
            {
                MarkInk(pixels.Slice((y - y0) * width, width), threshold, inkIsBright, ink);
                ReadOnlySpan<byte> inkRow = ink.AsSpan(0, width);
                mask.StoreInkRow(y, inkRow);

                // The ink runs of this row.
                int runs = 0;
                for (int x = 0; x < width;)
                {
                    int s = inkRow[x..].IndexOf((byte)1);
                    if (s < 0)
                    {
                        break;
                    }

                    s += x;
                    int length = inkRow[s..].IndexOfAnyExcept((byte)1);
                    int e = length < 0 ? width : s + length;
                    current[2 * runs] = s;
                    current[(2 * runs) + 1] = e;
                    runs++;
                    x = e;
                }

                // Columns that had ink in the last row and have none now end a vertical run.
                for (int r = 0; r < previousRuns; r++)
                {
                    for (int x = previous[2 * r]; x < previous[(2 * r) + 1]; x++)
                    {
                        if (inkRow[x] != 0)
                        {
                            continue;
                        }

                        int v = vertical[x];
                        if (y - v == y0)
                        {
                            first[x] = v;
                        }
                        else if (v >= longRun)
                        {
                            MarkColumn(mask, x, y - v, y);
                            rules = true;
                        }

                        vertical[x] = 0;
                    }
                }

                for (int r = 0; r < runs; r++)
                {
                    int s = current[2 * r], e = current[(2 * r) + 1];
                    for (int x = s; x < e; x++)
                    {
                        vertical[x]++;
                    }

                    if (e - s >= longRun)
                    {
                        mask.SetRun(y, s, e, PageMask.Rule);
                        rules = true;
                    }
                }

                (previous, current) = (current, previous);
                previousRuns = runs;
            }

            int rows = y1 - y0;
            for (int r = 0; r < previousRuns; r++)
            {
                for (int x = previous[2 * r]; x < previous[(2 * r) + 1]; x++)
                {
                    int v = vertical[x];
                    if (v == rows)
                    {
                        first[x] = rows;
                    }

                    last[x] = v;
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(ink);
            ArrayPool<int>.Shared.Return(previous);
            ArrayPool<int>.Shared.Return(current);
            ArrayPool<int>.Shared.Return(vertical);
        }

        return new Band(first, last, rules);
    }

    // ink[x] = 1 where the pixel is on the ink side of the threshold, else 0.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MarkInk(ReadOnlySpan<byte> row, int threshold, bool inkIsBright, byte[] ink)
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var limit = new Vector<byte>((byte)threshold);
            for (; x + Vector<byte>.Count <= row.Length; x += Vector<byte>.Count)
            {
                var v = new Vector<byte>(row[x..]);
                Vector<byte> hit = inkIsBright ? Vector.GreaterThan(v, limit) : Vector.LessThanOrEqual(v, limit);
                (hit & Vector<byte>.One).CopyTo(ink.AsSpan(x));
            }
        }

        for (; x < row.Length; x++)
        {
            ink[x] = (byte)((inkIsBright ? row[x] > threshold : row[x] <= threshold) ? 1 : 0);
        }
    }

    private static void MarkColumn(PageMask mask, int x, int fromRow, int toRowExclusive)
    {
        for (int y = fromRow; y < toRowExclusive; y++)
        {
            mask.Set(x, y, PageMask.Rule);
        }
    }

    // Rule pixels and their direct neighbours (the anti-aliased edge of a thin line) become background, except pixels of text ink.
    private static void EraseRules(Gray8[] gray, PageMask mask, bool inkIsBright)
    {
        int width = mask.Width, height = mask.Height;
        var background = new Gray8(inkIsBright ? (byte)0 : (byte)255);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(width * 3);
        try
        {
            Span<byte> above = buffer.AsSpan(0, width), here = buffer.AsSpan(width, width), below = buffer.AsSpan(2 * width, width);
            for (int y = 0; y < height; y++)
            {
                if (y > 0)
                {
                    mask.ReadRow(y - 1, 0, width, above);
                }

                mask.ReadRow(y, 0, width, here);
                if (y + 1 < height)
                {
                    mask.ReadRow(y + 1, 0, width, below);
                }

                int rowBase = y * width;
                for (int x = 0; x < width; x++)
                {
                    if (here[x] != PageMask.Rule)
                    {
                        continue;
                    }

                    gray[rowBase + x] = background;
                    if (x > 0 && here[x - 1] != PageMask.Ink)
                    {
                        gray[rowBase + x - 1] = background;
                    }

                    if (x + 1 < width && here[x + 1] != PageMask.Ink)
                    {
                        gray[rowBase + x + 1] = background;
                    }

                    if (y > 0 && above[x] != PageMask.Ink)
                    {
                        gray[rowBase - width + x] = background;
                    }

                    if (y + 1 < height && below[x] != PageMask.Ink)
                    {
                        gray[rowBase + width + x] = background;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    // The page's typical line height: the median height of the lines found in vertical bands a quarter of the page wide, so that
    // staggered columns do not run into each other. 0 when no line is found.
    private static int TypicalHeight(PageMask mask, int degree)
    {
        int width = mask.Width, height = mask.Height;
        int bandWidth = Math.Max(64, width / 4);
        int step = Math.Max(1, bandWidth / 2);
        var starts = new List<int>();
        for (int x = 0; x < width; x += step)
        {
            if (Math.Min(bandWidth, width - x) < Math.Min(32, width))
            {
                break;
            }

            starts.Add(x);
            if (x + bandWidth >= width)
            {
                break;
            }
        }

        var found = new List<int>[starts.Count];
        ParallelRun.For(starts.Count, degree, i =>
        {
            int x = starts[i];
            var list = new List<int>();
            foreach (LineBox line in LineFinder.FindInMask(mask, new LineBox(x, 0, Math.Min(bandWidth, width - x), height), int.MaxValue))
            {
                if (line.Height >= 4)
                {
                    list.Add(line.Height);
                }
            }

            found[i] = list;
        }, CancellationToken.None);

        var heights = new List<int>();
        foreach (List<int> list in found)
        {
            heights.AddRange(list);
        }

        if (heights.Count == 0)
        {
            return 0;
        }

        heights.Sort();
        return heights[heights.Count / 2];
    }

    private static List<Rect> CutIntoLeaves(PageMask mask, int line, int maxLeaves, out bool cut)
    {
        int width = mask.Width, height = mask.Height;
        cut = false;
        var leaves = new List<Rect>();
        int gutter = Math.Max(3, (int)Math.Ceiling(GutterLines * line));
        int blockGap = Math.Max(2, (int)(BlockGapLines * line));
        int minRows = (int)(MinRegionLines * line);
        int[] rows = ArrayPool<int>.Shared.Rent(height);
        int[] cols = ArrayPool<int>.Shared.Rent(width);
        int[] colsScratch = ArrayPool<int>.Shared.Rent(width);
        var parts = new List<(int Start, int End)>();
        var stack = new Stack<Rect>();
        stack.Push(new Rect(0, 0, width, height));
        while (stack.Count > 0)
        {
            Rect r = stack.Pop();
            Profile(mask, r, rows, cols);
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
                (gutters[i], empties[i], inkFrom[i], inkTo[i]) = Gutters(mask, new Rect(t.X, r.Y + slices[i].Start, t.W, slices[i].End - slices[i].Start + 1), colsScratch, gutter, minRows);
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

        ArrayPool<int>.Shared.Return(rows);
        ArrayPool<int>.Shared.Return(cols);
        ArrayPool<int>.Shared.Return(colsScratch);
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

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Profile(PageMask mask, Rect r, int[] rows, int[] cols)
    {
        Array.Clear(rows, 0, r.H);
        Array.Clear(cols, 0, r.W);
        byte[] scratch = ArrayPool<byte>.Shared.Rent(r.W);
        try
        {
            for (int y = 0; y < r.H; y++)
            {
                mask.ReadRow(r.Y + y, r.X, r.W, scratch);
                int n = scratch.AsSpan(0, r.W).Count(Ink);
                rows[y] = n;
                if (n == 0)
                {
                    continue;
                }

                for (int x = 0; x < r.W; x++)
                {
                    if (scratch[x] == Ink)
                    {
                        cols[x]++;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(scratch);
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
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static (List<(int Start, int End)> Between, List<(int Start, int End)> Free, int FirstInk, int LastInk) Gutters(
        PageMask mask, Rect slice, int[] cols, int gutter, int minRows)
    {
        var between = new List<(int, int)>();
        var free = new List<(int, int)>();
        Array.Clear(cols, 0, slice.W);
        byte[] scratch = ArrayPool<byte>.Shared.Rent(slice.W);
        try
        {
            for (int y = 0; y < slice.H; y++)
            {
                mask.ReadRow(slice.Y + y, slice.X, slice.W, scratch);
                if (!scratch.AsSpan(0, slice.W).Contains(Ink))
                {
                    continue;
                }

                for (int x = 0; x < slice.W; x++)
                {
                    if (scratch[x] == Ink)
                    {
                        cols[x]++;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(scratch);
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
