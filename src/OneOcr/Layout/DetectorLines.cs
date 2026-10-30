using OneImage;
using OneOcr.Detection;

namespace OneOcr.Layout;

/// <summary>
/// Where the page's orientation decision lives. The detector's orientation head gives a rotation and a confidence; this class turns
/// them into the quarter turns the page is rotated by before it is read, and does the rotation. A different source of the decision
/// (for example one that also looks at the text) replaces <see cref="QuarterTurnsClockwise"/> and nothing else changes.
/// </summary>
internal static class PageOrientation
{
    /// <summary>
    /// Clockwise quarter turns (0 to 3) that make the page upright: the head says the content is rotated that many quarter turns
    /// counter-clockwise from upright, which is undone by turning it clockwise by as many. 0 when the head is not sure enough.
    /// </summary>
    /// <param name="detection">What the detector found on the page as given.</param>
    /// <returns>The number of clockwise quarter turns to apply.</returns>
    public static int QuarterTurnsClockwise(PageDetection detection) =>
        detection.OrientationConfidence >= DetectorPostprocess.OrientationConfidenceMinimum ? detection.Quarter : 0;

    /// <summary>
    /// Turns an 8-bit gray page clockwise by <paramref name="quarterTurns"/> quarter turns into a pooled buffer. Pixel (x, y) of a
    /// page of width w and height h goes to (h-1-y, x) after one turn, (w-1-x, h-1-y) after two and (y, w-1-x) after three.
    /// </summary>
    /// <param name="source">The page, row-major.</param>
    /// <param name="width">Page width.</param>
    /// <param name="height">Page height.</param>
    /// <param name="quarterTurns">1, 2 or 3.</param>
    /// <param name="pool">Where the result's buffer comes from; the caller gives it back.</param>
    /// <param name="newWidth">Width of the result.</param>
    /// <param name="newHeight">Height of the result.</param>
    /// <returns>The rotated page.</returns>
    public static Gray8[] Rotate(ReadOnlySpan<Gray8> source, int width, int height, int quarterTurns, BufferPool<Gray8> pool, out int newWidth, out int newHeight)
    {
        if (quarterTurns is < 1 or > 3)
        {
            throw new ArgumentOutOfRangeException(nameof(quarterTurns));
        }

        newWidth = quarterTurns == 2 ? width : height;
        newHeight = quarterTurns == 2 ? height : width;
        Gray8[] result = pool.Rent((long)width * height);
        Span<Gray8> to = result.AsSpan(0, width * height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                (int nx, int ny) = quarterTurns switch
                {
                    1 => (height - 1 - y, x),
                    2 => (width - 1 - x, height - 1 - y),
                    _ => (y, width - 1 - x),
                };
                to[(ny * newWidth) + nx] = source[(y * width) + x];
            }
        }

        return result;
    }
}

/// <summary>The page as the detector located it: possibly turned upright, with its lines as strips and its regions.</summary>
internal sealed class LocatedPage
{
    /// <summary>The page pixels the lines refer to; the caller's own array unless the page was rotated.</summary>
    public required Gray8[] Gray { get; init; }

    public required bool OwnsGray { get; init; }
    public required int Width { get; init; }
    public required int Height { get; init; }
    public required List<StripLine> Strips { get; init; }
    public required List<DetectedRegion> Regions { get; init; }
    public required int Threshold { get; init; }
    public required bool InkIsBright { get; init; }
    public required List<string> Notes { get; init; }

    /// <summary>Gives the rotated copy back to <paramref name="pool"/>, if there is one.</summary>
    public void Release(BufferPool<Gray8> pool)
    {
        if (OwnsGray)
        {
            pool.Return(Gray);
        }
    }
}

/// <summary>Turns detector output into the line strips the reader works on.</summary>
internal static class DetectorLines
{
    /// <summary>The diagnostic that says the orientation head turned the page upright.</summary>
    public static string TurnedNote(int turns, float confidence) =>
        string.Create(System.Globalization.CultureInfo.InvariantCulture, $"The page was turned {turns * 90} degrees clockwise to read it upright (orientation confidence {confidence:0.00}); positions refer to the turned page.");

    // Region classes whose lines read together as one column or paragraph run (one leaf); lines of other classes, and lines outside
    // every region, are leaves of their own, so cells of a table row on one baseline still join into a row.
    private static readonly HashSet<string> RunningText = new(StringComparer.Ordinal) { "text", "title", "list", "header_footer" };

    /// <summary>
    /// Detects on the page, turns it upright when the orientation head is sure it is rotated (detecting again on the turned page),
    /// and returns the lines as ink-tight boxes with the regions that contain them.
    /// </summary>
    /// <remarks>
    /// <c>first</c> is a detection already made on exactly these pixels (used instead of detecting again); <c>turn</c> is false when the
    /// caller has already turned the page upright, so the head is not acted on.
    /// </remarks>
    public static LocatedPage Locate(
        IPageDetector detector, Gray8[] gray, int width, int height, BufferPool<Gray8> pixels, ResizerPool resizers, int degree, CancellationToken cancellationToken,
        PageDetection? first = null, bool turn = true, (int Level, bool InkIsBright)? known = null)
    {
        var notes = new List<string>();
        int count = checked(width * height);
        long clock = StageClock.Start();
        PageDetection detection = first ?? detector.Detect(gray.AsSpan(0, count), width, height, pixels, resizers, degree, cancellationToken);
        StageClock.Stop("6 detect again", clock);
        Gray8[] work = gray;
        bool owns = false;
        int turns = turn ? PageOrientation.QuarterTurnsClockwise(detection) : 0;
        if (turns != 0)
        {
            work = PageOrientation.Rotate(gray.AsSpan(0, count), width, height, turns, pixels, out int rw, out int rh);
            owns = true;
            width = rw;
            height = rh;
            notes.Add(TurnedNote(turns, detection.OrientationConfidence));
            try
            {
                detection = detector.Detect(work.AsSpan(0, count), width, height, pixels, resizers, degree, cancellationToken);
            }
            catch
            {
                pixels.Return(work);
                throw;
            }
        }

        try
        {
            clock = StageClock.Start();
            // The binarization is the caller's when it made it on these pixels (a page that was turned has new pixels).
            (int threshold, bool inkIsBright) = owns ? LineFinder.Binarize(work, count, degree) : known ?? LineFinder.Binarize(work, count, degree);
            var strips = new List<StripLine>(detection.Lines.Count);
            ReadOnlySpan<byte> page = Kernels.PixelSpans.AsBytes(work.AsSpan(0, count));
            // Only camera photos get line-by-line binarization (see LocalInk); null leaves every line to the page's.
            bool? darkInk = null;
            if (threshold >= 0 && detection.Sigma > LocalInk.MinPageNoise)
            {
                darkInk = true;
            }

            if (darkInk is true && inkIsBright)
            {
                var lineBoxes = new List<(int X0, int Y0, int X1, int Y1)>(detection.Lines.Count);
                foreach (DetectedLine l in detection.Lines)
                {
                    lineBoxes.Add((Math.Clamp((int)Math.Floor(l.Left), 0, width), Math.Clamp((int)Math.Floor(l.Top), 0, height), Math.Clamp((int)Math.Ceiling(l.Right), 0, width), Math.Clamp((int)Math.Ceiling(l.Bottom), 0, height)));
                }

                darkInk = LocalInk.PageIsDarkInk(page, width, lineBoxes);
            }

            // Page lines the ink of a line may not be followed into (see ExtendEnds); photos keep their own end handling.
            (int[]? Left, int[]? Right) limits = EndGap > 0 && darkInk is null ? NeighbourLimits(detection.Lines, width) : (null, null);
            int typicalBox = darkInk is null ? TypicalBoxHeight(detection.Lines) : 0;
            for (int i = 0; i < detection.Lines.Count; i++)
            {
                int leaf = LeafOf(detection, i);
                int before = strips.Count;
                // A box tall enough to hold several lines is tried upright first: the axis of a component that joins lines leans with
                // the lines it joins, and straightening along it smears the rows between them. When the upright cut finds several
                // lines they are used; otherwise the box goes on as any other (along its axis if it leans).
                var boxes = new List<LineBox>(1);
                LocalInk? local = null;
                bool cut = false;
                if (typicalBox > 0 && detection.Lines[i].Bottom - detection.Lines[i].Top >= TallBand * typicalBox)
                {
                    var cutFlag = new bool[1];
                    local = InkBoxes(detection.Lines[i], page, width, height, threshold, inkIsBright, boxes, darkInk, limits.Left?[i] ?? 0, limits.Right?[i] ?? width, typicalBox, RowNeighbours(detection.Lines, i), cutFlag);
                    cut = cutFlag[0];
                }

                if (!cut && TiltedLines.TryInkBoxes(detection.Lines[i], page, width, height, threshold, inkIsBright, strips, leaf, darkInk, typicalBox))
                {
                    MarkRecovered(strips, before, detection.Lines[i].MinConfidence);
                    continue;
                }

                if (boxes.Count == 0)
                {
                    local = InkBoxes(detection.Lines[i], page, width, height, threshold, inkIsBright, boxes, darkInk, limits.Left?[i] ?? 0, limits.Right?[i] ?? width, typicalBox);
                }

                foreach (LineBox b in boxes)
                {
                    strips.Add(new StripLine(b, leaf, Local: local, MinConfidence: detection.Lines[i].MinConfidence));
                }
            }

            StageClock.Stop("7 binarize+ink boxes", clock);
            return new LocatedPage
            {
                Gray = work, OwnsGray = owns, Width = width, Height = height, Strips = strips, Regions = detection.Regions,
                Threshold = threshold, InkIsBright = inkIsBright, Notes = notes,
            };
        }
        catch
        {
            if (owns)
            {
                pixels.Return(work);
            }

            throw;
        }
    }

    // A line recovered from faint text carries the confidence its words need; its strips (a tilted line makes one) are marked here.
    private static void MarkRecovered(List<StripLine> strips, int from, float minConfidence)
    {
        if (minConfidence <= 0)
        {
            return;
        }

        for (int s = from; s < strips.Count; s++)
        {
            strips[s] = strips[s] with { MinConfidence = minConfidence };
        }
    }

    /// <summary>
    /// How far a blank stretch may be, in band heights, for ink beyond a detected box's end to still belong to the line (0 turns the
    /// extension off). A letter gap is well under this; a word gap is not.
    /// </summary>
    internal static double EndGap = 0.25;

    /// <summary>The most ink, as a share of the area it was followed over, that still reads as letters.</summary>
    internal static double EndFill = 0.65;

    /// <summary>The least an end must move, in band heights, for the extension to be used.</summary>
    internal static double EndMin = 0.25;

    /// <summary>The most the ink may be followed past the detected box's end, in band heights.</summary>
    internal static double EndReach = 1.0;

    // The detector's box can end inside the last letter or just before it (its edge is a few pixels uncertain), so the line's ink band
    // is followed outwards from each end through ink that continues with only letter-sized gaps, never into the box of a neighbouring
    // line (the limits) and never farther than EndReach.
    private static void ExtendEnds(ReadOnlySpan<byte> page, int width, int rowFrom, int rowTo, int threshold, bool inkIsBright, int leftLimit, int rightLimit, ref int minX, ref int maxX)
    {
        int bandHeight = rowTo - rowFrom + 1;
        int gap = Math.Max(1, (int)Math.Round(EndGap * bandHeight));
        int reach = (int)(EndReach * bandHeight);
        int least = (int)Math.Ceiling(EndMin * bandHeight);

        // Ink to the right: columns up to the last one with ink, so long as the blank stretches stay within a letter gap.
        int last = maxX;
        long ink = 0, inkAtLast = 0;
        for (int x = maxX + 1; x < rightLimit && x - last <= gap && x - maxX <= reach; x++)
        {
            int n = ColumnInk(page, width, x, rowFrom, rowTo, threshold, inkIsBright);
            ink += n;
            if (n > 0)
            {
                last = x;
                inkAtLast = ink;
            }
        }

        // Small differences are the edge of the letters' anti-aliasing, not a cut letter: moving the crop for them only shifts what the
        // recognizer sees by a pixel, so only a real cut (the ink goes on for a good part of a letter) moves the end. A stretch that is
        // mostly ink is a button, a rule or a border, not letters.
        if (last - maxX >= least && IsLetterLike(inkAtLast, last - maxX, bandHeight))
        {
            maxX = last;
        }

        int first = minX;
        ink = 0;
        long inkAtFirst = 0;
        for (int x = minX - 1; x >= leftLimit && first - x <= gap && minX - x <= reach; x--)
        {
            int n = ColumnInk(page, width, x, rowFrom, rowTo, threshold, inkIsBright);
            ink += n;
            if (n > 0)
            {
                first = x;
                inkAtFirst = ink;
            }
        }

        if (minX - first >= least && IsLetterLike(inkAtFirst, minX - first, bandHeight))
        {
            minX = first;
        }
    }

    private static bool IsLetterLike(long inkPixels, int columns, int bandHeight) => inkPixels <= EndFill * columns * bandHeight;

    private static int ColumnInk(ReadOnlySpan<byte> page, int width, int x, int rowFrom, int rowTo, int threshold, bool inkIsBright)
    {
        int n = 0;
        for (int y = rowFrom; y <= rowTo; y++)
        {
            byte v = page[(y * width) + x];
            if (inkIsBright ? v > threshold : v <= threshold)
            {
                n++;
            }
        }

        return n;
    }

    // For every detected line, the nearest neighbouring detected line on each side on the same text row (the page edge when there is
    // none): the ink of a line is never followed into it. Lines are visited in order of their vertical centre and only those whose
    // centre is close enough to share a row are compared.
    private static (int[] Left, int[] Right) NeighbourLimits(List<DetectedLine> lines, int width)
    {
        int n = lines.Count;
        var left = new int[n];
        var right = new int[n];
        var order = new int[n];
        var centre = new double[n];
        double tallest = 0;
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
            right[i] = width;
            centre[i] = (lines[i].Top + lines[i].Bottom) / 2;
            tallest = Math.Max(tallest, lines[i].Bottom - lines[i].Top);
        }

        Array.Sort(order, (a, b) => centre[a] != centre[b] ? centre[a].CompareTo(centre[b]) : a.CompareTo(b));
        for (int p = 0; p < n; p++)
        {
            int i = order[p];
            DetectedLine a = lines[i];
            double middle = (a.Left + a.Right) / 2, ah = a.Bottom - a.Top;
            for (int dir = -1; dir <= 1; dir += 2)
            {
                for (int q = p + dir; q >= 0 && q < n; q += dir)
                {
                    int j = order[q];
                    if (Math.Abs(centre[j] - centre[i]) > ((ah + tallest) / 2))
                    {
                        break;
                    }

                    DetectedLine b = lines[j];
                    double overlap = Math.Min(a.Bottom, b.Bottom) - Math.Max(a.Top, b.Top);
                    if (overlap < 0.5 * Math.Min(ah, b.Bottom - b.Top))
                    {
                        continue;
                    }

                    if (b.Left >= middle)
                    {
                        right[i] = Math.Min(right[i], (int)Math.Floor(b.Left));
                    }
                    else if (b.Right <= middle)
                    {
                        left[i] = Math.Max(left[i], (int)Math.Ceiling(b.Right));
                    }
                }
            }
        }

        return (left, right);
    }

    /// <summary>
    /// A band of ink rows at least this many typical detected-box heights tall is two or more lines the detector joined and is split
    /// where the rows between the lines are nearly empty (see <see cref="SplitTallBands"/>).
    /// </summary>
    internal static double TallBand = 1.5;

    /// <summary>A cut row's ink may be at most this share of the weaker side's densest row for the cut to count as a gap between lines.</summary>
    internal static double ValleyShare = 0.2;

    /// <summary>The fewest detected lines a page needs for its typical box height to mean anything.</summary>
    private const int TypicalBoxMinimumLines = 8;

    // The median height of the page's detected boxes (0 when there are too few to say): what one line of this page's running text looks like.
    private static int TypicalBoxHeight(List<DetectedLine> lines)
    {
        if (TallBand <= 0 || lines.Count < TypicalBoxMinimumLines)
        {
            return 0;
        }

        var heights = new List<double>(lines.Count);
        foreach (DetectedLine l in lines)
        {
            if (l.MinConfidence <= 0)
            {
                heights.Add(l.Bottom - l.Top);
            }
        }

        if (heights.Count < TypicalBoxMinimumLines)
        {
            return 0;
        }

        heights.Sort();
        return (int)Math.Round(heights[heights.Count / 2]);
    }

    // Lines set solid (leading barely above the type size) touch row by row, so the blank-gap test above cannot separate two lines the
    // detector boxed together. A band that is much taller than a typical line is cut at its emptiest row, if that row is a real valley
    // (almost no ink compared with the denser side, so it lies between two dense bands and not inside one line's ascender or descender
    // rows) and both parts are at least half a line tall. A part is cut again while it is still much taller than the smaller of the
    // typical line and its sibling, so a box of three lines of small type is cut twice. The cut row belongs to neither part.
    private static bool SplitTallBands(int[] counts, List<(int Start, int End, long Mass)> bands, int typicalBox)
    {
        bool split = false;
        var result = new List<(int Start, int End, long Mass)>(bands.Count + 2);
        var work = new Stack<(int Start, int End, int Reference)>();
        foreach ((int start, int end, long mass) in bands)
        {
            if (end - start + 1 < (int)Math.Ceiling(TallBand * typicalBox))
            {
                result.Add((start, end, mass));
                continue;
            }

            work.Push((start, end, typicalBox));
            while (work.Count > 0)
            {
                (int s, int e, int reference) = work.Pop();
                int cut = e - s + 1 >= (int)Math.Ceiling(TallBand * reference) ? Valley(counts, s, e, Math.Max(2, reference / 2)) : -1;
                if (cut < 0)
                {
                    long total = 0;
                    for (int r = s; r <= e; r++)
                    {
                        total += counts[r];
                    }

                    result.Add((s, e, total));
                    continue;
                }

                split = true;
                int e1 = cut - 1, s2 = cut + 1;
                while (e1 > s && counts[e1] == 0)
                {
                    e1--;
                }

                while (s2 < e && counts[s2] == 0)
                {
                    s2++;
                }

                work.Push((s2, e, Math.Min(reference, e1 - s + 1)));
                work.Push((s, e1, Math.Min(reference, e - s2 + 1)));
            }
        }

        result.Sort((x, y) => x.Start.CompareTo(y.Start));
        bands.Clear();
        bands.AddRange(result);
        return split;
    }

    /// <summary>The share of a band's rows that another detected line's box must cover for the band to be that line (read there, not here).</summary>
    internal static double CoverShare = 0.5;

    // A box that spans several lines is often drawn over a line that has a box of its own (the lines of a shaded note beside running
    // text, say). A part cut from it that another detected line already covers would be read twice, so it is dropped (when every
    // part is covered, all are kept).
    private static void DropCoveredBands(List<(int Start, int End, long Mass)> bands, List<(double Top, double Bottom)> neighbours, int y0)
    {
        var keep = new List<(int Start, int End, long Mass)>(bands.Count);
        foreach ((int start, int end, long mass) in bands)
        {
            double from = y0 + start, to = y0 + end + 1, best = 0;
            foreach ((double top, double bottom) in neighbours)
            {
                best = Math.Max(best, Math.Min(to, bottom) - Math.Max(from, top));
            }

            if (best < CoverShare * (to - from))
            {
                keep.Add((start, end, mass));
            }
        }

        if (keep.Count > 0)
        {
            bands.Clear();
            bands.AddRange(keep);
        }
    }

    // The row of the deepest valley in rows [start, end] with at least `least` rows on each side, or -1 when no row is empty enough.
    // A row's depth is its ink over the weaker side's densest row; of equal depths the one nearest the middle wins.
    private static int Valley(int[] counts, int start, int end, int least)
    {
        int cut = -1, centre = (start + end) / 2;
        double bestDepth = double.MaxValue;
        var above = new int[end - start + 1];
        int peak = 0;
        for (int r = start; r <= end; r++)
        {
            above[r - start] = peak;
            peak = Math.Max(peak, counts[r]);
        }

        int below = 0;
        for (int r = end; r >= start + least; r--)
        {
            if (r <= end - least)
            {
                int weaker = Math.Min(above[r - start], below);
                double depth = (double)counts[r] / Math.Max(1, weaker);
                if (depth <= ValleyShare && (depth < bestDepth - 1e-9 || (Math.Abs(depth - bestDepth) <= 1e-9 && Math.Abs(r - centre) < Math.Abs(cut - centre))))
                {
                    bestDepth = depth;
                    cut = r;
                }
            }

            below = Math.Max(below, counts[r]);
        }

        return cut;
    }

    // The vertical extents of the other detected lines that share most of line i's width: what already reads the rows beside and over it.
    private static List<(double Top, double Bottom)> RowNeighbours(List<DetectedLine> lines, int i)
    {
        DetectedLine a = lines[i];
        var result = new List<(double Top, double Bottom)>();
        for (int j = 0; j < lines.Count; j++)
        {
            DetectedLine b = lines[j];
            if (j == i || b.MinConfidence > 0)
            {
                continue;
            }

            double shared = Math.Min(a.Right, b.Right) - Math.Max(a.Left, b.Left);
            if (shared >= 0.5 * Math.Min(a.Right - a.Left, b.Right - b.Left) && b.Bottom - b.Top < a.Bottom - a.Top)
            {
                result.Add((b.Top, b.Bottom));
            }
        }

        return result;
    }

    // How many line heights short of the detector's extent the ink may stop on a photo before the detector's end is used.
    private const double EndGapHeights = 0.8;

    // The ink inside the detected box, as the classical finder's boxes are (the recognizer's crop rules are written for ink-tight
    // lines); the detected box itself when the page has no contrast or no ink falls inside it. Returns the line's own binarization when
    // the page-wide one does not fit the line (see LocalInk), which the line's word boxes must then use too; null otherwise.
    internal static LocalInk? InkBoxes(
        DetectedLine line, ReadOnlySpan<byte> page, int width, int height, int threshold, bool inkIsBright, List<LineBox> result, bool? pageIsDarkInk = null,
        int leftLimit = 0, int rightLimit = int.MaxValue, int typicalBox = 0, List<(double Top, double Bottom)>? neighbours = null, bool[]? cutFlag = null)
    {
        int x0 = Math.Clamp((int)Math.Floor(line.Left), 0, width), x1 = Math.Clamp((int)Math.Ceiling(line.Right), 0, width);
        int y0 = Math.Clamp((int)Math.Floor(line.Top), 0, height), y1 = Math.Clamp((int)Math.Ceiling(line.Bottom), 0, height);
        if (x1 <= x0 || y1 <= y0)
        {
            return null;
        }

        LocalInk? local = null;
        if (threshold >= 0)
        {
            local = pageIsDarkInk is { } dark ? LocalInk.Choose(page, width, x0, x1, y0, y1, threshold, inkIsBright, dark) : null;
            if (local is { } own)
            {
                threshold = own.Threshold;
                inkIsBright = own.InkIsBright;
            }

            // Ink per row. A detected box is a little taller than the line (the training shrink is undone generously), so it can reach
            // the descenders of the line above or the ascenders of the line below; only the band of rows that holds most of the ink
            // belongs to this line. Rows separated by a short blank gap (the dot of an i, an accent) stay one band; a gap of more than
            // an eighth of the box height ends it, which is less than the space between two lines of running text.
            int rows = y1 - y0;
            int[] counts = System.Buffers.ArrayPool<int>.Shared.Rent(rows);
            try
            {
                Array.Clear(counts, 0, rows);
                for (int y = y0; y < y1; y++)
                {
                    ReadOnlySpan<byte> row = page.Slice((y * width) + x0, x1 - x0);
                    int n = 0;
                    for (int x = 0; x < row.Length; x++)
                    {
                        if (inkIsBright ? row[x] > threshold : row[x] <= threshold)
                        {
                            n++;
                        }
                    }

                    counts[y - y0] = n;
                }

                int gap = Math.Max(2, (rows + 4) / 8);
                var bands = new List<(int Start, int End, long Mass)>(2);
                long bestMass = 0;
                for (int r = 0; r < rows;)
                {
                    if (counts[r] == 0)
                    {
                        r++;
                        continue;
                    }

                    int start = r, end = r, blank = 0;
                    long mass = 0;
                    for (; r < rows && blank <= gap; r++)
                    {
                        if (counts[r] == 0)
                        {
                            blank++;
                            continue;
                        }

                        blank = 0;
                        end = r;
                        mass += counts[r];
                    }

                    bands.Add((start, end, mass));
                    bestMass = Math.Max(bestMass, mass);
                }

                bool cutIntoLines = false;
                if (typicalBox > 0)
                {
                    cutIntoLines = SplitTallBands(counts, bands, typicalBox);
                    if (cutIntoLines && neighbours is not null)
                    {
                        DropCoveredBands(bands, neighbours, y0);
                    }

                    if (cutFlag is { Length: > 0 })
                    {
                        cutFlag[0] = cutIntoLines;
                    }

                    bestMass = 0;
                    foreach ((int _, int _, long m) in bands)
                    {
                        bestMass = Math.Max(bestMass, m);
                    }
                }

                // The main band, and any other band that holds at least 40% as much ink (two lines the detector joined); a smaller
                // one is the edge of a neighbouring line.
                foreach ((int start, int end, long mass) in bands)
                {
                    // Parts cut from a joined box are lines in their own right (a short last line has little ink beside a full one), so
                    // they only need a tenth of the heaviest part's ink to not be specks.
                    if (cutIntoLines ? mass * 10 < bestMass : mass * 5 < bestMass * 2)
                    {
                        continue;
                    }

                    int minX = int.MaxValue, maxX = -1;
                    for (int y = y0 + start; y <= y0 + end; y++)
                    {
                        ReadOnlySpan<byte> row = page.Slice((y * width) + x0, x1 - x0);
                        for (int x = 0; x < row.Length; x++)
                        {
                            if (inkIsBright ? row[x] > threshold : row[x] <= threshold)
                            {
                                minX = Math.Min(minX, x0 + x);
                                maxX = Math.Max(maxX, x0 + x);
                            }
                        }
                    }

                    if (pageIsDarkInk is not null)
                    {
                        // On a photo the ends of a line are where blur and uneven light leave the faintest print, so the ink can stop
                        // well short of the line's end: an end that the ink leaves more than about one letter height short of the
                        // detector's extent is taken from the detector. (A detected box reaches about half a height past the ink.)
                        int reach = (int)(EndGapHeights * (end - start + 1));
                        if (minX - x0 > reach)
                        {
                            minX = x0;
                        }

                        if (x1 - 1 - maxX > reach)
                        {
                            maxX = x1 - 1;
                        }
                    }

                    if (EndGap > 0 && pageIsDarkInk is null)
                    {
                        ExtendEnds(page, width, y0 + start, y0 + end, threshold, inkIsBright, Math.Max(0, leftLimit), Math.Min(width, rightLimit), ref minX, ref maxX);
                    }

                    result.Add(new LineBox(minX, y0 + start, maxX - minX + 1, end - start + 1));
                }

                if (result.Count > 0)
                {
                    return local;
                }
            }
            finally
            {
                System.Buffers.ArrayPool<int>.Shared.Return(counts);
            }
        }

        result.Add(new LineBox(x0, y0, x1 - x0, y1 - y0));
        return null;
    }

    // The leaf of a line: the smallest running-text region that holds its centre (offset by one so 0 is never a region), otherwise a
    // leaf of its own.
    private static int LeafOf(PageDetection detection, int lineIndex)
    {
        DetectedLine line = detection.Lines[lineIndex];
        double cx = (line.Left + line.Right) / 2, cy = (line.Top + line.Bottom) / 2;
        int best = -1;
        double bestArea = double.MaxValue;
        for (int r = 0; r < detection.Regions.Count; r++)
        {
            DetectedRegion region = detection.Regions[r];
            if (!RunningText.Contains(region.Name) || cx < region.Left || cx > region.Right || cy < region.Top || cy > region.Bottom)
            {
                continue;
            }

            double area = (region.Right - region.Left) * (region.Bottom - region.Top);
            if (area < bestArea)
            {
                best = r;
                bestArea = area;
            }
        }

        return best >= 0 ? 1 + best : 1_000_000 + lineIndex;
    }
}
