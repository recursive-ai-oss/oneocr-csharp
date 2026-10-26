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

            for (int i = 0; i < detection.Lines.Count; i++)
            {
                int leaf = LeafOf(detection, i);
                if (TiltedLines.TryInkBoxes(detection.Lines[i], page, width, height, threshold, inkIsBright, strips, leaf, darkInk))
                {
                    continue;
                }

                var boxes = new List<LineBox>(1);
                LocalInk? local = InkBoxes(detection.Lines[i], page, width, height, threshold, inkIsBright, boxes, darkInk);
                foreach (LineBox b in boxes)
                {
                    strips.Add(new StripLine(b, leaf, Local: local));
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

    // How many line heights short of the detector's extent the ink may stop on a photo before the detector's end is used.
    private const double EndGapHeights = 0.8;

    // The ink inside the detected box, as the classical finder's boxes are (the recognizer's crop rules are written for ink-tight
    // lines); the detected box itself when the page has no contrast or no ink falls inside it. Returns the line's own binarization when
    // the page-wide one does not fit the line (see LocalInk), which the line's word boxes must then use too; null otherwise.
    internal static LocalInk? InkBoxes(DetectedLine line, ReadOnlySpan<byte> page, int width, int height, int threshold, bool inkIsBright, List<LineBox> result, bool? pageIsDarkInk = null)
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

                // The main band, and any other band that holds at least 40% as much ink (two lines the detector joined); a smaller
                // one is the edge of a neighbouring line.
                foreach ((int start, int end, long mass) in bands)
                {
                    if (mass * 5 < bestMass * 2)
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
