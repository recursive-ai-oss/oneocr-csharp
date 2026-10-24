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
    // Region classes whose lines read together as one column or paragraph run (one leaf); lines of other classes, and lines outside
    // every region, are leaves of their own, so cells of a table row on one baseline still join into a row.
    private static readonly HashSet<string> RunningText = new(StringComparer.Ordinal) { "text", "title", "list", "header_footer" };

    /// <summary>
    /// Detects on the page, turns it upright when the orientation head is sure it is rotated (detecting again on the turned page),
    /// and returns the lines as ink-tight boxes with the regions that contain them.
    /// </summary>
    public static LocatedPage Locate(
        IPageDetector detector, Gray8[] gray, int width, int height, BufferPool<Gray8> pixels, ResizerPool resizers, int degree, CancellationToken cancellationToken)
    {
        var notes = new List<string>();
        int count = checked(width * height);
        PageDetection detection = detector.Detect(gray.AsSpan(0, count), width, height, pixels, resizers, degree, cancellationToken);
        Gray8[] work = gray;
        bool owns = false;
        int turns = PageOrientation.QuarterTurnsClockwise(detection);
        if (turns != 0)
        {
            work = PageOrientation.Rotate(gray.AsSpan(0, count), width, height, turns, pixels, out int rw, out int rh);
            owns = true;
            width = rw;
            height = rh;
            notes.Add($"The page was turned {turns * 90} degrees clockwise to read it upright (orientation confidence {detection.OrientationConfidence:0.00}); positions refer to the turned page.");
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
            (int threshold, bool inkIsBright) = LineFinder.Binarize(work, count, degree);
            var strips = new List<StripLine>(detection.Lines.Count);
            ReadOnlySpan<byte> page = Kernels.PixelSpans.AsBytes(work.AsSpan(0, count));
            for (int i = 0; i < detection.Lines.Count; i++)
            {
                LineBox? box = InkBox(detection.Lines[i], page, width, height, threshold, inkIsBright);
                if (box is not { } b)
                {
                    continue;
                }

                strips.Add(new StripLine(b, LeafOf(detection, i)));
            }

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

    // The ink inside the detected box, as the classical finder's boxes are (the recognizer's crop rules are written for ink-tight
    // lines); the detected box itself when the page has no contrast or no ink falls inside it.
    private static LineBox? InkBox(DetectedLine line, ReadOnlySpan<byte> page, int width, int height, int threshold, bool inkIsBright)
    {
        int x0 = Math.Clamp((int)Math.Floor(line.Left), 0, width), x1 = Math.Clamp((int)Math.Ceiling(line.Right), 0, width);
        int y0 = Math.Clamp((int)Math.Floor(line.Top), 0, height), y1 = Math.Clamp((int)Math.Ceiling(line.Bottom), 0, height);
        if (x1 <= x0 || y1 <= y0)
        {
            return null;
        }

        if (threshold >= 0)
        {
            int minX = int.MaxValue, maxX = -1, minY = int.MaxValue, maxY = -1;
            for (int y = y0; y < y1; y++)
            {
                ReadOnlySpan<byte> row = page.Slice((y * width) + x0, x1 - x0);
                for (int x = 0; x < row.Length; x++)
                {
                    if (inkIsBright ? row[x] > threshold : row[x] <= threshold)
                    {
                        minX = Math.Min(minX, x0 + x);
                        maxX = Math.Max(maxX, x0 + x);
                        minY = Math.Min(minY, y);
                        maxY = Math.Max(maxY, y);
                    }
                }
            }

            if (maxX >= 0)
            {
                return new LineBox(minX, minY, maxX - minX + 1, maxY - minY + 1);
            }
        }

        return new LineBox(x0, y0, x1 - x0, y1 - y0);
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
