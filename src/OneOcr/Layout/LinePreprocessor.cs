using OneImage;

namespace OneOcr.Layout;

/// <summary>
/// Turns a found line into the model's input: a grayscale crop around the ink with margins, scaled to the model's input height.
/// The recognizer is trained on crops that hold a line box about one em tall, plus 0.05 to 0.35 em above and below and 0.1 to 0.9 em
/// left and right, scaled with <c>Image.Resize(height: h)</c> and default filtering. Ink (ascender top to descender bottom) is
/// about 0.9 em tall, so this class takes margins of a quarter of the ink height above and below and half of it left and right,
/// which puts a line on the same scale and with the same surroundings. All of these conventions are in this one class.
/// </summary>
internal static class LinePreprocessor
{
    // Lines shorter than this fraction of the page's typical line (no descenders, only digits) are treated as that tall,
    // so their scale matches lines of the same font that do have descenders.
    private const double MinHeightFraction = 0.85;
    private const double VerticalMargin = 0.25;
    private const double HorizontalMargin = 0.5;

    /// <summary>The page's typical line height: the median of the found lines' heights (0 when there are none).</summary>
    public static int TypicalHeight(IReadOnlyList<LineBox> lines)
    {
        if (lines.Count == 0)
        {
            return 0;
        }

        var heights = new int[lines.Count];
        for (int i = 0; i < heights.Length; i++)
        {
            heights[i] = lines[i].Height;
        }

        Array.Sort(heights);
        return heights[heights.Length / 2];
    }

    /// <summary>The crop rectangle for a line, clamped to the page, always at least one pixel.</summary>
    public static LineBox CropBox(LineBox line, int typicalHeight, int pageWidth, int pageHeight)
    {
        double effective = Math.Max(line.Height, MinHeightFraction * typicalHeight);
        double centerY = line.Y + (line.Height / 2.0);
        double halfHeight = effective * (0.5 + VerticalMargin);
        double margin = effective * HorizontalMargin;
        int x0 = Math.Max(0, (int)Math.Floor(line.X - margin));
        int x1 = Math.Min(pageWidth, (int)Math.Ceiling(line.Right + margin));
        int y0 = Math.Max(0, (int)Math.Floor(centerY - halfHeight));
        int y1 = Math.Min(pageHeight, (int)Math.Ceiling(centerY + halfHeight));
        return new LineBox(x0, y0, Math.Max(1, x1 - x0), Math.Max(1, y1 - y0));
    }

    /// <summary>The crop rectangle for a line of a tilted strip, with the same margins as <see cref="CropBox"/> but in the line's own frame, where nothing is clamped to the page.</summary>
    public static LineBox CropBoxInFrame(LineBox line, int typicalHeight)
    {
        double effective = Math.Max(line.Height, MinHeightFraction * typicalHeight);
        double centerY = line.Y + (line.Height / 2.0);
        double halfHeight = effective * (0.5 + VerticalMargin);
        double margin = effective * HorizontalMargin;
        int x0 = (int)Math.Floor(line.X - margin);
        int x1 = (int)Math.Ceiling(line.Right + margin);
        int y0 = (int)Math.Floor(centerY - halfHeight);
        int y1 = (int)Math.Ceiling(centerY + halfHeight);
        return new LineBox(x0, y0, Math.Max(1, x1 - x0), Math.Max(1, y1 - y0));
    }

    /// <summary>
    /// Crops <paramref name="box"/> out of an 8-bit grayscale page (row-major, <paramref name="pageWidth"/> pixels per row) and
    /// scales it to <paramref name="height"/> rows. Returns a buffer from <paramref name="pool"/> whose first
    /// <c>height * width</c> pixels are the scaled line; the caller gives it back to the pool.
    /// Page columns from <paramref name="blankFrom"/> up to but not including <paramref name="blankTo"/> are painted with the crop's
    /// background (its median gray value) before scaling, which hides the other side of a line that was cut at a dot leader.
    /// </summary>
    public static Gray8[] Prepare(ReadOnlySpan<byte> page, int pageWidth, LineBox box, int height, BufferPool<Gray8> pool, ResizerPool resizers, out int width, int blankFrom = int.MaxValue, int blankTo = int.MinValue)
    {
        int count = checked(box.Width * box.Height);
        (int scaledWidth, int scaledHeight) = PixelResizer.GetSize(box.Width, box.Height, null, height);
        Gray8[] crop = pool.Rent(count);
        Gray8[]? result = null;
        try
        {
            for (int y = 0; y < box.Height; y++)
            {
                ReadOnlySpan<byte> row = page.Slice(((box.Y + y) * pageWidth) + box.X, box.Width);
                Span<Gray8> to = crop.AsSpan(y * box.Width, box.Width);
                for (int x = 0; x < row.Length; x++)
                {
                    to[x] = new Gray8(row[x]);
                }
            }

            if (blankFrom < blankTo && blankFrom < box.Right && blankTo > box.X)
            {
                Blank(crop.AsSpan(0, count), box, Math.Max(blankFrom, box.X) - box.X, Math.Min(blankTo, box.Right) - box.X);
            }

            result = pool.Rent(checked(scaledWidth * scaledHeight));
            PixelResizer resizer = resizers.Rent();
            try
            {
                resizer.Resize<Gray8>(crop.AsSpan(0, count), box.Width, box.Height, result.AsSpan(0, scaledWidth * scaledHeight), scaledWidth, scaledHeight);
            }
            finally
            {
                resizers.Return(resizer);
            }
        }
        catch
        {
            if (result is not null)
            {
                pool.Return(result);
            }

            throw;
        }
        finally
        {
            pool.Return(crop);
        }

        width = scaledWidth;
        return result;
    }

    // Paints crop columns [from, to) with the median gray value of the whole crop.
    private static void Blank(Span<Gray8> crop, LineBox box, int from, int to)
    {
        Span<int> histogram = stackalloc int[256];
        foreach (Gray8 g in crop)
        {
            histogram[g.V]++;
        }

        int seen = 0, median = 0;
        for (; median < 255 && seen + histogram[median] <= crop.Length / 2; median++)
        {
            seen += histogram[median];
        }

        for (int y = 0; y < box.Height; y++)
        {
            crop.Slice((y * box.Width) + from, to - from).Fill(new Gray8((byte)median));
        }
    }
}
