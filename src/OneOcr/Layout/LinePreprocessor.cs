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

    /// <summary>
    /// Crops <paramref name="box"/> out of a grayscale page image and scales it to <paramref name="height"/> rows; returns the
    /// 8-bit pixels and the scaled width.
    /// </summary>
    public static byte[] Prepare(Image page, LineBox box, int height, out int width)
    {
        Image scaled = page.Crop(box.X, box.Y, box.Width, box.Height).Resize(height: height);
        width = scaled.Width;
        Gray8[] px = scaled.ToArray<Gray8>();
        var bytes = new byte[px.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            bytes[i] = px[i].V;
        }

        return bytes;
    }
}
