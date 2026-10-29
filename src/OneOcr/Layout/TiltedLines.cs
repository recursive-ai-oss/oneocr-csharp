using OneImage;

namespace OneOcr.Layout;

/// <summary>
/// A line read along its own axis. <c>Ink</c> is the line's ink rectangle in the line frame: the origin is (<c>CenterX</c>,
/// <c>CenterY</c>) in page pixels, <c>u</c> runs along the text and <c>v</c> across it, pointing down, so the page point of
/// (u, v) is <c>(CenterX + u*Cos - v*Sin, CenterY + u*Sin + v*Cos)</c>. Pixel (i, j) of the straightened line covers
/// [i, i+1) x [j, j+1) of the line frame.
/// </summary>
internal sealed record TiltedLine(double CenterX, double CenterY, double Cos, double Sin, LineBox Ink)
{
    /// <summary>The page point of a point of the line frame. Only +, - and *, so every platform gets the same bits.</summary>
    public (double X, double Y) ToPage(double u, double v) => (CenterX + (u * Cos) - (v * Sin), CenterY + (u * Sin) + (v * Cos));
}

/// <summary>
/// Lines whose own tilt differs from the page's. After the page is turned upright and deskewed once, lines of a photographed page still
/// lean by up to a couple of degrees (perspective), more than a line crop of one text height can absorb. The detector's text map gives
/// every line component a principal axis; when that axis is more than <see cref="ToleranceSin"/> from horizontal, the line is cut along
/// its axis: the crop is sampled from the page in the line's own frame (bilinear, edge pixels repeated) and scaled like any other crop,
/// and the words' boxes are rectangles in that frame, reported with their corners as quads. Lines within the tolerance, and every line of
/// a page that is already straight, take the axis-aligned path unchanged (docs/spec/detector-v1.md, "Line axis").
/// </summary>
internal static class TiltedLines
{
    /// <summary>A line is cut along its axis when the sine of its angle from horizontal is at least this (about 0.3 degrees).</summary>
    internal static double ToleranceSin = 0.0052;

    /// <summary>Lines leaning by more than this sine (about 15 degrees) are not straightened: the page-level turn is for those.</summary>
    internal static double MaxSin = 0.26;

    /// <summary>Components with fewer text-map cells have no angle worth trusting.</summary>
    internal static int MinCells = 20;

    /// <summary>The ratio of the variance along the axis to the variance across it that marks a single line (about 10:1 in length).</summary>
    internal static double MinElongation = 100;

    /// <summary>How many times thickness over length an axis angle (as a sine) must exceed to be believed.</summary>
    internal static double NoiseFactor = 0.15;

    /// <summary>Most pixels of the straightened strip the ink band is looked for in; larger strips take the axis-aligned path.</summary>
    private const long MaxStripPixels = 1L << 23;

    /// <summary>Whether a detected line is cut along its axis.</summary>
    public static bool IsTilted(Detection.LineAxis a)
    {
        double s = Math.Abs(a.Sin);
        // A short, thick component has a ragged outline and an unreliable axis: its angle must clear a noise floor that grows with
        // thickness over length, on top of the fixed tolerance.
        double noise = NoiseFactor * (a.V1 - a.V0) / (a.U1 - a.U0);
        return a.Cells >= MinCells && a.Elongation >= MinElongation && s >= Math.Max(ToleranceSin, noise) && s <= MaxSin;
    }

    /// <summary>
    /// For a detected line leaning more than the tolerance, finds its ink band(s) in the straightened strip and appends one tilted strip
    /// per band to <paramref name="result"/>. Returns false, appending nothing, when the line is not tilted (or its strip is not usable);
    /// the caller then takes the axis-aligned path.
    /// </summary>
    public static bool TryInkBoxes(
        Detection.DetectedLine line, ReadOnlySpan<byte> page, int width, int height, int threshold, bool inkIsBright, List<StripLine> result, int leaf, bool? pageIsDarkInk = null)
    {
        if (line.Axis is not { } a || threshold < 0 || !IsTilted(a))
        {
            return false;
        }

        int u0 = (int)Math.Floor(a.U0), u1 = (int)Math.Ceiling(a.U1), v0 = (int)Math.Floor(a.V0), v1 = (int)Math.Ceiling(a.V1);
        long w = (long)u1 - u0, h = (long)v1 - v0;
        if (w < 1 || h < 1 || w * h > MaxStripPixels)
        {
            return false;
        }

        var frame = new TiltedLine(a.CenterX, a.CenterY, a.Cos, a.Sin, new LineBox(0, 0, 0, 0));
        byte[] strip = System.Buffers.ArrayPool<byte>.Shared.Rent((int)(w * h));
        try
        {
            int sw = (int)w, sh = (int)h;
            for (int j = 0; j < sh; j++)
            {
                for (int i = 0; i < sw; i++)
                {
                    (double px, double py) = frame.ToPage(u0 + i + 0.5, v0 + j + 0.5);
                    strip[(j * sw) + i] = Sample(page, width, height, px, py);
                }
            }

            var boxes = new List<LineBox>(1);
            LocalInk? local = DetectorLines.InkBoxes(new Detection.DetectedLine(0, 0, sw, sh, line.Score), strip.AsSpan(0, sw * sh), sw, sh, threshold, inkIsBright, boxes, pageIsDarkInk);
            foreach (LineBox b in boxes)
            {
                var ink = new LineBox(u0 + b.X, v0 + b.Y, b.Width, b.Height);
                (double cx, double cy) = frame.ToPage(ink.X + (ink.Width / 2.0), ink.Y + (ink.Height / 2.0));
                // The strip's box for reading order: the straightened ink rectangle, unrotated, at the tilted rectangle's centre.
                var box = new LineBox(
                    (int)Math.Round(cx - (ink.Width / 2.0), MidpointRounding.ToEven), (int)Math.Round(cy - (ink.Height / 2.0), MidpointRounding.ToEven),
                    ink.Width, ink.Height);
                result.Add(new StripLine(box, leaf, frame with { Ink = ink }, local));
            }

            return boxes.Count > 0;
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(strip);
        }
    }

    /// <summary>
    /// The gray value at a page point (continuous coordinates, pixel (x, y) covering [x, x+1)): bilinear between the four pixel centres
    /// around it, edge pixels repeated beyond the page. Fixed operation order, so it is the same on every path.
    /// </summary>
    internal static byte Sample(ReadOnlySpan<byte> page, int width, int height, double px, double py)
    {
        double fx = px - 0.5, fy = py - 0.5;
        double flx = Math.Floor(fx), fly = Math.Floor(fy);
        double wx = fx - flx, wy = fy - fly;
        int x0 = (int)Math.Clamp(flx, -1, width), y0 = (int)Math.Clamp(fly, -1, height);
        int xa = Math.Clamp(x0, 0, width - 1), xb = Math.Clamp(x0 + 1, 0, width - 1);
        int ya = Math.Clamp(y0, 0, height - 1), yb = Math.Clamp(y0 + 1, 0, height - 1);
        double top = (page[(ya * width) + xa] * (1 - wx)) + (page[(ya * width) + xb] * wx);
        double bottom = (page[(yb * width) + xa] * (1 - wx)) + (page[(yb * width) + xb] * wx);
        double v = (top * (1 - wy)) + (bottom * wy);
        return (byte)Math.Clamp((int)(v + 0.5), 0, 255);
    }

    /// <summary>
    /// Samples the crop rectangle <paramref name="box"/> (line frame) of a tilted line from the page and scales it to
    /// <paramref name="height"/> rows, like <see cref="LinePreprocessor.Prepare"/>. The buffer comes from <paramref name="pool"/>.
    /// </summary>
    public static Gray8[] Prepare(
        ReadOnlySpan<byte> page, int pageWidth, int pageHeight, TiltedLine tilt, LineBox box, int height, BufferPool<Gray8> pool, ResizerPool resizers, out int width)
    {
        int count = checked(box.Width * box.Height);
        (int scaledWidth, int scaledHeight) = PixelResizer.GetSize(box.Width, box.Height, null, height);
        Gray8[] crop = pool.Rent(count);
        Gray8[]? result = null;
        try
        {
            for (int j = 0; j < box.Height; j++)
            {
                Span<Gray8> to = crop.AsSpan(j * box.Width, box.Width);
                for (int i = 0; i < to.Length; i++)
                {
                    (double px, double py) = tilt.ToPage(box.X + i + 0.5, box.Y + j + 0.5);
                    to[i] = new Gray8(Sample(page, pageWidth, pageHeight, px, py));
                }
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
}
