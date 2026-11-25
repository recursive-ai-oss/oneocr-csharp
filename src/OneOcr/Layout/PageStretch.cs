using OneImage;
using OneOcr.Kernels;

namespace OneOcr.Layout;

/// <summary>
/// Pixels that are not square. A capture can declare different resolutions across and down the page (a standard-mode fax is 200 by 100
/// dots per inch, so every pixel is twice as tall as it is wide). The detector and the recognizer were trained on square pixels, so such
/// a page is stretched along its sparser axis to square pixels before it is read, and everything the reader found is mapped back to the
/// pixels of the image the caller passed in.
/// </summary>
internal static class PageStretch
{
    /// <summary>Resolutions that differ by less than this fraction are read as they are (scanners declare 300 by 300 and sometimes 300 by 298).</summary>
    public const double Tolerance = 0.05;

    /// <summary>
    /// Passes of the [1 2 1] smoothing a stretched 1-bit page gets. The resampler's cubic overshoots at the hard 0/255 edges of a bilevel page
    /// and leaves a visible row pattern where rows were repeated by the fax; two passes were the point after which more passes stopped helping.
    /// </summary>
    public const int SmoothPasses = 2;

    /// <summary>A ratio beyond this is taken as a bad header.</summary>
    public const double MaxRatio = 4.0;

    /// <summary>
    /// Decides whether a page of <paramref name="width"/> by <paramref name="height"/> pixels declared at <paramref name="dpi"/> across and
    /// <paramref name="verticalDpi"/> down is stretched, and to what size.
    /// </summary>
    /// <returns>True when the page is stretched to <paramref name="stretchedWidth"/> by <paramref name="stretchedHeight"/>.</returns>
    public static bool Plan(double? dpi, double? verticalDpi, int width, int height, long maxPixels, List<OcrDiagnostic> notes, out int stretchedWidth, out int stretchedHeight)
    {
        stretchedWidth = width;
        stretchedHeight = height;
        if (dpi is not double x)
        {
            return false;
        }

        double y = verticalDpi ?? x;

        // A pixel that is k times taller than it is wide (k = x dpi / y dpi) needs k times more rows.
        double k = x / y;
        if (Math.Abs(k - 1) <= Tolerance)
        {
            return false;
        }

        if (!double.IsFinite(k) || k > MaxRatio || k < 1 / MaxRatio)
        {
            notes.Add(Diag.Warn(OcrDiagnosticCodes.PixelsNotStretched, "The declared resolutions across and down the page differ by more than 4 to 1, which is taken as a bad header, so the page was read as it is."));
            return false;
        }

        if (k > 1)
        {
            stretchedHeight = Math.Max(1, (int)Math.Round(height * k, MidpointRounding.ToEven));
        }
        else
        {
            stretchedWidth = Math.Max(1, (int)Math.Round(width / k, MidpointRounding.ToEven));
        }

        if ((long)stretchedWidth * stretchedHeight > maxPixels)
        {
            notes.Add(Diag.Warn(OcrDiagnosticCodes.PixelsNotStretched, "The page has non-square pixels but the stretched page would exceed the pixel limit, so it was read as it is."));
            stretchedWidth = width;
            stretchedHeight = height;
            return false;
        }

        notes.Add(Diag.Info(OcrDiagnosticCodes.PixelsStretched, string.Create(System.Globalization.CultureInfo.InvariantCulture, $"The page declares {x:0.##} by {y:0.##} pixels per inch; it was stretched from {width}x{height} to {stretchedWidth}x{stretchedHeight} square pixels before reading. Boxes are in the pixels of the image given.")));
        return true;
    }

    /// <summary>Stretches the page with the same resampling the detector uses for its own resizes.</summary>
    public static void Resize(ReadOnlySpan<Gray8> source, int width, int height, Span<Gray8> target, int stretchedWidth, int stretchedHeight, ResizerPool resizers, CancellationToken cancellationToken)
    {
        PixelResizer resizer = resizers.Rent();
        try
        {
            // Gamma-space resampling, as the detector's own resize: the models saw gamma-space resizes of pages.
            resizer.Resize<Gray8>(source, width, height, target, stretchedWidth, stretchedHeight, ResamplingFilter.Auto, linear: false, cancellationToken: cancellationToken);
        }
        finally
        {
            resizers.Return(resizer);
        }
    }

    /// <summary>True when nearly every pixel is pure black or white: a 1-bit capture (fax, bilevel scan).</summary>
    public static bool IsBilevel(ReadOnlySpan<byte> page)
    {
        if (page.Length == 0)
        {
            return false;
        }

        long extreme = 0;
        foreach (byte b in page)
        {
            if (b < 8 || b > 247)
            {
                extreme++;
            }
        }

        return extreme * 100 >= page.Length * 99L;
    }

    /// <summary>
    /// Runs the binomial [1 2 1] / 4 filter across the page and then down it, <see cref="SmoothPasses"/> times, in integers with the sum
    /// rounded half up; the outermost pixels repeat. The result is the same on every platform.
    /// </summary>
    public static void Smooth(Span<byte> page, int width, int height)
    {
        byte[] prev = new byte[width], cur = new byte[width];
        for (int pass = 0; pass < SmoothPasses; pass++)
        {
            for (int y = 0; y < height; y++)
            {
                Span<byte> r = page.Slice(y * width, width);
                r.CopyTo(cur);
                for (int x = 0; x < width; x++)
                {
                    int a = cur[x > 0 ? x - 1 : 0], c = cur[x + 1 < width ? x + 1 : width - 1];
                    r[x] = (byte)((a + (2 * cur[x]) + c + 2) >> 2);
                }
            }

            // Down the page: the row above is read from the copy of its unfiltered values, the row below is not yet changed.
            page[..width].CopyTo(prev);
            for (int y = 0; y < height; y++)
            {
                Span<byte> r = page.Slice(y * width, width);
                r.CopyTo(cur);
                ReadOnlySpan<byte> above = y > 0 ? prev : cur, below = y + 1 < height ? page.Slice((y + 1) * width, width) : cur;
                for (int x = 0; x < width; x++)
                {
                    r[x] = (byte)((above[x] + (2 * cur[x]) + below[x] + 2) >> 2);
                }

                (prev, cur) = (cur, prev);
            }
        }
    }

    /// <summary>
    /// Carries a page read in the stretched frame back to the image given: sizes become the image's, every box and quad is scaled, and the
    /// two resolutions are recorded.
    /// </summary>
    public static OcrPage MapBack(OcrPage page, int width, int height, int stretchedWidth, int stretchedHeight, double dpi, double verticalDpi, IReadOnlyList<OcrDiagnostic> notes)
    {
        double fx = width / (double)stretchedWidth, fy = height / (double)stretchedHeight;
        OcrBox B(OcrBox b) => new(b.Left * fx, b.Top * fy, b.Right * fx, b.Bottom * fy);
        OcrPoint P(OcrPoint p) => new(p.X * fx, p.Y * fy);
        OcrQuad? Q(OcrQuad? q) => q is { } v ? new OcrQuad(P(v.P0), P(v.P1), P(v.P2), P(v.P3)) : null;

        var blocks = new List<OcrBlock>(page.Blocks.Count);
        foreach (OcrBlock b in page.Blocks)
        {
            var lines = new List<OcrLine>(b.Lines.Count);
            foreach (OcrLine l in b.Lines)
            {
                var words = new List<OcrWord>(l.Words.Count);
                foreach (OcrWord w in l.Words)
                {
                    var chars = new List<OcrChar>(w.Chars.Count);
                    foreach (OcrChar c in w.Chars)
                    {
                        chars.Add(c with { Box = B(c.Box), Quad = Q(c.Quad) });
                    }

                    words.Add(w with { Box = B(w.Box), Quad = Q(w.Quad), Chars = chars });
                }

                lines.Add(l with { Words = words, Box = B(l.Box), Quad = Q(l.Quad) });
            }

            blocks.Add(b with { Lines = lines, Box = B(b.Box), Quad = Q(b.Quad) });
        }

        var tables = new List<OcrTable>(page.Tables.Count);
        foreach (OcrTable t in page.Tables)
        {
            var cells = new List<OcrTableCell>(t.Cells.Count);
            foreach (OcrTableCell c in t.Cells)
            {
                cells.Add(c with { Box = B(c.Box) });
            }

            tables.Add(t with { Cells = cells, Box = B(t.Box), Quad = Q(t.Quad) });
        }

        var regions = new List<OcrRegion>(page.Regions.Count);
        foreach (OcrRegion r in page.Regions)
        {
            regions.Add(r with { Box = B(r.Box), Quad = Q(r.Quad) });
        }

        var keyValues = new List<OcrKeyValue>(page.KeyValues.Count);
        foreach (OcrKeyValue k in page.KeyValues)
        {
            keyValues.Add(k with { Box = B(k.Box) });
        }

        var diagnostics = new List<OcrDiagnostic>(page.Diagnostics);
        diagnostics.AddRange(notes);
        return new OcrPage(
            page.Index, width, height, page.UnitsPerInch, blocks, regions, tables, keyValues, diagnostics, page.ToPdf, page.Orientation,
            MapSkew(page.SkewDegrees, fy / fx), verticalDpi == dpi ? 0 : verticalDpi);
    }

    // A line that descends t per unit across descends t * ratio per unit across in the other frame. tan and arctan are fixed-order series (the
    // skew is within 45 degrees; the reader only straightens up to about 15), so the result does not depend on the platform's math library.
    private static double MapSkew(double degrees, double ratio)
    {
        if (degrees == 0)
        {
            return 0;
        }

        double x = degrees * (Math.PI / 180.0), x2 = x * x;
        // tan x = x + x^3/3 + 2x^5/15 + 17x^7/315 + 62x^9/2835 + 1382x^11/155925 + 21844x^13/6081075 + 929569x^15/638512875
        double poly = 929569.0 / 638512875;
        poly = (poly * x2) + (21844.0 / 6081075);
        poly = (poly * x2) + (1382.0 / 155925);
        poly = (poly * x2) + (62.0 / 2835);
        poly = (poly * x2) + (17.0 / 315);
        poly = (poly * x2) + (2.0 / 15);
        poly = (poly * x2) + (1.0 / 3);
        double tan = x * (1 + (x2 * poly));
        // arctan t = 2 arctan(t / (1 + sqrt(1 + t^2))) keeps the series argument small even when the ratio is 4 (sqrt is exact in IEEE 754).
        double t = tan * ratio;
        return Math.Clamp(2 * PageFrame.ArcTan(t / (1 + Math.Sqrt(1 + (t * t)))) * (180.0 / Math.PI), -45, 45);
    }
}
