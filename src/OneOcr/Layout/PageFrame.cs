namespace OneOcr.Layout;

/// <summary>
/// The turns applied to a page before reading it, and the way back: a quarter-turn count, then a small deskew turn on a larger
/// canvas. Text is read in the turned frame; <see cref="MapPage"/> carries every box back to the input image's coordinates and
/// gives rotated elements their quad.
/// </summary>
internal sealed class PageFrame
{
    public PageFrame(int width, int height, int quarterTurns, int slope)
    {
        Width = width;
        Height = height;
        QuarterTurns = quarterTurns & 3;
        Slope = slope;
        bool sideways = (QuarterTurns & 1) != 0;
        TurnedWidth = sideways ? height : width;
        TurnedHeight = sideways ? width : height;
        double t = slope / (double)SkewEstimator.Scale;
        // Exact to one rounding each: only +, *, / and square root, which every platform computes the same way.
        Cos = 1.0 / Math.Sqrt(1.0 + (t * t));
        Sin = t * Cos;
        (DeskewedWidth, DeskewedHeight) = slope == 0 ? (TurnedWidth, TurnedHeight) : PageRotation.CanvasFor(TurnedWidth, TurnedHeight, Sin, Cos);
    }

    /// <summary>Size of the input image.</summary>
    public int Width { get; }

    public int Height { get; }

    /// <summary>Clockwise quarter turns applied to the input (0 to 3).</summary>
    public int QuarterTurns { get; }

    /// <summary>The deskew slope times 1024: the text's lines descend by this many units per 1024 across.</summary>
    public int Slope { get; }

    public int TurnedWidth { get; }

    public int TurnedHeight { get; }

    public int DeskewedWidth { get; }

    public int DeskewedHeight { get; }

    public double Sin { get; }

    public double Cos { get; }

    public bool IsIdentity => QuarterTurns == 0 && Slope == 0;

    /// <summary>How far the page content is turned clockwise from upright: 0, 90, 180 or 270.</summary>
    public int Orientation => (360 - (90 * QuarterTurns)) % 360;

    /// <summary>The tilt of the text lines in degrees, positive when they descend to the right.</summary>
    public double SkewDegrees => ArcTan(Slope / (double)SkewEstimator.Scale) * (180.0 / Math.PI);

    // Series x - x^3/3 + x^5/5 - ... in a fixed order; |x| is at most tan 15.3 degrees, so 16 terms are far below double rounding.
    private static double ArcTan(double x)
    {
        double x2 = x * x, term = x, sum = 0;
        for (int k = 0; k < 16; k++)
        {
            sum += term / ((2 * k) + 1);
            term = -term * x2;
        }

        return sum;
    }

    /// <summary>The point of the input image that the point (u, v) of the deskewed frame came from.</summary>
    public OcrPoint ToInput(double u, double v)
    {
        double qx = u, qy = v;
        if (Slope != 0)
        {
            double dx = u - (DeskewedWidth / 2.0), dy = v - (DeskewedHeight / 2.0);
            qx = ((dx * Cos) - (dy * Sin)) + (TurnedWidth / 2.0);
            qy = ((dx * Sin) + (dy * Cos)) + (TurnedHeight / 2.0);
        }

        return QuarterTurns switch
        {
            1 => new OcrPoint(qy, Height - qx),
            2 => new OcrPoint(Width - qx, Height - qy),
            3 => new OcrPoint(Width - qy, qx),
            _ => new OcrPoint(qx, qy),
        };
    }

    private (OcrBox Box, OcrQuad Quad) Map(OcrBox b)
    {
        OcrPoint p0 = ToInput(b.Left, b.Top), p1 = ToInput(b.Right, b.Top), p2 = ToInput(b.Right, b.Bottom), p3 = ToInput(b.Left, b.Bottom);
        var box = new OcrBox(
            Math.Min(Math.Min(p0.X, p1.X), Math.Min(p2.X, p3.X)), Math.Min(Math.Min(p0.Y, p1.Y), Math.Min(p2.Y, p3.Y)),
            Math.Max(Math.Max(p0.X, p1.X), Math.Max(p2.X, p3.X)), Math.Max(Math.Max(p0.Y, p1.Y), Math.Max(p2.Y, p3.Y)));
        return (box, new OcrQuad(p0, p1, p2, p3));
    }

    /// <summary>
    /// Carries a page read in the deskewed frame back to the input image: sizes become the input's, every box becomes the box
    /// around its mapped corners, and every element that carries a quad gets the mapped corners.
    /// </summary>
    public OcrPage MapPage(OcrPage page, IReadOnlyList<string> notes)
    {
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
                        (OcrBox box, OcrQuad quad) = Map(c.Box);
                        chars.Add(c with { Box = box, Quad = quad });
                    }

                    (OcrBox wb, OcrQuad wq) = Map(w.Box);
                    words.Add(w with { Box = wb, Quad = wq, Chars = chars });
                }

                (OcrBox lb, OcrQuad lq) = Map(l.Box);
                lines.Add(l with { Words = words, Box = lb, Quad = lq });
            }

            (OcrBox bb, OcrQuad bq) = Map(b.Box);
            blocks.Add(b with { Lines = lines, Box = bb, Quad = bq });
        }

        var tables = new List<OcrTable>(page.Tables.Count);
        foreach (OcrTable t in page.Tables)
        {
            var cells = new List<OcrTableCell>(t.Cells.Count);
            foreach (OcrTableCell c in t.Cells)
            {
                cells.Add(c with { Box = Map(c.Box).Box });
            }

            (OcrBox tb, OcrQuad tq) = Map(t.Box);
            tables.Add(t with { Cells = cells, Box = tb, Quad = tq });
        }

        var regions = new List<OcrRegion>(page.Regions.Count);
        foreach (OcrRegion r in page.Regions)
        {
            (OcrBox rb, OcrQuad rq) = Map(r.Box);
            regions.Add(r with { Box = rb, Quad = rq });
        }

        var keyValues = new List<OcrKeyValue>(page.KeyValues.Count);
        foreach (OcrKeyValue k in page.KeyValues)
        {
            keyValues.Add(k with { Box = Map(k.Box).Box });
        }

        var diagnostics = new List<string>(page.Diagnostics);
        diagnostics.AddRange(notes);
        return new OcrPage(
            page.Index, Width, Height, page.UnitsPerInch, blocks, regions, tables, keyValues, diagnostics, page.ToPdf, Orientation, SkewDegrees);
    }
}
