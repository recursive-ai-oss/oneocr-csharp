using System.Globalization;
using OneOcr.Pdf;
using OnePdf;
using OnePdf.Text;

namespace OneOcr.Tests;

/// <summary>The invisible text layer as OnePDF reads it back: text, order and word boxes.</summary>
public class SearchablePdfWriterTests
{
    internal sealed record Placed(string Text, double[] Quad);

    internal static OcrWord Word(string text, double[] q) => new()
    {
        Text = text,
        Box = new OcrBox(new[] { q[0], q[2], q[4], q[6] }.Min(), new[] { q[1], q[3], q[5], q[7] }.Min(), new[] { q[0], q[2], q[4], q[6] }.Max(), new[] { q[1], q[3], q[5], q[7] }.Max()),
        Confidence = 1,
        RawConfidence = 1,
        Source = OcrSource.Ocr,
        Chars = [],
        Quad = new OcrQuad(new OcrPoint(q[0], q[1]), new OcrPoint(q[2], q[3]), new OcrPoint(q[4], q[5]), new OcrPoint(q[6], q[7])),
    };

    /// <summary>A page of lines of words laid out on a grid (page units: pixels at 144 dpi), then turned by <paramref name="degrees"/> about the page centre.</summary>
    internal static OcrPage Page(string[] lines, double degrees, double width = 1200, double height = 1600, bool quadsForUpright = false)
    {
        double rad = degrees * Math.PI / 180, cos = Math.Cos(rad), sin = Math.Sin(rad), cx = width / 2, cy = height / 2;
        OcrPoint Turn(double x, double y) => new(cx + ((x - cx) * cos) - ((y - cy) * sin), cy + ((x - cx) * sin) + ((y - cy) * cos));
        var blocks = new List<OcrBlock>();
        var ocrLines = new List<OcrLine>();
        double top = 300;
        foreach (string text in lines)
        {
            double x = 250;
            var words = new List<OcrWord>();
            foreach (string t in text.Split(' '))
            {
                double w = (t.Length * 22) + 6, h = 34;
                OcrPoint p0 = Turn(x, top), p1 = Turn(x + w, top), p2 = Turn(x + w, top + h), p3 = Turn(x, top + h);
                OcrWord word = Word(t, [p0.X, p0.Y, p1.X, p1.Y, p2.X, p2.Y, p3.X, p3.Y]);
                if (degrees == 0 && !quadsForUpright)
                {
                    word = word with { Quad = null };
                }

                words.Add(word);
                x += w + 16;
            }

            OcrQuad? lineQuad = null;
            if (degrees != 0 || quadsForUpright)
            {
                OcrPoint l0 = Turn(250, top), l1 = Turn(x - 16, top), l2 = Turn(x - 16, top + 34), l3 = Turn(250, top + 34);
                lineQuad = new OcrQuad(l0, l1, l2, l3);
            }

            ocrLines.Add(new OcrLine { Text = text, Words = words, Cells = [0], Box = words.Select(static w => w.Box).Aggregate(OcrBox.Union), Confidence = 1, Source = OcrSource.Ocr, Quad = lineQuad });
            top += 70;
        }

        blocks.Add(new OcrBlock { Role = OcrBlockRole.Body, IsTable = false, Lines = ocrLines, Box = ocrLines.Select(static l => l.Box).Aggregate(OcrBox.Union), Confidence = 1 });
        return new OcrPage { Index = 0, Width = width, Height = height, UnitsPerInch = 144, Blocks = blocks };
    }

    internal static byte[] Write(OcrPage page)
    {
        using var doc = PdfDocument.CreateEmpty();
        double s = 72.0 / page.UnitsPerInch;
        PdfPage p = doc.InsertBlankPage(0, page.Width * s, page.Height * s);
        var writer = new PdfTextLayerWriter(doc);
        using (PdfPageContentBuilder b = p.BeginContent())
        {
            writer.Draw(b, page);
            b.Commit();
        }

        using var ms = new MemoryStream();
        doc.Save(ms, new PdfSaveOptions { DocumentId = new byte[16] });
        return ms.ToArray();
    }

    internal static List<PdfTextWord> Extract(byte[] pdf, int page = 0, bool collapseDuplicates = true)
    {
        using var doc = PdfDocument.Open(pdf);
        return [.. doc.Pages[page].ExtractText(new PdfTextOptions { CollapseDuplicates = collapseDuplicates }).Words];
    }

    /// <summary>Every word with its line, in reading order.</summary>
    internal static List<(OcrLine Line, OcrWord Word)> WordsWithLines(OcrPage page) =>
        [.. page.Blocks.SelectMany(static b => b.Lines).SelectMany(static l => l.Words.Select(w => (l, w)))];

    /// <summary>The axis-aligned box, in user space (y up), around the frame a word is drawn in.</summary>
    internal static PdfRectangle Expected(OcrPage page, OcrLine line, OcrWord w, OcrMatrix? toPdf = null)
    {
        OcrMatrix m = toPdf ?? page.ToPdf ?? PdfTextLayerWriter.ImageToPdf(page);
        var (bl, br, tr, tl) = PdfTextLayerWriter.Frame(line, w);
        OcrPoint[] ps = [bl, br, tr, tl];
        double[] xs = ps.Select(p => (m.A * p.X) + (m.C * p.Y) + m.E).ToArray();
        double[] ys = ps.Select(p => (m.B * p.X) + (m.D * p.Y) + m.F).ToArray();
        return new PdfRectangle(xs.Min(), ys.Min(), xs.Max(), ys.Max());
    }

    private static readonly string[] Lines =
    [
        "Quarterly harbor report 2024",
        "\u0394\u03B5\u03BB\u03C4\u03AF\u03BF \u0431\u0443\u0445\u0433\u0430\u043B\u0442\u0435\u0440\u0438\u044F caf\u00E9 na\u00EFve",
        "Total: $1,234.56 (EUR) 99%",
    ];

    /// <summary>The largest distance between a word's expected box edge and the one OnePDF reports, in points; also checks text and order.</summary>
    internal static double MaxEdgeError(OcrPage page, byte[] pdf) => MaxEdgeErrorOn(page, pdf, 0, null);

    /// <summary>Like <see cref="MaxEdgeError"/> for page <paramref name="index"/> of a file, with the map from page units to user space (null: the page's own).</summary>
    internal static double MaxEdgeErrorOn(OcrPage page, byte[] pdf, int index, OcrMatrix? toPdf = null)
    {
        List<PdfTextWord> got = Extract(pdf, index);
        List<(OcrLine Line, OcrWord Word)> want = WordsWithLines(page);
        Assert.Equal(want.Select(static w => w.Word.Text), got.Select(static w => w.Text));
        double worst = 0;
        for (int i = 0; i < got.Count; i++)
        {
            PdfRectangle e = Expected(page, want[i].Line, want[i].Word, toPdf), g = got[i].Bounds;
            worst = Math.Max(worst, Math.Max(Math.Max(Math.Abs(e.Left - g.Left), Math.Abs(e.Right - g.Right)), Math.Max(Math.Abs(e.Bottom - g.Bottom), Math.Abs(e.Top - g.Top))));
        }

        return worst;
    }

    [Fact]
    public void UprightWordsComeBackExactly()
    {
        OcrPage page = Page(Lines, 0);
        double worst = MaxEdgeError(page, Write(page));
        Console.WriteLine($"upright max edge error {worst:0.0000} pt");
        Assert.True(worst <= 0.05, worst.ToString(CultureInfo.InvariantCulture));
    }

    /// <summary>
    /// Reads the glyphs OnePDF finds in content order and compares each word's glyph corners with the word's quad: the largest
    /// difference along the baseline and across it, in points. Also checks the text, glyph by glyph.
    /// </summary>
    internal static (double Along, double Across) GlyphError(OcrPage page, byte[] pdf)
    {
        using var doc = PdfDocument.Open(pdf);
        PdfPageText text = doc.Pages[0].ExtractText(new PdfTextOptions { RetainGlyphs = true });
        IReadOnlyList<PdfTextGlyph> glyphs = text.Glyphs;
        int at = 0;
        double along = 0, across = 0;
        OcrMatrix m = page.ToPdf ?? PdfTextLayerWriter.ImageToPdf(page);
        foreach ((OcrLine line, OcrWord w) in WordsWithLines(page))
        {
            (double X, double Y) U(OcrPoint p) => ((m.A * p.X) + (m.C * p.Y) + m.E, (m.B * p.X) + (m.D * p.Y) + m.F);
            var (fbl, fbr, _, ftl) = PdfTextLayerWriter.Frame(line, w);
            var (x0, y0) = U(ftl);
            var (x2, y2) = U(fbr);
            var (x3, y3) = U(fbl);
            double bl = Math.Sqrt(((x2 - x3) * (x2 - x3)) + ((y2 - y3) * (y2 - y3))), ul = Math.Sqrt(((x0 - x3) * (x0 - x3)) + ((y0 - y3) * (y0 - y3)));
            double bx = (x2 - x3) / bl, by = (y2 - y3) / bl, ux = (x0 - x3) / ul, uy = (y0 - y3) / ul;
            double aMin = 0, aMax = bl, cMin = 0, cMax = ul;
            double gaMin = double.MaxValue, gaMax = double.MinValue, gcMin = double.MaxValue, gcMax = double.MinValue;
            string joined = "";
            foreach (char _ in w.Text)
            {
                PdfTextGlyph g = glyphs[at++];
                joined += g.Text;
                PdfQuad gq = g.Quad;
                foreach ((double X, double Y) c in new[] { (gq.X0, gq.Y0), (gq.X1, gq.Y1), (gq.X2, gq.Y2), (gq.X3, gq.Y3) })
                {
                    double pa = ((c.X - x3) * bx) + ((c.Y - y3) * by), pc = ((c.X - x3) * ux) + ((c.Y - y3) * uy);
                    gaMin = Math.Min(gaMin, pa);
                    gaMax = Math.Max(gaMax, pa);
                    gcMin = Math.Min(gcMin, pc);
                    gcMax = Math.Max(gcMax, pc);
                }
            }

            Assert.Equal(w.Text, joined);
            along = Math.Max(along, Math.Max(Math.Abs(gaMin - aMin), Math.Abs(gaMax - aMax)));
            across = Math.Max(across, Math.Max(Math.Abs(gcMin - cMin), Math.Abs(gcMax - cMax)));
        }

        Assert.Equal(glyphs.Count, at);
        return (along, across);
    }

    [Theory]
    [InlineData(90)]
    [InlineData(180)]
    [InlineData(270)]
    public void QuarterTurnedWordsComeBackWithTheirBoxes(int degrees)
    {
        OcrPage page = Page(Lines, degrees);
        double worst = MaxEdgeError(page, Write(page));
        Assert.True(worst <= 0.05, worst.ToString(CultureInfo.InvariantCulture));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(90)]
    [InlineData(180)]
    [InlineData(270)]
    [InlineData(2)]
    [InlineData(-2)]
    [InlineData(5)]
    [InlineData(-5)]
    [InlineData(10)]
    [InlineData(-10)]
    [InlineData(37.5)]
    public void GlyphsLieOnTheWordQuad(double degrees)
    {
        OcrPage page = Page(Lines, degrees);
        (double along, double across) = GlyphError(page, Write(page));
        Console.WriteLine($"turn {degrees}: glyph corners vs quad: along {along:0.0000} pt, across {across:0.0000} pt");
        Assert.True(along <= 0.05 && across <= 0.05, $"{along} {across}");
    }
}
