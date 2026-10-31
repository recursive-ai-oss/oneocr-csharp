using System.Globalization;
using System.Text;
using OneImage;
using OneOcr.Pdf;
using OnePdf;

namespace OneOcr.Tests;

public class PdfTests
{
    private static OcrEngine? s_random;

    /// <summary>A random-weight engine: enough for paths that never recognize anything (the text layer only).</summary>
    private static OcrEngine RandomEngine() =>
        s_random ??= OcrEngine.Load(new MemoryStream(TestModels.RandomContainer(
            new TestModels.Shape(32, [(8, 2, 2), (16, 2, 2), (16, 2, 1), (32, 2, 1), (32, 2, 1)], 24, 2, 41), 3)));

    /// <summary>The trained model named by ONEOCR_MODEL, or null.</summary>
    private static OcrEngine? RealEngine()
    {
        string? path = Environment.GetEnvironmentVariable("ONEOCR_MODEL");
        return path is null || !File.Exists(path) ? null : OcrEngine.Load(path);
    }

    private static readonly PdfOcrOptions NoRender = new() { ReadImages = false, CrossCheckLines = 0 };

    private static (double X, double Y) ToUser(OcrMatrix m, double x, double y) => ((m.A * x) + (m.C * y) + m.E, (m.B * x) + (m.D * y) + m.F);

    [Fact]
    public void TextOnlyPageKeepsTheTextLayerWithoutRendering()
    {
        (byte[] pdf, PageTruth[] truth, _) = PdfFixtures.Sample();
        IReadOnlyList<OcrPage> pages = PdfOcr.Read(pdf, RandomEngine(), NoRender);
        Assert.Equal(truth.Length, pages.Count);
        OcrPage p = pages[0];
        Assert.Equal(PdfFixtures.BodyA, p.ToText().Split('\n'));
        Assert.All(p.Words, w => Assert.Equal(OcrSource.TextLayer, w.Source));
        Assert.Equal(72, p.UnitsPerInch);
        Assert.NotNull(p.ToPdf);
        Assert.Equal(612, p.Width);
        Assert.Equal(792, p.Height);
        // Characters have exact boxes.
        Assert.All(p.Words, w => Assert.All(w.Chars, c => Assert.False(c.BoxIsApproximate)));
    }

    [Fact]
    public void A_pdf_with_no_page_tree_is_a_diagnostic_when_lenient_and_a_format_error_when_strict()
    {
        byte[] pdf = Encoding.Latin1.GetBytes("%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n");
        IReadOnlyList<OcrPage> pages = PdfOcr.Read(pdf, RandomEngine(), NoRender);
        OcrPage only = Assert.Single(pages);
        Assert.Contains(only.Diagnostics, d => d.Message.Contains("no readable pages", StringComparison.Ordinal));
        Assert.Empty(only.Words);
        Assert.Throws<OcrFormatException>(() => PdfOcr.Read(pdf, RandomEngine(), new PdfOcrOptions { ReadImages = false, CrossCheckLines = 0, Strict = true }));
    }

    [Fact]
    public void ExtensionMethodsMatchTheStaticEntryPoint()
    {
        (byte[] pdf, _, _) = PdfFixtures.Sample();
        using PdfDocument doc = PdfDocument.Open(pdf);
        IReadOnlyList<OcrPage> viaDoc = doc.Ocr(RandomEngine(), NoRender);
        IReadOnlyList<OcrPage> viaBytes = PdfOcr.Read(pdf, RandomEngine(), NoRender);
        Assert.Equal(OcrPageJson.ToJson(viaBytes), OcrPageJson.ToJson(viaDoc));
        OcrPage single = doc.Pages[0].Ocr(RandomEngine(), NoRender);
        Assert.Equal(viaBytes[0].Text, single.Text);
        Assert.Equal(viaBytes[0].ToText(), doc.Pages[0].ToText(RandomEngine(), NoRender));
        Assert.Equal(string.Join('\f', viaBytes.Select(x => x.ToText())), doc.ToText(RandomEngine(), NoRender));
    }

    [Fact]
    public void PageRangeSelectsPages()
    {
        (byte[] pdf, _, _) = PdfFixtures.Sample();
        IReadOnlyList<OcrPage> two = PdfOcr.Read(pdf, RandomEngine(), new PdfOcrOptions { ReadImages = false, CrossCheckLines = 0, FirstPage = 4, PageCount = 2 });
        Assert.Equal([4, 5], two.Select(p => p.Index));
        Assert.Empty(PdfOcr.Read(pdf, RandomEngine(), new PdfOcrOptions { FirstPage = 50 }));
    }

    [Theory]
    [InlineData("")]
    [InlineData("/CropBox [20 30 572 742]")]
    [InlineData("/Rotate 90")]
    [InlineData("/Rotate 180")]
    [InlineData("/Rotate 270")]
    [InlineData("/CropBox [50 60 500 700] /Rotate 90")]
    [InlineData("/CropBox [10 40 600 780] /Rotate 270")]
    [InlineData("/UserUnit 2")]
    public void WordBoxesMapBackToTheGlyphPositions(string pageEntries)
    {
        (byte[] pdf, PlacedLine[] lines) = PdfFixtures.GeometryPage(pageEntries);
        OcrPage page = PdfOcr.Read(pdf, RandomEngine(), NoRender)[0];
        Assert.NotNull(page.ToPdf);
        OcrMatrix m = page.ToPdf.Value;
        foreach (PlacedLine line in lines)
        {
            string first = line.Text.Split(' ')[0];
            OcrWord word = Assert.Single(page.Words, w => w.Text == first);
            // The word's box in user space: the glyph origin is on its left edge and the baseline is 0.2 em above its bottom edge
            // (OnePDF's glyph box runs from 0.9 em above the baseline to 0.2 em below it).
            (double ax, double ay) = ToUser(m, word.Box.Left, word.Box.Top);
            (double bx, double by) = ToUser(m, word.Box.Right, word.Box.Bottom);
            double minX = Math.Min(ax, bx), maxX = Math.Max(ax, bx), minY = Math.Min(ay, by), maxY = Math.Max(ay, by);
            Assert.InRange(line.X, minX - PdfTolerance, maxX);
            Assert.InRange(minX, line.X - PdfTolerance, line.X + PdfTolerance);
            Assert.InRange(minY, line.Baseline - (0.2 * line.Size) - PdfTolerance, line.Baseline - (0.2 * line.Size) + PdfTolerance);
            Assert.InRange(maxY, line.Baseline + (0.9 * line.Size) - PdfTolerance, line.Baseline + (0.9 * line.Size) + PdfTolerance);
        }
    }

    /// <summary>Documented tolerance of the box round trip, in points of user space.</summary>
    private const double PdfTolerance = 0.6;

    [Fact]
    public void MalformedFilesGiveTypedExceptionsOrPages()
    {
        (byte[] pdf, _, _) = PdfFixtures.Sample();
        var rng = new Random(12);
        int pagesRead = 0, rejected = 0;
        for (int i = 0; i < 60; i++)
        {
            byte[] mutated = (byte[])pdf.Clone();
            switch (i % 4)
            {
                case 0:
                    mutated = mutated[..rng.Next(mutated.Length)];
                    break;
                case 1:
                    for (int k = 0; k < 20; k++)
                    {
                        mutated[rng.Next(mutated.Length)] = (byte)rng.Next(256);
                    }

                    break;
                case 2:
                    mutated = [.. mutated.Take(rng.Next(mutated.Length)), .. mutated.Skip(rng.Next(mutated.Length))];
                    break;
                default:
                    mutated = new byte[rng.Next(1, 400)];
                    rng.NextBytes(mutated);
                    break;
            }

            try
            {
                pagesRead += PdfOcr.Read(mutated, RandomEngine(), new PdfOcrOptions { RenderDpi = 72, CrossCheckLines = 1 }).Count;
            }
            catch (OcrFormatException)
            {
                rejected++;
            }
            catch (OcrLimitException)
            {
                rejected++;
            }
        }

        Assert.True(pagesRead + rejected > 0);
    }

    [Fact]
    public void ADocumentWithoutACatalogIsAFormatError()
    {
        // The page tree is read lazily: this file opens, and fails when the pages are counted.
        byte[] noRoot = "%PDF-1.4\n1 0 obj\n<< /Type /Nothing >>\nendobj\ntrailer\n<< /Size 2 >>\nstartxref\n0\n%%EOF\n"u8.ToArray();
        Assert.Throws<OcrFormatException>(() => PdfOcr.Read(noRoot, RandomEngine(), NoRender));
    }

    [Fact]
    public void LimitsAreCheckedBeforeWork()
    {
        (byte[] pdf, _, _) = PdfFixtures.Sample();
        Assert.Throws<OcrLimitException>(() => PdfOcr.Read(pdf, RandomEngine(), new PdfOcrOptions { MaxFileBytes = 100 }));
        Assert.Throws<OcrLimitException>(() => PdfOcr.Read(new MemoryStream(pdf), RandomEngine(), new PdfOcrOptions { MaxFileBytes = 100 }));
        var small = OcrEngine.Load(new MemoryStream(TestModels.RandomContainer(
            new TestModels.Shape(32, [(8, 2, 2), (16, 2, 2), (16, 2, 1), (32, 2, 1), (32, 2, 1)], 24, 2, 41), 3)), new OcrLimits { MaxPagesPerCall = 3 });
        Assert.Throws<OcrLimitException>(() => PdfOcr.Read(pdf, small, NoRender));
        Assert.Throws<ArgumentException>(() => PdfOcr.Read(pdf, RandomEngine(), new PdfOcrOptions { RenderDpi = 5 }));
        Assert.Throws<OcrFormatException>(() => PdfOcr.Read("not a pdf"u8.ToArray(), RandomEngine(), NoRender));
    }

    /// <summary>Per-page result of comparing a page with what it should read as.</summary>
    internal sealed record PageScore(string Category, int TruthWords, double TextAccuracy, double SourceAccuracy, double Precision);

    internal static PageScore Score(OcrPage page, PageTruth truth)
    {
        var want = new Dictionary<(OcrSource, string), int>();
        var wantText = new Dictionary<string, int>();
        int total = 0;
        foreach ((OcrSource s, string w) in truth.Words)
        {
            want[(s, w)] = want.GetValueOrDefault((s, w)) + 1;
            wantText[w] = wantText.GetValueOrDefault(w) + 1;
            total++;
        }

        var got = new Dictionary<(OcrSource, string), int>();
        var gotText = new Dictionary<string, int>();
        foreach (OcrWord w in page.Words)
        {
            got[(w.Source, w.Text)] = got.GetValueOrDefault((w.Source, w.Text)) + 1;
            gotText[w.Text] = gotText.GetValueOrDefault(w.Text) + 1;
        }

        int textHits = wantText.Sum(kv => Math.Min(kv.Value, gotText.GetValueOrDefault(kv.Key)));
        int sourceHits = want.Sum(kv => Math.Min(kv.Value, got.GetValueOrDefault(kv.Key)));
        return new PageScore(truth.Category, total, (double)textHits / total, (double)sourceHits / total, page.Words.Count == 0 ? 0 : (double)textHits / page.Words.Count);
    }

    private static string Table(string title, IEnumerable<PageScore> rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine(title);
        sb.AppendLine("| Category | Truth words | Word accuracy | Source correct | Precision |");
        sb.AppendLine("|---|---|---|---|---|");
        foreach (PageScore r in rows)
        {
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"| {r.Category} | {r.TruthWords} | {r.TextAccuracy:P1} | {r.SourceAccuracy:P1} | {r.Precision:P1} |"));
        }

        return sb.ToString();
    }

    [Fact]
    public void EveryPageCategoryMergesCorrectly()
    {
        OcrEngine? engine = RealEngine();
        Assert.SkipWhen(engine is null, "ONEOCR_MODEL (a trained model container) is not set.");
        (byte[] pdf, PageTruth[] truth, _) = PdfFixtures.Sample();
        IReadOnlyList<OcrPage> pages = PdfOcr.Read(pdf, engine, new PdfOcrOptions());
        var scores = new List<PageScore>();
        for (int i = 0; i < truth.Length; i++)
        {
            scores.Add(Score(pages[i], truth[i]));
        }

        string report = Table("default options", scores);
        foreach (OcrPage p in pages)
        {
            report += $"page {p.Index} diagnostics: {string.Join(" | ", p.Diagnostics)}\n";
        }

        // The invisible layer kept on request: wrong words from the text layer, the scan is not read again.
        IReadOnlyList<OcrPage> trusted = PdfOcr.Read(pdf, engine, new PdfOcrOptions { TrustExistingOcrLayer = true });
        PageTruth wrong = new("invisible layer, trusted", "layer kept", PdfFixtures.WrongLayer.Select(l => (OcrSource.TextLayer, l)).ToArray());
        PageScore trustedScore = Score(trusted[3], wrong);
        report += Table("TrustExistingOcrLayer on page 3", [trustedScore]);
        Environment.GetEnvironmentVariable("ONEOCR_PDF_REPORT").AppendReport(report);

        foreach (PageScore s in scores)
        {
            Assert.True(s.TextAccuracy >= 0.95, $"{s.Category}: word accuracy {s.TextAccuracy}");
            Assert.True(s.SourceAccuracy >= 0.95, $"{s.Category}: source correct {s.SourceAccuracy}");
        }

        Assert.Equal(1.0, scores[0].SourceAccuracy);
        Assert.Equal(1.0, trustedScore.SourceAccuracy);
        Assert.All(trusted[3].Words, w => Assert.Equal(OcrSource.TextLayer, w.Source));

        // Same pages at another degree of parallelism: identical JSON.
        IReadOnlyList<OcrPage> one = PdfOcr.Read(pdf, engine, new PdfOcrOptions { MaxDegreeOfParallelism = 1 });
        Assert.Equal(OcrPageJson.ToJson(one), OcrPageJson.ToJson(pages));
    }

    [Fact]
    public void RecognizedBoxesLandOnTheScannedLines()
    {
        OcrEngine? engine = RealEngine();
        Assert.SkipWhen(engine is null, "ONEOCR_MODEL (a trained model container) is not set.");
        (byte[] pdf, _, _) = PdfFixtures.Sample();
        OcrPage scan = PdfOcr.Read(pdf, engine, new PdfOcrOptions { FirstPage = 1, PageCount = 1 })[0];
        OcrMatrix m = scan.ToPdf!.Value;
        for (int i = 0; i < PdfFixtures.BodyB.Length; i++)
        {
            string first = PdfFixtures.BodyB[i].Split(' ')[0];
            double baseline = 700 - (22 * i);
            OcrWord w = scan.Words.First(x => x.Text == first && ToUser(m, x.Box.Left, x.Box.Bottom).Item2 < baseline + 8 && ToUser(m, x.Box.Left, x.Box.Bottom).Item2 > baseline - 8);
            (double x0, double y0) = ToUser(m, w.Box.Left, w.Box.Bottom);
            (double x1, double y1) = ToUser(m, w.Box.Right, w.Box.Top);
            Assert.True(Math.Abs(x0 - 72) < 4, $"line {i} {first}: left {x0}");
            Assert.InRange(baseline, Math.Min(y0, y1) - 1, Math.Max(y0, y1) + 1);
            Assert.True(Math.Abs(y1 - y0) < 24, "box is not taller than two lines");
            Assert.True(x1 > x0);
        }
    }
}
