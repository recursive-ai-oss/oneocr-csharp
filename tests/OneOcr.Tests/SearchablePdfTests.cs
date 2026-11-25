using System.Globalization;
using System.Text;
using OneImage;
using OneImage.Pdf;
using OneOcr.Pdf;
using OnePdf;
using OnePdf.Text;

namespace OneOcr.Tests;

/// <summary>Searchable PDFs end to end with a real recognizer: images and PDFs in, OnePDF extraction and OneImage rendering to check what comes out. Needs ONEOCR_MODEL.</summary>
public class SearchablePdfTests
{
    private static OcrEngine? Engine()
    {
        string? path = Environment.GetEnvironmentVariable("ONEOCR_MODEL");
        return path is null || !File.Exists(path) ? null : OcrEngine.Load(path);
    }

    private static OcrEngine Need()
    {
        OcrEngine? e = Engine();
        Assert.SkipWhen(e is null, "ONEOCR_MODEL (a trained model container) is not set.");
        return e!;
    }

    private static Image ScanImage(string[] lines, double width = 612, double height = 792)
    {
        (byte[] gray, int w, int h) = PdfFixtures.RenderLines(lines, width, height, 72, 12, 22, 92);
        return Image.FromPixels<Gray8>(gray.Select(static b => new Gray8(b)).ToArray(), w, h);
    }

    private static byte[] Encode(Image image, IImageFormat format)
    {
        using var ms = new MemoryStream();
        image.Save(ms, format);
        return ms.ToArray();
    }

    private static readonly OcrOptions Dpi300 = new() { Dpi = 300 };

    private static SearchablePdfOptions Opts => new() { Ocr = Dpi300 };

    /// <summary>All pages of a PDF rendered at <paramref name="dpi"/> as 8-bit gray.</summary>
    internal static List<(byte[] Gray, int Width, int Height)> Render(byte[] pdf, double dpi)
    {
        var configuration = new ImageConfiguration().Add(new PdfFormat(new PdfRenderOptions { Dpi = dpi }));
        Image image = Image.Load(pdf, new LoadOptions { Configuration = configuration });
        var pages = new List<(byte[], int, int)>();
        for (int i = 0; i < image.FrameCount; i++)
        {
            ImageFrame f = image.Frames[i];
            Gray8[] px = f.ToArray<Gray8>(1, CancellationToken.None);
            pages.Add((px.Select(static p => p.V).ToArray(), f.Width, f.Height));
        }

        return pages;
    }

    private static string Describe(SearchablePdfResult r) => string.Join(" | ", r.Pages.Select(p => p.ToText().Replace('\n', '/')));

    // ---- images ----

    [Theory]
    [InlineData("clean", 0)]
    [InlineData("quarter 90", 90)]
    [InlineData("half 180", 180)]
    [InlineData("quarter 270", 270)]
    [InlineData("skew +3", 3)]
    [InlineData("skew -3", -3)]
    [InlineData("skew +8", 8)]
    [InlineData("skew -8", -8)]
    public void ImagePagesGetTheirWordsBack(string name, double degrees)
    {
        OcrEngine engine = Need();
        Image scan = ScanImage(PdfFixtures.BodyA);
        Image page = degrees == 0 ? scan : scan.Rotate(degrees, true, new Rgba32(255, 255, 255, 255));
        byte[] png = Encode(page, PngFormat.Instance);
        using var ms = new MemoryStream();
        SearchablePdfResult result = SearchablePdf.FromData([png], ms, engine, Opts);
        byte[] pdf = ms.ToArray();

        OcrPage model = Assert.Single(result.Pages);
        Assert.True(model.Words.Count >= 40, $"{name}: only {model.Words.Count} words read");
        Assert.Equal(model.Words.Count, result.WordsWritten);
        AssertFramesCoverWords(model);
        if (degrees % 90 == 0)
        {
            double worst = SearchablePdfWriterTests.MaxEdgeError(model, pdf);
            Console.WriteLine($"{name}: word boxes within {worst:0.000} pt of the page model, {model.Words.Count} words");
            Assert.True(worst <= 1.0, $"{name}: {worst}");
            Assert.Equal(Words(model.ToText()), Words(PageText(pdf, 0)));
        }
        else
        {
            (double along, double across) = SearchablePdfWriterTests.GlyphError(model, pdf);
            Console.WriteLine($"{name}: glyph corners within {along:0.000} / {across:0.000} pt of the word quads, {model.Words.Count} words, skew {model.SkewDegrees}");
            Assert.True(along <= 0.1 && across <= 0.1, $"{name}: {along} {across}");
        }
    }

    /// <summary>A word's frame spans the word along the line and the line across it.</summary>
    internal static void AssertFramesCoverWords(OcrPage page)
    {
        foreach ((OcrLine line, OcrWord word) in SearchablePdfWriterTests.WordsWithLines(page))
        {
            var (bl, br, tr, tl) = PdfTextLayerWriter.Frame(line, word);
            OcrQuad lq = line.Quad ?? new OcrQuad(new OcrPoint(line.Box.Left, line.Box.Top), new OcrPoint(line.Box.Right, line.Box.Top), new OcrPoint(line.Box.Right, line.Box.Bottom), new OcrPoint(line.Box.Left, line.Box.Bottom));
            double h = Math.Sqrt(Math.Pow(lq.P0.X - lq.P3.X, 2) + Math.Pow(lq.P0.Y - lq.P3.Y, 2));
            Assert.Equal(h, Math.Sqrt(Math.Pow(tl.X - bl.X, 2) + Math.Pow(tl.Y - bl.Y, 2)), 6);
            // Every corner of the word lies between the frame's ends (up to the line's own extent).
            OcrQuad wq = word.Quad ?? new OcrQuad(new OcrPoint(word.Box.Left, word.Box.Top), new OcrPoint(word.Box.Right, word.Box.Top), new OcrPoint(word.Box.Right, word.Box.Bottom), new OcrPoint(word.Box.Left, word.Box.Bottom));
            double bx = br.X - bl.X, by = br.Y - bl.Y, len = Math.Sqrt((bx * bx) + (by * by));
            foreach (OcrPoint p in new[] { wq.P0, wq.P1, wq.P2, wq.P3 })
            {
                double t = (((p.X - bl.X) * bx) + ((p.Y - bl.Y) * by)) / len;
                Assert.InRange(t, -1e-6, len + 1e-6);
            }
        }
    }

    // Paragraph breaks are each reader's own call; the words and their order are what has to agree.
    private static string[] Words(string text) => text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

    private static string PageText(byte[] pdf, int index)
    {
        using var doc = PdfDocument.Open(pdf);
        return doc.Pages[index].ExtractText().Text;
    }

    [Fact]
    public void ImageIsReproducedPixelForPixelAtItsOwnResolution()
    {
        OcrEngine engine = Need();
        Image scan = ScanImage(PdfFixtures.BodyB);
        byte[] png = Encode(scan, PngFormat.Instance);
        using var ms = new MemoryStream();
        SearchablePdfResult r = SearchablePdf.FromData([png], ms, engine, Opts);
        (byte[] want, int w, int h) = Pixels(scan);
        (byte[] got, int gw, int gh) = Render(ms.ToArray(), 300)[0];
        Assert.Equal((w, h), (gw, gh));
        Assert.True(want.AsSpan().SequenceEqual(got), "rendered page differs from the image");
        Assert.Equal(1, r.PagesWithTextLayer);
    }

    private static (byte[] Gray, int W, int H) Pixels(Image image)
    {
        Gray8[] px = image.ToArray<Gray8>(1, CancellationToken.None);
        return (px.Select(static p => p.V).ToArray(), image.Width, image.Height);
    }

    [Fact]
    public void JpegIsPlacedAsItsOwnBytes()
    {
        OcrEngine engine = Need();
        byte[] jpeg = Encode(ScanImage(PdfFixtures.BodyB), JpegFormat.Instance);
        using var ms = new MemoryStream();
        SearchablePdf.FromData([jpeg], ms, engine, Opts);
        using var doc = PdfDocument.Open(ms.ToArray());
        PdfStream image = FirstImage(doc.Pages[0]);
        Assert.Equal("DCTDecode", Assert.Single(image.FilterNames));
        Assert.True(image.GetRawBytes().Span.SequenceEqual(jpeg), "the JPEG stream is not the original file");

        // The same file with passthrough off is stored losslessly instead.
        using var ms2 = new MemoryStream();
        SearchablePdf.FromData([jpeg], ms2, engine, new SearchablePdfOptions { Ocr = Dpi300, EmbedOriginalJpeg = false });
        using var doc2 = PdfDocument.Open(ms2.ToArray());
        Assert.DoesNotContain("DCTDecode", FirstImage(doc2.Pages[0]).FilterNames);
        Assert.True(ms2.Length > ms.Length);
    }

    internal static PdfStream FirstImage(PdfPage page)
    {
        PdfDictionary? xo = page.Resources.GetDictionary("XObject");
        Assert.NotNull(xo);
        foreach (string key in xo.Keys)
        {
            if (xo.Get(key) is PdfStream s && s.Dictionary.GetName("Subtype") == "Image")
            {
                return s;
            }
        }

        throw new InvalidOperationException("no image on the page");
    }

    private static byte[] Tiff1(IReadOnlyList<(byte[] Gray, int W, int H)> frames, int dpi)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write((byte)'I');
        w.Write((byte)'I');
        w.Write((ushort)42);
        w.Write(0u);
        long patch = 4;
        foreach ((byte[] gray, int width, int height) in frames)
        {
            int stride = (width + 7) / 8;
            byte[] packed = new byte[stride * height];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (gray[(y * width) + x] >= 128)
                    {
                        packed[(y * stride) + (x >> 3)] |= (byte)(0x80 >> (x & 7));
                    }
                }
            }

            long dataAt = ms.Position;
            w.Write(packed);
            long rationalAt = ms.Position;
            w.Write((uint)dpi);
            w.Write(1u);
            long ifdAt = ms.Position;
            long back = ms.Position;
            ms.Position = patch;
            w.Write((uint)ifdAt);
            ms.Position = back;
            (ushort, ushort, uint, uint)[] entries =
            [
                (256, 4, 1, (uint)width), (257, 4, 1, (uint)height), (258, 3, 1, 1), (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, (uint)dataAt),
                (277, 3, 1, 1), (278, 4, 1, (uint)height), (279, 4, 1, (uint)packed.Length), (282, 5, 1, (uint)rationalAt), (283, 5, 1, (uint)rationalAt), (296, 3, 1, 2),
            ];
            w.Write((ushort)entries.Length);
            foreach ((ushort tag, ushort type, uint count, uint value) in entries)
            {
                w.Write(tag);
                w.Write(type);
                w.Write(count);
                w.Write(value);
            }

            patch = ms.Position;
            w.Write(0u);
        }

        return ms.ToArray();
    }

    [Fact]
    public void MultiFrameBilevelTiffBecomesOneBitPages()
    {
        OcrEngine engine = Need();
        string[][] sets = [PdfFixtures.BodyA, PdfFixtures.BodyB, PdfFixtures.Garbled];
        var frames = new List<(byte[], int, int)>();
        foreach (string[] lines in sets)
        {
            // Threshold the anti-aliased raster: a fax-like page of pure black and white.
            (byte[] gray, int w, int h) = PdfFixtures.RenderLines(lines, 612, 792, 72, 12, 22, 92);
            frames.Add((gray.Select(static b => b >= 128 ? (byte)255 : (byte)0).ToArray(), w, h));
        }

        byte[] tiff = Tiff1(frames, 300);
        using var ms = new MemoryStream();
        SearchablePdfResult r = SearchablePdf.FromData([tiff], ms, engine);
        byte[] pdf = ms.ToArray();
        Assert.Equal(3, r.Pages.Count);
        using var doc = PdfDocument.Open(pdf);
        Assert.Equal(3, doc.PageCount);
        for (int i = 0; i < 3; i++)
        {
            PdfStream image = FirstImage(doc.Pages[i]);
            Assert.Equal(1, image.Dictionary.GetInt("BitsPerComponent"));
            Assert.Equal("DeviceGray", image.Dictionary.GetName("ColorSpace"));
            Assert.Equal(frames[i].Item2, image.Dictionary.GetInt("Width"));
            Assert.Equal(Words(r.Pages[i].ToText()), Words(PageText(pdf, i)));
            Assert.True(SearchablePdfWriterTests.MaxEdgeErrorOn(r.Pages[i], pdf, i) <= 1.0);
            // The TIFF said 300 dpi, so the page is letter sized.
            Assert.Equal(612, r.Pages[i].Width * 72 / r.Pages[i].UnitsPerInch, 0.5);
        }

        // Every frame comes back pixel for pixel.
        List<(byte[] Gray, int Width, int Height)> rendered = Render(pdf, 300);
        for (int i = 0; i < 3; i++)
        {
            Assert.True(frames[i].Item1.AsSpan().SequenceEqual(rendered[i].Gray), $"frame {i} differs");
        }

        Console.WriteLine($"1-bit TIFF, 3 frames, {tiff.Length} bytes in, PDF {pdf.Length} bytes out");
    }

    [Fact]
    public void TheLayerCostsLittleSpace()
    {
        OcrEngine engine = Need();
        byte[] jpeg = Encode(ScanImage(PdfFixtures.BodyA), JpegFormat.Instance);
        using var withLayer = new MemoryStream();
        SearchablePdfResult r = SearchablePdf.FromData([jpeg], withLayer, engine, Opts);
        long overhead = withLayer.Length - jpeg.Length;
        Console.WriteLine($"JPEG scan page: {jpeg.Length} bytes in, {withLayer.Length} bytes out, {r.WordsWritten} words, overhead {overhead} bytes ({overhead / (double)r.WordsWritten:0.0} per word)");
        Assert.InRange(overhead, 1, 20_000);
    }

    [Fact]
    public void SameInputGivesTheSameBytesAtAnyDegree()
    {
        OcrEngine engine = Need();
        byte[] png = Encode(ScanImage(PdfFixtures.BodyB), PngFormat.Instance);
        byte[] Run(OcrOptions o)
        {
            using var ms = new MemoryStream();
            SearchablePdf.FromData([png], ms, engine, new SearchablePdfOptions { Ocr = o });
            return ms.ToArray();
        }

        byte[] one = Run(new OcrOptions { Dpi = 300, MaxDegreeOfParallelism = 1 });
        byte[] again = Run(new OcrOptions { Dpi = 300, MaxDegreeOfParallelism = 1 });
        byte[] four = Run(new OcrOptions { Dpi = 300, MaxDegreeOfParallelism = 4 });
        Assert.True(one.AsSpan().SequenceEqual(again));
        Assert.True(one.AsSpan().SequenceEqual(four));
    }

    [Fact]
    public void ExtensionAndAsyncTwinAgreeWithTheStaticCall()
    {
        OcrEngine engine = Need();
        Image scan = ScanImage(PdfFixtures.BodyB);
        using var a = new MemoryStream();
        scan.ToSearchablePdf(engine, a, Opts);
        using var b = new MemoryStream();
        scan.ToSearchablePdfAsync(engine, b, Opts).GetAwaiter().GetResult();
        Assert.True(a.ToArray().AsSpan().SequenceEqual(b.ToArray()));
        string path = Path.Combine(Path.GetTempPath(), "oneocr-" + Guid.NewGuid().ToString("N") + ".pdf");
        try
        {
            scan.ToSearchablePdf(engine, path, Opts);
            Assert.True(File.ReadAllBytes(path).AsSpan().SequenceEqual(a.ToArray()));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void PdfA2bOutputIsAccepted()
    {
        OcrEngine engine = Need();
        byte[] png = Encode(ScanImage(PdfFixtures.BodyB), PngFormat.Instance);
        using var ms = new MemoryStream();
        SearchablePdf.FromData([png], ms, engine, new SearchablePdfOptions { Ocr = Dpi300, Conformance = SearchablePdfConformance.PdfA2b, Title = "Scan" });
        using var doc = PdfDocument.Open(ms.ToArray());
        string text = Encoding.Latin1.GetString(ms.ToArray());
        Assert.Contains("pdfaid", text);
        Assert.Contains("<pdfaid:part>2</pdfaid:part>", text);
        Assert.Contains("<pdfaid:conformance>B</pdfaid:conformance>", text);
    }

    [Fact]
    public void CancellationIsHonored()
    {
        OcrEngine engine = Need();
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        byte[] png = Encode(ScanImage(PdfFixtures.BodyB), PngFormat.Instance);
        Assert.ThrowsAny<OperationCanceledException>(() => SearchablePdf.FromData([png], new MemoryStream(), engine, Opts, cts.Token));
    }

    // ---- PDF in, PDF out ----

    [Fact]
    public void MakeSearchableAddsLayersOnlyWhereNeeded()
    {
        OcrEngine engine = Need();
        (byte[] input, PageTruth[] truth, _) = PdfFixtures.Sample();
        using var ms = new MemoryStream();
        SearchablePdfResult r = SearchablePdf.MakeSearchable(input, ms, engine);
        byte[] output = ms.ToArray();
        Console.WriteLine($"diagnostics: {string.Join(" || ", r.Diagnostics)}");

        using var before = PdfDocument.Open(input);
        using var after = PdfDocument.Open(output);
        Assert.Equal(before.PageCount, after.PageCount);

        // Rendering is unchanged on every page.
        List<(byte[] Gray, int Width, int Height)> was = Render(input, 100), now = Render(output, 100);
        for (int i = 0; i < before.PageCount; i++)
        {
            Assert.True(was[i].Gray.AsSpan().SequenceEqual(now[i].Gray), $"page {i} renders differently");
        }

        // 0 text only: no OCR words, nothing written, the content stream is the same bytes.
        Assert.Equal(Content(before.Pages[0]), Content(after.Pages[0]));
        Assert.Single(after.Pages[0].ContentStreams);
        Assert.DoesNotContain(r.Pages[0].Words, static w => w.Source == OcrSource.Ocr);

        var layered = new List<int>();
        for (int i = 1; i < 6; i++)
        {
            OcrPage model = r.Pages[i];
            string[] ocr = [.. model.Words.Where(static w => w.Source == OcrSource.Ocr).Select(static w => w.Text)];
            Assert.NotEmpty(ocr);

            // The invisible glyphs, whatever the reader makes of words where they lie over visible text, spell exactly the recognized words.
            using (var d = PdfDocument.Open(output))
            {
                PdfPageText t = d.Pages[i].ExtractText(new PdfTextOptions { RetainGlyphs = true, CollapseDuplicates = false });
                Assert.Equal(string.Concat(ocr), string.Concat(t.Glyphs.Where(static g => !g.IsVisible).Select(static g => g.Text)));
            }

            // Where nothing visible lies under the layer, the words come back as words, in order, next to the text layer's own visible ones.
            if (i <= 3)
            {
                List<PdfTextWord> words = SearchablePdfWriterTests.Extract(output, i);
                Assert.Equal(ocr, words.Where(static w => !w.IsVisible).Select(static w => w.Text));
                string[] visibleBefore = [.. SearchablePdfWriterTests.Extract(input, i).Where(static w => w.IsVisible).Select(static w => w.Text)];
                Assert.Equal(visibleBefore, words.Where(static w => w.IsVisible).Select(static w => w.Text));
            }

            layered.Add(i);
        }

        // Pages that keep their own content: it is still there, byte for byte, in the same order.
        foreach (int i in new[] { 1, 2, 4, 5 })
        {
            AssertContainsInOrder(after.Pages[i], before.Pages[i]);
        }

        // 3 invisible wrong layer: gone, replaced by ours; the image is still drawn.
        List<PdfTextWord> third = SearchablePdfWriterTests.Extract(output, 3);
        Assert.DoesNotContain(third, w => PdfFixtures.WrongLayer.Any(l => l.Split(' ').Contains(w.Text) && w.Text.Length > 4 && !PdfFixtures.BodyA.Any(b => b.Contains(w.Text))));
        Assert.Contains(r.Diagnostics, static d => d.StartsWith("Page 4: removed", StringComparison.Ordinal));
        Assert.Equal(r.Pages[3].Words.Count(static w => w.Source == OcrSource.Ocr), third.Count(static w => !w.IsVisible));

        // 4 and 5: the unreliable visible text stays (rendering must not change) and the result says so.
        Assert.Contains(r.Diagnostics, static d => d.StartsWith("Page 5:", StringComparison.Ordinal) && d.Contains("visible", StringComparison.Ordinal));
        Assert.Contains(r.Diagnostics, static d => d.StartsWith("Page 6:", StringComparison.Ordinal) && d.Contains("visible", StringComparison.Ordinal));

        // The scan page: boxes within tolerance of the page model.
        OcrPage scan = r.Pages[1];
        double worst = SearchablePdfWriterTests.MaxEdgeErrorOn(scan, output, 1, scan.ToPdf);
        Console.WriteLine($"scan page: word boxes within {worst:0.000} pt");
        Assert.True(worst <= 1.0, worst.ToString(CultureInfo.InvariantCulture));
        Assert.Equal(layered.Count, r.PagesWithTextLayer);

        // A second pass replaces our own layer instead of stacking another on it.
        using var again = new MemoryStream();
        SearchablePdf.MakeSearchable(output, again, engine);
        Assert.Equal(
            SearchablePdfWriterTests.Extract(output, 1).Select(static w => w.Text),
            SearchablePdfWriterTests.Extract(again.ToArray(), 1).Select(static w => w.Text));
    }

    [Fact]
    public void TrustedLayerAndNothingToReadLeaveTheFileAsItWas()
    {
        OcrEngine engine = Need();
        (byte[] input, _, _) = PdfFixtures.Sample();
        using var ms = new MemoryStream();
        SearchablePdfResult r = SearchablePdf.MakeSearchable(input, ms, engine, new SearchablePdfOptions { Pdf = new PdfOcrOptions { ReadImages = false, CrossCheckLines = 0, TrustExistingOcrLayer = true, FirstPage = 0, PageCount = 1 } });
        Assert.Equal(0, r.PagesWithTextLayer);
        Assert.True(ms.ToArray().AsSpan().SequenceEqual(input), "a file with nothing to add is written exactly as it came");
    }

    [Fact]
    public void MakeSearchableCanWritePdfA2b()
    {
        OcrEngine engine = Need();
        byte[] png = Encode(ScanImage(PdfFixtures.BodyB), PngFormat.Instance);
        using var plain = new MemoryStream();
        SearchablePdf.FromData([png], plain, engine, Opts);
        using var a = new MemoryStream();
        using var b = new MemoryStream();
        var options = new SearchablePdfOptions { Conformance = SearchablePdfConformance.PdfA2b };
        SearchablePdf.MakeSearchable(plain.ToArray(), a, engine, options);
        SearchablePdf.MakeSearchable(plain.ToArray(), b, engine, options);
        Assert.Contains("<pdfaid:part>2</pdfaid:part>", Encoding.Latin1.GetString(a.ToArray()), StringComparison.Ordinal);
        Assert.True(a.ToArray().AsSpan().SequenceEqual(b.ToArray()), "conversion is not deterministic");
        Assert.Equal(Words(SearchablePdfWriterTests.Extract(plain.ToArray()).Count.ToString()), Words(SearchablePdfWriterTests.Extract(a.ToArray()).Count.ToString()));
    }

    [Fact]
    public void MakeSearchableIsDeterministic()
    {
        OcrEngine engine = Need();
        (byte[] input, _, _) = PdfFixtures.Sample();
        using var a = new MemoryStream();
        using var b = new MemoryStream();
        SearchablePdf.MakeSearchable(input, a, engine, new SearchablePdfOptions { Pdf = new PdfOcrOptions { MaxDegreeOfParallelism = 1 } });
        SearchablePdf.MakeSearchable(input, b, engine, new SearchablePdfOptions { Pdf = new PdfOcrOptions { MaxDegreeOfParallelism = 4 } });
        Assert.True(a.ToArray().AsSpan().SequenceEqual(b.ToArray()));
    }

    [Fact]
    public void MixedSampleCanBeWrittenAsPdfA2bToo()
    {
        OcrEngine engine = Need();
        (byte[] input, _, _) = PdfFixtures.Sample();
        using var ms = new MemoryStream();
        SearchablePdfResult r = SearchablePdf.MakeSearchable(input, ms, engine, new SearchablePdfOptions { Conformance = SearchablePdfConformance.PdfA2b });
        Assert.Contains("<pdfaid:part>2</pdfaid:part>", Encoding.Latin1.GetString(ms.ToArray()), StringComparison.Ordinal);
        string[] ocr = [.. r.Pages[1].Words.Select(static w => w.Text)];
        Assert.Equal(ocr, SearchablePdfWriterTests.Extract(ms.ToArray(), 1).Where(static w => !w.IsVisible).Select(static w => w.Text));
    }

    private static void AssertContainsInOrder(PdfPage after, PdfPage before)
    {
        List<byte[]> a = after.ContentStreams.Select(static s => s.GetRawBytes().ToArray()).ToList();
        int at = 0;
        foreach (PdfStream s in before.ContentStreams)
        {
            byte[] want = s.GetRawBytes().ToArray();
            while (at < a.Count && !a[at].AsSpan().SequenceEqual(want))
            {
                at++;
            }

            Assert.True(at < a.Count, "an original content stream is missing or changed");
            at++;
        }
    }

    private static byte[] Content(PdfPage page)
    {
        using var ms = new MemoryStream();
        foreach (PdfStream s in page.ContentStreams)
        {
            ms.Write(s.GetRawBytes().Span);
        }

        return ms.ToArray();
    }
}
