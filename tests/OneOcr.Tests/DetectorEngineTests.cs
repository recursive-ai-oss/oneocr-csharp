using OneImage;
using OneOcr.Decoding;
using OneOcr.Detection;
using OneOcr.Kernels;
using OneOcr.Layout;
using OneOcr.Model;

namespace OneOcr.Tests;

public class DetectorEngineTests
{
    private sealed class StubDetector(Func<int, PageDetection> detect) : IPageDetector
    {
        public List<(int Width, int Height)> Calls { get; } = [];

        public PageDetection Detect(
            ReadOnlySpan<Gray8> page, int width, int height, BufferPool<Gray8> pixels, ResizerPool resizers, int degree, CancellationToken cancellationToken)
        {
            Calls.Add((width, height));
            return detect(Calls.Count - 1);
        }
    }

    private static (OcrEngine Classical, Image Page, List<LineBox> Planned, byte[] Pixels) Setup()
    {
        var (px, planned) = EngineTests.SyntheticPage(900, 420, [60, 130, 200, 270], 14, 8);
        Image page = Image.FromPixels<Gray8>(px.Select(b => new Gray8(b)).ToArray(), 900, 420);
        var engine = OcrEngine.Create(ModelContainer.Parse(TestModels.RandomContainer(TestModels.Small, 5)), null, KernelPath.Scalar);
        return (engine, page, planned, px);
    }

    private static DetectedLine Around(LineBox b) => new(b.X - 5, b.Y - 4, b.Right + 5, b.Bottom + 4, 0.9f);

    private static PageDetection Detection(List<LineBox> planned, int quarter = 0, float confidence = 0.99f, List<DetectedRegion>? regions = null) =>
        new() { Lines = planned.Select(Around).ToList(), Regions = regions ?? [], Quarter = quarter, OrientationConfidence = confidence };

    [Fact]
    public void DetectedLinesReadLikeTheClassicalFinderOnACleanPage()
    {
        var (classical, page, planned, _) = Setup();
        Assert.False(classical.HasDetector);
        OcrEngine engine = classical.WithDetector(new StubDetector(_ => Detection(planned)));
        Assert.True(engine.HasDetector);
        OcrPage viaDetector = engine.Ocr(page);
        OcrPage viaFinder = classical.Ocr(page);
        Assert.Equal(viaFinder.Text, viaDetector.Text);
        Assert.Equal(viaFinder.Words.Count, viaDetector.Words.Count);
        for (int i = 0; i < viaFinder.Words.Count; i++)
        {
            Assert.Equal(viaFinder.Words[i].Box, viaDetector.Words[i].Box);
        }

        Assert.False(engine.WithoutDetector().HasDetector);
        // Plain text and beam decoding go through the detector too.
        Assert.Equal(classical.ToText(page), engine.ToText(page));
        Assert.Equal(classical.ToText(page, DecodeOptions.Greedy), engine.ToText(page, DecodeOptions.Greedy));
    }

    [Fact]
    public void RegionsBecomePageRegionsAndFurnitureRoles()
    {
        var (classical, page, planned, _) = Setup();
        DetectedRegion R(string name, LineBox from, LineBox to) =>
            new(Array.IndexOf(TestDetectorModels.AllClasses, name), name, from.X - 10, from.Y - 8, to.Right + 10, to.Bottom + 8, 0.8f);
        var regions = new List<DetectedRegion>
        {
            R("title", planned[0], planned[0]),
            R("text", planned[1], planned[2]),
            R("header_footer", planned[3], planned[3]),
            R("table", planned[1], planned[2]),
            new(99, "something_new", 0, 0, 10, 10, 0.5f),
        };
        OcrPage result = classical.WithDetector(new StubDetector(_ => Detection(planned, regions: regions))).Ocr(page);
        Assert.Equal([OcrRegionRole.Title, OcrRegionRole.Text, OcrRegionRole.Footer, OcrRegionRole.Table], result.Regions.Select(r => r.Role).ToArray());
        Assert.All(result.Regions, r => Assert.Equal(OcrSource.Ocr, r.Source));
        Assert.Equal(0.8f, result.Regions[0].Confidence);
        // Every block lies in one of the regions it lists; the last line is in the footer region (lower half of the page).
        foreach (OcrRegion region in result.Regions)
        {
            Assert.NotEmpty(region.BlockIndices);
            foreach (int index in region.BlockIndices)
            {
                OcrBox box = result.Blocks[index].Box;
                Assert.InRange((box.Left + box.Right) / 2, region.Box.Left, region.Box.Right);
                Assert.InRange((box.Top + box.Bottom) / 2, region.Box.Top, region.Box.Bottom);
            }
        }

        Assert.Contains(result.Blocks, b => b.Role == OcrBlockRole.Footer);
        Assert.DoesNotContain(result.Blocks, b => b.Role == OcrBlockRole.Header);

        // A header_footer region in the upper half is a header.
        var upper = new List<DetectedRegion> { R("header_footer", planned[0], planned[0]) };
        OcrPage top = classical.WithDetector(new StubDetector(_ => Detection(planned, regions: upper))).Ocr(page);
        Assert.Equal(OcrRegionRole.Header, top.Regions[0].Role);
        Assert.Contains(top.Blocks, b => b.Role == OcrBlockRole.Header);
    }

    [Fact]
    public void ARotatedPageIsTurnedUprightBeforeReading()
    {
        var (classical, upright, planned, pixels) = Setup();
        Gray8[] source = pixels.Select(b => new Gray8(b)).ToArray();
        // The content rotated 90 degrees counter-clockwise is the upright page turned clockwise three times.
        var pool = new BufferPool<Gray8>(1 << 20, 1);
        Gray8[] ccw = PageOrientation.Rotate(source, 900, 420, 3, pool, out int cw, out int ch);
        Assert.Equal((420, 900), (cw, ch));
        Image rotated = Image.FromPixels<Gray8>(ccw.AsSpan(0, cw * ch).ToArray(), cw, ch);

        var stub = new StubDetector(call => call == 0 ? Detection([], quarter: 1) : Detection(planned));
        OcrPage page = classical.WithDetector(stub, confirmTurns: false).Ocr(rotated, new OcrOptions { SearchAngle = false });
        Assert.Equal([(420, 900), (900, 420)], stub.Calls);
        // Coordinates stay in the image as given; the turn is reported as the orientation.
        Assert.Equal(420, page.Width);
        Assert.Equal(900, page.Height);
        Assert.Equal(270, page.Orientation);
        OcrDiagnostic turnedNote = Assert.Single(page.Diagnostics, d => d.Code == OcrDiagnosticCodes.PageTurned);
        Assert.Contains("90 degrees clockwise", turnedNote.Message, StringComparison.Ordinal);
        Assert.Contains("image as given", turnedNote.Message, StringComparison.Ordinal);
        Assert.Equal(classical.Ocr(upright).Text, page.Text);

        // With the classical check on (the default), a turn the head asks for on a page the line reading finds upright is not made.
        var wrong = new StubDetector(_ => Detection(planned, quarter: 2, confidence: 0.99f));
        OcrPage kept = classical.WithDetector(wrong).Ocr(upright, new OcrOptions { SearchAngle = false });
        Assert.DoesNotContain(kept.Diagnostics, d => d.Message.Contains("turned", StringComparison.Ordinal));
        Assert.Equal((900, 420), (kept.Width, kept.Height));

        // Not sure enough: read as it is.
        var unsure = new StubDetector(_ => Detection(planned, quarter: 2, confidence: 0.6f));
        OcrPage asIs = classical.WithDetector(unsure).Ocr(upright, new OcrOptions { SearchAngle = false });
        Assert.Single(unsure.Calls);
        Assert.DoesNotContain(asIs.Diagnostics, d => d.Message.Contains("turned", StringComparison.Ordinal));
        Assert.Equal(classical.Ocr(upright).Text, asIs.Text);
    }

    // The content turned clockwise by k quarter turns is read upright by turning it back; every box must then sit on ink of the image
    // that was passed in, and carry the quad of the text's own frame.
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void RotatedPagesReportBoxesAndQuadsInTheImageAsGiven(int turnedBy)
    {
        var (classical, upright, planned, pixels) = Setup();
        var source = pixels.Select(b => new Gray8(b)).ToArray();
        var pool = new BufferPool<Gray8>(1 << 20, 1);
        Gray8[] turned = PageOrientation.Rotate(source, 900, 420, turnedBy, pool, out int w, out int h);
        byte[] ink = turned.AsSpan(0, w * h).ToArray().Select(g => g.V).ToArray();
        Image rotated = Image.FromPixels<Gray8>(turned.AsSpan(0, w * h).ToArray(), w, h);

        var stub = new StubDetector(call => call == 0 ? Detection([], quarter: 4 - turnedBy) : Detection(planned));
        OcrPage page = classical.WithDetector(stub, confirmTurns: false).Ocr(rotated, new OcrOptions { SearchAngle = false });
        Assert.Equal((w, h), (page.Width, page.Height));
        Assert.Equal(90 * turnedBy, page.Orientation);
        Assert.Equal(classical.Ocr(upright).Text, page.Text);
        Assert.NotEmpty(page.Words);

        double Dark(OcrBox b)
        {
            int x0 = Math.Max(0, (int)Math.Floor(b.Left)), x1 = Math.Min(w, (int)Math.Ceiling(b.Right));
            int y0 = Math.Max(0, (int)Math.Floor(b.Top)), y1 = Math.Min(h, (int)Math.Ceiling(b.Bottom));
            long dark = 0, total = 0;
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    total++;
                    if (ink[(y * w) + x] < 128)
                    {
                        dark++;
                    }
                }
            }

            return total == 0 ? 0 : (double)dark / total;
        }

        foreach (OcrWord word in page.Words)
        {
            Assert.InRange(word.Box.Left, 0, w);
            Assert.InRange(word.Box.Right, 0, w);
            Assert.InRange(word.Box.Top, 0, h);
            Assert.InRange(word.Box.Bottom, 0, h);
            Assert.True(Dark(word.Box) > 0.25, $"word '{word.Text}' box {word.Box} covers {Dark(word.Box):0.00} ink on the input image");
            OcrQuad q = Assert.NotNull(word.Quad);
            foreach (OcrPoint pt in new[] { q.P0, q.P1, q.P2, q.P3 })
            {
                Assert.InRange(pt.X, -1, w + 1);
                Assert.InRange(pt.Y, -1, h + 1);
            }
        }

        foreach (OcrLine line in page.Blocks.SelectMany(b => b.Lines))
        {
            Assert.NotNull(line.Quad);
        }
    }

    [Fact]
    public void RotationMovesEveryPixelWhereTheTurnSays()
    {
        var pool = new BufferPool<Gray8>(1 << 20, 1);
        const int w = 5, h = 3;
        Gray8[] src = Enumerable.Range(0, w * h).Select(i => new Gray8((byte)i)).ToArray();
        Gray8[] once = PageOrientation.Rotate(src, w, h, 1, pool, out int w1, out int h1);
        Assert.Equal((3, 5), (w1, h1));
        // One clockwise turn: the top-left pixel ends up top-right, the bottom-left pixel top-left.
        Assert.Equal(src[0].V, once[w1 - 1].V);
        Assert.Equal(src[2 * w].V, once[0].V);
        Gray8[] twice = PageOrientation.Rotate(once.AsSpan(0, w * h), w1, h1, 1, pool, out int w2, out int h2);
        Gray8[] half = PageOrientation.Rotate(src, w, h, 2, pool, out _, out _);
        Assert.Equal((w, h), (w2, h2));
        Assert.Equal(half.AsSpan(0, w * h).ToArray().Select(p => p.V), twice.AsSpan(0, w * h).ToArray().Select(p => p.V));
        Gray8[] back = PageOrientation.Rotate(twice.AsSpan(0, w * h), w2, h2, 1, pool, out int w3, out int h3);
        Gray8[] three = PageOrientation.Rotate(src, w, h, 3, pool, out int w4, out int h4);
        Assert.Equal((w3, h3), (w4, h4));
        Assert.Equal(three.AsSpan(0, w * h).ToArray().Select(p => p.V), back.AsSpan(0, w * h).ToArray().Select(p => p.V));
        Assert.Throws<ArgumentOutOfRangeException>(() => PageOrientation.Rotate(src, w, h, 0, pool, out _, out _));
    }

    // A detector whose maps are constants: no text, every layout class on over the whole page.
    private static byte[] ConstantMapsModel(string? trainedJson)
    {
        List<TestTensor> tensors = TestDetectorModels.RandomTensors(TestDetectorModels.Small, 31);
        foreach (TestTensor t in tensors)
        {
            if (t.Name == "layout_out.weight" || t.Name == "prob_out.weight")
            {
                Array.Clear(t.Values);
            }
            else if (t.Name == "layout_out.bias")
            {
                Array.Fill(t.Values, 2f);
            }
            else if (t.Name == "prob_out.bias")
            {
                Array.Fill(t.Values, -3f);
            }
        }

        return TestDetectorModels.Random(TestDetectorModels.Small, 0, trainedJson, tensors);
    }

    [Fact]
    public void UntrainedLayoutClassesAreNeverReportedUnlessTheModelSaysTheyAreTrained()
    {
        var (classical, page, _, _) = Setup();
        OcrPage byDefault = classical.WithDetector(ModelContainer.Parse(ConstantMapsModel(null)), KernelPath.Scalar).Ocr(page);
        Assert.Equal(
            [OcrRegionRole.Text, OcrRegionRole.Title, OcrRegionRole.List, OcrRegionRole.Table, OcrRegionRole.Figure, OcrRegionRole.Footer],
            byDefault.Regions.Select(r => r.Role).Select(r => r == OcrRegionRole.Header ? OcrRegionRole.Footer : r).ToArray());
        string all = "[\"text\",\"title\",\"list\",\"table\",\"figure\",\"header_footer\",\"form_field\",\"checkbox\"]";
        OcrPage allTrained = classical.WithDetector(ModelContainer.Parse(ConstantMapsModel(all)), KernelPath.Scalar).Ocr(page);
        Assert.Contains(allTrained.Regions, r => r.Role == OcrRegionRole.FormField);
        Assert.Contains(allTrained.Regions, r => r.Role == OcrRegionRole.Checkbox);
        Assert.Equal(8, allTrained.Regions.Count);
        OcrPage some = classical.WithDetector(ModelContainer.Parse(ConstantMapsModel("[\"table\"]")), KernelPath.Scalar).Ocr(page);
        Assert.Equal([OcrRegionRole.Table], some.Regions.Select(r => r.Role).ToArray());
        Assert.Throws<OcrModelException>(() => DbNetModel.FromContainer(ModelContainer.Parse(ConstantMapsModel("[\"nothing\"]")), OcrLimits.Default));
    }

    [Fact]
    public void PagesFromARandomDetectorAreIdenticalOnEveryPathAndDegree()
    {
        var (classical, page, _, _) = Setup();
        byte[] file = TestDetectorModels.Random(TestDetectorModels.Small, 32);
        var container = ModelContainer.Parse(file);
        string? first = null;
        foreach (KernelPath path in Dispatch.Available())
        {
            OcrEngine engine = OcrEngine.Create(ModelContainer.Parse(TestModels.RandomContainer(TestModels.Small, 5)), null, path).WithDetector(container, path);
            foreach (int degree in new[] { 1, 4 })
            {
                string json = OcrPageJson.ToJson([engine.Ocr(page, new OcrOptions { MaxDegreeOfParallelism = degree })]);
                first ??= json;
                Assert.Equal(first, json);
            }
        }

        _ = classical;
    }

    [Fact]
    public void ADetectorFileIsNotARecognizerAndTheOtherWayRound()
    {
        byte[] detector = TestDetectorModels.Random(TestDetectorModels.Small, 33);
        byte[] recognizer = TestModels.RandomContainer(TestModels.Small, 5);
        Assert.Throws<OcrModelException>(() => OcrEngine.Load(new MemoryStream(detector)));
        OcrEngine engine = OcrEngine.Load(new MemoryStream(recognizer));
        Assert.True(engine.WithDetector(new MemoryStream(detector)).HasDetector);
        Assert.Throws<OcrModelException>(() => engine.WithDetector(new MemoryStream(recognizer)));
        Assert.Throws<ArgumentNullException>(() => engine.WithDetector((string)null!));
        // The recognizer's container rules still demand a positive input height.
        byte[] zeroHeight = TestContainerWriter.Write(
            "crnn-v0", 0, TestModels.Charset(TestModels.Small.Classes), TestModels.ConfigJson(TestModels.Small), TestModels.RandomTensors(TestModels.Small, 5));
        Assert.Throws<OcrModelException>(() => ModelContainer.Parse(zeroHeight));
    }
}
