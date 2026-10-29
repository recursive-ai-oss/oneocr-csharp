using OneImage;
using OneOcr.Kernels;
using OneOcr.Detection;
using OneOcr.Layout;
using OneOcr.Model;

namespace OneOcr.Tests;

public class OrientationTests
{
    private const int W = 1000, H = 700;

    // Text-like lines; about half of the ink pixels, scattered by a hash, are paper again, so glyphs are as thin as the strokes of real text.
    private static byte[] TextPage() => ScatterInk(EngineTests.SyntheticPage(W, H, [.. Enumerable.Range(0, 20).Select(i => 60 + (30 * i))], 6, 8, plain: true).Pixels);

    // Widely spaced thin lines: their profile is so much sharper than the page turned that a sideways copy is turned upright
    // without the (random-weight) recognizer having to read anything with confidence.
    private static byte[] SparseTextPage() => ScatterInk(EngineTests.SyntheticPage(W, H, [.. Enumerable.Range(0, 10).Select(i => 60 + (60 * i))], 4, 8, plain: true).Pixels);

    private static byte[] ScatterInk(byte[] px)
    {
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                if (((x * 73856093) ^ (y * 19349663)) % 100 < 50)
                {
                    px[(y * W) + x] = 255;
                }
            }
        }

        return px;
    }

    // Ink moved down by slope * x (slope in 1/1024), onto a taller canvas: lines of text tilted clockwise by atan(slope / 1024).
    private static (byte[] Pixels, int Height) Shear(byte[] pixels, int width, int height, int slope)
    {
        int extra = Math.Abs(slope) * width / 1024 + 2;
        int h2 = height + extra;
        var result = new byte[width * h2];
        Array.Fill(result, (byte)255);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (pixels[(y * width) + x] < 128)
                {
                    int shift = slope * x / 1024;
                    result[((y + shift + (slope < 0 ? extra : 0)) * width) + x] = pixels[(y * width) + x];
                }
            }
        }

        return (result, h2);
    }

    private static Gray8[] Gray(byte[] pixels) => [.. pixels.Select(b => new Gray8(b))];

    private static Image ToImage(byte[] pixels, int width, int height) => Image.FromPixels<Gray8>(Gray(pixels), width, height);

    private static OcrEngine Engine(KernelPath path = KernelPath.Scalar) =>
        OcrEngine.Create(ModelContainer.Parse(TestModels.RandomContainer(TestModels.Small, 5)), null, path);

    [Theory]
    [InlineData(-120)]
    [InlineData(-30)]
    [InlineData(0)]
    [InlineData(45)]
    [InlineData(200)]
    public void TheSkewSearchFindsTheSlopeOfTiltedLines(int slope)
    {
        (byte[] px, int h) = Shear(TextPage(), W, H, slope);
        (int threshold, bool inkIsBright) = LineFinder.Binarize(px);
        SkewEstimate estimate = SkewEstimator.Estimate(InkGrid.Build(px, W, h, threshold, inkIsBright));
        Assert.True(Math.Abs(estimate.Slope - slope) <= 3, $"{estimate} thr {threshold} bright {inkIsBright}");
    }

    // The page turned clockwise by quarter turns with the engine's shared rotation step, in an array of exactly its size.
    private static Gray8[] Turn(Gray8[] source, int width, int height, int quarterTurns) =>
        quarterTurns == 0
            ? (Gray8[])source.Clone()
            : PageOrientation.Rotate(source, width, height, quarterTurns, new BufferPool<Gray8>(0, 1), out _, out _).AsSpan(0, width * height).ToArray();

    [Fact]
    public void TheTurnedEstimateReadsTheGridWithSwappedAxes()
    {
        byte[] px = TextPage();
        (int threshold, bool inkIsBright) = LineFinder.Binarize(px);
        using InkGrid grid = InkGrid.Build(px, W, H, threshold, inkIsBright);
        using InkGrid copy = grid.TurnClockwise();
        SkewEstimate viaCopy = SkewEstimator.Estimate(copy), viaAxes = SkewEstimator.EstimateTurned(grid);
        // Same cells, same bins: the contrast of the turned page is the same up to the coarse search's block alignment.
        Assert.InRange(viaAxes.Contrast / viaCopy.Contrast, 0.9, 1.1);
    }

    [Fact]
    public void ASidewaysPageHasTheSharperProfileTurnedAQuarter()
    {
        byte[] px = TextPage();
        Gray8[] turned = Turn(Gray(px), W, H, 1);
        byte[] sideways = PixelSpans.AsBytes(turned).ToArray();
        (int threshold, bool inkIsBright) = LineFinder.Binarize(sideways);
        InkGrid grid = InkGrid.Build(sideways, H, W, threshold, inkIsBright);
        SkewEstimate flat = SkewEstimator.Estimate(grid), upright = SkewEstimator.Estimate(grid.TurnClockwise());
        Assert.True(upright.Contrast > 3 * flat.Contrast, $"{upright} vs {flat}");
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 0)]
    [InlineData(2, 0)]
    [InlineData(3, 0)]
    [InlineData(0, 54)]
    [InlineData(1, -120)]
    [InlineData(2, 200)]
    [InlineData(3, 77)]
    public void ThePageFrameMapsPointsBackToTheInput(int quarter, int slope)
    {
        const int w = 300, h = 180;
        // A dark square at a known place on a white page.
        var px = new byte[w * h];
        Array.Fill(px, (byte)255);
        const int dotX = 230, dotY = 40;
        for (int y = dotY; y < dotY + 5; y++)
        {
            for (int x = dotX; x < dotX + 5; x++)
            {
                px[(y * w) + x] = 0;
            }
        }

        var frame = new PageFrame(w, h, quarter, slope);
        Gray8[] turned = Turn(Gray(px), w, h, quarter);
        Gray8[] final = turned;
        int fw = frame.TurnedWidth, fh = frame.TurnedHeight;
        if (slope != 0)
        {
            final = new Gray8[frame.DeskewedWidth * frame.DeskewedHeight];
            PageRotation.Deskew(turned, fw, fh, frame.Sin, frame.Cos, 255, final, frame.DeskewedWidth, frame.DeskewedHeight, 2, CancellationToken.None);
            fw = frame.DeskewedWidth;
            fh = frame.DeskewedHeight;
        }

        // The dark pixels' centre of mass in the final frame, mapped back, is the square's centre.
        double sx = 0, sy = 0, total = 0;
        ReadOnlySpan<byte> bytes = PixelSpans.AsBytes(final.AsSpan(0, fw * fh));
        for (int y = 0; y < fh; y++)
        {
            for (int x = 0; x < fw; x++)
            {
                double weight = 255 - bytes[(y * fw) + x];
                sx += weight * (x + 0.5);
                sy += weight * (y + 0.5);
                total += weight;
            }
        }

        Assert.True(total > 0);
        OcrPoint back = frame.ToInput(sx / total, sy / total);
        Assert.Equal(dotX + 2.5, back.X, 0.6);
        Assert.Equal(dotY + 2.5, back.Y, 0.6);
        Assert.Equal((360 - (90 * quarter)) % 360, frame.Orientation);
    }

    [Fact]
    public void AnUprightPageIsReadExactlyAsBefore()
    {
        Image image = ToImage(TextPage(), W, H);
        OcrPage page = Engine().Ocr(image);
        Assert.Equal(0, page.Orientation);
        Assert.Equal(0, page.SkewDegrees);
        Assert.All(page.Words, w => Assert.Null(w.Quad));
        Assert.DoesNotContain(page.Diagnostics, d => d.Contains("turned", StringComparison.Ordinal) || d.Contains("tilted", StringComparison.Ordinal));
        OcrPage plain = Engine().Ocr(image, new OcrOptions { DetectOrientation = false, Deskew = false });
        Assert.Equal(OcrPageJson.ToJson([plain]), OcrPageJson.ToJson([page]));
    }

    [Fact]
    public void ATiltedPageIsStraightenedAndItsBoxesStayInTheInputImage()
    {
        (byte[] px, int h) = Shear(TextPage(), W, H, 45);
        Image image = ToImage(px, W, h);
        OcrPage page = Engine().Ocr(image);
        Assert.InRange(page.SkewDegrees, 2.3, 2.9);
        Assert.Equal(0, page.Orientation);
        Assert.Equal(W, page.Width);
        Assert.Equal(h, page.Height);
        Assert.NotEmpty(page.Words);
        Assert.All(page.Words, w =>
        {
            Assert.NotNull(w.Quad);
            Assert.InRange(w.Box.Left, -1, W + 1);
            Assert.InRange(w.Box.Top, -1, h + 1);
        });
        // Lines run down to the right: the line's quad top edge descends.
        OcrQuad q = page.Blocks[0].Lines[0].Quad!.Value;
        Assert.True(q.P1.Y > q.P0.Y);
        Assert.Contains(page.Diagnostics, d => d.Contains("straightened", StringComparison.Ordinal));
        // Turned off, nothing is done.
        OcrPage off = Engine().Ocr(image, new OcrOptions { Deskew = false });
        Assert.Equal(0, off.SkewDegrees);
    }

    [Fact]
    public void ASidewaysPageIsTurnedUprightWithQuadsInInputCoordinates()
    {
        Gray8[] turned = Turn(Gray(SparseTextPage()), W, H, 1);
        Image image = Image.FromPixels<Gray8>(turned, H, W);
        OcrPage page = Engine().Ocr(image);
        // The random-weight recognizer cannot tell the two upright candidates apart; the line direction is what is tested.
        Assert.True(page.Orientation is 90 or 270, $"orientation {page.Orientation}");
        Assert.Equal(H, page.Width);
        Assert.Equal(W, page.Height);
        Assert.NotEmpty(page.Blocks);
        double tall = 0, wide = 0;
        foreach (OcrBlock block in page.Blocks)
        {
            foreach (OcrLine line in block.Lines)
            {
                Assert.NotNull(line.Quad);
                tall += line.Box.Height;
                wide += line.Box.Width;
                Assert.InRange(line.Box.Left, -1, H + 1);
                Assert.InRange(line.Box.Bottom, -1, W + 1);
            }
        }

        Assert.True(tall > 3 * wide, "lines are vertical strips in the input");
        Assert.Contains(page.Diagnostics, d => d.Contains("clockwise", StringComparison.Ordinal));
        OcrPage off = Engine().Ocr(image, new OcrOptions { DetectOrientation = false });
        Assert.Equal(0, off.Orientation);
    }

    [Fact]
    public void ResultsAreIdenticalOnEveryPathAndDegree()
    {
        (byte[] px, int h) = Shear(SparseTextPage(), W, H, -60);
        Gray8[] turned = Turn(Gray(px), W, h, 3);
        Image image = Image.FromPixels<Gray8>(turned, h, W);
        string baseline = OcrPageJson.ToJson([Engine().Ocr(image, new OcrOptions { MaxDegreeOfParallelism = 1 })]);
        foreach (KernelPath path in Dispatch.Available())
        {
            OcrEngine engine = Engine(path);
            Assert.Equal(baseline, OcrPageJson.ToJson([engine.Ocr(image, new OcrOptions { MaxDegreeOfParallelism = 1 })]));
            Assert.Equal(baseline, OcrPageJson.ToJson([engine.Ocr(image, new OcrOptions { MaxDegreeOfParallelism = 8 })]));
        }

        Assert.Contains("\"orientation\"", baseline, StringComparison.Ordinal);
        Assert.Contains("\"skew\"", baseline, StringComparison.Ordinal);
    }

    [Fact]
    public void OrientationAndSkewSurviveJsonAndBadValuesAreRejected()
    {
        OcrPage page = SamplePage.Build();
        var turned = new OcrPage(page.Index, page.Width, page.Height, page.UnitsPerInch, page.Blocks, page.Regions, page.Tables, page.KeyValues, page.Diagnostics, page.ToPdf, 270, -3.25);
        string json = OcrPageJson.ToJson([turned]);
        Assert.Contains("\"orientation\":270", json.Replace(" ", "", StringComparison.Ordinal), StringComparison.Ordinal);
        OcrPage back = OcrPageJson.Parse(json)[0];
        Assert.Equal(270, back.Orientation);
        Assert.Equal(-3.25, back.SkewDegrees, 3);
        Assert.Equal(json, OcrPageJson.ToJson([back]));
        // Absent when zero, so pages that need no turn keep their bytes.
        Assert.DoesNotContain("orientation", OcrPageJson.ToJson([page]), StringComparison.Ordinal);
        Assert.Throws<OcrFormatException>(() => OcrPageJson.Parse(json.Replace("\"orientation\":270", "\"orientation\":45", StringComparison.Ordinal)));
        Assert.Throws<OcrFormatException>(() => OcrPageJson.Parse(json.Replace("\"skew\":-3.25", "\"skew\":80", StringComparison.Ordinal)));
        Assert.Throws<ArgumentException>(() => new OcrPage(0, 1, 1, 72, [], null, null, null, null, null, 45, 0));
        Assert.Throws<ArgumentException>(() => new OcrPage(0, 1, 1, 72, [], null, null, null, null, null, 0, double.NaN));
    }

    private sealed class HeadStub(Func<int, PageDetection> detect) : IPageDetector
    {
        public List<(int Width, int Height)> Calls { get; } = [];

        public PageDetection Detect(
            ReadOnlySpan<Gray8> page, int width, int height, BufferPool<Gray8> pixels, ResizerPool resizers, int degree, CancellationToken cancellationToken)
        {
            Calls.Add((width, height));
            return detect(Calls.Count - 1);
        }
    }

    private static PageDetection Head(int quarter, float confidence) =>
        new() { Lines = [], Regions = [], Quarter = quarter, OrientationConfidence = confidence };

    [Fact]
    public void WithADetectorTheHeadDecidesAndTheClassicalDecisionIsSkipped()
    {
        // A sideways page: the classical decision would turn it, the head says it is upright and sure, so it is read as it is.
        Gray8[] turned = Turn(Gray(SparseTextPage()), W, H, 1);
        Image image = Image.FromPixels<Gray8>(turned, H, W);
        var stub = new HeadStub(_ => Head(0, 0.99f));
        OcrPage page = Engine().WithDetector(stub).Ocr(image, new OcrOptions { SearchAngle = false });
        Assert.Single(stub.Calls);
        Assert.Equal(0, page.Orientation);
        Assert.Equal((H, W), (page.Width, page.Height));
        Assert.DoesNotContain(page.Diagnostics, d => d.Contains("turned", StringComparison.Ordinal));

        // The same page without a detector goes through the classical decision.
        Assert.True(Engine().Ocr(image).Orientation is 90 or 270);

        // Detection of orientation off: the head is not acted on either.
        var sure = new HeadStub(_ => Head(1, 0.99f));
        OcrPage off = Engine().WithDetector(sure).Ocr(image, new OcrOptions { DetectOrientation = false, SearchAngle = false });
        Assert.Single(sure.Calls);
        Assert.DoesNotContain(off.Diagnostics, d => d.Contains("turned", StringComparison.Ordinal));
    }

    [Fact]
    public void WithADetectorTheHeadTurnsAndTheDeskewComposesOnTheTurnedPage()
    {
        // Text tilted by about 3.4 degrees, then the page turned three quarters clockwise; the head undoes it with one more.
        (byte[] sheared, int h) = Shear(TextPage(), W, H, -60);
        Gray8[] sideways = Turn(Gray(sheared), W, h, 3);
        Image image = Image.FromPixels<Gray8>(sideways, h, W);
        var stub = new HeadStub(call => call == 0 ? Head(1, 0.97f) : Head(0, 0.99f));
        OcrPage page = Engine().WithDetector(stub, confirmTurns: false).Ocr(image, new OcrOptions { SearchAngle = false });

        // First look at the page as given, then a look at the page turned and straightened (a larger canvas).
        Assert.Equal(2, stub.Calls.Count);
        Assert.Equal((h, W), stub.Calls[0]);
        Assert.True(stub.Calls[1].Width > W && stub.Calls[1].Height >= h);
        // The page model is in the turned (upright) frame, as without deskew; the tilt is reported, the quarter turn is not.
        Assert.Equal((W, h), (page.Width, page.Height));
        Assert.Equal(0, page.Orientation);
        Assert.NotEqual(0, page.SkewDegrees);
        Assert.Contains(page.Diagnostics, d => d.Contains("90 degrees clockwise", StringComparison.Ordinal) && d.Contains("0.97", StringComparison.Ordinal));
        Assert.Contains(page.Diagnostics, d => d.Contains("straightened", StringComparison.Ordinal));
    }
}
