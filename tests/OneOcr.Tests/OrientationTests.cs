using OneImage;
using OneOcr.Kernels;
using OneOcr.Layout;
using OneOcr.Model;

namespace OneOcr.Tests;

public class OrientationTests
{
    private const int W = 1000, H = 700;

    // Text-like lines; about half of the ink pixels, scattered by a hash, are paper again, so glyphs are as thin as the strokes of real text.
    private static byte[] TextPage()
    {
        byte[] px = EngineTests.SyntheticPage(W, H, [.. Enumerable.Range(0, 20).Select(i => 60 + (30 * i))], 6, 8, plain: true).Pixels;
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

    [Fact]
    public void ASidewaysPageHasTheSharperProfileTurnedAQuarter()
    {
        byte[] px = TextPage();
        var turned = new Gray8[W * H];
        PageRotation.QuarterTurn(Gray(px), W, H, 1, turned, 1, CancellationToken.None);
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
        var turned = new Gray8[w * h];
        PageRotation.QuarterTurn(Gray(px), w, h, quarter, turned, 1, CancellationToken.None);
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
        var turned = new Gray8[W * H];
        PageRotation.QuarterTurn(Gray(TextPage()), W, H, 1, turned, 1, CancellationToken.None);
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
        (byte[] px, int h) = Shear(TextPage(), W, H, -60);
        var turned = new Gray8[W * h];
        PageRotation.QuarterTurn(Gray(px), W, h, 3, turned, 1, CancellationToken.None);
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
}
