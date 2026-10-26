using System.Numerics;
using OneImage;
using OneImage.Text;
using OneOcr.Detection;
using OneOcr.Kernels;
using OneOcr.Layout;

namespace OneOcr.Tests;

/// <summary>Lines read along their own axis: the axis the detector's text map gives, the straightened crop, the quads, and a tilted rendered line read exactly.</summary>
public class TiltedLineTests
{
    private sealed class StubDetector(PageDetection detection) : IPageDetector
    {
        public PageDetection Detect(
            ReadOnlySpan<Gray8> page, int width, int height, BufferPool<Gray8> pixels, ResizerPool resizers, int degree, CancellationToken cancellationToken) => detection;
    }

    // A band of text-map cells that leans by `slope` cells per cell, `half` cells thick on each side of its centre line.
    private static float[] Band(int width, int height, double slope, int half, int x0 = 4, int x1 = -1)
    {
        var map = new float[width * height];
        Array.Fill(map, -6f);
        x1 = x1 < 0 ? width - 4 : x1;
        for (int x = x0; x < x1; x++)
        {
            int cy = (int)Math.Round((height / 2.0) + (slope * (x - (width / 2.0))), MidpointRounding.ToEven);
            for (int y = cy - half; y <= cy + half; y++)
            {
                map[(y * width) + x] = 4f;
            }
        }

        return map;
    }

    private static PageDetection Decode(float[] prob, int w, int h) =>
        DetectorPostprocess.Decode(prob, w, h, new float[16], 1, 1, 16, TestDetectorModels.AllClasses, new bool[8], new float[4], 1.0, 2 * w, 2 * h, OcrLimits.Default);

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.02)]
    [InlineData(-0.035)]
    public void TheAxisOfALineComponentFollowsItsLean(double slope)
    {
        PageDetection d = Decode(Band(160, 60, slope, 2), 160, 60);
        DetectedLine line = Assert.Single(d.Lines);
        LineAxis axis = line.Axis!.Value;
        double expected = slope / Math.Sqrt(1 + (slope * slope));
        Assert.InRange(axis.Sin, expected - 0.003, expected + 0.003);
        Assert.True(axis.Cos > 0.99);
        Assert.True(axis.Elongation > 100);
        // The line frame's extent along the line is about the component's length (140 cells of 2 pixels), across it about 5 cells.
        Assert.InRange(axis.U1 - axis.U0, 280, 340);
        Assert.InRange(axis.V1 - axis.V0, 10, 40);
        Assert.InRange(axis.CenterX, 158, 164);
        Assert.InRange(axis.CenterY, 58, 64);
    }

    [Fact]
    public void ABlobAndATinyComponentHaveNoUsableAxis()
    {
        var blob = new float[40 * 40];
        Array.Fill(blob, -6f);
        for (int y = 10; y < 30; y++)
        {
            for (int x = 10; x < 30; x++)
            {
                blob[(y * 40) + x] = 4f;
            }
        }

        LineAxis? blobAxis = Assert.Single(Decode(blob, 40, 40).Lines).Axis;
        Assert.False(blobAxis is { } b && TiltedLines.IsTilted(b));
        var tiny = new float[40 * 40];
        Array.Fill(tiny, -6f);
        tiny[(20 * 40) + 20] = 4f;
        tiny[(20 * 40) + 21] = 4f;
        Assert.Null(Assert.Single(Decode(tiny, 40, 40).Lines).Axis);
    }

    [Fact]
    public void ShortThickAndNearlyStraightAxesAreNotTilted()
    {
        LineAxis Make(double angleDegrees, double length, double thickness, int cells = 100, double elongation = 500)
        {
            double a = angleDegrees * Math.PI / 180;
            return new LineAxis(0, 0, Math.Cos(a), Math.Sin(a), -length / 2, length / 2, -thickness / 2, thickness / 2, cells, elongation);
        }

        Assert.True(TiltedLines.IsTilted(Make(1.5, 800, 40)));
        Assert.False(TiltedLines.IsTilted(Make(0.1, 800, 40)));
        Assert.False(TiltedLines.IsTilted(Make(1.5, 800, 40, cells: 5)));
        Assert.False(TiltedLines.IsTilted(Make(1.5, 80, 40)));
        Assert.False(TiltedLines.IsTilted(Make(1.5, 800, 40, elongation: 8)));
        Assert.False(TiltedLines.IsTilted(Make(40, 800, 40)));
        // The noise floor grows with thickness over length: 1 degree is believed on a long thin line, not on a short thick one.
        Assert.True(TiltedLines.IsTilted(Make(1.0, 1000, 40)));
        Assert.False(TiltedLines.IsTilted(Make(1.0, 300, 40)));
    }

    // A page of bars (a stand-in for text lines) leaning by the given angle about (cx, cy): the upright page sampled through the inverse turn.
    private static byte[] Lean(byte[] upright, int width, int height, double degrees, double cx, double cy)
    {
        double a = degrees * Math.PI / 180, c = Math.Cos(a), s = Math.Sin(a);
        var result = new byte[width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
                double sx = cx + (dx * c) + (dy * s), sy = cy - (dx * s) + (dy * c);
                result[(y * width) + x] = TiltedLines.Sample(upright, width, height, sx, sy);
            }
        }

        return result;
    }

    [Fact]
    public void ACropAlongTheAxisHasHorizontalStrokes()
    {
        const int W = 600, H = 200;
        var upright = new byte[W * H];
        Array.Fill(upright, (byte)255);
        // Two thick horizontal strokes 60 pixels apart, long enough to show a lean.
        for (int x = 50; x < 550; x++)
        {
            for (int y = 96; y < 104; y++)
            {
                upright[(y * W) + x] = 0;
                upright[((y + 40) * W) + x] = 0;
            }
        }

        double degrees = 2.0, a = degrees * Math.PI / 180;
        byte[] tilted = Lean(upright, W, H, degrees, W / 2.0, H / 2.0);
        // The strokes now lean by about 17 pixels end to end.
        var column = (int x) => Enumerable.Range(0, H).Where(y => tilted[(y * W) + x] < 128).ToArray();
        Assert.True(Math.Abs(column(60).Min() - column(540).Min()) > 12);

        var pool = new BufferPool<Gray8>(1 << 24, 1);
        var resizers = new ResizerPool();
        var tilt = new TiltedLine(W / 2.0, H / 2.0, Math.Cos(a), Math.Sin(a), new LineBox(-250, -60, 500, 120));
        Gray8[] crop = TiltedLines.Prepare(tilted, W, H, tilt, new LineBox(-250, -60, 500, 120), 120, pool, resizers, out int width);
        Assert.Equal(500, width);
        // In the straightened crop every column of a stroke starts on the same rows, to within a pixel (the sampling's own blur).
        int[] starts = [.. Enumerable.Range(20, width - 40).Select(x => Enumerable.Range(0, 120).First(y => crop[(y * width) + x].V < 128))];
        Assert.InRange(starts.Max() - starts.Min(), 0, 2);
        pool.Return(crop);
    }

    [Fact]
    public void TheCornersOfAQuadAreMappedThroughTheFrame()
    {
        var frame = new PageFrame(400, 300, 0, 40);
        var word = new OcrWord("a", new OcrBox(10, 10, 50, 30), 1f, 1f, OcrSource.Ocr, [], new OcrQuad(new OcrPoint(10, 12), new OcrPoint(50, 10), new OcrPoint(50, 30), new OcrPoint(10, 32)));
        var line = new OcrLine("a", [word], [0], word.Box, 1f, OcrSource.Ocr, Quad: word.Quad);
        var page = new OcrPage(0, frame.DeskewedWidth, frame.DeskewedHeight, 72, [new OcrBlock(OcrBlockRole.Body, false, word.Box, [line], 1f, OcrSource.Ocr)]);
        OcrPage mapped = frame.MapPage(page, []);
        OcrQuad q = mapped.Blocks[0].Lines[0].Words[0].Quad!.Value;
        Assert.Equal(frame.ToInput(10, 12), q.P0);
        Assert.Equal(frame.ToInput(50, 10), q.P1);
        Assert.Equal(frame.ToInput(50, 30), q.P2);
        Assert.Equal(frame.ToInput(10, 32), q.P3);
        // A word with no quad of its own gets its box's corners, as before.
        var plain = word with { Quad = null };
        var line2 = line with { Words = [plain], Quad = null };
        var page2 = new OcrPage(0, frame.DeskewedWidth, frame.DeskewedHeight, 72, [page.Blocks[0] with { Lines = [line2] }]);
        Assert.Equal(frame.ToInput(10, 10), frame.MapPage(page2, []).Blocks[0].Lines[0].Words[0].Quad!.Value.P0);
    }

    private static OcrEngine? Engine()
    {
        string? model = Environment.GetEnvironmentVariable("ONEOCR_TABLE_MODEL");
        if (model is null || !File.Exists(model))
        {
            return null;
        }

        using FileStream stream = File.OpenRead(model);
        return OcrEngine.Load(stream);
    }

    private static string? FontFile()
    {
        string path = Environment.GetEnvironmentVariable("ONEOCR_TEST_FONT") ?? "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
        return File.Exists(path) ? path : null;
    }

    [Fact]
    public void ATiltedRenderedLineIsReadExactlyAlongItsAxisAndCarriesQuads()
    {
        OcrEngine? engine = Engine();
        string? font = FontFile();
        Assert.SkipWhen(engine is null || font is null, "ONEOCR_TABLE_MODEL (a recognizer container) or the test font is not available.");
        const int W = 1500, H = 300;
        const string Text = "Invoice 4711 was issued on 12 March 2024 to Acme Holdings Ltd";
        var px = new Rgba32[W * H];
        Array.Fill(px, new Rgba32(255, 255, 255, 255));
        Image drawn = Image.FromPixels<Rgba32>(px, W, H)
            .DrawText(Text, new TextOptions(new FontCollection().Add(font!).CreateFont(32)) { Origin = new Vector2(60, 120) }, new Rgba32(0, 0, 0, 255));
        var gray = new Gray8[W * H];
        drawn.CopyPixelsTo<Gray8>(gray, new ImageRect(0, 0, W, H), 1);
        byte[] upright = [.. gray.Select(g => g.V)];

        Assert.Equal(Text, engine!.Ocr(drawn, new OcrOptions { DetectOrientation = false, Deskew = false, MaxDegreeOfParallelism = 1 }).Text.Trim());
        const double Degrees = 1.8;
        double a = Degrees * Math.PI / 180;
        byte[] lean = Lean(upright, W, H, Degrees, W / 2.0, H / 2.0);
        Image page = Image.FromPixels<Gray8>([.. lean.Select(b => new Gray8(b))], W, H);

        // The detector's view: the line's centre (the text's middle turned about the page centre) and its axis.
        double tx = 60 + 600, ty = 140;
        double dx = tx - (W / 2.0), dy = ty - (H / 2.0);
        double cx = (W / 2.0) + (dx * Math.Cos(a)) - (dy * Math.Sin(a)), cy = (H / 2.0) + (dx * Math.Sin(a)) + (dy * Math.Cos(a));
        var axis = new LineAxis(cx, cy, Math.Cos(a), Math.Sin(a), -640, 640, -40, 40, 400, 800);
        var around = new DetectedLine(20, 60, 1380, 260, 0.9f, axis);
        PageDetection Detection(DetectedLine l) => new() { Lines = [l], Regions = [], Quarter = 0, OrientationConfidence = 0.99f };
        var options = new OcrOptions { DetectOrientation = false, Deskew = false, MaxDegreeOfParallelism = 1, IncludeChars = true };

        OcrPage along = engine!.WithDetector(new StubDetector(Detection(around))).Ocr(page, options);
        Assert.Equal(Text, along.Text.Trim());
        OcrLine line = Assert.Single(along.Blocks.SelectMany(b => b.Lines));
        Assert.NotNull(line.Quad);
        foreach (OcrWord w in line.Words)
        {
            OcrQuad q = w.Quad!.Value;
            // The top edge leans like the text, and the box is the box around the quad, inside the page.
            Assert.Equal(Math.Tan(a), (q.P1.Y - q.P0.Y) / (q.P1.X - q.P0.X), 6);
            Assert.Equal(Math.Min(Math.Min(q.P0.X, q.P1.X), Math.Min(q.P2.X, q.P3.X)), w.Box.Left, 6);
            Assert.Equal(Math.Max(Math.Max(q.P0.Y, q.P1.Y), Math.Max(q.P2.Y, q.P3.Y)), w.Box.Bottom, 6);
            Assert.InRange(w.Box.Left, 0, W);
            Assert.InRange(w.Box.Bottom, 0, H);
            Assert.All(w.Chars, c => Assert.NotNull(c.Quad));
        }

        // The words follow the lean: the last one sits lower than the first by about the line's length times the tangent.
        Assert.InRange(line.Words[^1].Box.Top - line.Words[0].Box.Top, 25, 40);

        // The same line without an axis (the axis-aligned path) is not read exactly, and carries no quad.
        OcrPage flat = engine.WithDetector(new StubDetector(Detection(around with { Axis = null }))).Ocr(page, options);
        Assert.NotEqual(Text, flat.Text.Trim());
        Assert.All(flat.Words, w => Assert.Null(w.Quad));

        // The result does not depend on the degree of parallelism, and survives the page JSON.
        OcrPage again = engine.WithDetector(new StubDetector(Detection(around))).Ocr(page, new OcrOptions { DetectOrientation = false, Deskew = false, MaxDegreeOfParallelism = 4, IncludeChars = true });
        Assert.Equal(OcrPageJson.ToJson([along]), OcrPageJson.ToJson([again]));
        OcrPage read = OcrPageJson.Parse(OcrPageJson.ToJson([along]))[0];
        // The page JSON rounds to three decimals.
        Assert.Equal(line.Words[0].Quad!.Value.P0.X, read.Words[0].Quad!.Value.P0.X, 3);
        Assert.Equal(line.Words[0].Quad!.Value.P2.Y, read.Words[0].Quad!.Value.P2.Y, 3);
    }
}
