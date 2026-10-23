using OneImage;
using OneOcr.Kernels;
using OneOcr.Layout;
using OneOcr.Model;

namespace OneOcr.Tests;

public class EngineTests
{
    /// <summary>
    /// A page of text-like lines: each line is a row of glyph-like blocks (x-height body, some with an ascender above and some
    /// with a descender below), in black on white. Returns the pixels and the planned ink boxes.
    /// </summary>
    internal static (byte[] Pixels, List<LineBox> Planned) SyntheticPage(int width, int height, int[] tops, int body, int pitchGap, bool plain = false, int seed = 1)
    {
        var rng = new Random(seed);
        var px = new byte[width * height];
        Array.Fill(px, (byte)255);
        var planned = new List<LineBox>();
        foreach (int top in tops)
        {
            int asc = body, desc = body / 2;
            int x = 100;
            int minX = int.MaxValue, maxX = 0;
            int y0 = top + asc, y1 = top + asc + body;
            while (x < width - 200)
            {
                int w = 6 + rng.Next(8);
                bool hasAsc = !plain && rng.Next(3) == 0, hasDesc = !plain && rng.Next(4) == 0;
                int yTop = hasAsc ? top : y0;
                int yBottom = hasDesc ? y1 + desc : y1;
                for (int y = yTop; y < yBottom; y++)
                {
                    for (int xx = x; xx < x + w; xx++)
                    {
                        px[(y * width) + xx] = 20;
                    }
                }

                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x + w - 1);
                x += w + 2 + (rng.Next(5) == 0 ? pitchGap : 0);
            }

            // Make sure the planned box is the real ink box: one block with an ascender and one with a descender.
            for (int y = top; y < top + asc; y++) { for (int c = 100; c < 103; c++) { px[(y * width) + c] = 20; } }
            for (int y = y1; y < y1 + desc; y++) { for (int c = 104; c < 107; c++) { px[(y * width) + c] = 20; } }
            planned.Add(new LineBox(minX, top, maxX - minX + 1, asc + body + desc));
        }

        return (px, planned);
    }

    [Fact]
    public void FindsEveryPlannedLineTopToBottom()
    {
        int body = 14;
        int[] tops = [100, 190, 280, 370, 460];
        var (px, planned) = SyntheticPage(1200, 600, tops, body, 8);
        List<LineBox> found = LineFinder.Find(px, 1200, 600, 100);
        Assert.Equal(planned.Count, found.Count);
        for (int i = 0; i < found.Count; i++)
        {
            Assert.Equal(planned[i].Y, found[i].Y);
            Assert.Equal(planned[i].Height, found[i].Height);
            Assert.Equal(planned[i].X, found[i].X);
            Assert.Equal(planned[i].Width, found[i].Width);
        }
    }

    [Fact]
    public void SplitsLinesWhoseDescendersOverlapTheNextLinesAscenders()
    {
        int body = 14;
        int lineHeight = body + body + (body / 2);
        // The third line starts three rows above the bottom of the second: the rows where a descender column of one line and an
        // ascender column of the next line overlap never drop to zero ink.
        int[] tops = [100, 200, 200 + lineHeight - 3, 330, 400];
        var (px, planned) = SyntheticPage(1200, 500, tops, body, 8, plain: true);
        List<LineBox> found = LineFinder.Find(px, 1200, 500, 100);
        Assert.Equal(planned.Count, found.Count);
    }

    [Fact]
    public void KeepsALargeHeadingAsOneLine()
    {
        // Same glyph blocks, drawn twice as tall: a heading among normal lines.
        var (small, _) = SyntheticPage(1200, 700, [300, 390, 480, 570], 14, 8);
        var (big, plannedBig) = SyntheticPage(1200, 700, [60], 28, 16, seed: 5);
        for (int i = 0; i < small.Length; i++)
        {
            if (big[i] < 255)
            {
                small[i] = big[i];
            }
        }

        List<LineBox> found = LineFinder.Find(small, 1200, 700, 100);
        Assert.Equal(5, found.Count);
        Assert.Equal(plannedBig[0].Y, found[0].Y);
        Assert.Equal(plannedBig[0].Height, found[0].Height);
    }

    [Fact]
    public void LightOnDarkAndNoisyPagesGiveTheSameLines()
    {
        var (px, planned) = SyntheticPage(1200, 600, [100, 190, 280], 14, 8);
        var inverted = px.Select(p => (byte)(255 - p)).ToArray();
        Assert.Equal(planned.Count, LineFinder.Find(inverted, 1200, 600, 100).Count);

        var rng = new Random(3);
        var noisy = (byte[])px.Clone();
        for (int i = 0; i < noisy.Length; i++)
        {
            noisy[i] = (byte)Math.Clamp(noisy[i] + rng.Next(-12, 13), 0, 255);
        }

        Assert.Equal(planned.Count, LineFinder.Find(noisy, 1200, 600, 100).Count);
    }

    [Fact]
    public void BlankPagesAndLimitsAreHandled()
    {
        Assert.Empty(LineFinder.Find(new byte[100 * 100], 100, 100, 10));
        var white = new byte[100 * 100];
        Array.Fill(white, (byte)255);
        Assert.Empty(LineFinder.Find(white, 100, 100, 10));
        var (px, _) = SyntheticPage(1200, 600, [100, 190, 280], 14, 8);
        Assert.Throws<OcrLimitException>(() => LineFinder.Find(px, 1200, 600, 2));
        Assert.Throws<ArgumentException>(() => LineFinder.Find(new byte[10], 4, 4, 10));
    }

    [Fact]
    public void CropBoxSurroundsTheInkAndStaysOnThePage()
    {
        var line = new LineBox(100, 200, 500, 20);
        LineBox crop = LinePreprocessor.CropBox(line, 20, 1000, 1000);
        Assert.Equal(1.5 * 20, crop.Height, 1.0);
        Assert.Equal(100 - 10, crop.X);
        Assert.Equal(500 + 20, crop.Width);
        // A short line (digits only) is cropped as tall as the page's typical line, centred on its ink.
        LineBox shortLine = LinePreprocessor.CropBox(new LineBox(5, 300, 50, 10), 20, 1000, 1000);
        Assert.Equal(0, shortLine.X);
        Assert.InRange(shortLine.Height, 25, 27);
        Assert.Equal(1, LinePreprocessor.CropBox(new LineBox(0, 0, 1, 1), 1, 1, 1).Width);
    }

    private static (byte[] File, Image Page) Setup(out TestModels.Shape shape)
    {
        shape = TestModels.Small;
        var (px, _) = SyntheticPage(900, 420, [60, 130, 200, 270], 14, 8);
        Image page = Image.FromPixels<Gray8>(px.Select(b => new Gray8(b)).ToArray(), 900, 420);
        return (TestModels.RandomContainer(shape, 5), page);
    }

    [Fact]
    public void ToTextIsIdenticalAtEveryDegreeAndOnBothPaths()
    {
        var (file, page) = Setup(out _);
        var container = ModelContainer.Parse(file);
        var scalar = OcrEngine.Create(container, null, KernelPath.Scalar);
        string baseline = scalar.ToText(page, 1);
        Assert.Equal(3, baseline.Count(c => c == '\n'));
        Assert.Equal(baseline, scalar.ToText(page, 4));
        Assert.Equal(baseline, scalar.ToText(page));
        if (Dispatch.Avx2Available)
        {
            var avx2 = OcrEngine.Create(container, null, KernelPath.Avx2);
            Assert.Equal(baseline, avx2.ToText(page, 1));
            Assert.Equal(baseline, avx2.ToText(page, 8));
        }

        Assert.Equal(baseline, page.ToText(scalar));
    }

    [Fact]
    public void DefaultEngineMustBeConfiguredAndOnlyOnce()
    {
        var (file, page) = Setup(out _);
        Assert.Throws<OcrModelException>(() => page.ToText());
        using var stream = new MemoryStream(file);
        var engine = OcrEngine.Load(stream);
        OneOcr.Configure(engine);
        Assert.Same(engine, OneOcr.Default);
        Assert.Equal(engine.ToText(page, 1), page.ToText());
        Assert.Throws<InvalidOperationException>(() => OneOcr.Configure(engine));
    }

    [Fact]
    public void ToTextRejectsOversizedImagesBeforeReadingPixels()
    {
        var (file, page) = Setup(out _);
        var engine = OcrEngine.Create(ModelContainer.Parse(file), new OcrLimits { MaxImagePixels = 1000 }, KernelPath.Scalar);
        Assert.Throws<OcrLimitException>(() => engine.ToText(page));
        Assert.Throws<ArgumentNullException>(() => engine.ToText(null!));
    }

    [Fact]
    public void BlankImageGivesEmptyText()
    {
        var (file, _) = Setup(out _);
        var engine = OcrEngine.Create(ModelContainer.Parse(file), null, KernelPath.Scalar);
        Image blank = Image.FromPixels<Gray8>(Enumerable.Repeat(new Gray8(255), 300 * 200).ToArray(), 300, 200);
        Assert.Equal(string.Empty, engine.ToText(blank));
    }
}
