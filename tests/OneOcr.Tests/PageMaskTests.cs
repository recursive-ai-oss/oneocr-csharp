using OneImage;
using OneOcr.Layout;

namespace OneOcr.Tests;

public class PageMaskTests
{
    [Fact]
    public void PackedValuesComeBackAtEveryOffset()
    {
        var rng = new Random(3);
        foreach ((int w, int h) in new[] { (1, 1), (3, 5), (4, 4), (7, 9), (130, 17), (257, 3) })
        {
            var reference = new byte[w * h];
            var mask = new PageMask(new byte[PageMask.BytesFor(w, h)], w, h);
            for (int i = 0; i < w * h * 2; i++)
            {
                int x = rng.Next(w), y = rng.Next(h);
                byte v = (byte)rng.Next(3);
                mask.Set(x, y, v);
                reference[(y * w) + x] = v;
            }

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    Assert.Equal(reference[(y * w) + x], mask.Get(x, y));
                }

                for (int trial = 0; trial < 20; trial++)
                {
                    int from = rng.Next(w), count = rng.Next(w - from + 1);
                    var row = new byte[count];
                    mask.ReadRow(y, from, count, row);
                    Assert.True(reference.AsSpan((y * w) + from, count).SequenceEqual(row));
                }
            }
        }
    }

    [Fact]
    public void StoredInkRowsReplaceTheWholeRowAndRowsDoNotShareBytes()
    {
        int w = 11, h = 3;
        var mask = new PageMask(new byte[PageMask.BytesFor(w, h)], w, h);
        mask.SetRun(1, 0, 11, PageMask.Rule);
        var ink = new byte[] { 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1 };
        mask.StoreInkRow(1, ink);
        var back = new byte[w];
        mask.ReadRow(1, 0, w, back);
        Assert.Equal(ink, back);
        mask.ReadRow(0, 0, w, back);
        Assert.All(back, b => Assert.Equal(0, b));
        mask.ReadRow(2, 0, w, back);
        Assert.All(back, b => Assert.Equal(0, b));
        Assert.Throws<ArgumentException>(() => new PageMask(new byte[3], 40, 4));
    }

    // The ink mask the way a plain scan builds it: one byte per pixel, ruling lines (runs of ink of at least `longRun` pixels
    // along a row or a column) marked 2.
    private static byte[] ReferenceMask(byte[] gray, int w, int h, int threshold, bool inkIsBright, out bool rules)
    {
        int longRun = Math.Max(48, Math.Min(w, h) / 16);
        var ink = new bool[w * h];
        for (int i = 0; i < ink.Length; i++)
        {
            ink[i] = inkIsBright ? gray[i] > threshold : gray[i] <= threshold;
        }

        var mask = new byte[w * h];
        for (int i = 0; i < mask.Length; i++)
        {
            mask[i] = ink[i] ? (byte)1 : (byte)0;
        }

        rules = false;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w;)
            {
                if (!ink[(y * w) + x])
                {
                    x++;
                    continue;
                }

                int e = x;
                while (e < w && ink[(y * w) + e])
                {
                    e++;
                }

                if (e - x >= longRun)
                {
                    for (int k = x; k < e; k++)
                    {
                        mask[(y * w) + k] = 2;
                    }

                    rules = true;
                }

                x = e;
            }
        }

        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h;)
            {
                if (!ink[(y * w) + x])
                {
                    y++;
                    continue;
                }

                int e = y;
                while (e < h && ink[(e * w) + x])
                {
                    e++;
                }

                if (e - y >= longRun)
                {
                    for (int k = y; k < e; k++)
                    {
                        mask[(k * w) + x] = 2;
                    }

                    rules = true;
                }

                y = e;
            }
        }

        return mask;
    }

    [Theory]
    [InlineData(1, true)]
    [InlineData(2, true)]
    [InlineData(4, false)]
    [InlineData(8, true)]
    public void MaskWithRulesEqualsAPlainScanAtEveryDegree(int degree, bool inkIsBright)
    {
        var rng = new Random(degree * 17);
        foreach ((int w, int h) in new[] { (203, 157), (300, 400), (64, 1000), (1000, 70), (51, 51) })
        {
            // Text-like blocks, a few ruling lines of different lengths, some that cross band edges, and noise.
            var gray = new byte[w * h];
            Array.Fill(gray, inkIsBright ? (byte)0 : (byte)255);
            byte inkValue = inkIsBright ? (byte)255 : (byte)0;
            for (int b = 0; b < 40; b++)
            {
                int bx = rng.Next(w), by = rng.Next(h), bw = 1 + rng.Next(20), bh = 1 + rng.Next(12);
                for (int y = by; y < Math.Min(h, by + bh); y++)
                {
                    for (int x = bx; x < Math.Min(w, bx + bw); x++)
                    {
                        gray[(y * w) + x] = inkValue;
                    }
                }
            }

            for (int r = 0; r < 4; r++)
            {
                int pos = rng.Next(Math.Min(w, h)), from = rng.Next(w / 2), to = from + 40 + rng.Next(w);
                for (int k = from; k < Math.Min(w, to); k++)
                {
                    gray[(pos % h * w) + k] = inkValue;
                }

                int fromY = rng.Next(h / 2), toY = fromY + 40 + rng.Next(h);
                for (int k = fromY; k < Math.Min(h, toY); k++)
                {
                    gray[(k * w) + (pos % w)] = inkValue;
                }
            }

            var pixels = gray.Select(v => new Gray8(v)).ToArray();
            int threshold = 128;
            byte[] expected = ReferenceMask(gray, w, h, threshold, inkIsBright, out bool expectedRules);
            var mask = new PageMask(new byte[PageMask.BytesFor(w, h)], w, h);
            bool rules = StripFinder.BuildMask(pixels, w, h, threshold, inkIsBright, mask, degree);
            Assert.Equal(expectedRules, rules);
            var row = new byte[w];
            for (int y = 0; y < h; y++)
            {
                mask.ReadRow(y, 0, w, row);
                Assert.True(expected.AsSpan(y * w, w).SequenceEqual(row), $"row {y} of {w}x{h} at degree {degree}");
            }
        }
    }

    [Fact]
    public void ThresholdDoesNotDependOnTheDegree()
    {
        var rng = new Random(9);
        int count = (3 << 20) + 12345;
        var bytes = new byte[count];
        for (int i = 0; i < count; i++)
        {
            bytes[i] = (byte)(rng.Next(5) == 0 ? rng.Next(60) : 200 + rng.Next(50));
        }

        Gray8[] pixels = bytes.Select(b => new Gray8(b)).ToArray();
        var expected = LineFinder.Binarize(bytes);
        foreach (int degree in new[] { 1, 2, 4, 8 })
        {
            Assert.Equal(expected, LineFinder.Binarize(pixels, count, degree));
        }
    }

    [Fact]
    public void LinesInAMaskAreTheLinesOfTheSamePixels()
    {
        // The single-column path reads the mask instead of the page; both must find the same lines.
        var (px, _) = EngineTests.SyntheticPage(900, 420, [60, 130, 200, 270], 14, 8);
        List<LineBox> fromPage = LineFinder.Find(px, 900, 420, 4096);
        (int threshold, bool inkIsBright) = LineFinder.Binarize(px);
        var mask = new PageMask(new byte[PageMask.BytesFor(900, 420)], 900, 420);
        bool rules = StripFinder.BuildMask(px.Select(b => new Gray8(b)).ToArray(), 900, 420, threshold, inkIsBright, mask, 1);
        Assert.False(rules);
        Assert.Equal(fromPage, LineFinder.FindInMask(mask, new LineBox(0, 0, 900, 420), 4096));
    }
}
