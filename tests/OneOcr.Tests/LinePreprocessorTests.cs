using OneImage;
using OneOcr.Layout;

namespace OneOcr.Tests;

public class LinePreprocessorTests
{
    // The prepared line must be the bytes of Image.Resize(height:) on the cropped pixels: that call is what the recognizer was trained with.
    [Fact]
    public void PreparedLinesEqualImageResize()
    {
        var rng = new Random(11);
        int w = 400, h = 120;
        var page = new byte[w * h];
        rng.NextBytes(page);
        var pool = new BufferPool<Gray8>(1L << 26, 1);
        var resizers = new ResizerPool();
        foreach (var box in new[] { new LineBox(3, 5, 301, 47), new LineBox(0, 0, 400, 120), new LineBox(10, 30, 7, 19), new LineBox(50, 10, 120, 12), new LineBox(0, 90, 399, 30) })
        {
            var crop = new Gray8[box.Width * box.Height];
            for (int y = 0; y < box.Height; y++)
            {
                for (int x = 0; x < box.Width; x++)
                {
                    crop[(y * box.Width) + x] = new Gray8(page[((box.Y + y) * w) + box.X + x]);
                }
            }

            Image expected = Image.FromPixels<Gray8>(crop, box.Width, box.Height).Resize(height: 32);
            var want = new Gray8[expected.Width * expected.Height];
            expected.CopyPixelsTo<Gray8>(want, new ImageRect(0, 0, expected.Width, expected.Height), 1);
            for (int round = 0; round < 2; round++)
            {
                Gray8[] got = LinePreprocessor.Prepare(page, w, box, 32, pool, resizers, out int width);
                Assert.Equal(expected.Width, width);
                Assert.True(want.AsSpan().SequenceEqual(got.AsSpan(0, want.Length)));
                pool.Return(got);
            }
        }
    }
}
