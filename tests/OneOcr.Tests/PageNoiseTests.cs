using OneImage;
using OneOcr.Detection;
using Xunit;

namespace OneOcr.Tests;

public sealed class PageNoiseTests
{
    private static Gray8[] Page(int w, int h, Func<int, int, int> f)
    {
        var page = new Gray8[w * h];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                page[(y * w) + x] = new Gray8((byte)f(x, y));
            }
        }

        return page;
    }

    [Fact]
    public void SigmaOfFlatAndSmoothPagesIsZero()
    {
        Assert.Equal(0, PageNoise.Sigma(Page(40, 30, (_, _) => 200), 40, 30));
        // The kernel is a second difference in both directions: a plane (a ramp) gives zero everywhere.
        Assert.Equal(0, PageNoise.Sigma(Page(40, 30, (x, y) => x + y), 40, 30));
        Assert.Equal(0, PageNoise.Sigma(Page(2, 50, (_, _) => 9), 2, 50));
    }

    [Fact]
    public void SigmaIsTheLowerMedianOverFourPointZeroFourSeven()
    {
        // A checkerboard of 0 and 10.
        Gray8[] board = Page(8, 8, (x, y) => ((x + y) & 1) == 0 ? 10 : 0);
        // Centre 10: corners 4 x 10, edges 0 -> 40 + 40 = 80. Centre 0: corners 0, edges 4 x 10 -> -80 -> 80.
        Assert.Equal(80 / PageNoise.SigmaDivisor, PageNoise.Sigma(board, 8, 8));

        // One bright pixel touches 9 of the 25 interior responses; the median is still 0.
        Gray8[] dot = Page(7, 7, (x, y) => x == 3 && y == 3 ? 100 : 0);
        Assert.Equal(0, PageNoise.Sigma(dot, 7, 7));
    }

    [Fact]
    public void SigmaHandlesTheFullResponseRange()
    {
        // Alternating 0 and 255 columns/rows reach responses of +-2040.
        Gray8[] board = Page(8, 8, (x, y) => ((x + y) & 1) == 0 ? 255 : 0);
        Assert.Equal(2040 / PageNoise.SigmaDivisor, PageNoise.Sigma(board, 8, 8));
    }

    [Fact]
    public void JunkDensityCountsSmallWeakComponents()
    {
        const int n = 20;
        var logits = new float[n * n];
        Array.Fill(logits, -6f);
        // Weak 3-cell component (probability 0.38 < 0.5, above 0.3): counted.
        logits[(2 * n) + 2] = logits[(2 * n) + 3] = logits[(2 * n) + 4] = -0.5f;
        // Strong 2-cell component (probability 0.88): not counted.
        logits[(6 * n) + 2] = logits[(6 * n) + 3] = 2f;
        // Single weak cell: not counted (under 2 cells).
        logits[(10 * n) + 10] = -0.5f;
        // Weak component of 13 cells: not counted.
        for (int x = 0; x < 13; x++)
        {
            logits[(15 * n) + x] = -0.5f;
        }

        Assert.Equal(1 / (n * n * 4 / 1e6), PageNoise.JunkDensity(logits, n, n), 6);
    }

    [Fact]
    public void RuleChoosesTheTextThresholdPerPage()
    {
        // A text line of 4 x 2 cells at probability 0.5: on at 0.3, off at 0.5.
        var prob = new float[8 * 8];
        Array.Fill(prob, -6f);
        for (int x = 2; x < 6; x++)
        {
            prob[(3 * 8) + x] = 0f;
            prob[(4 * 8) + x] = 0f;
        }

        var layout = new float[8 * 8 * 16];
        Array.Fill(layout, -6f);
        var options = new DetectorPostprocess.Options(0.3, 1.0, 0.6, 0.85, RuleSigmaMin: 0.1235, RuleJunkMax: 0.6358, RuleTextThreshold: 0.5);

        int Lines(double? sigma, DetectorPostprocess.Options? o = null) => DetectorPostprocess.Decode(
            prob, 8, 8, layout, 8, 8, 16, TestDetectorModels.AllClasses, new bool[8], new float[4], 1.0, 128, 128, OcrLimits.Default, o ?? options, sigma).Lines.Count;

        Assert.Equal(1, Lines(null)); // not measured: unchanged
        Assert.Equal(1, Lines(0.1)); // quiet page
        Assert.Equal(0, Lines(0.5)); // noisy page, no specks: the stricter threshold
        Assert.Equal(1, Lines(0.5, options with { RuleSigmaMin = null })); // rule off
    }
}
