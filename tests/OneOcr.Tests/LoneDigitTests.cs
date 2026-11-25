using OneOcr.Detection;

namespace OneOcr.Tests;

public sealed class LoneDigitTests
{
    private const int N = 64;

    // Four ordinary lines (probability 0.5, four cells tall) and, on the rows of the second, a weak isolated mark (probability about
    // 0.15, under the ordinary threshold) far to its right: a quantity cell of a table row that the text map crosses only weakly.
    private static float[] Map(int markX = 40, int markWidth = 2, int markHeight = 4, float markLogit = -1.73f)
    {
        var prob = new float[N * N];
        Array.Fill(prob, -8f);
        for (int row = 0; row < 4; row++)
        {
            for (int y = 4 + (row * 10); y < 8 + (row * 10); y++)
            {
                for (int x = 2; x < 20; x++)
                {
                    prob[(y * N) + x] = 0f;
                }
            }
        }

        for (int y = 14; y < 14 + markHeight; y++)
        {
            for (int x = markX; x < markX + markWidth; x++)
            {
                prob[(y * N) + x] = markLogit;
            }
        }

        return prob;
    }

    private static PageDetection Decode(float[] prob) =>
        DetectorPostprocess.Decode(
            prob, N, N, new float[N * N * 16], N, N, 16, TestDetectorModels.AllClasses, new bool[8], new float[4], 1.0, 2 * N, 2 * N, OcrLimits.Default);

    [Fact]
    public void AWeakIsolatedMarkInATextRowBecomesALoneDigitLine()
    {
        PageDetection d = Decode(Map());
        Assert.Equal(5, d.Lines.Count);
        DetectedLine mark = d.Lines[4];
        Assert.True(mark.LoneDigits);
        Assert.Equal(0.9f, mark.MinConfidence);
        Assert.True(mark.Left > 70 && mark.Right < 100);
        Assert.All(d.Lines.Take(4), l => Assert.False(l.LoneDigits));
    }

    [Fact]
    public void AMarkThatIsTooSmallTooTallOrTooWideIsLeftAlone()
    {
        Assert.Equal(4, Decode(Map(markHeight: 2)).Lines.Count);
        Assert.Equal(4, Decode(Map(markHeight: 12)).Lines.Count);
        Assert.Equal(4, Decode(Map(markWidth: 14)).Lines.Count);
        Assert.Equal(4, Decode(Map(markLogit: -3.5f)).Lines.Count);
    }

    [Fact]
    public void AMarkOutsideEveryTextRowIsLeftAlone()
    {
        float[] prob = Map();
        // Move the mark's cells (rows 14 to 17) to rows 20 to 23, between two text rows.
        for (int y = 14; y < 18; y++)
        {
            for (int x = 40; x < 42; x++)
            {
                prob[(y * N) + x] = -8f;
                prob[((y + 6) * N) + x] = -1.73f;
            }
        }

        Assert.Equal(4, Decode(prob).Lines.Count);
    }

    [Fact]
    public void ABarePageOrOneWithFewLinesHasNoReferenceRowAndKeepsItsLines()
    {
        var prob = new float[N * N];
        Array.Fill(prob, -8f);
        for (int y = 14; y < 18; y++)
        {
            prob[(y * N) + 40] = prob[(y * N) + 41] = -1.73f;
        }

        Assert.Empty(Decode(prob).Lines);
        for (int y = 4; y < 8; y++)
        {
            for (int x = 2; x < 20; x++)
            {
                prob[(y * N) + x] = 0f;
            }
        }

        Assert.Single(Decode(prob).Lines);
    }
}
