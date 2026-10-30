using OneOcr.Detection;
using OneOcr.Layout;

namespace OneOcr.Tests;

/// <summary>Following a line's ink past the end of the detected box when the box cuts a letter, and only then.</summary>
public class LineEndTests
{
    private const int Width = 240, Height = 40;

    // White page; dark bars 3 px wide every 5 px (letters with 2 px gaps) over x in [from, to), 20 rows tall (rows 10..29).
    private static byte[] Page(params (int From, int To)[] runs)
    {
        var page = new byte[Width * Height];
        Array.Fill(page, (byte)255);
        foreach ((int from, int to) in runs)
        {
            for (int y = 10; y < 30; y++)
            {
                for (int x = from; x < to; x++)
                {
                    if ((x - from) % 5 < 3)
                    {
                        page[(y * Width) + x] = 20;
                    }
                }
            }
        }

        return page;
    }

    private static LineBox Ink(byte[] page, double right, int leftLimit = 0, int rightLimit = Width, bool photo = false)
    {
        var result = new List<LineBox>();
        DetectorLines.InkBoxes(new DetectedLine(0, 5, right, 35, 0.9f), page, Width, Height, 128, false, result, photo ? true : null, leftLimit, rightLimit);
        Assert.Single(result);
        return result[0];
    }

    [Fact]
    public void InkThatGoesOnPastTheBoxIsFollowedToTheEndOfTheLetters()
    {
        // The box stops at x 90, inside a run of letters that goes on to x 100 (the last bar covers 95..97).
        LineBox box = Ink(Page((20, 100)), 90);
        Assert.Equal(20, box.X);
        Assert.Equal(98, box.Right);
    }

    [Fact]
    public void InkIsFollowedLeftwardsToo()
    {
        var result = new List<LineBox>();
        DetectorLines.InkBoxes(new DetectedLine(30, 5, 140, 35, 0.9f), Page((20, 100)), Width, Height, 128, false, result);
        Assert.Equal(20, result[0].X);
    }

    [Fact]
    public void AWordGapStopsTheExtension()
    {
        // 15 blank pixels (0.75 of the line's height) follow the box: the next word is not part of this line.
        LineBox box = Ink(Page((20, 60), (75, 100)), 60);
        Assert.Equal(58, box.Right);
    }

    [Fact]
    public void ANeighbouringLinesBoxIsNeverEntered()
    {
        LineBox box = Ink(Page((20, 100)), 90, rightLimit: 94);
        Assert.True(box.Right <= 94, $"right edge {box.Right}");
    }

    [Fact]
    public void ASolidBlockIsNotLetters()
    {
        // A button or rule: every column is ink, so the stretch beyond the box is not followed.
        byte[] page = Page((20, 60));
        for (int y = 10; y < 30; y++)
        {
            for (int x = 60; x < 80; x++)
            {
                page[(y * Width) + x] = 20;
            }
        }

        Assert.Equal(58, Ink(page, 60).Right);
    }

    [Fact]
    public void ADifferenceOfAFewPixelsIsLeftAlone()
    {
        // The ink goes on for 2 px past the box (less than a quarter of the line's height): that is anti-aliasing, not a cut letter.
        LineBox box = Ink(Page((20, 100)), 96);
        Assert.Equal(96, box.Right);
        LineBox near = Ink(Page((20, 100)), 98);
        Assert.Equal(98, near.Right);
    }

    [Fact]
    public void PhotosKeepTheirOwnEndRule()
    {
        // On a camera photo the end comes from the ink (or the detector) as before: the last bar inside the box ends at x 88.
        LineBox flat = Ink(Page((20, 100)), 90);
        LineBox photo = Ink(Page((20, 100)), 90, photo: true);
        Assert.Equal(98, flat.Right);
        Assert.Equal(88, photo.Right);
    }
}
