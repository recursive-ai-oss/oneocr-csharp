using OneOcr.Detection;
using OneOcr.Layout;

namespace OneOcr.Tests;

/// <summary>Line-by-line binarization of camera photos: when the page's threshold is wrong for a line, and that a screenshot is never touched.</summary>
public class LocalInkTests
{
    // A page of `paper` gray with dark text-like bars (gray `ink`, 3 px wide, every 6 px) over x in [from, to) and 6 rows in the middle.
    private static byte[] Page(int width, int height, byte paper, byte ink, int from, int to)
    {
        var page = new byte[width * height];
        Array.Fill(page, paper);
        for (int y = height / 2 - 3; y < height / 2 + 3; y++)
        {
            for (int x = from; x < to; x++)
            {
                if ((x - from) % 6 < 3)
                {
                    page[(y * width) + x] = ink;
                }
            }
        }

        return page;
    }

    [Fact]
    public void AShadowedLineOnAPageThatReadsInkAsBrightIsReadAsDarkInk()
    {
        // The page's threshold (150) sits above this line's paper (130) and ink (40): everything is "dark", the bright class is the page's ink.
        byte[] page = Page(120, 20, 130, 40, 10, 110);
        LocalInk? local = LocalInk.Choose(page, 120, 0, 120, 0, 20, 150, inkIsBright: true, pageIsDarkInk: true);
        Assert.NotNull(local);
        Assert.False(local.Value.InkIsBright);
        Assert.InRange(local.Value.Threshold, 40, 129);
    }

    [Fact]
    public void ALineOfAPageThatReallyReadsInkAsBrightKeepsThePagesReading()
    {
        // A dark-mode screen: bright text on dark, and most of its lines say so, so a light button on it is left alone.
        byte[] page = Page(120, 20, 130, 40, 10, 110);
        Assert.Null(LocalInk.Choose(page, 120, 0, 120, 0, 20, 150, inkIsBright: true, pageIsDarkInk: false));
    }

    [Fact]
    public void ALineTheDarkInkThresholdFitsIsNotTouched()
    {
        byte[] page = Page(120, 20, 220, 40, 10, 110);
        Assert.Null(LocalInk.Choose(page, 120, 0, 120, 0, 20, 128, inkIsBright: false, pageIsDarkInk: true));
    }

    [Fact]
    public void ALineWhoseThresholdLiesInsideItsPaperGetsItsOwnThreshold()
    {
        // Paper 110, ink 20, page threshold 128: the whole line is "ink" for the page.
        byte[] page = Page(120, 20, 110, 20, 10, 110);
        LocalInk? local = LocalInk.Choose(page, 120, 0, 120, 0, 20, 128, inkIsBright: false, pageIsDarkInk: true);
        Assert.NotNull(local);
        Assert.InRange(local.Value.Threshold, 20, 109);
    }

    [Fact]
    public void OnAPhotoAFaintLineEndComesFromTheDetectorOnAFlatPageItComesFromTheInk()
    {
        // Ink covers x 20..100 of a line the detector found over x 0..300: print further right is lighter than the threshold.
        byte[] page = Page(300, 40, 230, 30, 20, 100);
        var line = new DetectedLine(0, 0, 300, 40, 0.9f);
        var flat = new List<LineBox>();
        DetectorLines.InkBoxes(line, page, 300, 40, 128, false, flat);
        var photo = new List<LineBox>();
        DetectorLines.InkBoxes(line, page, 300, 40, 128, false, photo, pageIsDarkInk: true);
        Assert.Equal(20, flat[0].X);
        Assert.Equal(80, flat[0].Width);
        Assert.Equal(0, photo[0].X);
        Assert.Equal(300, photo[0].Right);
    }
}
