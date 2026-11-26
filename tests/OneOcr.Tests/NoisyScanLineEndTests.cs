using OneOcr.Detection;
using OneOcr.Layout;

namespace OneOcr.Tests;

/// <summary>
/// The ends of a line on a noisy scan. A photocopied or scanned page is noisy enough to take the camera-photo path of the line finder, and a
/// detector box there stops a letter or two before the ink does (the text map is cut at a higher threshold); the last digit or a trailing
/// minus sign was then never read. Lines that lean on their own are cut in a strip of their own, which has to reach past the box as well.
/// </summary>
public class NoisyScanLineEndTests
{
    private const int Width = 520, Height = 200;

    // White page with dark bars 3 px wide every 5 px (letters with 2 px gaps), 20 rows tall, from x = from to x = to, leaning by slope rows per column.
    private static byte[] LeaningLine(int from, int to, double slope)
    {
        var page = new byte[Width * Height];
        Array.Fill(page, (byte)255);
        for (int x = from; x < to; x++)
        {
            if ((x - from) % 5 >= 3)
            {
                continue;
            }

            int shift = (int)Math.Round(slope * (x - 260), MidpointRounding.ToEven);
            for (int y = 90; y < 110; y++)
            {
                page[((y + shift) * Width) + x] = 20;
            }
        }

        return page;
    }

    [Fact]
    public void AnUprightLineOnANoisyPageIsFollowedPastTheDetectorsEnd()
    {
        byte[] page = LeaningLine(60, 460, 0);
        var flat = new List<LineBox>();
        var noisy = new List<LineBox>();
        // The box ends at x 445, inside the run of letters that goes on to x 458.
        var line = new DetectedLine(55, 70, 445, 130, 0.9f);
        DetectorLines.InkBoxes(line, page, Width, Height, 128, false, flat);
        DetectorLines.InkBoxes(line, page, Width, Height, 128, false, noisy, pageIsDarkInk: true);
        Assert.Equal(458, flat[0].Right);
        Assert.Equal(458, noisy[0].Right);
        Assert.Equal(60, noisy[0].X);
    }

    [Fact]
    public void ALeaningLineOnANoisyPageIsFollowedPastTheEndOfItsComponent()
    {
        const double Slope = 0.03;
        byte[] page = LeaningLine(60, 460, Slope);
        double cos = 1 / Math.Sqrt(1 + (Slope * Slope)), sin = Slope * cos;
        // The detector's component covers u from -200 to +185 (x 60 to about 445) of a line whose letters go on to u = +198.
        var axis = new LineAxis(260, 100, cos, sin, -200, 185, -14, 14, 400, 800);
        var detected = new DetectedLine(50, 70, 450, 130, 0.9f, axis);
        foreach (bool? noisy in new bool?[] { null, true })
        {
            var strips = new List<StripLine>();
            Assert.True(TiltedLines.TryInkBoxes(detected, page, Width, Height, 128, false, strips, 1, noisy));
            TiltedLine frame = Assert.Single(strips).Tilt!;
            // The ink in the line's own frame spans every letter, to within a pixel: the first bar starts at u = -200, the last ends at 198.
            Assert.InRange(frame.Ink.X, -202, -198);
            Assert.InRange(frame.Ink.Right, 196, 201);
        }
    }
}
