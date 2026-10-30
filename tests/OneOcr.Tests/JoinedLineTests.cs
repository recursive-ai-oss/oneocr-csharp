using OneOcr.Detection;
using OneOcr.Layout;

namespace OneOcr.Tests;

/// <summary>Cutting a detected box that holds several solid-set lines into its lines, and leaving single lines alone.</summary>
public class JoinedLineTests
{
    private const int Width = 300, Height = 400;

    // White page; a text line is a 14-row dense body (rows body..body+13) with sparse 4-row ascender and descender stubs around it, so
    // consecutive lines at a pitch of 24 leave two or three sparse rows between bodies but never an empty one.
    private static byte[] Page(params int[] bodies) => Page(280, bodies);

    // Same, with the first line ending at x `firstEnd` (a short last line of a paragraph above a full one).
    private static byte[] Page(int firstEnd, int[] bodies)
    {
        var page = new byte[Width * Height];
        Array.Fill(page, (byte)255);
        for (int k = 0; k < bodies.Length; k++)
        {
            int body = bodies[k], end = k == 0 ? firstEnd : 280;
            for (int y = body - 4; y < body + 18; y++)
            {
                bool dense = y >= body && y < body + 14;
                for (int x = 20; x < end; x++)
                {
                    if (dense ? (x % 4 < 3) : (x % 24 < 3))
                    {
                        page[(y * Width) + x] = 20;
                    }
                }
            }
        }

        return page;
    }

    private static List<LineBox> Boxes(byte[] page, DetectedLine line, int typical, List<(double Top, double Bottom)>? neighbours = null)
    {
        var result = new List<LineBox>();
        DetectorLines.InkBoxes(line, page, Width, Height, 128, false, result, null, 0, Width, typical, neighbours);
        return result;
    }

    [Fact]
    public void TwoTouchingLinesInOneBoxAreCutIntoTwoStrips()
    {
        byte[] page = Page(100, 124);
        List<LineBox> boxes = Boxes(page, new DetectedLine(10, 92, 290, 146, 0.9f), typical: 30);
        Assert.Equal(2, boxes.Count);
        Assert.True(boxes[0].Y + boxes[0].Height <= 120 && boxes[1].Y >= 118, $"{boxes[0].Y}+{boxes[0].Height} / {boxes[1].Y}+{boxes[1].Height}");
    }

    [Fact]
    public void AShortLineAboveAFullOneIsKept()
    {
        byte[] page = Page(70, [100, 124]);
        Assert.Equal(2, Boxes(page, new DetectedLine(10, 92, 290, 146, 0.9f), typical: 30).Count);
    }

    [Fact]
    public void ThreeLinesAreCutTwice()
    {
        byte[] page = Page(100, 124, 148);
        List<LineBox> boxes = Boxes(page, new DetectedLine(10, 92, 290, 170, 0.9f), typical: 30);
        Assert.Equal(3, boxes.Count);
    }

    [Fact]
    public void ASingleLineIsNotCut()
    {
        byte[] page = Page(100);
        Assert.Single(Boxes(page, new DetectedLine(10, 92, 290, 126, 0.9f), typical: 30));
    }

    [Fact]
    public void ATallSingleLineWithNoValleyIsNotCut()
    {
        // A heading three times the page's typical line: dense all the way, no row between two dense bands.
        var page = new byte[Width * Height];
        Array.Fill(page, (byte)255);
        for (int y = 100; y < 160; y++)
        {
            for (int x = 20; x < 280; x++)
            {
                if (x % 20 < 14)
                {
                    page[(y * Width) + x] = 20;
                }
            }
        }

        Assert.Single(Boxes(page, new DetectedLine(10, 96, 290, 164, 0.9f), typical: 30));
    }

    [Fact]
    public void APartThatAnotherDetectedLineAlreadyCoversIsDropped()
    {
        byte[] page = Page(100, 124);
        // The upper line has a detected box of its own.
        List<LineBox> boxes = Boxes(page, new DetectedLine(10, 92, 290, 146, 0.9f), typical: 30, neighbours: [(92, 120)]);
        LineBox only = Assert.Single(boxes);
        Assert.True(only.Y >= 119, $"strip starts at {only.Y}");
    }

    [Fact]
    public void NothingIsCutWithoutATypicalHeight()
    {
        byte[] page = Page(100, 124);
        Assert.Single(Boxes(page, new DetectedLine(10, 92, 290, 146, 0.9f), typical: 0));
    }
}
