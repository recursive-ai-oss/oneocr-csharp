using OneOcr.Layout;

namespace OneOcr.Tests;

/// <summary>
/// The minus sign of an amount that lies outside the crop the recognizer reads (see SignRescue). A phone capture of a statement leaves
/// the sign a thin grey stroke a few pixels from the last digit, and the line then comes back as the bare number.
/// </summary>
public class SignRescueTests
{
    private const int Width = 120, Rows = 32, Paper = 230, Ink = 40;

    // A crop of paper with a block of "digits" (columns 40 to 79, rows 6 to 25) and whatever the caller draws.
    private static byte[] Crop(Action<byte[]> draw)
    {
        var crop = new byte[Width * Rows];
        Array.Fill(crop, (byte)Paper);
        for (int y = 6; y <= 25; y++)
        {
            for (int x = 40; x < 80; x++)
            {
                if (x % 5 < 3)
                {
                    crop[(y * Width) + x] = Ink;
                }
            }
        }

        draw(crop);
        return crop;
    }

    private static void Bar(byte[] crop, int x0, int x1, int y0, int y1, byte value = Ink)
    {
        for (int y = y0; y <= y1; y++)
        {
            for (int x = x0; x <= x1; x++)
            {
                crop[(y * Width) + x] = value;
            }
        }
    }

    private static int Right(byte[] crop) => SignRescue.FindBeyond(crop, Width, 6, 25, 79, 1, 128);

    private static int Left(byte[] crop) => SignRescue.FindBeyond(crop, Width, 6, 25, 40, -1, 128);

    [Fact]
    public void AThinStrokeAtMidHeightPastTheEndIsFoundOnBothSides()
    {
        Assert.Equal(13, Right(Crop(c => Bar(c, 86, 92, 15, 16))));
        Assert.Equal(12, Left(Crop(c => Bar(c, 28, 33, 15, 16))));
    }

    [Fact]
    public void AFaintStrokeThatScalingLightenedIsStillFound()
    {
        // Gray 150 against paper 230 and a threshold of 128: only the relaxed threshold (158) sees it.
        Assert.True(Right(Crop(c => Bar(c, 86, 92, 15, 16, 150))) > 0);
    }

    [Fact]
    public void ThingsThatAreNotASignAreNotFound()
    {
        // A full stop at the baseline, a leader of dots, a tall rule, the fringe of the last digit, and a stroke too far away.
        Assert.Equal(0, Right(Crop(c => Bar(c, 84, 87, 23, 25))));
        Assert.Equal(0, Right(Crop(c => { Bar(c, 84, 86, 14, 16); Bar(c, 90, 92, 14, 16); Bar(c, 96, 98, 14, 16); })));
        Assert.Equal(0, Right(Crop(c => Bar(c, 86, 88, 4, 27))));
        Assert.Equal(0, Right(Crop(c => Bar(c, 80, 81, 14, 17))));
        Assert.Equal(0, Right(Crop(c => Bar(c, 100, 106, 15, 16))));
    }

    private static List<OcrWord> Words(params string[] texts) =>
        texts.Select(t => new OcrWord(t, new OcrBox(0, 0, 1, 1), 0.9f, 0.9f, OcrSource.Ocr, [])).ToList();

    [Fact]
    public void ASecondReadingIsKeptOnlyWhenItDiffersByTheSign()
    {
        Assert.True(SignRescue.DiffersByASign(Words("4,981.95"), Words("4,981.95-"), false, true));
        Assert.True(SignRescue.DiffersByASign(Words("USD", "12.00"), Words("USD", "12.00", "-"), false, true));
        Assert.True(SignRescue.DiffersByASign(Words("1,370.35"), Words("-1,370.35"), true, false));
        Assert.False(SignRescue.DiffersByASign(Words("4,981.95"), Words("4,981.96-"), false, true));
        Assert.False(SignRescue.DiffersByASign(Words("4,981.95"), Words("-4,981.95"), false, true));
        Assert.False(SignRescue.DiffersByASign(Words("4,981.95"), Words("4,981.95"), false, true));
    }

    [Fact]
    public void OnlyNumbersWithoutASignWantOne()
    {
        Assert.True(SignRescue.WantsTrailingSign(Words("Total", "1,200.00")));
        Assert.False(SignRescue.WantsTrailingSign(Words("1,200.00-")));
        Assert.False(SignRescue.WantsTrailingSign(Words("(1,200.00)")));
        Assert.False(SignRescue.WantsTrailingSign(Words("Total")));
        Assert.True(SignRescue.WantsLeadingSign(Words("1,200.00", "USD")));
        Assert.False(SignRescue.WantsLeadingSign(Words("-1,200.00")));
        Assert.False(SignRescue.WantsLeadingSign(Words("7")));
    }
}
