using OneImage;
using OneOcr.Layout;

namespace OneOcr.Tests;

public class LeaderTests
{
    private const int W = 300, H = 40;

    // A white page with a line of text in a 25-pixel-high box: two letters (8x14), a run of 3x3 dots on the baseline and a number
    // (8x14) after it. Dot k of the run is centred at x = firstDot + 1 + k * pitch.
    private static byte[] Page(int firstDot, int pitch, int dots, int skip = -1, Action<byte[]>? extra = null)
    {
        var page = new byte[W * H];
        Array.Fill(page, (byte)255);
        Rect(page, 5, 10, 8, 14);
        Rect(page, 16, 10, 8, 14);
        for (int k = 0; k < dots; k++)
        {
            if (k != skip)
            {
                Rect(page, firstDot + (k * pitch), 21, 3, 3);
            }
        }

        Rect(page, 270, 10, 8, 14);
        extra?.Invoke(page);
        return page;
    }

    private static void Rect(byte[] page, int x, int y, int w, int h)
    {
        for (int j = y; j < y + h; j++)
        {
            for (int i = x; i < x + w; i++)
            {
                page[(j * W) + i] = 0;
            }
        }
    }

    private static Leader? Find(byte[] page) => LeaderFinder.Find(page, W, H, new LineBox(0, 5, W, 25), 128, false, 20);

    [Fact]
    public void FindsARunOfDotsAndTheInkAroundIt()
    {
        Leader? leader = Find(Page(50, 8, 20));
        Assert.NotNull(leader);
        Assert.Equal(20, leader.Count);
        Assert.Equal(50, leader.From);
        Assert.Equal(50 + (19 * 8) + 3, leader.To);
        Assert.Equal(8.0, leader.Pitch, 0.01);
        Assert.False(leader.Spaced);
        Assert.Equal(5, leader.Left!.Value.X);
        Assert.Equal(270, leader.Right!.Value.X);
        Assert.Equal(20, leader.Centers.Length);
    }

    [Fact]
    public void WidePitchIsSpaced()
    {
        Leader? leader = Find(Page(50, 12, 15));
        Assert.NotNull(leader);
        Assert.True(leader.Spaced);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    public void ShortRunsAreNotLeaders(int dots) => Assert.Null(Find(Page(50, 8, dots)));

    [Fact]
    public void ADotTooFaintToSeeIsCounted()
    {
        Leader? leader = Find(Page(50, 8, 20, skip: 10));
        Assert.NotNull(leader);
        Assert.Equal(20, leader.Count);
    }

    [Fact]
    public void ALetterInTheMiddleEndsTheRun()
    {
        // Five dots, a letter, five dots: neither side is long enough.
        byte[] page = Page(50, 8, 11, skip: 5, extra: p => Rect(p, 90, 10, 8, 14));
        Assert.Null(Find(page));
    }

    [Fact]
    public void ASentenceFullStopBeforeTheLeaderIsNotCounted()
    {
        // The first "dot" sits right behind the letter at x 16..23, at the same pitch as the rest.
        byte[] page = Page(25, 8, 21);
        Leader? leader = Find(page);
        Assert.NotNull(leader);
        Assert.Equal(20, leader.Count);
        Assert.Equal(33, leader.From);
    }

    [Fact]
    public void BlankPageAndTextWithoutDotsAreNotLeaders()
    {
        Assert.Null(Find(Page(50, 8, 0)));
        Assert.Null(LeaderFinder.Find(new byte[W * H], W, H, new LineBox(0, 5, W, 25), 128, false, 20));
    }

    private static OcrWord Word(string text, double left, double right) =>
        new(text, new OcrBox(left, 10, right, 24), 0.9f, 0.9f, OcrSource.Ocr, []);

    [Fact]
    public void NumberRightBehindTheLeaderIsOneWordWithIt()
    {
        Leader leader = Find(Page(50, 8, 20))!;
        List<OcrWord> words = LeaderWords.Join([Word("Intro", 5, 24)], [Word("12", leader.To + 6, leader.To + 20)], leader, false);
        Assert.Equal(["Intro", new string('.', 20) + "12"], words.Select(w => w.Text));
        Assert.Equal((double)leader.From, words[1].Box.Left);
        Assert.Equal((double)(leader.To + 20), words[1].Box.Right);
    }

    [Fact]
    public void NumberFarFromTheLeaderIsAWordOfItsOwn()
    {
        Leader leader = Find(Page(50, 8, 20))!;
        List<OcrWord> words = LeaderWords.Join([Word("Intro", 5, 24)], [Word("12", leader.To + 40, leader.To + 54)], leader, true);
        Assert.Equal(["Intro", new string('.', 20), "12"], words.Select(w => w.Text));
        Assert.Equal(LeaderWords.Confidence, words[1].Confidence);
        Assert.Equal(20, words[1].Chars.Count);
    }

    [Fact]
    public void StopsLeftOverRightOfTheRunAreDropped()
    {
        Leader leader = Find(Page(50, 8, 20))!;
        List<OcrWord> words = LeaderWords.Join([], [Word("..", leader.To + 1, leader.To + 8), Word("7", leader.To + 40, leader.To + 50)], leader, false);
        Assert.Equal([new string('.', 20), "7"], words.Select(w => w.Text));
    }

    [Fact]
    public void SpacedStopsAreWordsOfTheirOwn()
    {
        Leader leader = Find(Page(50, 12, 15))!;
        List<OcrWord> words = LeaderWords.Join([Word("Line", 5, 24)], [Word("3.", leader.To + 12, leader.To + 25)], leader, false);
        Assert.Equal(17, words.Count);
        Assert.All(words.Skip(1).Take(15), w => Assert.Equal(".", w.Text));
        Assert.Equal("3.", words[^1].Text);
    }

    [Fact]
    public void BlankedColumnsReadAsBackground()
    {
        var plain = new byte[200 * 20];
        Array.Fill(plain, (byte)200);
        byte[] inked = (byte[])plain.Clone();
        for (int y = 0; y < 20; y++)
        {
            for (int x = 30; x < 60; x++)
            {
                inked[(y * 200) + x] = 10;
            }
        }

        var pool = new BufferPool<Gray8>(1L << 26, 1);
        var resizers = new ResizerPool();
        var box = new LineBox(0, 0, 200, 20);
        Gray8[] want = LinePreprocessor.Prepare(plain, 200, box, 32, pool, resizers, out int w1);
        Gray8[] got = LinePreprocessor.Prepare(inked, 200, box, 32, pool, resizers, out int w2, 30, 60);
        Assert.Equal(w1, w2);
        Assert.True(want.AsSpan(0, 32 * w1).SequenceEqual(got.AsSpan(0, 32 * w2)));
    }
}
