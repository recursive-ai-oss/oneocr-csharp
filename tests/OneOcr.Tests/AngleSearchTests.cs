using OneImage;
using OneOcr.Kernels;
using OneOcr.Layout;
using OneOcr.Model;

namespace OneOcr.Tests;

public class AngleSearchTests
{
    private static OcrPage PageOf(params (string Text, float Confidence)[] words)
    {
        var list = new List<OcrWord>();
        foreach ((string text, float confidence) in words)
        {
            list.Add(new OcrWord(text, new OcrBox(0, 0, 10, 10), confidence, confidence, OcrSource.Ocr, []));
        }

        var line = new OcrLine(string.Join(' ', list.Select(w => w.Text)), list, [0], new OcrBox(0, 0, 10, 10), 1f, OcrSource.Ocr);
        return new OcrPage(0, 100, 100, 72, [new OcrBlock(OcrBlockRole.Body, false, new OcrBox(0, 0, 10, 10), [line], 1f)]);
    }

    // A page whose score is the given number of confident characters.
    private static OcrPage Scored(int chars) => PageOf((new string('a', chars), 1f));

    [Fact]
    public void TheScoreCountsOnlyConfidentWordsByLength()
    {
        OcrPage page = PageOf(("hello", 0.95f), ("noise", 0.5f), ("ab", 1f));
        Assert.Equal((0.95f * 5) + 2.0, PageAngleSearch.Score(page), 6);
        Assert.True(PageAngleSearch.IsWeak(page, out double share));
        Assert.Equal(7 / 12.0, share, 9);
        Assert.False(PageAngleSearch.IsWeak(PageOf(("hello", 0.99f)), out _));
        Assert.True(PageAngleSearch.IsWeak(PageOf(), out _));
    }

    [Fact]
    public void ASearchKeepsTheFirstReadingUnlessAnAngleClearlyBeatsIt()
    {
        OcrPage first = Scored(100);
        var tried = new List<int>();
        OcrPage? Read(int slope)
        {
            tried.Add(slope);
            return Scored(105);
        }

        (OcrPage page, int? angle) = PageAngleSearch.Search(first, Read, default);
        Assert.Same(first, page);
        Assert.Null(angle);
        // Coarse: 10, 20, 30 degrees, both signs, positive first; no fine step without a coarse winner.
        Assert.Equal([181, -181, 373, -373, 591, -591], tried);
    }

    [Fact]
    public void ADoubtfulReadingGetsTheLightSearchAndAWeakOneTheFullSearch()
    {
        // 8 of 10 characters are confident: searched, but only lightly. 5 of 10: the full search. 19 of 20: not searched.
        OcrPage doubtful = PageOf(("aaaaaaaa", 0.95f), ("bb", 0.3f));
        Assert.True(PageAngleSearch.IsWeak(doubtful, out double share));
        Assert.True(PageAngleSearch.IsLight(share));
        Assert.True(PageAngleSearch.IsWeak(PageOf(("aaaaa", 0.95f), ("bbbbb", 0.3f)), out share));
        Assert.False(PageAngleSearch.IsLight(share));
        Assert.False(PageAngleSearch.IsWeak(PageOf((new string('a', 19), 0.95f), ("b", 0.3f)), out _));

        var tried = new List<int>();
        OcrPage? Read(int slope)
        {
            tried.Add(slope);
            return Scored(slope == 181 ? 60 : 5);
        }

        (OcrPage page, int? angle) = PageAngleSearch.Search(Scored(10), Read, default, light: true);
        Assert.Equal(10, angle);
        // 10 and -10 degrees only, then 15 and 5 degrees around the winner (no 20 or 30 degrees).
        Assert.Equal([181, -181, 274, 90], tried);
        Assert.Equal(60, PageAngleSearch.Score(page), 6);

        tried.Clear();
        (_, angle) = PageAngleSearch.Search(Scored(100), s => { tried.Add(s); return Scored(105); }, default, light: true);
        Assert.Null(angle);
        Assert.Equal([181, -181], tried);
    }

    [Fact]
    public void ASearchRefinesAroundTheBestCoarseAngle()
    {
        OcrPage first = Scored(10);
        var scores = new Dictionary<int, int> { [-373] = 60, [-274] = 90, [-477] = 70 };
        var tried = new List<int>();
        OcrPage? Read(int slope)
        {
            tried.Add(slope);
            return Scored(scores.GetValueOrDefault(slope, 5));
        }

        (OcrPage page, int? angle) = PageAngleSearch.Search(first, Read, default);
        Assert.Equal(-15, angle);
        Assert.Equal(90, PageAngleSearch.Score(page), 6);
        // The fine step looks 5 degrees either side of the coarse winner (-20 degrees): -15 and -25.
        Assert.Equal([-274, -477], tried.TakeLast(2));
    }

    [Fact]
    public void AnUnreadableAngleIsSkippedAndTiesGoToTheSmallerAngle()
    {
        OcrPage first = Scored(10);
        OcrPage? Read(int slope) => Math.Abs(slope) == 373 ? null : Scored(50);
        (OcrPage page, int? angle) = PageAngleSearch.Search(first, Read, default);
        // 10 degrees (positive) is tried first and equal scores do not replace it.
        Assert.Equal(10, angle);
        Assert.Equal(50, PageAngleSearch.Score(page), 6);
    }

    [Fact]
    public void TheSearchStopsWhenCancelled()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => PageAngleSearch.Search(Scored(1), _ => Scored(1), cts.Token));
    }

    // A tilted page with a random-weight recognizer: nothing is read with confidence, so the page is weak, the search runs, finds nothing
    // that beats the first reading, and the result is the one without the search, on every path and degree.
    [Fact]
    public void ASearchThatFindsNothingBetterChangesNothingOnAnyPathOrDegree()
    {
        byte[] pixels = EngineTests.SyntheticPage(600, 400, [.. Enumerable.Range(0, 10).Select(i => 40 + (30 * i))], 6, 8, plain: true).Pixels;
        Image image = Image.FromPixels<Gray8>([.. pixels.Select(b => new Gray8(b))], 600, 400);
        byte[] model = TestModels.RandomContainer(TestModels.Small, 5);
        OcrEngine scalar = OcrEngine.Create(ModelContainer.Parse(model), null, KernelPath.Scalar);
        OcrEngine best = OcrEngine.Create(ModelContainer.Parse(model), null, Dispatch.Best);
        string off = OcrPageJson.ToJson([scalar.Ocr(image, new OcrOptions { SearchAngle = false, MaxDegreeOfParallelism = 1 })]);
        Assert.Equal(off, OcrPageJson.ToJson([scalar.Ocr(image, new OcrOptions { MaxDegreeOfParallelism = 1 })]));
        Assert.Equal(off, OcrPageJson.ToJson([best.Ocr(image, new OcrOptions { MaxDegreeOfParallelism = 4 })]));
    }
}
