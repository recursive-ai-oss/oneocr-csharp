using OneOcr.Layout;

namespace OneOcr.Tests;

/// <summary>The noise filter: unsure words of small lines go, confident small print and small pages stay.</summary>
public class JunkFilterTests
{
    private static OcrWord Word(string text, float confidence) => new(text, new OcrBox(0, 0, 10, 5), confidence, confidence, OcrSource.Ocr, []);

    private static OrderStrip Strip(double top, double height, params (string Text, float Confidence)[] words) => new()
    {
        Leaf = 0,
        Box = new OcrBox(0, top, 200, top + height),
        Size = height,
        Words = [.. words.Select(w => Word(w.Text, w.Confidence))],
    };

    // Four body lines of 20 pixels read with confidence, so the reference height is 20.
    private static List<OrderStrip?> Body() => [.. Enumerable.Range(0, 4).Select(i => (OrderStrip?)Strip(i * 30, 20, ("alpha", 0.99f), ("beta", 0.98f)))];

    [Fact]
    public void UnsureWordsOfSmallLinesAreDropped()
    {
        List<OrderStrip?> list = Body();
        list.Add(Strip(200, 10, ("x.", 0.2f), ("ab", 0.9f)));
        list.Add(Strip(220, 5, ("..", 0.1f)));
        OrderStrip?[] strips = [.. list];

        int dropped = JunkFilter.Apply(strips, new JunkParameters());

        Assert.Equal(2, dropped);
        Assert.Equal(["ab"], strips[4]!.Words.Select(w => w.Text));
        Assert.Null(strips[5]);
        Assert.All(strips.Take(4), s => Assert.Equal(2, s!.Words.Count));
    }

    [Fact]
    public void ConfidentSmallPrintStays()
    {
        List<OrderStrip?> list = Body();
        list.Add(Strip(200, 8, ("footnote", 0.95f), ("text", 0.8f)));
        OrderStrip?[] strips = [.. list];

        Assert.Equal(0, JunkFilter.Apply(strips, new JunkParameters()));
        Assert.Equal(2, strips[4]!.Words.Count);
    }

    [Fact]
    public void UnsureWordsOfFullSizeLinesStay()
    {
        List<OrderStrip?> list = Body();
        list.Add(Strip(200, 18, ("blurry", 0.1f)));
        OrderStrip?[] strips = [.. list];

        Assert.Equal(0, JunkFilter.Apply(strips, new JunkParameters()));
    }

    [Fact]
    public void LargeTextIsNeverSmallInAbsoluteTerms()
    {
        // 100-pixel body text: a 40-pixel line is under 60% of it, but taller than the absolute limit.
        OrderStrip?[] strips = [.. Enumerable.Range(0, 4).Select(i => (OrderStrip?)Strip(i * 150, 100, ("big", 0.99f))), Strip(700, 40, ("sub", 0.2f))];

        Assert.Equal(0, JunkFilter.Apply(strips, new JunkParameters()));
    }

    [Fact]
    public void PagesWithoutConfidentReferenceLinesAreLeftAlone()
    {
        OrderStrip?[] strips = [Strip(0, 20, ("alpha", 0.99f)), Strip(30, 20, ("beta", 0.99f)), Strip(100, 5, ("x", 0.1f))];

        Assert.Equal(0, JunkFilter.Apply(strips, new JunkParameters()));
    }

    [Fact]
    public void AllSmallTextIsKept()
    {
        // A page set entirely in small type has a small reference, so nothing is small next to it.
        OrderStrip?[] strips = [.. Enumerable.Range(0, 5).Select(i => (OrderStrip?)Strip(i * 12, 7, ("tiny", i == 4 ? 0.2f : 0.99f)))];

        Assert.Equal(0, JunkFilter.Apply(strips, new JunkParameters()));
    }

    [Fact]
    public void TinyLinesNeedAlmostCertainReading()
    {
        List<OrderStrip?> list = Body();
        list.Add(Strip(200, 3, (".", 0.8f), ("-", 0.995f)));
        list.Add(Strip(220, 10, ("ok", 0.8f)));
        OrderStrip?[] strips = [.. list];

        // The speck's 0.8 word goes, its 0.995 word stays; the 10-pixel line is small but not tiny, so 0.8 is enough there.
        Assert.Equal(1, JunkFilter.Apply(strips, new JunkParameters()));
        Assert.Equal(["-"], strips[4]!.Words.Select(w => w.Text));
        Assert.Equal(["ok"], strips[5]!.Words.Select(w => w.Text));
    }

    [Fact]
    public void DotsAndTicksAreDroppedOnPagesThatHaveSeveral()
    {
        List<OrderStrip?> list = Body();
        list.Add(Strip(200, 20, (".", 0.3f), ("word", 0.99f), ("'", 0.5f)));
        list.Add(Strip(240, 20, ("..", 0.2f), ("-", 0.2f), ("\u00B7", 0.4f), (".", 0.95f)));
        OrderStrip?[] strips = [.. list];

        // Three dots and ticks are unsure; the dash and the middle dot are words of their own, and a confident dot stays.
        Assert.Equal(3, JunkFilter.Apply(strips, new JunkParameters()));
        Assert.Equal(["word"], strips[4]!.Words.Select(w => w.Text));
        Assert.Equal(["-", "\u00B7", "."], strips[5]!.Words.Select(w => w.Text));
    }

    [Fact]
    public void DotsAndTicksStayOnPagesThatHaveFewOfThem()
    {
        List<OrderStrip?> list = Body();
        list.Add(Strip(200, 20, (".", 0.3f), ("word", 0.99f), ("'", 0.5f)));
        OrderStrip?[] strips = [.. list];

        Assert.Equal(0, JunkFilter.Apply(strips, new JunkParameters()));
        Assert.Equal(3, strips[4]!.Words.Count);
        Assert.Equal(2, JunkFilter.Apply(strips, new JunkParameters(SpeckPageMin: 2)));
    }
}
