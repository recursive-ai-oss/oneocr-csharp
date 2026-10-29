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

    [Fact]
    public void ControlsReadAsLettersAreDropped()
    {
        // Radio buttons and checkboxes: lines that are only a ring or box glyph read with little confidence.
        OrderStrip?[] strips =
        [
            Strip(0, 30, ("O", 0.3f)),
            Strip(40, 30, ("L]", 0.6f)),
            Strip(80, 30, ("[]", 0.1f)),
            Strip(120, 30, ("D", 0.84f)),
            Strip(160, 30, ("Remember", 0.99f), ("me", 0.99f)),
        ];

        Assert.Equal(4, JunkFilter.Apply(strips, new JunkParameters()));
        Assert.All(strips.Take(4), s => Assert.Null(s));
        Assert.NotNull(strips[4]);
    }

    [Fact]
    public void AControlApartFromItsLabelIsDroppedButALetterInTextStays()
    {
        static OcrWord At(string text, double left, double right, float confidence) =>
            new(text, new OcrBox(left, 0, right, 20), confidence, confidence, OcrSource.Ocr, []);

        OrderStrip Line(params OcrWord[] words) => new() { Leaf = 0, Box = new OcrBox(0, 0, 400, 20), Size = 20, Words = words };

        // A radio button (height 20) and its label 14 px to the right; "O" then a word one space (4 px) away is running text.
        OrderStrip?[] strips = [Line(At("O", 0, 12, 0.4f), At("Remember", 26, 120, 0.99f)), Line(At("Hail", 0, 40, 0.99f), At("O", 44, 56, 0.4f), At("Captain", 60, 130, 0.99f))];

        Assert.Equal(1, JunkFilter.Apply(strips, new JunkParameters()));
        Assert.Equal(["Remember"], strips[0]!.Words.Select(w => w.Text));
        Assert.Equal(3, strips[1]!.Words.Count);
    }

    [Fact]
    public void ConfidentLettersAndRealWordsAreNotControls()
    {
        OrderStrip?[] strips =
        [
            Strip(0, 30, ("C", 0.97f)), // a grade, read with confidence
            Strip(40, 30, ("Do", 0.4f)), // a word
            Strip(80, 30, ("OK", 0.5f)), // second letter outside the control set
            Strip(120, 30, ("O", 0.3f), ("Yes", 0.99f)), // a label on the same line: not a lone glyph
            Strip(160, 30, ("0", 0.2f)), // a digit
        ];

        Assert.Equal(0, JunkFilter.Apply(strips, new JunkParameters()));
        Assert.Equal(0, JunkFilter.Apply([Strip(0, 30, ("O", 0.3f))], new JunkParameters(ControlConfidence: 0)));
    }
}
