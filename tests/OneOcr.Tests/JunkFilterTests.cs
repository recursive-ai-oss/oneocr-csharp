using OneOcr.Layout;

namespace OneOcr.Tests;

/// <summary>The noise filter: unsure words of small lines go, confident small print and small pages stay.</summary>
public class JunkFilterTests
{
    private static OcrWord Word(string text, float confidence, OcrBox? box = null) => new(text, box ?? new OcrBox(0, 0, 10, 5), confidence, confidence, OcrSource.Ocr, []);

    private static OrderStrip Strip(double top, double height, params (string Text, float Confidence)[] words) => new()
    {
        Leaf = 0,
        Box = new OcrBox(0, top, 200, top + height),
        Size = height,
        // Words stand side by side, as printed words do; boxes that coincide would read as one word drawn twice.
        Words = [.. words.Select((w, k) => Word(w.Text, w.Confidence, new OcrBox(20 * k, top, (20 * k) + 16, top + height)))],
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
        static OcrWord At(string text, double left, double right, float confidence, double top = 0) =>
            new(text, new OcrBox(left, top, right, top + 20), confidence, confidence, OcrSource.Ocr, []);

        OrderStrip Line(params OcrWord[] words) => new() { Leaf = 0, Box = new OcrBox(0, words[0].Box.Top, 400, words[0].Box.Bottom), Size = 20, Words = words };

        // A radio button (height 20) and its label 14 px to the right; "O" then a word one space (4 px) away is running text.
        OrderStrip?[] strips = [Line(At("O", 0, 12, 0.4f), At("Remember", 26, 120, 0.99f)), Line(At("Hail", 0, 40, 0.99f, 50), At("O", 44, 56, 0.4f, 50), At("Captain", 60, 130, 0.99f, 50))];

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

    private static OrderStrip Placed(params (string Text, float Confidence, OcrBox Box)[] words) => new()
    {
        Leaf = 0,
        Box = new OcrBox(words.Min(w => w.Box.Left), words.Min(w => w.Box.Top), words.Max(w => w.Box.Right), words.Max(w => w.Box.Bottom)),
        Size = 30,
        Words = [.. words.Select(w => Word(w.Text, w.Confidence, w.Box))],
    };

    [Fact]
    public void ACheckboxReadWithItsBracketIsDroppedBesideALabelThatStartsCloseToIt()
    {
        // The box is 38 wide and 43 high; its label starts 20 pixels away, under half a height: a lone letter would stay.
        OrderStrip?[] strips =
        [
            Placed(("Dl]", 0.4f, new OcrBox(288, 807, 325, 850)), ("Send", 0.99f, new OcrBox(345, 810, 400, 845)), ("a", 0.99f, new OcrBox(410, 810, 425, 845))),
            Placed(("E", 0.4f, new OcrBox(288, 907, 325, 950)), ("Send", 0.99f, new OcrBox(345, 910, 400, 945))),
        ];

        Assert.Equal(1, JunkFilter.Apply(strips, new JunkParameters()));
        Assert.Equal(["Send", "a"], strips[0]!.Words.Select(w => w.Text));
        Assert.Equal(["E", "Send"], strips[1]!.Words.Select(w => w.Text));
    }

    // A page of confident text: the stroke and contained-fragment rules only apply where most of the page was read with confidence.
    private static OrderStrip Confident() =>
        Placed((new string('a', 200), 0.99f, new OcrBox(500, 500, 1500, 530)));

    [Fact]
    public void ANarrowUnsureCharacterIsAStrokeButAConfidentOneOrAWideOneIsNot()
    {
        OrderStrip?[] strips =
        [
            Placed(("1", 0.03f, new OcrBox(0, 0, 6, 23))), // a table rule's edge
            Placed(("1", 0.9f, new OcrBox(0, 40, 6, 63))), // a real digit
            Placed(("o", 0.1f, new OcrBox(0, 80, 17, 103))), // wide: a letter o
            Placed(("-", 0.1f, new OcrBox(0, 120, 6, 123))), // not a letter or digit
            Placed(("1", 0.3f, new OcrBox(0, 160, 6, 183))), // unsure, but not as unsure as a stroke
            Confident(),
        ];

        Assert.Equal(1, JunkFilter.Apply(strips, new JunkParameters()));
        Assert.Null(strips[0]);
        Assert.All(strips.Skip(1), s => Assert.NotNull(s));
        Assert.Equal(0, JunkFilter.Apply([Placed(("1", 0.03f, new OcrBox(0, 0, 6, 23))), Confident()], new JunkParameters(StrokeConfidence: 0)));
    }

    [Fact]
    public void StrokesAndFragmentsStayOnPagesThatAreNotMostlyConfident()
    {
        // A photo of a receipt: thin real characters are read with little confidence, so nothing is judged a stroke.
        OrderStrip?[] strips =
        [
            Placed(("1", 0.03f, new OcrBox(0, 0, 6, 23))),
            Placed(("perfectly", 0.5f, new OcrBox(100, 100, 300, 130)), ("p", 0.5f, new OcrBox(102, 102, 112, 130))),
        ];

        Assert.Equal(0, JunkFilter.Apply(strips, new JunkParameters()));
    }

    [Fact]
    public void AShortWordInsideALongerWordsBoxIsAFragmentOfIt()
    {
        OrderStrip?[] strips =
        [
            Placed(("perfectly", 0.99f, new OcrBox(100, 100, 300, 130)), ("p", 0.95f, new OcrBox(102, 102, 112, 130))),
            // Side by side words, and a short word that overlaps a neighbour only a little, stay.
            Placed(("a", 0.99f, new OcrBox(100, 200, 112, 230)), ("road", 0.99f, new OcrBox(110, 200, 180, 230))),
            Placed(("in", 0.99f, new OcrBox(500, 100, 520, 130)), ("it", 0.99f, new OcrBox(500, 100, 520, 130))), // equal length: neither is the longer
            // Two characters are a real word more often than a fragment ("If" beside a larger word); only single characters are judged.
            Placed(("offset", 0.99f, new OcrBox(700, 100, 900, 130)), ("of", 0.99f, new OcrBox(702, 102, 740, 130))),
            Confident(),
        ];

        Assert.Equal(1, JunkFilter.Apply(strips, new JunkParameters(RepeatFraction: 0)));
        Assert.Equal(["perfectly"], strips[0]!.Words.Select(w => w.Text));
        Assert.Equal(2, strips[1]!.Words.Count);
        Assert.Equal(2, strips[2]!.Words.Count);
        Assert.Equal(2, strips[3]!.Words.Count);
        Assert.Equal(0, JunkFilter.Apply([Placed(("perfectly", 0.99f, new OcrBox(100, 100, 300, 130)), ("p", 0.95f, new OcrBox(102, 102, 112, 130)))], new JunkParameters(ContainedFraction: 0, RepeatFraction: 0)));
    }

    [Fact]
    public void AStretchOfALineBoxedAgainIsDroppedAsARepeat()
    {
        OrderStrip?[] strips =
        [
            Placed(("firearms.", 0.91f, new OcrBox(1132, 1137, 1269, 1177)), ("in", 1.0f, new OcrBox(1093, 1140, 1116, 1164))),
            // The same stretch read again from a box of its own: pieces of the word and a second "in".
            Placed(("firea", 0.98f, new OcrBox(1136, 1150, 1193, 1165)), ("arm", 0.92f, new OcrBox(1188, 1150, 1226, 1164)), ("ms.", 0.95f, new OcrBox(1231, 1150, 1269, 1164)), ("in", 0.94f, new OcrBox(1092, 1140, 1116, 1156))),
            // Words of a line that merely sit side by side stay, and so does a word that is not made of the other's letters.
            Placed(("the", 0.99f, new OcrBox(100, 200, 140, 230)), ("then", 0.99f, new OcrBox(150, 200, 210, 230)), ("xyz", 0.99f, new OcrBox(1136, 1150, 1193, 1165))),
            Confident(),
        ];

        Assert.Equal(4, JunkFilter.Apply(strips, new JunkParameters()));
        Assert.Equal(["firearms.", "in"], strips[0]!.Words.Select(w => w.Text));
        Assert.Equal(["the", "then", "xyz"], strips[2]!.Words.Select(w => w.Text));
        Assert.Null(strips[1]);
    }
}
