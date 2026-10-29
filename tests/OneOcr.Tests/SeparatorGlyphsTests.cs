using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

/// <summary>Middle dots, bullets and vertical bars read from their pixels, and nothing else changed.</summary>
public class SeparatorGlyphsTests
{
    private const int Height = 32, Width = 220;

    // A line image: light background, dark ink. Letters are 6-wide blocks 12 rows high (rows 10..21), as x-height letters would be.
    private sealed class Line
    {
        public readonly byte[] Pixels = new byte[Width * Height];
        public readonly List<RecognizedWord> Words = [];

        public Line() => Array.Fill(Pixels, (byte)235);

        public void Block(int x, int y, int w, int h)
        {
            for (int yy = y; yy < y + h; yy++)
            {
                for (int xx = x; xx < x + w; xx++)
                {
                    Pixels[(yy * Width) + xx] = 20;
                }
            }
        }

        // A word of letter blocks starting at x; returns the x after it.
        public int Letters(string text, int x)
        {
            for (int i = 0; i < text.Length; i++)
            {
                Block(x + (i * 8), 10, 6, 12);
            }

            Words.Add(Make(text, x, x + (text.Length * 8) - 2));
            return x + (text.Length * 8);
        }

        // A one-character word read as `read` whose ink is the rectangle (x, y, w, h).
        public void Mark(string read, int x, int y, int w, int h, float confidence = 0.2f)
        {
            Block(x, y, w, h);
            Words.Add(Make(read, x - 1, x + w + 1, confidence));
        }

        public LineWords Result() => new("", 0.5f, Words);
    }

    private static RecognizedWord Make(string text, float x0, float x1, float confidence = 0.9f) =>
        new(text, x0, x1, confidence, confidence, OcrScript.Latin, text.Length, [.. text.Select((c, i) => new RecognizedChar(c.ToString(), x0 + i, x0 + i + 1, 0.5f))]);

    // "abcd", a mark, "efgh" with the mark of the given size and place; returns what the mark was read as afterwards.
    private static string Between(string read, int w, int h, int y, float confidence = 0.2f)
    {
        var line = new Line();
        int x = line.Letters("abcd", 10);
        line.Mark(read, x + 6, y, w, h, confidence);
        line.Letters("efgh", x + 6 + w + 8);
        LineWords after = SeparatorGlyphs.Apply(line.Result(), line.Pixels, Width, Height);
        return after.Words[1].Text;
    }

    [Theory]
    [InlineData("-", 3, 3, 14, "\u00B7")] // a small round blob in the middle of the letters
    [InlineData(".", 3, 3, 14, "\u00B7")]
    [InlineData("*", 2, 3, 14, "\u00B7")]
    [InlineData("-", 2, 1, 15, "\u00B7")] // a tiny dot is as wide as it is tall give or take a pixel
    [InlineData("-", 7, 7, 13, "\u2022")] // a large round one is a bullet
    [InlineData("=", 6, 6, 13, "\u2022")]
    [InlineData("\u00B7", 7, 7, 13, "\u2022")] // the network's dot is changed when the size clearly says bullet
    [InlineData("\u2022", 3, 3, 14, "\u00B7")] // ... and its bullet when it clearly says dot
    public void ADotOrBulletIsReadBySize(string read, int w, int h, int y, string expected) =>
        Assert.Equal(expected, Between(read, w, h, y));

    [Theory]
    [InlineData("-", 7, 2, 15)] // a hyphen is wide and flat
    [InlineData("\u2013", 10, 2, 15)]
    [InlineData(".", 3, 3, 20)] // a full stop sits on the baseline, not in the middle of the letters
    [InlineData("-", 3, 3, 4)] // a mark above the letters
    [InlineData("-", 14, 14, 9)] // too big for a dot
    [InlineData("\u00B7", 4, 4, 14)] // the network's dot between the margins stays
    [InlineData("\u2022", 5, 5, 14)] // and so does its bullet
    [InlineData("a", 3, 3, 14)] // a letter is never a candidate
    public void HyphensDashesAndOtherMarksStay(string read, int w, int h, int y) =>
        Assert.Equal(read, Between(read, w, h, y));

    [Fact]
    public void ASureShortHyphenStaysAndAnUnsureOneBecomesADot()
    {
        // 3 x 2: as wide as 150% of its height.
        Assert.Equal("-", Between("-", 3, 2, 15, confidence: 0.95f));
        Assert.Equal("\u00B7", Between("-", 3, 2, 15, confidence: 0.2f));
    }

    [Theory]
    [InlineData("I")]
    [InlineData("l")]
    [InlineData("1")]
    [InlineData("!")]
    public void AThinBarTallerThanTheLettersIsAVerticalBar(string read) =>
        Assert.Equal("|", Between(read, 2, 24, 4));

    [Theory]
    [InlineData(2, 12, 10)] // as tall as the letters: a capital I or a one
    [InlineData(6, 24, 4)] // tall but thick: a letter
    [InlineData(2, 14, 9)] // barely taller than the letters
    public void ALetterLikeStrokeStays(int w, int h, int y) => Assert.Equal("I", Between("I", w, h, y));

    [Fact]
    public void ABarIsNotMeasuredAgainstOtherBarsOfTheLine()
    {
        var line = new Line();
        int x = line.Letters("abcd", 10);
        line.Mark("I", x + 6, 4, 2, 24);
        x = line.Letters("efgh", x + 20);
        line.Mark("l", x + 6, 4, 2, 24);
        line.Letters("ijkl", x + 20);
        LineWords after = SeparatorGlyphs.Apply(line.Result(), line.Pixels, Width, Height);
        Assert.Equal(["abcd", "|", "efgh", "|", "ijkl"], after.Words.Select(w => w.Text));
    }

    [Fact]
    public void OnlyTheTextChangesAndEveryCharacterKeepsItsExtent()
    {
        var line = new Line();
        int x = line.Letters("abcd", 10);
        line.Mark("-", x + 6, 14, 3, 3, confidence: 0.37f);
        line.Letters("efgh", x + 20);
        LineWords before = line.Result();
        LineWords after = SeparatorGlyphs.Apply(before, line.Pixels, Width, Height);
        RecognizedWord was = before.Words[1], now = after.Words[1];
        Assert.Equal("\u00B7", now.Text);
        Assert.Equal([new RecognizedChar("\u00B7", was.Chars[0].X0, was.Chars[0].X1, was.Chars[0].Probability)], now.Chars);
        Assert.Equal((was.X0, was.X1, was.Confidence, was.RawConfidence, was.Script, was.Characters), (now.X0, now.X1, now.Confidence, now.RawConfidence, now.Script, now.Characters));
        Assert.Equal(before.Words[0], after.Words[0]);
        Assert.Equal(before.Words[2], after.Words[2]);
        Assert.Equal(before.Text, after.Text);
        Assert.Equal(Between("-", 3, 3, 14), Between("-", 3, 3, 14)); // a pure function of the pixels
    }

    [Fact]
    public void WordsOfSeveralCharactersAndTheEndsOfALineAreNeverTouched()
    {
        // A mark at the start of the line, a two-character word "--" in the middle, and a mark at the end.
        var line = new Line();
        line.Mark("-", 10, 14, 3, 3);
        int x = line.Letters("abcd", 20);
        line.Block(x + 6, 14, 3, 3);
        line.Words.Add(Make("--", x + 5, x + 10));
        x = line.Letters("efgh", x + 20);
        line.Mark("-", x + 6, 14, 3, 3);
        LineWords before = line.Result();
        Assert.Same(before, SeparatorGlyphs.Apply(before, line.Pixels, Width, Height));

        // A hyphen inside a word: "a-b" is one three-character word.
        var inside = new Line();
        int h = inside.Letters("abcd", 10);
        inside.Block(h + 4, 14, 3, 3);
        inside.Words.Add(Make("a-b", h, h + 20));
        inside.Letters("efgh", h + 30);
        LineWords unchanged = inside.Result();
        Assert.Same(unchanged, SeparatorGlyphs.Apply(unchanged, inside.Pixels, Width, Height));
    }

    [Theory]
    [InlineData("\u00BA", 7, 7, 13, "\u2022")] // a list marker the network read as an ordinal sign
    [InlineData("\u00B0", 3, 3, 14, "\u00B7")]
    [InlineData("\u2022", 3, 3, 14, "\u00B7")]
    public void ABulletAtTheStartOrEndOfALineIsReadBySize(string read, int w, int h, int y, string expected)
    {
        var first = new Line();
        first.Mark(read, 10, y, w, h);
        first.Letters("abcd", 10 + w + 10);
        Assert.Equal(expected, SeparatorGlyphs.Apply(first.Result(), first.Pixels, Width, Height).Words[0].Text);

        var last = new Line();
        int x = last.Letters("abcd", 10);
        last.Mark(read, x + 8, y, w, h);
        Assert.Equal(expected, SeparatorGlyphs.Apply(last.Result(), last.Pixels, Width, Height).Words[1].Text);
    }

    [Theory]
    [InlineData("-")] // a list may be marked with a hyphen
    [InlineData(".")]
    [InlineData("I")] // a bar mark at a line end could be a letter
    [InlineData("a")]
    [InlineData("*")]
    public void OtherMarksAtTheEndsOfALineStayAsRead(string read)
    {
        var line = new Line();
        line.Mark(read, 10, 14, 3, 3);
        line.Letters("abcd", 30);
        LineWords before = line.Result();
        Assert.Same(before, SeparatorGlyphs.Apply(before, line.Pixels, Width, Height));
    }

    [Fact]
    public void FaintLinesAndLinesWithoutLettersAreLeftAsRead()
    {
        var faint = new Line();
        int x = faint.Letters("abcd", 10);
        faint.Mark("-", x + 6, 14, 3, 3);
        faint.Letters("efgh", x + 20);
        for (int i = 0; i < faint.Pixels.Length; i++)
        {
            faint.Pixels[i] = faint.Pixels[i] == 20 ? (byte)200 : faint.Pixels[i];
        }

        LineWords line = faint.Result();
        Assert.Same(line, SeparatorGlyphs.Apply(line, faint.Pixels, Width, Height));

        var bare = new Line();
        bare.Block(10, 10, 3, 12);
        bare.Words.Add(Make("a", 9, 14));
        bare.Mark("-", 30, 14, 3, 3);
        bare.Block(50, 10, 3, 12);
        bare.Words.Add(Make("b", 49, 54));
        LineWords two = bare.Result();
        Assert.Same(two, SeparatorGlyphs.Apply(two, bare.Pixels, Width, Height)); // too little ink to tell what the letters are
    }

    [Fact]
    public void TheStepIsOnByDefaultAndSwitchable()
    {
        Assert.True(new OcrOptions().FixSeparators);
        Assert.False(new OcrOptions { FixSeparators = false }.FixSeparators);
    }
}
