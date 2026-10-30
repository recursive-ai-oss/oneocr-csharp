using OneOcr.Decoding;
using OneOcr.Layout;
using OneOcr.Unicode;

namespace OneOcr.Tests;

/// <summary>Accents and dots that grain puts on letters are taken off; real accented text and unlisted words are left alone.</summary>
public class StrayAccentsTests
{
    private static readonly WordFrequencyModel Frequencies = WordFrequencyModel.Parse(TestWordFrequencies.Build(
    [
        ("the", 60000), ("down", 900), ("budget", 40), ("pronounced", 30), ("resume", 300), ("th\u00e9", 4), ("caf\u00e9", 60), ("cafe", 90), ("ete", 20), ("\u00e9t\u00e9", 8000),
    ]));

    private static OcrWord Word(string text, float confidence, int x = 0) =>
        new(text, new OcrBox(x, 0, x + 16, 5), confidence, confidence, OcrSource.Ocr, []);

    private static OrderStrip Strip(params (string Text, float Confidence)[] words) => new()
    {
        Leaf = 0,
        Box = new OcrBox(0, 0, 400, 10),
        Size = 10,
        Words = [.. words.Select((w, k) => Word(w.Text, w.Confidence, 20 * k))],
    };

    // Many plain words so that the page counts as unaccented.
    private static OrderStrip Plain() => Strip(Enumerable.Repeat(("alpha", 0.99f), 40).ToArray());

    private static string[] Run(OrderStrip body, out int changed)
    {
        OrderStrip?[] strips = [Plain(), body];
        JunkFilter.Apply(strips, new JunkParameters(), Frequencies, out changed);
        return [.. strips[1]!.Words.Select(w => w.Text)];
    }

    [Fact]
    public void StripMarksFindsThePlainLetter()
    {
        Assert.True(TextNormalization.TryStripMarks('\u00e9', out char e));
        Assert.Equal('e', e);
        Assert.True(TextNormalization.TryStripMarks('\u1ecc', out char o));
        Assert.Equal('O', o);
        Assert.True(TextNormalization.TryStripMarks('\u0219', out char s));
        Assert.Equal('s', s);
        Assert.False(TextNormalization.TryStripMarks('e', out _));
        Assert.False(TextNormalization.TryStripMarks('\u00df', out _));
        Assert.False(TextNormalization.TryStripMarks('\u0439', out _));
        Assert.False(TextNormalization.TryStripMarks('\u00e6', out _));
    }

    [Fact]
    public void UnsureAccentedWordsGoBackToTheirPlainSpelling()
    {
        string[] words = Run(Strip(("th\u00e9", 0.7f), ("d\u1ecdwn,", 0.6f), ("\u201cTh\u00e9", 0.8f), ("bl\u00e9ck", 0.5f)), out int changed);

        // The first three are plain words once the marks are off; the last is not a word the table knows.
        Assert.Equal(["the", "down,", "\u201cThe", "bl\u00e9ck"], words);
        Assert.Equal(3, changed);
    }

    [Fact]
    public void CaseAndBoxAndConfidenceAreKept()
    {
        OrderStrip?[] strips = [Plain(), Strip(("Th\u00e9", 0.7f))];
        JunkFilter.Apply(strips, new JunkParameters(), Frequencies, out int changed);

        OcrWord w = strips[1]!.Words[0];
        Assert.Equal(1, changed);
        Assert.Equal("The", w.Text);
        Assert.Equal(0.7f, w.Confidence);
        Assert.Equal(new OcrBox(0, 0, 16, 5), w.Box);
    }

    [Fact]
    public void ConfidentWordsAndCommonAccentedWordsStay()
    {
        // 0.99 is a word the network is sure of; \u00e9t\u00e9 is far more frequent than ete; caf\u00e9 is not eight times rarer than cafe.
        string[] words = Run(Strip(("th\u00e9", 0.99f), ("\u00e9t\u00e9", 0.5f), ("caf\u00e9", 0.5f)), out int changed);

        Assert.Equal(["th\u00e9", "\u00e9t\u00e9", "caf\u00e9"], words);
        Assert.Equal(0, changed);
    }

    [Fact]
    public void WordsWithDigitsInnerMarksOrOtherScriptsStay()
    {
        string[] words = Run(Strip(("th\u00e92", 0.5f), ("d\u1ecd-wn", 0.5f), ("th\u0435", 0.5f), ("\u00e9", 0.5f)), out int changed);

        Assert.Equal(["th\u00e92", "d\u1ecd-wn", "th\u0435", "\u00e9"], words);
        Assert.Equal(0, changed);
    }

    [Fact]
    public void APageInALanguageWithAccentsIsLeftAlone()
    {
        // Every second word has an accent: the page is French, whatever the table says about single words.
        OrderStrip?[] strips = [Strip(("th\u00e9", 0.5f), ("the", 0.9f), ("th\u00e9", 0.5f), ("the", 0.9f))];
        JunkFilter.Apply(strips, new JunkParameters(), Frequencies, out int changed);

        Assert.Equal(0, changed);
        Assert.Equal("th\u00e9", strips[0]!.Words[0].Text);
    }

    [Fact]
    public void CharactersFollowTheWord()
    {
        var chars = new List<OcrChar>();
        for (int i = 0; i < 3; i++)
        {
            chars.Add(new OcrChar("th\u00e9"[i].ToString(), new OcrBox(i * 5, 0, (i * 5) + 5, 5), 0.5f));
        }

        var word = new OcrWord("th\u00e9", new OcrBox(0, 0, 15, 5), 0.5f, 0.5f, OcrSource.Ocr, chars);
        OrderStrip?[] strips = [Plain(), new OrderStrip { Leaf = 0, Box = new OcrBox(0, 0, 100, 10), Size = 10, Words = [word] }];
        JunkFilter.Apply(strips, new JunkParameters(), Frequencies, out _);

        OcrWord result = strips[1]!.Words[0];
        Assert.Equal("the", result.Text);
        Assert.Equal(["t", "h", "e"], result.Chars.Select(c => c.Text));
    }

    [Fact]
    public void NoFrequenciesOrZeroConfidenceBoundMeansOff()
    {
        OrderStrip?[] strips = [Plain(), Strip(("th\u00e9", 0.5f))];
        JunkFilter.Apply(strips, new JunkParameters(), null, out int none);
        Assert.Equal(0, none);

        JunkFilter.Apply(strips, new JunkParameters(AccentConfidence: 0), Frequencies, out int off);
        Assert.Equal(0, off);
        Assert.Equal("th\u00e9", strips[1]!.Words[0].Text);
        Assert.True(new OcrOptions().FixStrayAccents);
    }
}
