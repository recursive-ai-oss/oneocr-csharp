using OneOcr.Decoding;
using OneOcr.Models;
using OneOcr.Recognition;

namespace OneOcr.Tests;

/// <summary>Word breaks from the line's pixels: numbers and dates are joined, wide empty stretches are cut, letter-spaced print is left alone.</summary>
public class WordSegmenterTests
{
    [Theory]
    [InlineData("£96,", "221.95", true)]
    [InlineData("$40,", "181.65", true)]
    [InlineData("1,234,", "567", true)]
    [InlineData("16,0", "80.65", true)]
    [InlineData("2014-", "02-15.", true)]
    [InlineData("24/07", "/2010.", true)]
    [InlineData("total:", "1,000", false)]
    [InlineData("1,", "2345", false)]
    [InlineData("on", "5", false)]
    [InlineData("5", "3", false)]
    [InlineData("order", "-12", false)]
    [InlineData("Invoice", "12", false)]
    [InlineData("12,345", "6", false)]
    public void NumbersAndDatesAreJoinedOnlyWhenTheyAreHalvesOfOne(string left, string right, bool joined) =>
        Assert.Equal(joined, WordSegmenter.JoinsNumber(left, right));

    [Theory]
    [InlineData("elena.nair", "@example.com", true)]
    [InlineData("fatima.costa@corp.", "example", true)]
    [InlineData("dad@corp.example.", "net", true)]
    [InlineData("IRS.", "gov", true)]
    [InlineData("example.", "COM,", true)]
    [InlineData("https:", "//www.irs.gov", true)]
    [InlineData("https://www.irs.gov/", "forms", true)]
    [InlineData("$", "17,906.36", true)]
    [InlineData("US$", "50", true)]
    [InlineData("$", "9", false)]
    [InlineData("follow", "@example.com", true)]
    [InlineData("us", "@nasa", false)]
    [InlineData("us", "@", false)]
    [InlineData("mail", "@ me", false)]
    [InlineData("IRS.", "Gov", false)]
    [InlineData("It.", "us", false)]
    [InlineData("U.S.", "government", false)]
    [InlineData("done.", "orgé", false)]
    [InlineData("cost:", "17", false)]
    [InlineData("5$", "17", false)]
    [InlineData("visit", "//home", false)]
    [InlineData("docs/", "intro", false)]
    [InlineData("a@b.", "Net", false)]
    public void AddressesWebAddressesAndAmountsAreJoinedOnlyWhenTheyAreHalvesOfOne(string left, string right, bool joined) =>
        Assert.Equal(joined, WordSegmenter.JoinsCode(left, right));

    // A line image 32 pixels high, white with black blocks at the given [from, to] column ranges.
    private static byte[] Image(int width, params (int From, int To)[] ink)
    {
        var pixels = new byte[width * 32];
        Array.Fill(pixels, (byte)255);
        foreach ((int from, int to) in ink)
        {
            for (int y = 8; y < 24; y++)
            {
                for (int x = from; x <= to; x++)
                {
                    pixels[(y * width) + x] = 0;
                }
            }
        }

        return pixels;
    }

    // One character per ink block, with the block as its extent.
    private static List<RecognizedChar> Chars(string text, params (int From, int To)[] ink) =>
        [.. text.Select((c, i) => new RecognizedChar(c.ToString(), ink[i].From, ink[i].To + 1, 1f))];

    [Fact]
    public void ASpaceBeforeTheAtSignOfAnAddressIsTakenOutWhenTheInkIsClose()
    {
        // "jo" "@a.co": a four pixel stretch, no word model or table needed.
        (int, int)[] ink = [(5, 12), (15, 22), (26, 33), (36, 43), (46, 49), (52, 59)];
        bool[]? plan = WordSegmenter.Plan(Chars("jo@a.c", ink), [false, true, false, false, false], Image(70, ink), 70, 32, null);

        Assert.NotNull(plan);
        Assert.All(plan!, b => Assert.False(b));
    }

    [Fact]
    public void TheSameSpaceStaysWhenTheStretchIsWiderThanCodeGap()
    {
        (int, int)[] ink = [(5, 12), (15, 22), (31, 38), (41, 48), (51, 54), (57, 64)];

        Assert.Null(WordSegmenter.Plan(Chars("jo@a.c", ink), [false, true, false, false, false], Image(70, ink), 70, 32, null));
    }

    [Fact]
    public void ASpaceInsideAThousandsGroupIsTakenOutWhenTheInkIsClose()
    {
        (int, int)[] ink = [(5, 12), (15, 17), (22, 29), (32, 39), (42, 49)];
        bool[] was = [false, true, false, false];

        bool[]? plan = WordSegmenter.Plan(Chars("1,234", ink), was, Image(60, ink), 60, 32, null);

        Assert.NotNull(plan);
        Assert.All(plan!, b => Assert.False(b));
    }

    [Fact]
    public void TheSameSpaceStaysWhenTheStretchIsAsWideAsAWordGap()
    {
        (int, int)[] ink = [(5, 12), (15, 17), (30, 37), (40, 47), (50, 57)];

        Assert.Null(WordSegmenter.Plan(Chars("1,234", ink), [false, true, false, false], Image(70, ink), 70, 32, null));
    }

    [Fact]
    public void ASpaceBetweenDigitsThatAreNotAGroupStays()
    {
        (int, int)[] ink = [(5, 12), (15, 22), (25, 27), (30, 37), (40, 47)];

        Assert.Null(WordSegmenter.Plan(Chars("12,34", ink), [false, false, true, false], Image(60, ink), 60, 32, null));
    }

    [Fact]
    public void AWideEmptyStretchInsideAWordBecomesASpace()
    {
        (int, int)[] ink = [(5, 12), (15, 22), (45, 52), (55, 62), (65, 72)];

        bool[]? plan = WordSegmenter.Plan(Chars("Iagre", ink), [false, false, false, false], Image(80, ink), 80, 32, null);

        Assert.NotNull(plan);
        Assert.Equal([false, true, false, false], plan!);
    }

    [Fact]
    public void LetterSpacedPrintIsNotCut()
    {
        // Every letter gap is 10 pixels, as in spaced capitals; none stands out.
        (int, int)[] ink = [(5, 12), (23, 30), (41, 48), (59, 66), (77, 84)];

        Assert.Null(WordSegmenter.Plan(Chars("TOTAL", ink), [false, false, false, false], Image(90, ink), 90, 32, null));
    }

    [Fact]
    public void AStretchThatANetworkSpaceAccountsForIsNotACut()
    {
        (int, int)[] ink = [(5, 12), (15, 22), (45, 52), (55, 62)];

        Assert.Null(WordSegmenter.Plan(Chars("abcd", ink), [false, true, false], Image(70, ink), 70, 32, null));
    }

    [Fact]
    public void NumbersAreNeverCutBetweenDigits()
    {
        (int, int)[] ink = [(5, 12), (15, 22), (45, 52), (55, 62)];

        Assert.Null(WordSegmenter.Plan(Chars("1234", ink), [false, false, false], Image(70, ink), 70, 32, null));
    }

    [Fact]
    public void FaintLinesAreLeftAsRead()
    {
        (int, int)[] ink = [(5, 12), (15, 22), (45, 52), (55, 62)];
        byte[] image = Image(70, ink);
        for (int i = 0; i < image.Length; i++)
        {
            image[i] = (byte)(200 + (image[i] / 8)); // contrast under the floor
        }

        Assert.Null(WordSegmenter.Plan(Chars("abcd", ink), [false, false, false], image, 70, 32, null));
    }

    [Fact]
    public void AnImageShorterThanItsSizeGivesNoPlan() =>
        Assert.Null(WordSegmenter.Plan(Chars("ab", (1, 2), (5, 6)), [false], new byte[10], 70, 32, null));

    private static WordPlausibility Words() =>
        new(CharLanguageModel.Parse(TestLm.Build("test", ["a", "b", "c", "d"], [.. Enumerable.Repeat("abcd", 20), .. Enumerable.Repeat("dcba", 20)], order: 4)));

    [Fact]
    public void TheWordModelScoresWordsLikeItsTrainingWordsHigher()
    {
        WordPlausibility words = Words();

        Assert.True(words.Score("abcd") > words.Score("abdc"));
        Assert.True(words.Score("abcd") > words.Score("ab") + words.Score("cd"));
        Assert.Null(words.Score("ab1"));
        Assert.Null(words.Score("abz"));
        Assert.Null(words.Score(""));
    }

    [Fact]
    public void ASpaceBetweenFragmentsOfAWordIsTakenOutWhenTheModelPrefersTheJoinedWord()
    {
        (int, int)[] ink = [(5, 12), (15, 22), (25, 32), (35, 42)];
        byte[] image = Image(50, ink);

        bool[]? plan = WordSegmenter.Plan(Chars("abcd", ink), [false, true, false], image, 50, 32, Words());
        Assert.NotNull(plan);
        Assert.All(plan!, b => Assert.False(b));

        // Without the model there is no evidence that the fragments belong together.
        Assert.Null(WordSegmenter.Plan(Chars("abcd", ink), [false, true, false], image, 50, 32, null));
    }

    [Fact]
    public void CapitalsInsideAPieceKeepTheModelOut()
    {
        (int, int)[] ink = [(5, 12), (15, 22), (25, 32), (35, 42)];
        var chars = new List<RecognizedChar>
        {
            new("a", 5, 13, 1f), new("b", 15, 23, 1f), new("C", 25, 33, 1f), new("D", 35, 43, 1f),
        };

        Assert.Null(WordSegmenter.Plan(chars, [false, true, false], Image(50, ink), 50, 32, Words()));
    }

    // Ink blocks of 6 pixels per letter, 2 pixels apart, except the stretch of `stretch` pixels after letter `cut`.
    private static (int, int)[] Ink(int letters, int cut, int stretch)
    {
        var ink = new (int, int)[letters];
        int x = 5;
        for (int i = 0; i < letters; i++)
        {
            ink[i] = (x, x + 5);
            x += 6 + (i == cut ? stretch : 2);
        }

        return ink;
    }

    [Theory]
    [InlineData("Company", 2, 4, true)] // gain 3.5 nats at 4 px: plain evidence
    [InlineData("Company", 2, 6, false)] // the same gain at 6 px: over the plain bound, and neither "Com" nor "pany" is a non-word
    [InlineData("January", 3, 4, true)] // gain 1.66 nats: weak evidence, enough because "Janu" scores as no word
    [InlineData("taxat", 2, 4, false)] // gain 1.66 too, but "tax" is a rare real word: not bad enough to call it a fragment
    [InlineData("abough", 0, 4, false)] // "a" and "bough" are words: a gain under the plain margin does not join them
    public void WeakEvidenceJoinsAFragmentOnlyWhenAPieceIsNoWord(string text, int cut, int stretch, bool joins)
    {
        byte[] data = [];
        if (!BundledModels.HasLanguageModel || BundledModels.TryReadLanguageModel(out data) is not null)
        {
            return;
        }

        var words = new WordPlausibility(CharLanguageModel.Parse(data));
        (int, int)[] ink = Ink(text.Length, cut, stretch);
        int width = ink[^1].Item2 + 8;
        bool[] was = new bool[text.Length - 1];
        was[cut] = true;

        bool[]? plan = WordSegmenter.Plan(Chars(text, ink), was, Image(width, ink), width, 32, words);
        Assert.Equal(joins, plan is not null && !plan[cut]);
    }
}
