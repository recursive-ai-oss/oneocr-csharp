using System.Buffers.Binary;
using System.Text;
using OneOcr.Decoding;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

/// <summary>The word frequency file (parser, lookup, limits) and the rule that uses it to take back a space inside a word.</summary>
public class WordFrequencyTests
{
    private const int Total = TestWordFrequencies.Total;

    private static byte Quarter(int count) => TestWordFrequencies.Quarter(count);

    private static List<(string, int)> Vocabulary()
    {
        var list = new List<(string, int)>();
        for (int i = 0; i < 200; i++)
        {
            list.Add(("w" + new string((char)('a' + (i % 26)), 1 + (i / 26)), 3 + (i * 7)));
        }

        list.AddRange([("conditionally", 400), ("condition", 3000), ("ally", 20), ("daresay", 40), ("dare", 300), ("say", 2000), ("indebted", 120), ("in", 20000), ("united", 2500), ("unit", 400), ("ed", 30), ("the", 60000)]);
        return list;
    }

    [Fact]
    public void EveryWordIsFoundWithItsCountAndOthersAreNot()
    {
        List<(string Word, int Count)> words = Vocabulary();
        WordFrequencyModel model = WordFrequencyModel.Parse(TestWordFrequencies.Build(words));

        Assert.Equal(words.Count, model.WordCount);
        foreach ((string word, int count) in words)
        {
            Assert.True(model.TryFind(Encoding.UTF8.GetBytes(word), out int q), word);
            Assert.Equal(Quarter(count), q);
        }

        Assert.False(model.TryFind("zzzz"u8, out _));
        Assert.False(model.TryFind("condition_"u8, out _));
        Assert.False(model.TryFind([], out _));
        Assert.Equal(2, model.Quarter("nosuchword"));
    }

    [Fact]
    public void AnEmptyTableParsesAndFindsNothing()
    {
        WordFrequencyModel model = WordFrequencyModel.Parse(TestWordFrequencies.Build([]));

        Assert.Equal(0, model.WordCount);
        Assert.False(model.TryFind("a"u8, out _));
    }

    [Fact]
    public void MalformedFilesAreRefusedWithATypedException()
    {
        byte[] good = TestWordFrequencies.Build(Vocabulary());

        // Every prefix of the file is either refused or (never) accepted; nothing else may come out.
        for (int length = 0; length < good.Length; length += 7)
        {
            Assert.Throws<OcrModelException>(() => WordFrequencyModel.Parse(good.AsSpan(0, length)));
        }

        byte[] badMagic = (byte[])good.Clone();
        badMagic[0] = (byte)'X';
        Assert.Throws<OcrModelException>(() => WordFrequencyModel.Parse(badMagic));

        byte[] badVersion = (byte[])good.Clone();
        badVersion[4] = 2;
        Assert.Throws<OcrModelException>(() => WordFrequencyModel.Parse(badVersion));

        byte[] flags = (byte[])good.Clone();
        flags[6] = 1;
        Assert.Throws<OcrModelException>(() => WordFrequencyModel.Parse(flags));

        // A block count that does not fit the word count.
        byte[] blocks = (byte[])good.Clone();
        BinaryPrimitives.WriteUInt32LittleEndian(blocks.AsSpan(18), 1);
        Assert.Throws<OcrModelException>(() => WordFrequencyModel.Parse(blocks));

        // Trailing bytes.
        Assert.Throws<OcrModelException>(() => WordFrequencyModel.Parse([.. good, 0]));
    }

    [Fact]
    public void WordsOutOfOrderAreRefused()
    {
        byte[] good = TestWordFrequencies.Build([("apple", 10), ("banana", 10)]);
        // The index is 1 block (4 bytes) after the 22-byte header; the first entry is shared=0, suffix=5 "apple", q.
        int first = 22 + 4;
        byte[] swapped = (byte[])good.Clone();
        swapped[first + 2] = (byte)'c';

        Assert.Throws<OcrModelException>(() => WordFrequencyModel.Parse(swapped));
    }

    [Fact]
    public void MoreWordsThanTheLimitAllowsIsALimitErrorBeforeAnyWork()
    {
        byte[] good = TestWordFrequencies.Build(Vocabulary());

        Assert.Throws<OcrLimitException>(() => WordFrequencyModel.Parse(good, new OcrLimits { MaxWordModelWords = 10 }));
        Assert.Throws<OcrLimitException>(() => WordFrequencyModel.Parse(good, new OcrLimits { MaxLanguageModelFileBytes = 100 }));
        // A header that claims billions of words in a tiny file is refused without allocating for them.
        byte[] huge = (byte[])good.Clone();
        BinaryPrimitives.WriteUInt32LittleEndian(huge.AsSpan(12), uint.MaxValue);
        Assert.Throws<OcrLimitException>(() => WordFrequencyModel.Parse(huge));
    }

    // Characters laid out as blocks of ink 8 pixels wide with the given stretch between the pieces.
    private static (List<RecognizedChar> Chars, bool[] WasSpace, byte[] Image, int Width) Line(string left, string right, int stretch)
    {
        string text = left + right;
        var chars = new List<RecognizedChar>();
        var ink = new List<(int, int)>();
        int x = 5;
        for (int i = 0; i < text.Length; i++)
        {
            if (i == left.Length)
            {
                x += stretch;
            }

            chars.Add(new RecognizedChar(text[i].ToString(), x, x + 8, 1f));
            ink.Add((x, x + 5));
            x += 8;
        }

        int width = x + 5;
        var pixels = new byte[width * 32];
        Array.Fill(pixels, (byte)255);
        foreach ((int from, int to) in ink)
        {
            for (int y = 8; y < 24; y++)
            {
                for (int px = from; px <= to; px++)
                {
                    pixels[(y * width) + px] = 0;
                }
            }
        }

        var was = new bool[text.Length - 1];
        was[left.Length - 1] = true;
        return (chars, was, pixels, width);
    }

    private static bool Joins(WordFrequencyModel model, string left, string right, int stretch)
    {
        var line = Line(left, right, stretch);
        bool[]? plan = WordSegmenter.Plan(line.Chars, line.WasSpace, line.Image, line.Width, 32, null, model);
        return plan is not null && !plan[left.Length - 1];
    }

    [Theory]
    [InlineData("condition", "ally", 3, true)]
    [InlineData("united", "ally", 3, false)]
    [InlineData("Unit", "ed", 4, true)]
    [InlineData("in", "debted", 4, true)]
    [InlineData("dare", "say", 4, false)]
    [InlineData("condition", "ally", 20, false)]
    [InlineData("conDition", "ally", 3, false)]
    [InlineData("condition", "aLly", 3, false)]
    [InlineData("condition", "5ally", 3, false)]
    [InlineData("the", "re", 3, false)]
    public void ASpaceInsideAWordGoesOnlyWhenTheJoinedWordIsFarMoreCommonThanThePieces(string left, string right, int stretch, bool joins)
    {
        WordFrequencyModel model = WordFrequencyModel.Parse(TestWordFrequencies.Build(Vocabulary()));

        Assert.Equal(joins, Joins(model, left, right, stretch));
    }

    [Fact]
    public void WithoutATableNothingIsJoinedByFrequency()
    {
        var line = Line("condition", "ally", 3);

        Assert.Null(WordSegmenter.Plan(line.Chars, line.WasSpace, line.Image, line.Width, 32, null, null));
    }

    [Fact]
    public void ACodeWithADigitOrASymbolIsNeverJudged()
    {
        WordFrequencyModel model = WordFrequencyModel.Parse(TestWordFrequencies.Build(Vocabulary()));

        Assert.False(Joins(model, "condition", "ally2", 3));
        Assert.False(Joins(model, "condition", "al_ly", 3));
        Assert.False(Joins(model, "a@condition", "ally", 3));
    }

    [Fact]
    public void AnEngineReportsWhetherItHasFrequencies()
    {
        WordFrequencyModel model = WordFrequencyModel.Parse(TestWordFrequencies.Build(Vocabulary()));
        OcrEngine engine = OcrEngine.Create(ModelContainer.Parse(TestModels.RandomContainer(TestModels.Small, 5)), null, KernelPath.Scalar);

        Assert.False(engine.HasWordFrequencies);
        Assert.True(engine.WithWordFrequencies(model).HasWordFrequencies);
        Assert.False(engine.WithWordFrequencies(model).WithWordFrequencies(null).HasWordFrequencies);
    }
}
