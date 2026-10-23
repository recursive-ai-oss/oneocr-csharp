using OneOcr.Decoding;

namespace OneOcr.Tests;

public class LanguageModelTests
{
    private static readonly string[] Alphabet = ["a", "b", "c"];

    private static byte[] Model() => TestLm.Build("test", Alphabet, [.. Enumerable.Repeat("cab", 10), "ba"], order: 3);

    [Fact]
    public void ParsesAndScoresWithBackOff()
    {
        CharLanguageModel lm = CharLanguageModel.Parse(Model());
        ScriptModel m = lm.Scripts[0];
        Assert.Equal("test", m.Name);
        Assert.Equal(3, m.Order);
        Assert.Equal(Alphabet, m.Alphabet);
        Assert.Equal(-1.5f, m.MeanLogProb);

        const byte a = 2, b = 3, c = 4;
        // "cab" x10 and "ba": after the start, c has 10 of 11 words; after "c", a always follows; after "ca", b always follows.
        Assert.Equal(-MathF.Round(-MathF.Log(10f / 11f) / 0.1f) * 0.1f, m.LogProb([ScriptModel.Start], c), 3);
        Assert.Equal(0f, m.LogProb([ScriptModel.Start, c], a), 3);
        Assert.Equal(0f, m.LogProb([ScriptModel.Start, c, a], b), 3);

        // A trigram that was never seen backs off to the unigram of the same symbol (back-off weights are 0 in this file).
        float unigramC = m.LogProb([], c);
        Assert.Equal(unigramC, m.LogProb([b, b], c), 5);
        // Longer contexts than the order are cut to the last two symbols.
        Assert.Equal(m.LogProb([c, a], b), m.LogProb([a, b, c, a], b), 5);
        // The end of a word is a symbol too.
        Assert.True(m.LogProb([ScriptModel.Start, c, a, b], ScriptModel.End) > m.LogProb([ScriptModel.Start], ScriptModel.End));
    }

    [Fact]
    public void LoadsFromStreamsAndFilesUnderTheSizeLimit()
    {
        byte[] file = Model();
        using var stream = new MemoryStream(file);
        Assert.Single(CharLanguageModel.Load(stream).Scripts);
        string path = Path.Combine(Path.GetTempPath(), $"oneocr-lm-{Guid.NewGuid():N}.bin");
        File.WriteAllBytes(path, file);
        try
        {
            Assert.Single(CharLanguageModel.Load(path).Scripts);
        }
        finally
        {
            File.Delete(path);
        }

        var tight = new OcrLimits { MaxLanguageModelFileBytes = file.Length - 1 };
        Assert.Throws<OcrLimitException>(() => CharLanguageModel.Parse(file, tight));
        using var again = new MemoryStream(file);
        Assert.Throws<OcrLimitException>(() => CharLanguageModel.Load(again, tight));
    }

    [Fact]
    public void MalformedFilesAreRejectedWithTypedErrors()
    {
        byte[] good = Model();
        Assert.Throws<OcrModelException>(() => CharLanguageModel.Parse([]));
        Assert.Throws<OcrModelException>(() => CharLanguageModel.Parse("OCLX\x01\0\x01\0"u8));

        byte[] version = (byte[])good.Clone();
        version[4] = 2;
        Assert.Throws<OcrModelException>(() => CharLanguageModel.Parse(version));

        byte[] noScripts = (byte[])good.Clone();
        noScripts[6] = 0;
        Assert.Throws<OcrModelException>(() => CharLanguageModel.Parse(noScripts));

        // Every truncation fails cleanly.
        for (int cut = 0; cut < good.Length; cut += 7)
        {
            Assert.Throws<OcrModelException>(() => CharLanguageModel.Parse(good.AsSpan(0, cut)));
        }

        Assert.Throws<OcrModelException>(() => CharLanguageModel.Parse([.. good, 0]));

        // Out-of-order entries: swap the first two unigram rows (rows are n + 2 bytes: key, probability, back-off).
        int unigrams = FindFirstTable(good);
        byte[] swapped = (byte[])good.Clone();
        for (int i = 0; i < 3; i++)
        {
            (swapped[unigrams + i], swapped[unigrams + 3 + i]) = (swapped[unigrams + 3 + i], swapped[unigrams + i]);
        }

        Assert.Throws<OcrModelException>(() => CharLanguageModel.Parse(swapped));

        // A symbol outside the alphabet.
        byte[] outside = (byte[])good.Clone();
        outside[unigrams] = 200;
        Assert.Throws<OcrModelException>(() => CharLanguageModel.Parse(outside));
    }

    // Offset of the first unigram row: header (8), name (1 + 4), order and alphabet size (2), alphabet (3 x 2), mean (4), count (4).
    private static int FindFirstTable(byte[] file) => 8 + 5 + 2 + 6 + 4 + 4;

    [Fact]
    public void DeclaredSizesAreCheckedBeforeAnythingIsAllocated()
    {
        byte[] good = Model();
        int count = FindFirstTable(good) - 4;

        // 4 billion unigrams claimed in a file of a few hundred bytes.
        byte[] huge = (byte[])good.Clone();
        BitConverter.TryWriteBytes(huge.AsSpan(count), uint.MaxValue);
        Assert.Throws<OcrLimitException>(() => CharLanguageModel.Parse(huge));

        // Under the n-gram limit but more rows than the file holds.
        byte[] lying = (byte[])good.Clone();
        BitConverter.TryWriteBytes(lying.AsSpan(count), 3_000_000u);
        Assert.Throws<OcrModelException>(() => CharLanguageModel.Parse(lying));

        var few = new OcrLimits { MaxLanguageModelNgrams = 5 };
        Assert.Throws<OcrLimitException>(() => CharLanguageModel.Parse(good, few));
        Assert.Throws<OcrLimitException>(() => CharLanguageModel.Parse(good, new OcrLimits { MaxLanguageModelOrder = 2 }));
        Assert.Throws<OcrLimitException>(() => CharLanguageModel.Parse(
            TestLm.Assemble([("a", Alphabet, ["ab"]), ("b", Alphabet, ["ab"])]), new OcrLimits { MaxLanguageModelScripts = 1 }));
    }

    [Fact]
    public void LexiconListsAreDeduplicatedAndBounded()
    {
        Lexicon lexicon = Lexicon.FromWords(["Zo\u00EB", "zoe", "zoe"]);
        Assert.Equal(["Zo\u00EB", "zoe"], lexicon.Words);
        Assert.Equal(2, lexicon.Count);
        Assert.Throws<ArgumentException>(() => Lexicon.FromWords(["two words"]));
        Assert.Throws<ArgumentException>(() => Lexicon.FromWords([""]));
        Assert.Throws<OcrLimitException>(() => Lexicon.FromWords(["abcdef"], new OcrLimits { MaxLexiconWordLength = 5 }));
        Assert.Throws<OcrLimitException>(() => Lexicon.FromWords(["a", "b", "c"], new OcrLimits { MaxLexiconWords = 2 }));

        string path = Path.Combine(Path.GetTempPath(), $"oneocr-lex-{Guid.NewGuid():N}.txt");
        try
        {
            File.WriteAllText(path, "alpha\r\n\r\nbeta\n  gamma  \n");
            Assert.Equal(3, Lexicon.Load(path).Count);
            File.WriteAllText(path, "alpha beta\n");
            Assert.Throws<OcrModelException>(() => Lexicon.Load(path));
            File.WriteAllBytes(path, [0xFF, 0xFE, 0xFA]);
            Assert.Throws<OcrModelException>(() => Lexicon.Load(path));
            File.WriteAllText(path, "alpha\nbeta\n");
            Assert.Throws<OcrLimitException>(() => Lexicon.Load(path, new OcrLimits { MaxLexiconFileBytes = 3 }));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
