using OneOcr.Decoding;
using OneOcr.Kernels;
using OneOcr.Recognition;

namespace OneOcr.Tests;

public class DecodingTests
{
    // 0 blank, 1 space, 2 a, 3 b, 4 c, 5 d, 6 o, 7 '4', 8 '0', 9 'O'
    private static readonly string[] Charset = ["<blank>", " ", "a", "b", "c", "d", "o", "4", "0", "O"];
    private static readonly string[] Alphabet = ["a", "b", "c", "d", "o", "O"];

    private static float[] Frames(params (int Cls, float Logit)[][] steps)
    {
        var logits = new float[steps.Length * Charset.Length];
        Array.Fill(logits, -8f);
        for (int t = 0; t < steps.Length; t++)
        {
            foreach (var (cls, logit) in steps[t])
            {
                logits[(t * Charset.Length) + cls] = logit;
            }
        }

        return logits;
    }

    private static (int, float)[] S(int cls, float logit = 10f) => [(cls, logit)];

    private static RecognizedLine Greedy(float[] logits)
    {
        int steps = logits.Length / Charset.Length, last = 0;
        var text = new System.Text.StringBuilder();
        for (int t = 0; t < steps; t++)
        {
            int best = 0;
            for (int j = 1; j < Charset.Length; j++)
            {
                if (logits[(t * Charset.Length) + j] > logits[(t * Charset.Length) + best])
                {
                    best = j;
                }
            }

            if (best != 0 && best != last)
            {
                text.Append(Charset[best]);
            }

            last = best;
        }

        return new RecognizedLine(text.ToString(), 1f);
    }

    private static string Beam(float[] logits, DecodeOptions options, KernelPath path = KernelPath.Scalar) =>
        BeamLine(logits, options, path).Text;

    private static RecognizedLine BeamLine(float[] logits, DecodeOptions options, KernelPath path = KernelPath.Scalar) =>
        BeamDecoder.Decode(options, options.Prepare(Charset), logits, logits.Length / Charset.Length, Charset.Length, path, Greedy(logits));

    private static CharLanguageModel CabModel(float mean = -1.5f) => CharLanguageModel.Parse(TestLm.Build(
        "test", Alphabet, [.. Enumerable.Repeat("cab", 30), .. Enumerable.Repeat("dab", 30), "cod"], order: 3, mean: mean));

    [Fact]
    public void BeamReadsClearInputLikeGreedyAndCollapsesRepeatsOnlyWithoutABlank()
    {
        var beam = new DecodeOptions { Mode = CtcDecoding.Beam };
        Assert.Equal("cab", Beam(Frames(S(4), S(0), S(2), S(0), S(3)), beam));
        Assert.Equal("a", Beam(Frames(S(2), S(2)), beam));
        Assert.Equal("aa", Beam(Frames(S(2), S(0), S(2)), beam));
        Assert.Equal("ca b", Beam(Frames(S(4), S(2), S(1), S(3)), beam));
        Assert.Equal(string.Empty, Beam(Frames(S(0), S(0)), beam));
    }

    [Fact]
    public void LanguageModelBreaksCloseCallsTowardWordsOfTheScript()
    {
        float[] logits = Frames(S(4), S(0), [(2, 10f), (6, 10.4f)], S(0), S(3));
        Assert.Equal("cob", Greedy(logits).Text);
        Assert.Equal("cob", Beam(logits, new DecodeOptions { Mode = CtcDecoding.Beam }));
        var withLm = new DecodeOptions { Mode = CtcDecoding.Beam, LanguageModel = CabModel() };
        Assert.Equal("cab", Beam(logits, withLm));

        // Strong acoustic evidence is not overruled.
        float[] sure = Frames(S(4), S(0), [(2, 10f), (6, 16f)], S(0), S(3));
        Assert.Equal("cob", Beam(sure, withLm));

        // Weight 0 turns the model off.
        Assert.Equal("cob", Beam(logits, new DecodeOptions { Mode = CtcDecoding.Beam, LanguageModel = CabModel(), LanguageModelWeight = 0 }));
    }

    [Fact]
    public void WordsWithDigitsAreNeverChangedByTheLanguageModel()
    {
        // "c4b" against the model's favourite "cab": the digit word stays as the recognizer read it.
        float[] logits = Frames(S(4), S(0), [(2, 10f), (7, 10.4f)], S(0), S(3));
        var withLm = new DecodeOptions { Mode = CtcDecoding.Beam, LanguageModel = CabModel(), LanguageModelWeight = 5 };
        Assert.Equal("c4b", Greedy(logits).Text);
        Assert.Equal("c4b", Beam(logits, withLm));

        // And a letter word is not turned into a digit word either.
        float[] other = Frames(S(4), S(0), [(2, 10.4f), (7, 10f)], S(0), S(3));
        Assert.Equal("cab", Beam(other, withLm));
    }

    [Fact]
    public void LexiconBoostsCloseCallsButNeverForcesAReading()
    {
        var lexicon = Lexicon.FromWords(["cod"]);
        var options = new DecodeOptions { Mode = CtcDecoding.Beam, Lexicon = lexicon, LexiconBoost = 2 };
        float[] close = Frames(S(4), S(0), S(6), S(0), [(3, 10.5f), (5, 10f)]);
        Assert.Equal("cob", Greedy(close).Text);
        Assert.Equal("cod", Beam(close, options));

        float[] sure = Frames(S(4), S(0), S(6), S(0), [(3, 16f), (5, 10f)]);
        Assert.Equal("cob", Beam(sure, options));
        Assert.Equal("cob", Beam(close, new DecodeOptions { Mode = CtcDecoding.Beam, Lexicon = lexicon, LexiconBoost = 0 }));
    }

    [Fact]
    public void LexiconMayListCodesWithDigits()
    {
        var options = new DecodeOptions { Mode = CtcDecoding.Beam, Lexicon = Lexicon.FromWords(["c4b"]), LexiconBoost = 2 };
        float[] logits = Frames(S(4), S(0), [(2, 10.5f), (7, 10f)], S(0), S(3));
        Assert.Equal("cab", Greedy(logits).Text);
        Assert.Equal("c4b", Beam(logits, options));
    }

    [Fact]
    public void ResultIsIdenticalOnEveryKernelPathAndRepeatedCalls()
    {
        var rng = new Random(5);
        var options = new DecodeOptions { Mode = CtcDecoding.Beam, LanguageModel = CabModel(), Lexicon = Lexicon.FromWords(["cod", "dab"]) };
        for (int trial = 0; trial < 20; trial++)
        {
            var logits = new float[(10 + trial) * Charset.Length];
            for (int i = 0; i < logits.Length; i++)
            {
                logits[i] = (float)((rng.NextDouble() * 6) - 2);
            }

            RecognizedLine baseline = BeamLine(logits, options, KernelPath.Scalar);
            foreach (KernelPath path in Dispatch.Available())
            {
                RecognizedLine other = BeamLine(logits, options, path);
                Assert.Equal(baseline.Text, other.Text);
                Assert.Equal(BitConverter.SingleToInt32Bits(baseline.Confidence), BitConverter.SingleToInt32Bits(other.Confidence));
            }
        }
    }

    [Fact]
    public void ScriptWithTheMostLettersOfTheLineIsChosen()
    {
        // Two models over the same alphabet that disagree about "a" after "c": the line picks the model whose alphabet matches.
        byte[] file = TestLm.Assemble([
            ("other", ["x", "y"], ["xy"]),
            ("test", Alphabet, [.. Enumerable.Repeat("cab", 30), .. Enumerable.Repeat("dab", 30), "cod"]),
        ]);
        var lm = CharLanguageModel.Parse(file);
        Assert.Equal(["other", "test"], lm.ScriptNames);
        float[] logits = Frames(S(4), S(0), [(2, 10f), (6, 10.4f)], S(0), S(3));
        Assert.Equal("cab", Beam(logits, new DecodeOptions { Mode = CtcDecoding.Beam, LanguageModel = lm }));
    }

    [Fact]
    public void OptionsAreValidatedAgainstTheLimits()
    {
        var recognizer = new LineRecognizer(
            CrnnModel.FromContainer(Model.ModelContainer.Parse(TestModels.RandomContainer(TestModels.Small, 3)), OcrLimits.Default),
            OcrLimits.Default, KernelPath.Scalar);
        byte[] image = TestModels.RandomImage(recognizer.InputHeight, 120, 1);
        Assert.Throws<OcrLimitException>(() => recognizer.Recognize(image, 120, new DecodeOptions { Mode = CtcDecoding.Beam, BeamWidth = 65 }));
        Assert.Throws<ArgumentException>(() => recognizer.Recognize(image, 120, new DecodeOptions { Mode = CtcDecoding.Beam, BeamWidth = 0 }));
        Assert.Throws<ArgumentException>(() => recognizer.Recognize(image, 120, new DecodeOptions { Mode = CtcDecoding.Beam, LanguageModelWeight = -1 }));
        Assert.Throws<ArgumentException>(() => recognizer.Recognize(image, 120, new DecodeOptions { Mode = CtcDecoding.Beam, LexiconBoost = double.NaN }));

        var tight = new OcrLimits { MaxBeamNodes = 100 };
        var small = new LineRecognizer(
            CrnnModel.FromContainer(Model.ModelContainer.Parse(TestModels.RandomContainer(TestModels.Small, 3)), tight), tight, KernelPath.Scalar);
        Assert.Throws<OcrLimitException>(() => small.Recognize(image, 120, new DecodeOptions { Mode = CtcDecoding.Beam }));

        // Greedy settings need no checking and leave the result as it was.
        Assert.Equal(recognizer.Recognize(image, 120, 1), recognizer.Recognize(image, 120, DecodeOptions.Greedy, 1));
        Assert.Equal(recognizer.Recognize(image, 120, 1), recognizer.Recognize(image, 120, null, 1));
    }

    [Fact]
    public void CaseFoldingUsesTheSimpleLowercaseForm()
    {
        Assert.Equal("o", Prepared.Fold("O"));
        Assert.Equal("\u00e9", Prepared.Fold("\u00c9"));
        Assert.Equal("\u03c3", Prepared.Fold("\u03a3"));
        Assert.Equal("\u0436", Prepared.Fold("\u0416"));
        Assert.Equal("\u00df", Prepared.Fold("\u00df"));
        Assert.Equal("\u0130", Prepared.Fold("\u0130"));
        Assert.Equal("7", Prepared.Fold("7"));
        Assert.Equal("ab", Prepared.Fold("ab"));

        // A capital letter is scored as its lowercase form: 'O' in the charset reads as the alphabet's 'o'.
        float[] logits = Frames(S(4), S(0), [(2, 10f), (9, 10.4f)], S(0), S(3));
        Assert.Equal("cOb", Greedy(logits).Text);
        Assert.Equal("cab", Beam(logits, new DecodeOptions { Mode = CtcDecoding.Beam, LanguageModel = CabModel() }));
    }

    [Fact]
    public void EngineDecodingDefaultsToGreedyAndCanBeChangedPerEngineOrPerCall()
    {
        var shape = TestModels.Small;
        using var stream = new MemoryStream(TestModels.RandomContainer(shape, 3));
        OcrEngine engine = OcrEngine.Load(stream);
        Assert.Same(DecodeOptions.Greedy, engine.Decoding);

        var pixels = new OneImage.Gray8[500 * 200];
        Array.Fill(pixels, new OneImage.Gray8(255));
        foreach (int top in new[] { 30, 90, 150 })
        {
            for (int y = top; y < top + 24; y++)
            {
                for (int x = 40; x < 460; x++)
                {
                    if ((x / 5) % 3 != 0)
                    {
                        pixels[(y * 500) + x] = new OneImage.Gray8(20);
                    }
                }
            }
        }

        OneImage.Image page = OneImage.Image.FromPixels<OneImage.Gray8>(pixels, 500, 200);
        var beam = new DecodeOptions { Mode = CtcDecoding.Beam, BeamWidth = 4 };
        string greedy = engine.ToText(page, Opt.Degree(1));
        Assert.Equal(greedy, engine.ToText(page, DecodeOptions.Greedy, Opt.Degree(1)));
        string viaCall = engine.ToText(page, beam, Opt.Degree(1));
        OcrEngine other = engine.WithDecoding(beam);
        Assert.Same(beam, other.Decoding);
        Assert.Same(DecodeOptions.Greedy, engine.Decoding);
        Assert.Equal(viaCall, other.ToText(page, Opt.Degree(1)));
        Assert.Equal(viaCall, other.ToText(page, Opt.Degree(4)));
        Assert.Throws<ArgumentNullException>(() => engine.WithDecoding(null!));
        Assert.Throws<OcrLimitException>(() => engine.WithDecoding(new DecodeOptions { Mode = CtcDecoding.Beam, BeamWidth = 1000 }));
    }

    [Fact]
    public void BeamDecodingOfAModelRunsAtEveryDegreeWithTheSameText()
    {
        var recognizer = new LineRecognizer(
            CrnnModel.FromContainer(Model.ModelContainer.Parse(TestModels.RandomContainer(TestModels.Small, 3)), OcrLimits.Default),
            OcrLimits.Default, KernelPath.Scalar);
        byte[] image = TestModels.RandomImage(recognizer.InputHeight, 200, 2);
        var options = new DecodeOptions { Mode = CtcDecoding.Beam, BeamWidth = 4 };
        RecognizedLine a = recognizer.Recognize(image, 200, options, 1);
        RecognizedLine b = recognizer.Recognize(image, 200, options, 4);
        Assert.Equal(a, b);
    }
}

public class DetMathTests
{
    [Fact]
    public void ExpAndLogAgreeWithTheRuntimeToRoundingError()
    {
        for (double x = -60; x <= 60; x += 0.173)
        {
            double expect = Math.Exp(x);
            Assert.True(Math.Abs(Decoding.DetMath.Exp(x) - expect) <= 6e-15 * expect, $"exp({x})");
        }

        foreach (double x in new[] { 1e-300, 1e-12, 0.001, 0.5, 0.9999999, 1.0, 1.0000001, 1.5, 2, Math.E, 10, 12345.678, 1e100, 1e300 })
        {
            Assert.True(Math.Abs(Decoding.DetMath.Log(x) - Math.Log(x)) <= 1e-14 * Math.Max(1, Math.Abs(Math.Log(x))), $"log({x})");
        }

        Assert.Equal(double.NegativeInfinity, Decoding.DetMath.Log(0));
        Assert.Equal(0.0, Decoding.DetMath.Exp(-800));
        Assert.Equal(0.0, Decoding.DetMath.Exp(double.NaN));
    }

    [Fact]
    public void LogAddHandlesImpossibleAndDistantTerms()
    {
        double ninf = double.NegativeInfinity;
        Assert.Equal(-3.0, Decoding.DetMath.LogAdd(ninf, -3.0));
        Assert.Equal(ninf, Decoding.DetMath.LogAdd(ninf, ninf));
        Assert.True(Math.Abs(Decoding.DetMath.LogAdd(-1.0, -1.0) - (-1.0 + Math.Log(2))) < 1e-15);
        Assert.Equal(0.0, Decoding.DetMath.LogAdd(0.0, -100.0));
    }
}
