using System.Buffers.Binary;
using System.Text;
using OneImage;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

public class WordTests
{
    private static readonly TestModels.Shape Shape = TestModels.Small;

    private const string Calibration =
        "{\"raw\":\"word-mean-max-run\",\"scripts\":{" +
        "\"Latin\":{\"x\":[0.0,0.5,1.0],\"y\":[0.0,0.25,1.0]}," +
        "\"Greek\":{\"x\":[0.2,0.8],\"y\":[0.1,0.9]}," +
        "\"Cyrillic\":{\"x\":[0.5],\"y\":[0.5]}}}";

    private static string[] SpaceCharset()
    {
        string[] cs = TestModels.Charset(Shape.Classes);
        cs[1] = " ";
        cs[2] = "\u00A0";
        cs[3] = "\u2003";
        cs[4] = "a";
        cs[5] = "b";
        cs[6] = "\u03B1"; // alpha
        cs[7] = "\u0431"; // be
        cs[8] = "1";
        return cs;
    }

    private static byte[] Container(string? calibration)
    {
        List<TestTensor> tensors = TestModels.RandomTensors(Shape, 5);
        // Favour the space classes a little so that random weights produce lines with several words.
        float[] bias = tensors.Single(t => t.Name == "fc.bias").Values;
        for (int i = 1; i <= 3; i++)
        {
            bias[i] += SpaceBoost;
        }

        return TestContainerWriter.Write("crnn-v0", Shape.Height, SpaceCharset(), TestModels.ConfigJson(Shape), tensors, calibrationJson: calibration);
    }

    private const float SpaceBoost = 0.5f;

    private static ModelContainer Parse(string calibration) =>
        ModelContainer.Parse(TestModels.RandomContainerV1(Shape, 1, calibration));

    private static OcrModelException Rejected(string calibration) =>
        Assert.Throws<OcrModelException>(() => Parse(calibration));

    private static string Cal(string latin) => "{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":" + latin + "}}";

    // ---- calibration curve ----

    [Fact]
    public void CurveInterpolatesClampsAndFallsBackToIdentity()
    {
        WordCalibration c = Parse(Calibration).Calibration!;
        Assert.Equal(0f, c.Apply(OcrScript.Latin, 0f));
        Assert.Equal(0.125f, c.Apply(OcrScript.Latin, 0.25f));
        Assert.Equal(0.25f, c.Apply(OcrScript.Latin, 0.5f));
        Assert.Equal(0.625f, c.Apply(OcrScript.Latin, 0.75f));
        Assert.Equal(1f, c.Apply(OcrScript.Latin, 1f));
        Assert.Equal(0.1f, c.Apply(OcrScript.Greek, 0f));
        Assert.Equal(0.5f, c.Apply(OcrScript.Greek, 0.5f));
        Assert.Equal(0.9f, c.Apply(OcrScript.Greek, 1f));
        Assert.Equal(0.5f, c.Apply(OcrScript.Cyrillic, 0.01f));
        Assert.Equal(0.5f, c.Apply(OcrScript.Cyrillic, 0.99f));

        // A script without a curve is the identity; so is the absence of any scripts.
        WordCalibration partial = Parse(Cal("{\"x\":[0,1],\"y\":[0,1]}")).Calibration!;
        Assert.False(partial.HasCurve(OcrScript.Greek));
        Assert.Equal(0.37f, partial.Apply(OcrScript.Greek, 0.37f));
        Assert.Equal(0.37f, Parse("{\"raw\":\"word-mean-max-run\",\"scripts\":{}}").Calibration!.Apply(OcrScript.Latin, 0.37f));

        // Steps (equal y) and a vertical jump at the last knot are allowed.
        WordCalibration step = Parse(Cal("{\"x\":[0,0.4,0.6,1],\"y\":[0.2,0.2,0.9,0.9]}")).Calibration!;
        Assert.Equal(0.2f, step.Apply(OcrScript.Latin, 0.3f));
        Assert.Equal(0.55f, step.Apply(OcrScript.Latin, 0.5f), 5);
        Assert.Equal(0.9f, step.Apply(OcrScript.Latin, 0.7f));
    }

    [Fact]
    public void KnotsAtTheLimitLoadAndOneMoreIsALimitError()
    {
        string Knots(int n) =>
            Cal("{\"x\":[" + string.Join(',', Enumerable.Range(0, n).Select(i => (i / (double)n).ToString("R", System.Globalization.CultureInfo.InvariantCulture))) +
                "],\"y\":[" + string.Join(',', Enumerable.Range(0, n).Select(i => (i / (double)n).ToString("R", System.Globalization.CultureInfo.InvariantCulture))) + "]}");
        Assert.NotNull(Parse(Knots(64)).Calibration);
        Assert.Throws<OcrLimitException>(() => Parse(Knots(65)));
    }

    [Theory]
    [InlineData("{\"x\":[0,1],\"y\":[0]}")]            // lengths differ
    [InlineData("{\"x\":[0],\"y\":[0,1]}")]
    [InlineData("{\"x\":[],\"y\":[]}")]               // no knots
    [InlineData("{\"x\":[0.5,0.5],\"y\":[0,1]}")]     // x not strictly increasing
    [InlineData("{\"x\":[0.6,0.5],\"y\":[0,1]}")]
    [InlineData("{\"x\":[0,1],\"y\":[0.9,0.1]}")]     // y decreases
    [InlineData("{\"x\":[-0.1,1],\"y\":[0,1]}")]      // outside [0, 1]
    [InlineData("{\"x\":[0,1.1],\"y\":[0,1]}")]
    [InlineData("{\"x\":[0,1],\"y\":[0,1.5]}")]
    [InlineData("{\"x\":[0,1],\"y\":[-0.5,1]}")]
    [InlineData("{\"x\":[0,1e999],\"y\":[0,1]}")]     // not finite
    [InlineData("{\"x\":[0,1]}")]                     // missing y
    [InlineData("{\"x\":[0,\"1\"],\"y\":[0,1]}")]     // not a number
    [InlineData("{\"x\":[0,1],\"y\":[0,null]}")]
    [InlineData("null")]
    [InlineData("[]")]
    public void RejectsBadKnots(string latin)
    {
        Assert.Throws<OcrModelException>(() => Parse(Cal(latin)));
    }

    [Theory]
    [InlineData("{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Arabic\":{\"x\":[0],\"y\":[0]}}}")]
    [InlineData("{\"raw\":\"word-mean-max-run\",\"scripts\":{\"latin\":{\"x\":[0],\"y\":[0]}}}")]
    [InlineData("{\"raw\":\"word-mean-max-run\",\"scripts\":{\"0\":{\"x\":[0],\"y\":[0]}}}")]
    [InlineData("{\"raw\":\"line-mean\",\"scripts\":{}}")]
    [InlineData("{\"scripts\":{}}")]
    [InlineData("{\"raw\":\"word-mean-max-run\"}")]
    [InlineData("{\"raw\":\"word-mean-max-run\",\"scripts\":[]}")]
    [InlineData("{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[0],\"y\":[0]},\"Latin\":{\"x\":[0],\"y\":[1]}}}")]
    [InlineData("[]")]
    [InlineData("3")]
    [InlineData("null")]
    public void RejectsUnknownScriptsAndBadStructure(string calibration)
    {
        Assert.Throws<OcrModelException>(() => Parse(calibration));
    }

    [Fact]
    public void VersionOneNeedsCalibrationAndVersionZeroIgnoresIt()
    {
        // A version 1 header over a manifest without the member.
        byte[] v0 = TestModels.RandomContainer(Shape, 1);
        byte[] asV1 = (byte[])v0.Clone();
        BinaryPrimitives.WriteUInt32LittleEndian(asV1.AsSpan(8), 1);
        Assert.Throws<OcrModelException>(() => ModelContainer.Parse(asV1));

        // A version 0 header over a manifest that has a calibration member: not a member of that version, so ignored.
        byte[] good = TestModels.RandomContainerV1(Shape, 1, Calibration);
        byte[] asV0 = (byte[])good.Clone();
        BinaryPrimitives.WriteUInt32LittleEndian(asV0.AsSpan(8), 0);
        ModelContainer c = ModelContainer.Parse(asV0);
        Assert.Equal(0u, c.FormatVersion);
        Assert.Null(c.Calibration);

        Assert.Throws<OcrModelException>(() => ModelContainer.Parse(WithVersion(good, 2)));
    }

    private static byte[] WithVersion(byte[] file, uint version)
    {
        byte[] copy = (byte[])file.Clone();
        BinaryPrimitives.WriteUInt32LittleEndian(copy.AsSpan(8), version);
        return copy;
    }

    [Fact]
    public void ReadsBothVersionsAndVersionZeroGetsADiagnostic()
    {
        ModelContainer v0 = ModelContainer.Parse(TestModels.RandomContainer(Shape, 1));
        Assert.Equal(0u, v0.FormatVersion);
        Assert.Null(v0.Calibration);
        Assert.Single(v0.Diagnostics);

        ModelContainer v1 = Parse(Calibration);
        Assert.Equal(1u, v1.FormatVersion);
        Assert.NotNull(v1.Calibration);
        Assert.Empty(v1.Diagnostics);
        // Same tensors either way.
        Assert.Equal(v0.ReadTensor("fc.bias"), v1.ReadTensor("fc.bias"));
        Assert.NotNull(LineRecognizer.FromContainer(v1));
    }

    [Fact]
    public void EveryTruncationOfAVersionOneFileIsRejected()
    {
        byte[] file = TestModels.RandomContainerV1(Shape, 1, Calibration);
        for (int length = 0; length < file.Length; length += length < 4096 ? 1 : 997)
        {
            Exception ex = Assert.ThrowsAny<Exception>(() => ModelContainer.Parse(file[..length]));
            Assert.True(ex is OcrModelException or OcrLimitException);
        }
    }

    // ---- scripts ----

    [Theory]
    [InlineData("hello", OcrScript.Latin)]
    [InlineData("naïve", OcrScript.Latin)]
    [InlineData("\u03B1\u03B2\u03B3", OcrScript.Greek)]
    [InlineData("\u0431\u0443\u043A\u0432\u0430", OcrScript.Cyrillic)]
    [InlineData("12345", OcrScript.Latin)]
    [InlineData("-.,", OcrScript.Latin)]
    [InlineData("", OcrScript.Latin)]
    [InlineData("\u03B1\u03B2x1234", OcrScript.Greek)]      // 2 Greek letters beat 1 Latin; digits do not count
    [InlineData("ab\u03B1\u03B2", OcrScript.Latin)]         // tie: Latin first
    [InlineData("\u03B1\u03B2\u0431\u0432", OcrScript.Greek)] // tie: Greek before Cyrillic
    [InlineData("x\u0431\u0432", OcrScript.Cyrillic)]
    public void ScriptIsTheOneWithMostLetters(string word, OcrScript expected) =>
        Assert.Equal(expected, ScriptDetector.Of(word));

    // ---- word splitting on crafted network output ----

    private static float[] Frames(int[] classes)
    {
        int c = Shape.Classes;
        var logits = new float[classes.Length * c];
        for (int t = 0; t < classes.Length; t++)
        {
            logits[(t * c) + classes[t]] = 6f;
        }

        return logits;
    }

    [Fact]
    public void SplitsWordsAtSpacesWithBoxesAndConfidences()
    {
        var recognizer = LineRecognizer.FromContainer(ModelContainer.Parse(Container(Calibration)));
        // steps:    0  1  2  3  4  5  6  7  8  9 10 11 12 13
        int[] seq = [0, 4, 4, 0, 5, 0, 1, 6, 0, 0, 3, 8, 8, 0];
        //            a a     b     sp   α        em  1  1
        float[] logits = Frames(seq);
        LineWords line = recognizer.SplitWords(logits, seq.Length, 140);
        Assert.Equal("ab \u03B1\u2003" + "1", line.Text);
        Assert.Equal(RecognizedLineText(recognizer, logits, seq.Length), line.Text);
        Assert.Equal(3, line.Words.Count);

        // Raw: softmax of one 6 among 40 zeros.
        float p = (float)(Math.Exp(6) / (Math.Exp(6) + (Shape.Classes - 1)));
        Assert.Equal(p, line.Words[0].RawConfidence, 5);
        Assert.Equal("ab", line.Words[0].Text);
        Assert.Equal(OcrScript.Latin, line.Words[0].Script);
        Assert.Equal(2, line.Words[0].Characters);
        Assert.Equal(OcrScript.Greek, line.Words[1].Script);
        Assert.Equal(OcrScript.Latin, line.Words[2].Script);
        Assert.Equal("1", line.Words[2].Text);

        // Calibration is applied by script: Latin through (0,0),(0.5,0.25),(1,1); Greek through (0.2,0.1),(0.8,0.9).
        WordCalibration cal = Parse(Calibration).Calibration!;
        Assert.Equal(cal.Apply(OcrScript.Latin, line.Words[0].RawConfidence), line.Words[0].Confidence);
        Assert.Equal(cal.Apply(OcrScript.Greek, line.Words[1].RawConfidence), line.Words[1].Confidence);
        Assert.NotEqual(line.Words[0].RawConfidence, line.Words[0].Confidence);

        // Boxes: 10 px per step. "ab" fires on steps 1-2 and 4; the line starts at its first step, and the edge toward the
        // space (step 6) is halfway across the gap after step 4: (5 + 6) / 2 = 5.5 steps.
        Assert.Equal(10f, line.Words[0].X0);
        Assert.Equal(55f, line.Words[0].X1);
        // "α" is step 7: its left edge is halfway between the space (ends at 7) and itself (starts at 7), its right edge halfway
        // between its end (8) and the em space at step 10.
        Assert.Equal(70f, line.Words[1].X0);
        Assert.Equal(90f, line.Words[1].X1);
        // "1" is steps 11-12, after the em space at step 10; the line ends where its last run does.
        Assert.Equal(110f, line.Words[2].X0);
        Assert.Equal(130f, line.Words[2].X1);
    }

    private static string RecognizedLineText(LineRecognizer r, float[] logits, int steps) => r.Decode(logits, steps).Text;

    [Fact]
    public void WordPartsAgreeWithTheLineRule()
    {
        var recognizer = LineRecognizer.FromContainer(ModelContainer.Parse(Container(Calibration)));
        int[] seq = [1, 4, 4, 0, 4, 1, 1, 5, 0, 1, 0];
        float[] logits = Frames(seq);
        // Make step 2 less sure than step 1, to check that the run keeps its largest probability.
        logits[(2 * Shape.Classes) + 4] = 3f;
        LineWords line = recognizer.SplitWords(logits, seq.Length, 55);
        Assert.Equal(" aa b ", line.Text);
        Assert.Equal(new[] { "aa", "b" }, line.Words.Select(w => w.Text).ToArray());
        RecognizedLine whole = recognizer.Decode(logits, seq.Length);
        Assert.Equal(line.Text, whole.Text);
        Assert.Equal(whole.Confidence, line.RawConfidence);
        // Leading, doubled and trailing spaces make no empty words; every word has a box inside the line.
        foreach (RecognizedWord w in line.Words)
        {
            Assert.InRange(w.X0, 0f, 55f);
            Assert.InRange(w.X1, w.X0, 55f);
        }
    }

    [Fact]
    public void EmptyAndSpaceOnlyLinesHaveNoWords()
    {
        var recognizer = LineRecognizer.FromContainer(ModelContainer.Parse(Container(Calibration)));
        LineWords empty = recognizer.SplitWords(Frames([0, 0, 0]), 3, 30);
        Assert.Empty(empty.Words);
        Assert.Equal(0f, empty.RawConfidence);
        LineWords spaces = recognizer.SplitWords(Frames([1, 0, 2]), 3, 30);
        Assert.Empty(spaces.Words);
        Assert.Empty(recognizer.SplitWords([], 0, 30).Words);
    }

    [Fact]
    public void VersionZeroModelsReportRawAsCalibrated()
    {
        var recognizer = LineRecognizer.FromContainer(ModelContainer.Parse(Container(null)));
        LineWords line = recognizer.SplitWords(Frames([4, 0, 5]), 3, 30);
        Assert.Equal(line.Words[0].RawConfidence, line.Words[0].Confidence);
    }

    // ---- engine ----

    private static (ModelContainer Model, Image Page) EngineSetup()
    {
        var (px, _) = EngineTests.SyntheticPage(900, 420, [60, 130, 200, 270], 14, 8);
        Image page = Image.FromPixels<Gray8>(px.Select(b => new Gray8(b)).ToArray(), 900, 420);
        return (ModelContainer.Parse(Container(Calibration)), page);
    }

    private static string Flatten(OcrPage r)
    {
        var sb = new StringBuilder();
        foreach (OcrBlock b in r.Blocks)
        {
            foreach (OcrLine l in b.Lines)
            {
                sb.Append(CultureInvariant($"L {l.Text.Length} {BitConverter.SingleToInt32Bits(l.Confidence):x8} {l.Box.Left} {l.Box.Top} {l.Box.Right} {l.Box.Bottom}\n"));
                foreach (OcrWord w in l.Words)
                {
                    sb.Append(CultureInvariant($"  {w.Text} {w.Box.Left} {w.Box.Top} {w.Box.Right} {w.Box.Bottom} {BitConverter.SingleToInt32Bits(w.Confidence):x8} {BitConverter.SingleToInt32Bits(w.RawConfidence):x8}\n"));
                    foreach (OcrChar c in w.Chars)
                    {
                        sb.Append(CultureInvariant($"    {c.Text} {c.Box.Left} {c.Box.Right} {BitConverter.SingleToInt32Bits(c.Confidence):x8}\n"));
                    }
                }
            }
        }

        return sb.ToString();
    }

    private static string CultureInvariant(FormattableString s) => FormattableString.Invariant(s);

    private static OcrPage Ocr(OcrEngine e, Image page, int degree) => e.Ocr(page, new OcrOptions { MaxDegreeOfParallelism = degree });

    [Fact]
    public void OcrIsBitIdenticalOnEveryPathAndDegreeAndMatchesToText()
    {
        var (container, page) = EngineSetup();
        var scalar = OcrEngine.Create(container, null, KernelPath.Scalar);
        OcrPage baseline = Ocr(scalar, page, 1);
        string flat = Flatten(baseline);
        var lines = baseline.Blocks.SelectMany(b => b.Lines).ToList();
        Assert.Equal(4, lines.Count);
        Assert.Contains(lines, l => l.Words.Count > 1);
        Assert.Equal(scalar.ToText(page, Opt.Degree(1)), baseline.ToText());
        Assert.Equal(flat, Flatten(Ocr(scalar, page, 4)));
        Assert.Equal(flat, Flatten(scalar.Ocr(page)));
        string json = OcrPageJson.ToJson([baseline]);
        foreach (KernelPath path in Dispatch.Available())
        {
            var engine = OcrEngine.Create(container, null, path);
            Assert.Equal(flat, Flatten(Ocr(engine, page, 1)));
            Assert.Equal(flat, Flatten(Ocr(engine, page, 8)));
            Assert.Equal(json, OcrPageJson.ToJson([Ocr(engine, page, 3)]));
        }

        foreach (OcrLine line in lines)
        {
            double prev = line.Box.Left;
            foreach (OcrWord w in line.Words)
            {
                Assert.InRange(w.Confidence, 0f, 1f);
                Assert.InRange(w.RawConfidence, 0f, 1f);
                Assert.True(w.Box.Left <= w.Box.Right);
                Assert.True(w.Box.Left >= 0 && w.Box.Right <= 900);
                Assert.True(w.Box.Left >= prev - 0.01, "words run left to right");
                prev = w.Box.Right;
                Assert.All(w.Chars, c => Assert.True(c.BoxIsApproximate));
            }
        }
    }

    [Fact]
    public void VersionZeroEnginesSayTheyAreUncalibrated()
    {
        var (_, page) = EngineSetup();
        var engine = OcrEngine.Create(ModelContainer.Parse(Container(null)), null, KernelPath.Scalar);
        OcrPage r = Ocr(engine, page, 1);
        Assert.Contains(r.Diagnostics, d => d.Message == ModelContainer.NoCalibrationNote);
        foreach (OcrWord w in r.Words)
        {
            Assert.Equal(w.RawConfidence, w.Confidence);
        }

        Assert.Throws<ArgumentNullException>(() => engine.Ocr((Image)null!));
        Assert.Empty(engine.Ocr(Image.FromPixels<Gray8>(Enumerable.Repeat(new Gray8(255), 300 * 200).ToArray(), 300, 200)).Words);
    }

    [Fact]
    public void OcrRejectsOversizedImages()
    {
        var (container, page) = EngineSetup();
        var engine = OcrEngine.Create(container, new OcrLimits { MaxImagePixels = 1000 }, KernelPath.Scalar);
        Assert.Throws<OcrLimitException>(() => engine.Ocr(page));
    }
}
