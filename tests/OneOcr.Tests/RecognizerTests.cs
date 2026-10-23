using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

public class RecognizerTests
{
    // Per-layer tolerance against the float64 reference: |c - r| <= Atol + Rtol * |r|.
    // Measured on random-weight models of the intended size (K = 9..2304 per conv output, 4*256 gates over 30 steps):
    // largest absolute error 1.2e-6 (conv), 7.7e-7 (LSTM), 4.0e-7 (classifier); largest relative error 6.8e-6 where |r| > 0.1.
    // The tolerances leave 3-5x headroom over that. Sources of error: float32 rounding in the dot products,
    // and the 1e-7 absolute error of the tanh/sigmoid approximations in the recurrence.
    private static readonly Dictionary<string, (double Atol, double Rtol)> Tolerance = new()
    {
        ["input"] = (0, 0),
        ["conv"] = (4e-6, 2e-5),
        ["lstm"] = (3e-6, 2e-5),
        ["fc"] = (2e-6, 2e-5),
    };

    private static (double Atol, double Rtol) ToleranceFor(string layer) =>
        Tolerance[layer.StartsWith("conv", StringComparison.Ordinal) ? "conv" : layer.StartsWith("lstm", StringComparison.Ordinal) ? "lstm" : layer];

    private static LineRecognizer Create(byte[] file, KernelPath path) =>
        new(CrnnModel.FromContainer(ModelContainer.Parse(file), OcrLimits.Default), OcrLimits.Default, path);

    private static ForwardTrace Trace(LineRecognizer r, byte[] image, int width, int degree, out int steps)
    {
        var trace = new ForwardTrace();
        r.Forward(image, width, degree, trace, CancellationToken.None, out steps);
        return trace;
    }

    [Theory]
    [InlineData(32)]
    [InlineData(61)]
    [InlineData(100)]
    [InlineData(257)]
    public void MatchesDoublePrecisionReferenceLayerByLayer(int width)
    {
        var shape = TestModels.Small;
        byte[] file = TestModels.RandomContainer(shape, seed: width);
        byte[] image = TestModels.RandomImage(32, width, seed: 7);
        var container = ModelContainer.Parse(file);
        var reference = Reference64.Forward(container, shape, image, width, out int refSteps);
        ForwardTrace trace = Trace(Create(file, Dispatch.Best), image, width, 1, out int steps);
        Assert.Equal(refSteps, steps);

        foreach (var (name, (refShape, refData)) in reference)
        {
            var (gotShape, got) = trace.Layers[name];
            Assert.Equal(refShape, gotShape);
            var (atol, rtol) = ToleranceFor(name);
            double worst = 0;
            for (int i = 0; i < got.Length; i++)
            {
                double err = Math.Abs(got[i] - refData[i]);
                double allowed = atol + (rtol * Math.Abs(refData[i]));
                if (allowed == 0)
                {
                    Assert.Equal(refData[i], got[i]);
                    continue;
                }

                worst = Math.Max(worst, err / allowed);
            }

            Assert.True(worst <= 1, $"{name}: error is {worst:F2}x the tolerance (atol {atol}, rtol {rtol})");
        }
    }

    [Fact]
    public void ScalarAndAvx2AreBitIdenticalOnEveryLayerAndAtEveryDegree()
    {
        if (!Dispatch.Avx2Available)
        {
            return;
        }

        var shape = TestModels.Small;
        byte[] file = TestModels.RandomContainer(shape, seed: 11);
        LineRecognizer scalar = Create(file, KernelPath.Scalar);
        LineRecognizer avx2 = Create(file, KernelPath.Avx2);
        foreach (int width in new[] { 16, 33, 64, 97, 128, 301 })
        {
            byte[] image = TestModels.RandomImage(32, width, seed: width);
            ForwardTrace baseline = Trace(scalar, image, width, 1, out _);
            RecognizedLine baselineLine = scalar.Recognize(image, width, 1);
            foreach (var (recognizer, degree) in new[] { (scalar, 4), (avx2, 1), (avx2, 3), (avx2, 8) })
            {
                ForwardTrace other = Trace(recognizer, image, width, degree, out _);
                Assert.Equal(baseline.Layers.Keys.Order(), other.Layers.Keys.Order());
                foreach (var (name, (_, data)) in baseline.Layers)
                {
                    float[] o = other.Layers[name].Data;
                    for (int i = 0; i < data.Length; i++)
                    {
                        if (BitConverter.SingleToInt32Bits(data[i]) != BitConverter.SingleToInt32Bits(o[i]))
                        {
                            Assert.Fail($"width {width}, degree {degree}: {name}[{i}] differs: {data[i]:R} vs {o[i]:R}");
                        }
                    }
                }

                RecognizedLine line = recognizer.Recognize(image, width, degree);
                Assert.Equal(baselineLine.Text, line.Text);
                Assert.Equal(BitConverter.SingleToInt32Bits(baselineLine.Confidence), BitConverter.SingleToInt32Bits(line.Confidence));
            }
        }
    }

    [Fact]
    public void TooNarrowLinesReadAsEmptyAndBadSizesAreRejected()
    {
        byte[] file = TestModels.RandomContainer(TestModels.Small, seed: 5);
        LineRecognizer r = Create(file, Dispatch.Best);
        // Width 1 pools to zero columns after the first block.
        Assert.Equal(new RecognizedLine(string.Empty, 0f), r.Recognize(new byte[32], 1));
        Assert.Throws<ArgumentException>(() => r.Recognize(new byte[31 * 40], 40));
        Assert.Throws<ArgumentOutOfRangeException>(() => r.Recognize(new byte[32], 0));
        Assert.Throws<OcrLimitException>(() => r.Recognize(new byte[32 * 9000], 9000));
    }

    [Fact]
    public void ActivationLimitIsCheckedBeforeAllocating()
    {
        byte[] file = TestModels.RandomContainer(TestModels.Small, seed: 5);
        var tight = new OcrLimits { MaxActivationElements = 1000 };
        var r = new LineRecognizer(CrnnModel.FromContainer(ModelContainer.Parse(file), tight), tight, KernelPath.Scalar);
        Assert.Throws<OcrLimitException>(() => r.Recognize(new byte[32 * 200], 200));
    }

    [Fact]
    public void DecodingMergesRepeatsAndDropsBlanks()
    {
        var shape = TestModels.Small;
        LineRecognizer r = Create(TestModels.RandomContainer(shape, seed: 5), KernelPath.Scalar);
        int classes = shape.Classes;
        // Frames: A A blank A B B blank blank C  -> "AABC"
        int[] labels = [3, 3, 0, 3, 5, 5, 0, 0, 9];
        var logits = new float[labels.Length * classes];
        for (int t = 0; t < labels.Length; t++)
        {
            logits[(t * classes) + labels[t]] = 10f;
        }

        RecognizedLine line = r.Decode(logits, labels.Length);
        string[] cs = TestModels.Charset(classes);
        Assert.Equal(cs[3] + cs[3] + cs[5] + cs[9], line.Text);
        Assert.InRange(line.Confidence, 0.99f, 1f);

        // Ties go to the lowest index.
        var tie = new float[classes];
        tie[4] = 2f;
        tie[7] = 2f;
        Assert.Equal(cs[4], r.Decode(tie, 1).Text);
        Assert.Equal(string.Empty, r.Decode(new float[classes], 1).Text);
    }

    [Fact]
    public void WritesOutputHashForCrossConfigurationComparison()
    {
        // With ONEOCR_HASH_OUT set, appends a digest of every layer of a fixed run; the same run under different
        // DOTNET_Enable* settings must produce the same digest.
        string? output = Environment.GetEnvironmentVariable("ONEOCR_HASH_OUT");
        if (string.IsNullOrEmpty(output))
        {
            Assert.Skip("ONEOCR_HASH_OUT is not set.");
        }

        byte[] file = TestModels.RandomContainer(TestModels.Small, seed: 31);
        LineRecognizer r = Create(file, Dispatch.Best);
        using var sha = System.Security.Cryptography.SHA256.Create();
        var lines = new List<string>();
        foreach (int width in new[] { 40, 111, 300 })
        {
            byte[] image = TestModels.RandomImage(32, width, width);
            foreach (int degree in new[] { 1, 4 })
            {
                ForwardTrace trace = Trace(r, image, width, degree, out _);
                foreach (var (name, (_, data)) in trace.Layers.OrderBy(k => k.Key, StringComparer.Ordinal))
                {
                    var bytes = new byte[data.Length * 4];
                    Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
                    lines.Add($"w{width} d{degree} {name} {Convert.ToHexString(sha.ComputeHash(bytes))[..16]}");
                }

                RecognizedLine line = r.Recognize(image, width, degree);
                lines.Add($"w{width} d{degree} text {line.Text.Length} {BitConverter.SingleToInt32Bits(line.Confidence):X8} {Convert.ToHexString(sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(line.Text)))[..16]}");
            }
        }

        File.WriteAllLines(output, lines);
    }

    [Fact]
    public void PrintsSingleLineTimingWhenAsked()
    {
        if (Environment.GetEnvironmentVariable("ONEOCR_TIMING") is not "1")
        {
            Assert.Skip("ONEOCR_TIMING is not 1.");
        }

        byte[] file = TestModels.RandomContainer(TestModels.Full, seed: 1);
        byte[] image = TestModels.RandomImage(32, 400, 1);
        foreach (KernelPath path in Dispatch.Avx2Available ? new[] { KernelPath.Scalar, KernelPath.Avx2 } : new[] { KernelPath.Scalar })
        {
            foreach (int degree in new[] { 1, 4 })
            {
                LineRecognizer r = Create(file, path);
                r.Recognize(image, 400, degree);
                int runs = path == KernelPath.Scalar ? 5 : 20;
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < runs; i++)
                {
                    r.Recognize(image, 400, degree);
                }

                Console.Error.WriteLine($"TIMING {path} degree {degree}: {sw.Elapsed.TotalMilliseconds / runs:F1} ms per 32x400 line");
            }
        }
    }

    [Fact]
    public void RecognizesWithTheStatedParallelDefault()
    {
        byte[] file = TestModels.RandomContainer(TestModels.Small, seed: 21);
        LineRecognizer r = Create(file, Dispatch.Best);
        byte[] image = TestModels.RandomImage(32, 120, 3);
        RecognizedLine a = r.Recognize(image, 120);
        RecognizedLine b = r.Recognize(image, 120, 1);
        Assert.Equal(a, b);
    }

    /// <summary>Writes a fixture directory in the agreed layout from the float64 reference and runs the harness over it, so the harness itself is exercised.</summary>
    [Fact]
    public void FixtureHarnessAcceptsAReferenceGeneratedDirectory()
    {
        var shape = TestModels.Small;
        string dir = Path.Combine(Path.GetTempPath(), $"oneocr-parity-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            byte[] file = TestModels.RandomContainer(shape, seed: 77);
            File.WriteAllBytes(Path.Combine(dir, "model.onm"), file);
            var container = ModelContainer.Parse(file);
            var entries = new List<string>();
            foreach (int width in new[] { 64, 130 })
            {
                byte[] image = TestModels.RandomImage(32, width, width);
                var px = image.Select(b => new OneImage.Gray8(b)).ToArray();
                OneImage.Image.FromPixels<OneImage.Gray8>(px, width, 32).Save(Path.Combine(dir, $"l{width}.png"));
                var layers = new List<string>();
                foreach (var (name, (layerShape, data)) in Reference64.Forward(container, shape, image, width, out _))
                {
                    var raw = new byte[data.Length * 4];
                    for (int i = 0; i < data.Length; i++)
                    {
                        BitConverter.TryWriteBytes(raw.AsSpan(i * 4), (float)data[i]);
                    }

                    File.WriteAllBytes(Path.Combine(dir, $"l{width}.{name}.f32"), raw);
                    layers.Add($"{{\"name\":\"{name}\",\"shape\":[{string.Join(',', layerShape)}],\"file\":\"l{width}.{name}.f32\"}}");
                }

                entries.Add($"{{\"id\":\"l{width}\",\"png\":\"l{width}.png\",\"text\":\"\",\"layers\":[{string.Join(',', layers)}]}}");
            }

            File.WriteAllText(Path.Combine(dir, "fixtures.json"), $"{{\"lines\":[{string.Join(',', entries)}]}}");
            ParityFixtures.Run(dir, Dispatch.Best);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    /// <summary>Reads the fixture directory produced by the PyTorch side, when ONEOCR_PARITY_DIR points at one.</summary>
    [Fact]
    public void MatchesPyTorchFixturesWhenAvailable()
    {
        string? dir = Environment.GetEnvironmentVariable("ONEOCR_PARITY_DIR");
        if (string.IsNullOrEmpty(dir))
        {
            Assert.Skip("ONEOCR_PARITY_DIR is not set.");
        }

        ParityFixtures.Run(dir, Dispatch.Best);
    }
}
