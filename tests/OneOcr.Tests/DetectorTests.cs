using System.Text.Json;
using OneImage;
using OneOcr.Detection;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Tests;

public class DetectorTests
{
    // Tolerances against the float64 reference of the same network and fp16 weights: |ours - exact| <= Atol + Rtol * |exact|.
    // Measured on random-weight models (the shipped size and a narrow one, every path) with about 3x headroom over the worst error seen; the PyTorch
    // fixtures are checked at twice these values plus twice PyTorch's own recorded fp32 error.
    private static readonly Dictionary<string, (double Atol, double Rtol)> Tolerance = new()
    {
        ["input"] = (1e-6, 1e-6),
        ["stem"] = (1e-6, 1e-5),
        ["stage0"] = (1e-6, 1e-5),
        ["stage1"] = (2e-6, 1e-5),
        ["stage2"] = (2e-6, 1e-5),
        ["stage3"] = (2e-6, 1e-5),
        ["p8"] = (4e-6, 1e-5),
        ["p4"] = (5e-6, 1e-5),
        ["layout_feat"] = (5e-6, 1e-5),
        ["layout"] = (3e-6, 1e-5),
        ["text4"] = (5e-6, 1e-5),
        ["text2"] = (8e-6, 1e-5),
        ["prob"] = (5e-6, 1e-5),
        ["orient"] = (2e-7, 1e-5),
    };

    private static readonly string[] Layers = ["stem", "stage0", "stage1", "stage2", "stage3", "p8", "p4", "layout_feat", "layout", "text4", "text2", "prob", "orient"];

    private static (DbNetModel Model, ForwardTrace Trace, DetectorMaps Maps) Run(
        byte[] container, KernelPath path, int degree, float[] input, int h, int w)
    {
        var model = DbNetModel.FromContainer(ModelContainer.Parse(container), OcrLimits.Default);
        var trace = new ForwardTrace();
        DetectorMaps maps = DbNetForward.Run(
            model, path, degree, new BufferPool<float>(1 << 24, 4), input, h, w, OcrLimits.Default, trace, CancellationToken.None);
        return (model, trace, maps);
    }

    private static void AssertBitEqual(float[] a, float[] b, string what)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
            {
                Assert.Fail($"{what}: element {i} differs: {a[i]:R} vs {b[i]:R}");
            }
        }
    }

    private static readonly object ReportLock = new();

    private static void Report(string line)
    {
        if (Environment.GetEnvironmentVariable("ONEOCR_PARITY_REPORT") is { Length: > 0 } file)
        {
            lock (ReportLock)
            {
                File.AppendAllText(file, line + "\n");
            }
        }
    }

    [Theory]
    [InlineData(false, 64, 96)]
    [InlineData(true, 96, 64)]
    public void ForwardMatchesTheFloat64ReferenceOnEveryPath(bool small, int h, int w)
    {
        TestDetectorModels.Shape shape = small ? TestDetectorModels.Small : TestDetectorModels.Default;
        byte[] file = TestDetectorModels.Random(shape, 21);
        var container = ModelContainer.Parse(file);
        float[] input = TestDetectorModels.Normalize(TestDetectorModels.Image(h, w, 5));
        var reference = new DetectorReference64(container, shape).Forward(input, h, w);
        foreach (KernelPath path in Dispatch.Available())
        {
            var (_, trace, _) = Run(file, path, 1, input, h, w);
            foreach (string layer in Layers)
            {
                var (gotShape, got) = trace.Layers[layer];
                var (refShape, exact) = reference[layer];
                Assert.Equal(refShape, gotShape);
                var (atol, rtol) = Tolerance[layer];
                double worst = 0, maxAbs = 0;
                for (int i = 0; i < got.Length; i++)
                {
                    double diff = Math.Abs(got[i] - exact[i]);
                    maxAbs = Math.Max(maxAbs, diff);
                    worst = Math.Max(worst, diff / (atol + (rtol * Math.Abs(exact[i]))));
                }

                Report($"FLOAT64 {(small ? "small" : "default")} {path} {layer}: max abs {maxAbs:E2}, {worst:F2}x tolerance, max |ref| {exact.Max(Math.Abs):F2}");
                Assert.True(worst <= 1, $"{path}/{layer}: error is {worst:F2}x the tolerance (atol {atol}, rtol {rtol}).");
            }
        }
    }

    [Fact]
    public void OutputsAreBitIdenticalOnEveryPathAndDegree()
    {
        // The shipped size with widths that are not multiples of 16 elsewhere, on a size that needs several row chunks.
        foreach ((TestDetectorModels.Shape shape, int h, int w) in new[] { (TestDetectorModels.Default, 224, 288), (TestDetectorModels.Small, 96, 160) })
        {
            byte[] file = TestDetectorModels.Random(shape, 22);
            float[] input = TestDetectorModels.Normalize(TestDetectorModels.Image(h, w, 6));
            var (_, baseTrace, baseMaps) = Run(file, KernelPath.Scalar, 1, input, h, w);
            foreach (KernelPath path in Dispatch.Available())
            {
                foreach (int degree in new[] { 1, 3, 8 })
                {
                    var (_, trace, maps) = Run(file, path, degree, input, h, w);
                    foreach (string layer in Layers)
                    {
                        AssertBitEqual(baseTrace.Layers[layer].Data, trace.Layers[layer].Data, $"{path} degree {degree} {layer}");
                    }

                    AssertBitEqual(baseMaps.Prob, maps.Prob, $"{path} degree {degree} prob map");
                    AssertBitEqual(baseMaps.Layout, maps.Layout, $"{path} degree {degree} layout map");
                    AssertBitEqual(baseMaps.Orientation, maps.Orientation, $"{path} degree {degree} orientation");
                }
            }
        }
    }

    [Fact]
    public void DetectionsAreIdenticalOnEveryPathAndDegree()
    {
        byte[] file = TestDetectorModels.Random(TestDetectorModels.Small, 23);
        var model = DbNetModel.FromContainer(ModelContainer.Parse(file), OcrLimits.Default);
        byte[] gray = TestDetectorModels.Image(150, 230, 8);
        Gray8[] page = gray.Select(b => new Gray8(b)).ToArray();
        string? first = null;
        foreach (KernelPath path in Dispatch.Available())
        {
            foreach (int degree in new[] { 1, 4 })
            {
                var detector = new PageDetector(model, OcrLimits.Default, path);
                PageDetection d = detector.Detect(
                    page, 230, 150, new BufferPool<Gray8>(1 << 20, 1), new Layout.ResizerPool(), degree, CancellationToken.None);
                string text = string.Join(
                    ';', d.Lines.Select(l => $"{l.Left:R},{l.Top:R},{l.Right:R},{l.Bottom:R},{l.Score:R}")
                        .Concat(d.Regions.Select(r => $"{r.Class},{r.Left:R},{r.Top:R},{r.Right:R},{r.Bottom:R},{r.Score:R}"))
                        .Append($"{d.Quarter},{d.OrientationConfidence:R}"));
                first ??= text;
                Assert.Equal(first, text);
            }
        }
    }

    [Fact]
    public void MatchesPyTorchFixturesWhenAvailable()
    {
        string? dirs = Environment.GetEnvironmentVariable("ONEOCR_DETECTOR_PARITY_DIR");
        if (string.IsNullOrEmpty(dirs))
        {
            Assert.Skip("ONEOCR_DETECTOR_PARITY_DIR is not set.");
        }

        foreach (string dir in dirs.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            RunFixtures(dir);
        }
    }

    // Per layer: |ours - PyTorch| <= 2 * Atol + 2 * Floor + 2 * Rtol * |PyTorch|, Floor being PyTorch's own recorded fp32 error.
    private static void RunFixtures(string dir)
    {
        var container = ModelContainer.Load(Path.Combine(dir, "model.onm"));
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "fixtures.json")));
        var floors = new Dictionary<string, double>();
        foreach (JsonProperty f in doc.RootElement.GetProperty("fp32_vs_fp64_floor").EnumerateObject())
        {
            floors[f.Name] = f.Value.GetProperty("max_abs").GetDouble();
        }

        var worstByLayer = new Dictionary<string, double>();
        foreach (JsonElement sample in doc.RootElement.GetProperty("samples").EnumerateArray())
        {
            string id = sample.GetProperty("id").GetString()!;
            var layers = new Dictionary<string, (int[] Shape, float[] Data)>();
            foreach (JsonElement entry in sample.GetProperty("layers").EnumerateArray())
            {
                byte[] raw = File.ReadAllBytes(Path.Combine(dir, entry.GetProperty("file").GetString()!));
                var data = new float[raw.Length / 4];
                Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
                layers[entry.GetProperty("name").GetString()!] = (entry.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray(), data);
            }

            int h = layers["input"].Shape[1], w = layers["input"].Shape[2];

            // Pixels to the model's input tensor: our normalization must reproduce the fixture's input.
            if (sample.GetProperty("png").ValueKind == JsonValueKind.String)
            {
                Image image = Image.Load(Path.Combine(dir, sample.GetProperty("png").GetString()!));
                var px = new Gray8[w * h];
                image.CopyPixelsTo<Gray8>(px, new ImageRect(0, 0, w, h));
                var canvas = new float[w * h];
                PageDetector.Normalize(px, w, h, w, h, new ModelInputView(container.Input.Mean, container.Input.Std, container.Input.Invert), canvas);
                Compare(id, "input", canvas, layers["input"].Data, 1e-6, 1e-6, 0, worstByLayer);
            }

            foreach (KernelPath path in Dispatch.Available())
            {
                var model = DbNetModel.FromContainer(container, OcrLimits.Default);
                var trace = new ForwardTrace();
                DbNetForward.Run(model, path, 2, new BufferPool<float>(1 << 24, 4), layers["input"].Data, h, w, OcrLimits.Default, trace, CancellationToken.None);
                foreach ((string name, var expected) in layers)
                {
                    if (name == "input")
                    {
                        continue;
                    }

                    var (shape, got) = trace.Layers[name];
                    Assert.Equal(expected.Shape, shape);
                    var (atol, rtol) = Tolerance[name];
                    Compare($"{id}/{path}", name, got, expected.Data, 2 * atol, 2 * rtol, 2 * floors.GetValueOrDefault(name), worstByLayer);
                }
            }
        }

        foreach ((string layer, double abs) in worstByLayer)
        {
            Report($"PYTORCH {layer}: max abs difference {abs:E2} (PyTorch floor {floors.GetValueOrDefault(layer):E2})");
        }
    }

    private static void Compare(
        string where, string layer, float[] got, float[] expected, double atol, double rtol, double floor, Dictionary<string, double> worstByLayer)
    {
        Assert.Equal(expected.Length, got.Length);
        double worst = 0, maxAbs = 0;
        for (int i = 0; i < got.Length; i++)
        {
            double diff = Math.Abs(got[i] - expected[i]);
            maxAbs = Math.Max(maxAbs, diff);
            worst = Math.Max(worst, diff / (atol + floor + (rtol * Math.Abs(expected[i]))));
        }

        worstByLayer[layer] = Math.Max(worstByLayer.GetValueOrDefault(layer), maxAbs);
        Assert.True(worst <= 1, $"{where}/{layer}: error is {worst:F2}x the tolerance (atol {atol}, floor {floor}, rtol {rtol}).");
    }

    [Fact]
    public void PostprocessingGivesTheReferenceBoxes()
    {
        string? dirs = Environment.GetEnvironmentVariable("ONEOCR_DETECTOR_PARITY_DIR");
        if (string.IsNullOrEmpty(dirs))
        {
            Assert.Skip("ONEOCR_DETECTOR_PARITY_DIR is not set.");
        }

        foreach (string dir in dirs.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "postprocess.json")));
            string[] classes = doc.RootElement.GetProperty("classes").EnumerateArray().Select(e => e.GetString()!).ToArray();
            bool[] allTrained = classes.Select(_ => true).ToArray();
            foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
            {
                string name = c.GetProperty("name").GetString()!;
                int[] ps = c.GetProperty("prob_shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                int[] ls = c.GetProperty("layout_shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                float[] prob = ReadFloats(Path.Combine(dir, "postprocess", name + ".prob.f32"));
                float[] chw = ReadFloats(Path.Combine(dir, "postprocess", name + ".layout.f32"));
                int lh = ls[1], lw = ls[2], stride = 16;
                var layout = new float[lh * lw * stride];
                for (int k = 0; k < ls[0]; k++)
                {
                    for (int p = 0; p < lh * lw; p++)
                    {
                        layout[(p * stride) + k] = chw[(k * lh * lw) + p];
                    }
                }

                float[] orient = c.GetProperty("orient").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
                int pageW = ps[1] * 2, pageH = ps[0] * 2;
                PageDetection d = DetectorPostprocess.Decode(prob, ps[1], ps[0], layout, lw, lh, stride, classes, allTrained, orient, 1.0, pageW, pageH, OcrLimits.Default);

                var expectedLines = new List<(double[] Box, double Score)>();
                var lineScores = c.GetProperty("line_scores").EnumerateArray().Select(e => e.GetDouble()).ToArray();
                int li = 0;
                foreach (JsonElement b in c.GetProperty("lines").EnumerateArray())
                {
                    if (ClipBox(b.EnumerateArray().Select(e => e.GetDouble()).ToArray(), pageW, pageH) is { } box)
                    {
                        expectedLines.Add((box, lineScores[li]));
                    }

                    li++;
                }

                Assert.Equal(expectedLines.Count, d.Lines.Count);
                for (int i = 0; i < d.Lines.Count; i++)
                {
                    AssertBox(expectedLines[i].Box, d.Lines[i].Left, d.Lines[i].Top, d.Lines[i].Right, d.Lines[i].Bottom, $"{name} line {i}");
                    Assert.True(Math.Abs(expectedLines[i].Score - d.Lines[i].Score) < 1e-5, $"{name} line {i} score");
                }

                var expectedRegions = new List<(int Class, double[] Box, double Score)>();
                foreach (JsonElement r in c.GetProperty("regions").EnumerateArray())
                {
                    if (ClipBox(r.GetProperty("box").EnumerateArray().Select(e => e.GetDouble()).ToArray(), pageW, pageH) is { } box)
                    {
                        expectedRegions.Add((r.GetProperty("class").GetInt32(), box, r.GetProperty("score").GetDouble()));
                    }
                }

                Assert.Equal(expectedRegions.Count, d.Regions.Count);
                for (int i = 0; i < d.Regions.Count; i++)
                {
                    Assert.Equal(expectedRegions[i].Class, d.Regions[i].Class);
                    AssertBox(expectedRegions[i].Box, d.Regions[i].Left, d.Regions[i].Top, d.Regions[i].Right, d.Regions[i].Bottom, $"{name} region {i}");
                    Assert.True(Math.Abs(expectedRegions[i].Score - d.Regions[i].Score) < 1e-5, $"{name} region {i} score");
                }

                float[] probs = c.GetProperty("expected_orient").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
                int best = Array.IndexOf(probs, probs.Max());
                Assert.Equal(best, d.Quarter);
                Assert.True(Math.Abs(probs[best] - d.OrientationConfidence) < 1e-5);
                Report($"POSTPROCESS {name}: {d.Lines.Count} lines, {d.Regions.Count} regions match the reference");
            }
        }
    }

    private static double[]? ClipBox(double[] b, int w, int h)
    {
        double l = Math.Clamp(b[0], 0, w), t = Math.Clamp(b[1], 0, h), r = Math.Clamp(b[2], 0, w), bt = Math.Clamp(b[3], 0, h);
        return r > l && bt > t ? [l, t, r, bt] : null;
    }

    // The reference computes in float32 (numpy), we in double: boxes agree to far below a thousandth of a pixel.
    private static void AssertBox(double[] expected, double l, double t, double r, double b, string what)
    {
        double[] got = [l, t, r, b];
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(expected[i] - got[i]) < 1e-3, $"{what}: coordinate {i} is {got[i]}, reference {expected[i]}");
        }
    }

    private static float[] ReadFloats(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        var data = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return data;
    }
}
