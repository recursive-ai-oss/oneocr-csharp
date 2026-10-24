using System.Globalization;
using System.Text;
using System.Text.Json;
using OneOcr.Documents;
using OneOcr.Kernels;
using OneOcr.Model;
using DocInvoice = OneOcr.Documents.Invoice;
using DocReceipt = OneOcr.Documents.Receipt;
using DocModels = OneOcr.Models.Documents.DocumentModels;

namespace OneOcr.Tests;

/// <summary>
/// The document-extraction model: featurization and PyTorch parity on fixtures (<c>ONEOCR_DOCEXTRACT_PARITY_DIR</c>, directories joined by
/// ':'), bit-identical results across kernel paths and degrees, assembly, and the loader's limits.
/// Parity rule: our fp32 error against PyTorch's float64 run of the same fp16 weights stays within <see cref="ErrorFactor"/> times
/// PyTorch's own fp32 error (recorded per layer in the fixtures) plus <see cref="UlpFloor"/> ulps of the layer's largest value; and our
/// result stays within <see cref="ErrorFactor"/> + 1 times that same error of PyTorch's fp32 result.
/// </summary>
public class DocExtractTests
{
    private const double ErrorFactor = 2.0;
    private const double UlpFloor = 8;
    private const double Ulp = 1.1920929e-7;

    private static IEnumerable<string> FixtureDirs()
    {
        string? env = Environment.GetEnvironmentVariable("ONEOCR_DOCEXTRACT_PARITY_DIR");
        return string.IsNullOrEmpty(env) ? [] : env.Split(':', StringSplitOptions.RemoveEmptyEntries);
    }

    private static string[] RequireFixtures()
    {
        string[] dirs = [.. FixtureDirs()];
        if (dirs.Length == 0)
        {
            Assert.Skip("ONEOCR_DOCEXTRACT_PARITY_DIR is not set.");
        }

        return dirs;
    }

    private static float[] ReadF32(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        var v = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, v, 0, raw.Length);
        return v;
    }

    private static double[] ReadF64(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        var v = new double[raw.Length / 8];
        Buffer.BlockCopy(raw, 0, v, 0, raw.Length);
        return v;
    }

    private sealed record Sample(string Id, string[] Texts, OcrBox[] Boxes, double Width, double Height, JsonElement Json);

    private static (DocExtractModel Model, JsonDocument Doc, List<Sample> Samples) Open(string dir)
    {
        DocExtractModel model = DocExtractModel.FromContainer(ModelContainer.Load(Path.Combine(dir, "model.onm")), OcrLimits.Default);
        JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "fixtures.json")));
        var samples = new List<Sample>();
        foreach (JsonElement s in doc.RootElement.GetProperty("samples").EnumerateArray())
        {
            JsonElement[] words = [.. s.GetProperty("words").EnumerateArray()];
            string[] texts = [.. words.Select(w => w.GetProperty("text").GetString()!)];
            OcrBox[] boxes = [.. words.Select(w =>
            {
                double[] b = [.. w.GetProperty("box").EnumerateArray().Select(e => e.GetDouble())];
                return new OcrBox(b[0], b[1], b[2], b[3]);
            })];
            double width = s.TryGetProperty("width", out JsonElement we) ? we.GetDouble() : 800;
            double height = s.TryGetProperty("height", out JsonElement he) ? he.GetDouble() : 800;
            samples.Add(new Sample(s.GetProperty("id").GetString()!, texts, boxes, width, height, s));
        }

        return (model, doc, samples);
    }

    private static bool IsAssemblyOnly(Sample s) => s.Json.TryGetProperty("assemblyOnly", out _);

    [Fact]
    public void FeaturizationMatchesTheReferenceExactly()
    {
        foreach (string dir in RequireFixtures())
        {
            (DocExtractModel model, JsonDocument doc, List<Sample> samples) = Open(dir);
            using (doc)
            {
                foreach (Sample s in samples.Where(x => !IsAssemblyOnly(x)))
                {
                    DocFeatures f = DocFeatures.Build(model, s.Texts, s.Boxes, s.Width, s.Height);
                    JsonElement fj = s.Json.GetProperty("featurize");
                    Assert.Equal(fj.GetProperty("unit").GetDouble(), f.Unit);
                    int i = 0;
                    foreach (JsonElement row in fj.GetProperty("chars").EnumerateArray())
                    {
                        foreach (JsonElement c in row.EnumerateArray())
                        {
                            Assert.Equal(c.GetInt32(), f.Chars[i++]);
                        }
                    }

                    Assert.Equal(f.Chars.Length, i);
                    i = 0;
                    foreach (JsonElement row in fj.GetProperty("bins").EnumerateArray())
                    {
                        foreach (JsonElement c in row.EnumerateArray())
                        {
                            Assert.Equal(c.GetInt32(), f.Bins[i++]);
                        }
                    }

                    AssertBits(ReadF32(Path.Combine(dir, fj.GetProperty("feats_file").GetString()!)), f.Feats, $"{s.Id} feats");
                    AssertBits(ReadF32(Path.Combine(dir, fj.GetProperty("geo_file").GetString()!)), f.Geo, $"{s.Id} geo");
                }
            }
        }
    }

    private static void AssertBits(ReadOnlySpan<float> expected, ReadOnlySpan<float> got, string what)
    {
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(got[i]))
            {
                Assert.Fail($"{what}: element {i} differs: expected {expected[i]:R}, got {got[i]:R}");
            }
        }
    }

    [Fact]
    public void EveryLayerMatchesPyTorchAndTheFloat64ReferenceOnEveryPath()
    {
        var report = new StringBuilder();
        foreach (string dir in RequireFixtures())
        {
            (DocExtractModel model, JsonDocument doc, List<Sample> samples) = Open(dir);
            using (doc)
            {
                foreach (KernelPath path in Dispatch.Available())
                {
                    foreach (Sample s in samples.Where(x => !IsAssemblyOnly(x) && x.Json.GetProperty("windows").GetArrayLength() == 1))
                    {
                        DocFeatures f = DocFeatures.Build(model, s.Texts, s.Boxes, s.Width, s.Height);
                        var trace = new Dictionary<string, float[]>();
                        DocWindowOutputs w = DocForward.RunWindow(model, f, 0, f.Count, path, 1, trace, CancellationToken.None);
                        trace["tags"] = w.Tags;
                        trace["roles"] = w.Roles;
                        trace["doc"] = w.Doc;
                        trace["rel"] = w.Rel;
                        foreach (JsonElement layer in s.Json.GetProperty("layers").EnumerateArray())
                        {
                            string name = layer.GetProperty("name").GetString()!;
                            float[] pt = ReadF32(Path.Combine(dir, layer.GetProperty("file").GetString()!));
                            double[] exact = ReadF64(Path.Combine(dir, layer.GetProperty("file64").GetString()!));
                            double floor = s.Json.GetProperty("fp32_vs_fp64_floor").GetProperty(name).GetProperty("max_abs").GetDouble();
                            float[] got = trace[name];
                            Assert.Equal(pt.Length, got.Length);
                            double refMax = exact.Max(Math.Abs);
                            double slack = UlpFloor * Ulp * Math.Max(refMax, 1e-30);
                            double errExact = 0, errPyTorch = 0;
                            for (int i = 0; i < got.Length; i++)
                            {
                                errExact = Math.Max(errExact, Math.Abs(got[i] - exact[i]));
                                errPyTorch = Math.Max(errPyTorch, Math.Abs(got[i] - (double)pt[i]));
                            }

                            report.AppendLine(CultureInfo.InvariantCulture, $"{path} {s.Id} {name}: vs float64 {errExact:E2}, pytorch fp32 vs float64 {floor:E2}, vs pytorch {errPyTorch:E2}, max {refMax:E2}");
                            Assert.True(errExact <= (ErrorFactor * floor) + slack, $"{path} {s.Id} {name}: error vs float64 {errExact:E3} exceeds {ErrorFactor} x PyTorch's {floor:E3} + {slack:E3}");
                            Assert.True(errPyTorch <= ((ErrorFactor + 1) * floor) + slack, $"{path} {s.Id} {name}: differs from PyTorch by {errPyTorch:E3}, allowed {(ErrorFactor + 1) * floor + slack:E3}");
                        }
                    }
                }
            }
        }

        string? outFile = Environment.GetEnvironmentVariable("ONEOCR_DOCEXTRACT_REPORT");
        if (!string.IsNullOrEmpty(outFile))
        {
            File.WriteAllText(outFile, report.ToString());
        }
    }

    private static DocOutputs FromLogits(DocExtractModel m, Sample s, string dir)
    {
        int n = s.Texts.Length;
        float[] Read(string name) => ReadF32(Path.Combine(dir, $"data/{s.Id}.logits_{name}.f32"));
        float[] rel = Read("rel");
        var g = new int[n];
        var kv = new int[n];
        for (int i = 0; i < n; i++)
        {
            g[i] = DocForward.ArgMax(rel.AsSpan(i * n, n));
            kv[i] = DocForward.ArgMax(rel.AsSpan(((n + i) * n), n));
        }

        return new DocOutputs { N = n, Tags = Read("tags"), Roles = Read("roles"), Doc = Read("doc"), GroupPointer = g, KeyValuePointer = kv };
    }

    private static DocumentExtraction Assemble(DocExtractModel m, Sample s, DocOutputs o)
    {
        int[] index = [.. Enumerable.Range(0, s.Texts.Length)];
        OcrSource[] sources = [.. index.Select(_ => OcrSource.Ocr)];
        return DocAssembly.Assemble(m, o, s.Texts, s.Boxes, index, sources, null, NumberStyle.PointDecimal, DateOrder.Auto);
    }

    [Fact]
    public void AssemblyMatchesTheReferenceOnPlannedLogits()
    {
        foreach (string dir in RequireFixtures())
        {
            (DocExtractModel model, JsonDocument doc, List<Sample> samples) = Open(dir);
            using (doc)
            {
                int checkedPages = 0;
                foreach (Sample s in samples.Where(IsAssemblyOnly))
                {
                    AssertAssembled(s.Json.GetProperty("assembled"), Assemble(model, s, FromLogits(model, s, dir)), s.Id);
                    checkedPages++;
                }

                Assert.True(checkedPages >= 3);
            }
        }
    }

    [Fact]
    public void WholeModelAssemblyAndPointersMatchTheReferenceOnEveryPathAndPageSize()
    {
        foreach (string dir in RequireFixtures())
        {
            (DocExtractModel model, JsonDocument doc, List<Sample> samples) = Open(dir);
            using (doc)
            {
                foreach (KernelPath path in Dispatch.Available())
                {
                    foreach (Sample s in samples.Where(x => !IsAssemblyOnly(x)))
                    {
                        DocFeatures f = DocFeatures.Build(model, s.Texts, s.Boxes, s.Width, s.Height);
                        DocOutputs o = DocForward.Run(model, f, path, 2, CancellationToken.None);
                        double[][] margin = [.. s.Json.GetProperty("ptr_margin").EnumerateArray().Select(a => a.EnumerateArray().Select(e => e.GetDouble()).ToArray())];
                        int[][] expected = [
                            [.. s.Json.GetProperty("group_ptr").EnumerateArray().Select(e => e.GetInt32())],
                            [.. s.Json.GetProperty("kv_ptr").EnumerateArray().Select(e => e.GetInt32())],
                        ];
                        int flips = 0;
                        for (int i = 0; i < o.N; i++)
                        {
                            if (o.GroupPointer[i] != expected[0][i] && margin[0][i] > 1e-4)
                            {
                                Assert.Fail($"{path} {s.Id}: item pointer of word {i} is {o.GroupPointer[i]}, reference {expected[0][i]} (margin {margin[0][i]:E2})");
                            }

                            if (o.KeyValuePointer[i] != expected[1][i] && margin[1][i] > 1e-4)
                            {
                                Assert.Fail($"{path} {s.Id}: key/value pointer of word {i} is {o.KeyValuePointer[i]}, reference {expected[1][i]} (margin {margin[1][i]:E2})");
                            }

                            flips += (o.GroupPointer[i] != expected[0][i] ? 1 : 0) + (o.KeyValuePointer[i] != expected[1][i] ? 1 : 0);
                        }

                        Assert.True(flips <= 2, $"{path} {s.Id}: {flips} pointers differ within the near-tie margin");
                        AssertAssembled(s.Json.GetProperty("assembled"), Assemble(model, s, o), $"{path} {s.Id}");
                    }
                }
            }
        }
    }

    private static void AssertAssembled(JsonElement expected, DocumentExtraction got, string id)
    {
        Assert.Equal(expected.GetProperty("documentType").GetString(), got.DocumentType.ToString().ToLowerInvariant());
        Assert.Equal(expected.GetProperty("documentTypeConfidence").GetDouble(), got.DocumentTypeConfidence, 1e-5);

        JsonElement[] fields = [.. expected.GetProperty("fields").EnumerateArray()];
        Assert.Equal(fields.Length, got.Fields.Count);
        for (int i = 0; i < fields.Length; i++)
        {
            AssertField(fields[i], got.Fields[i], $"{id} field {i}");
        }

        AssertGroups(expected.GetProperty("lineItems"), got.LineItems, $"{id} lineItems");
        AssertGroups(expected.GetProperty("taxBreakdown"), got.TaxRows, $"{id} taxBreakdown");

        JsonElement[] pairs = [.. expected.GetProperty("keyValues").EnumerateArray()];
        Assert.Equal(pairs.Length, got.KeyValues.Count);
        for (int i = 0; i < pairs.Length; i++)
        {
            Assert.Equal(pairs[i].GetProperty("key").GetProperty("words").EnumerateArray().Select(e => e.GetInt32()), got.KeyValues[i].KeyWordIndices);
            Assert.Equal(pairs[i].GetProperty("value").GetProperty("words").EnumerateArray().Select(e => e.GetInt32()), got.KeyValues[i].ValueWordIndices);
            Assert.Equal(pairs[i].GetProperty("confidence").GetDouble(), got.KeyValues[i].Confidence, 1e-5);
        }
    }

    private static void AssertGroups(JsonElement expected, IReadOnlyList<ExtractedGroup> got, string id)
    {
        JsonElement[] groups = [.. expected.EnumerateArray()];
        Assert.Equal(groups.Length, got.Count);
        for (int i = 0; i < groups.Length; i++)
        {
            Assert.Equal(groups[i].GetProperty("anchor").GetInt32(), got[i].Anchor);
            JsonElement[] fields = [.. groups[i].GetProperty("fields").EnumerateArray()];
            Assert.Equal(fields.Length, got[i].Fields.Count);
            for (int k = 0; k < fields.Length; k++)
            {
                AssertField(fields[k], got[i].Fields[k], $"{id}[{i}].{k}");
            }
        }
    }

    private static void AssertField(JsonElement expected, ExtractedField got, string where)
    {
        Assert.True(expected.GetProperty("id").GetString() == got.Id, $"{where}: class");
        Assert.True(expected.GetProperty("text").GetString() == got.Text, $"{where}: text");
        Assert.Equal(expected.GetProperty("words").EnumerateArray().Select(e => e.GetInt32()), got.WordIndices);
        double[] b = [.. expected.GetProperty("bounds").EnumerateArray().Select(e => e.GetDouble())];
        Assert.Equal(new OcrBox(b[0], b[1], b[2], b[3]), got.Box);
        Assert.True(Math.Abs(expected.GetProperty("confidence").GetDouble() - got.Confidence) < 1e-5, $"{where}: confidence {got.Confidence} vs {expected.GetProperty("confidence").GetDouble()}");
    }

    [Fact]
    public void ResultsAreBitIdenticalAcrossPathsAndDegrees()
    {
        foreach (string dir in RequireFixtures())
        {
            (DocExtractModel model, JsonDocument doc, List<Sample> samples) = Open(dir);
            using (doc)
            {
                foreach (Sample s in samples.Where(x => !IsAssemblyOnly(x) && (x.Texts.Length is 64 or 513)))
                {
                    DocFeatures f = DocFeatures.Build(model, s.Texts, s.Boxes, s.Width, s.Height);
                    DocOutputs reference = DocForward.Run(model, f, KernelPath.Scalar, 1, CancellationToken.None);
                    foreach (KernelPath path in Dispatch.Available())
                    {
                        foreach (int degree in new[] { 1, 3 })
                        {
                            DocOutputs o = DocForward.Run(model, f, path, degree, CancellationToken.None);
                            AssertBits(reference.Tags, o.Tags, $"{path}/{degree} {s.Id} tags");
                            AssertBits(reference.Roles, o.Roles, $"{path}/{degree} {s.Id} roles");
                            AssertBits(reference.Doc, o.Doc, $"{path}/{degree} {s.Id} doc");
                            Assert.Equal(reference.GroupPointer, o.GroupPointer);
                            Assert.Equal(reference.KeyValuePointer, o.KeyValuePointer);
                        }
                    }
                }
            }
        }
    }

    // ---- kernels, without fixtures ----

    private static float[] Random(Random rng, int n, double scale = 1)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
        {
            v[i] = (float)(((rng.NextDouble() * 2) - 1) * scale);
        }

        return v;
    }

    [Fact]
    public void LayerNormGeluSoftmaxAndCharPoolAreBitIdenticalOnEveryPath()
    {
        var rng = new Random(5);
        foreach (int d in new[] { 1, 2, 7, 16, 17, 33, 192 })
        {
            float[] x = Random(rng, 5 * d, 3), w = Random(rng, d), b = Random(rng, d);
            var reference = new float[5 * d];
            DocKernels.LayerNorm(KernelPath.Scalar, x, reference, 5, d, w, b);
            float[] g = (float[])x.Clone();
            DocKernels.Gelu(KernelPath.Scalar, g);
            foreach (KernelPath path in Dispatch.Available())
            {
                var got = new float[5 * d];
                DocKernels.LayerNorm(path, x, got, 5, d, w, b);
                AssertBits(reference, got, $"layer norm d={d} {path}");
                float[] gv = (float[])x.Clone();
                DocKernels.Gelu(path, gv);
                AssertBits(g, gv, $"gelu n={5 * d} {path}");
            }
        }

        foreach (int n in new[] { 1, 2, 5, 16, 31, 64, 100 })
        {
            float[] s = Random(rng, 3 * n, 4), bias = Random(rng, 3 * n, 2);
            float[] reference = (float[])s.Clone();
            DocKernels.SoftmaxRowsWithBias(KernelPath.Scalar, reference, bias, 3, n, 5.656854f);
            for (int r = 0; r < 3; r++)
            {
                Assert.Equal(1.0, reference.AsSpan(r * n, n).ToArray().Sum(v => (double)v), 1e-5);
            }

            foreach (KernelPath path in Dispatch.Available())
            {
                float[] got = (float[])s.Clone();
                DocKernels.SoftmaxRowsWithBias(path, got, bias, 3, n, 5.656854f);
                AssertBits(reference, got, $"softmax n={n} {path}");
            }
        }

        foreach (int len in new[] { 1, 2, 5, 8 })
        {
            var ids = new int[8];
            for (int i = 0; i < len; i++)
            {
                ids[i] = 1 + rng.Next(40);
            }

            float[] table = Random(rng, 42 * 8), cw = Random(rng, 16 * 24), cb = Random(rng, 16);
            var reference = new float[16];
            DocKernels.CharPool(KernelPath.Scalar, ids, table, 42, 8, cw, cb, 16, reference);
            Assert.All(reference, v => Assert.True(v >= 0));
            foreach (KernelPath path in Dispatch.Available())
            {
                var got = new float[16];
                DocKernels.CharPool(path, ids, table, 42, 8, cw, cb, 16, got);
                AssertBits(reference, got, $"char pool len={len} {path}");
            }
        }
    }

    [Fact]
    public void LayerNormAndSoftmaxMatchADoublePrecisionReference()
    {
        var rng = new Random(9);
        int d = 192;
        float[] x = Random(rng, 4 * d, 5), w = Random(rng, d), b = Random(rng, d);
        var got = new float[4 * d];
        DocKernels.LayerNorm(Dispatch.Best, x, got, 4, d, w, b);
        for (int r = 0; r < 4; r++)
        {
            double mean = x.Skip(r * d).Take(d).Sum(v => (double)v) / d;
            double var = x.Skip(r * d).Take(d).Sum(v => ((double)v - mean) * ((double)v - mean)) / d;
            for (int i = 0; i < d; i++)
            {
                double expected = (((double)x[(r * d) + i] - mean) / Math.Sqrt(var + 1e-5) * w[i]) + b[i];
                Assert.Equal(expected, got[(r * d) + i], 3e-6);
            }
        }

        int n = 50;
        float[] s = Random(rng, n, 3), bias = Random(rng, n);
        float[] p = (float[])s.Clone();
        DocKernels.SoftmaxRowsWithBias(Dispatch.Best, p, bias, 1, n, 4f);
        double[] logits = [.. Enumerable.Range(0, n).Select(i => ((double)s[i] / 4) + bias[i])];
        double top = logits.Max();
        double z = logits.Sum(v => Math.Exp(v - top));
        for (int i = 0; i < n; i++)
        {
            Assert.Equal(Math.Exp(logits[i] - top) / z, p[i], 1e-6);
        }
    }

    // ---- container, limits ----

    private static void AssertRejected(byte[] file, OcrLimits? limits = null)
    {
        Exception ex = Assert.ThrowsAny<Exception>(() => DocExtractModel.FromContainer(ModelContainer.Parse(file, limits), limits ?? OcrLimits.Default));
        Assert.True(ex is OcrModelException or OcrLimitException, $"unexpected {ex.GetType().Name}: {ex.Message}");
    }

    [Fact]
    public void LoadsAValidSmallContainerAndRejectsMalformedOnes()
    {
        TestDocModels.Shape shape = TestDocModels.Small;
        byte[] valid = TestDocModels.RandomContainer(shape, 1);
        DocumentExtractor extractor = DocumentExtractor.Load(valid);
        Assert.Equal(shape.Classes, extractor.Classes);

        AssertRejected(TestDocModels.RandomContainer(shape, 1, drop: (n, _) => n == "tag_head.weight"));
        AssertRejected(TestDocModels.RandomContainer(shape, 1, editConfig: c => c.Replace("\"heads\":2", "\"heads\":5", StringComparison.Ordinal)));
        AssertRejected(TestDocModels.RandomContainer(shape, 1, editConfig: c => c.Replace("\"unk_char\":41", "\"unk_char\":40", StringComparison.Ordinal)));
        AssertRejected(TestDocModels.RandomContainer(shape, 1, editConfig: c => c.Replace("\"tags\":21", "\"tags\":22", StringComparison.Ordinal)));
        AssertRejected(TestDocModels.RandomContainer(shape, 1, editConfig: c => c.Replace("\"d\":32", "\"d\":48", StringComparison.Ordinal)));
        AssertRejected(TestDocModels.RandomContainer(shape, 1, editConfig: c => c.Replace("\"max_words\":16", "\"max_words\":17", StringComparison.Ordinal)));
        AssertRejected(TestDocModels.RandomContainer(shape, 1, editConfig: c => c.Replace("\"roles_list\":[\"O\"", "\"roles_list\":[\"X\"", StringComparison.Ordinal)));

        // A pixel model's input block on a word model, and the reverse.
        AssertRejected(TestContainerWriter.Write("docextract-v0", 32, TestDocModels.Charset(shape), TestDocModels.ConfigJson(shape), TestDocModels.RandomTensors(shape, 1), calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}"));
        AssertRejected(TestContainerWriter.Write("crnn-v0", 0, TestModels.Charset(41), TestModels.ConfigJson(TestModels.Small), TestModels.RandomTensors(TestModels.Small, 1), calibrationJson: "{\"raw\":\"word-mean-max-run\",\"scripts\":{}}", wordInputJson: TestDocModels.WordInputJson(shape)));
    }

    [Fact]
    public void DeclaredSizesAreCheckedBeforeAnythingIsAllocated()
    {
        TestDocModels.Shape shape = TestDocModels.Small;
        byte[] valid = TestDocModels.RandomContainer(shape, 1);
        var tight = new OcrLimits { MaxWordsPerWindow = 8 };
        Assert.Throws<OcrLimitException>(() => DocExtractModel.FromContainer(ModelContainer.Parse(valid, tight), tight));
        var chars = new OcrLimits { MaxCharsPerWord = 4 };
        Assert.Throws<OcrLimitException>(() => ModelContainer.Parse(valid, chars));
        var layers = new OcrLimits { MaxTransformerLayers = 1 };
        Assert.Throws<OcrModelException>(() => DocExtractModel.FromContainer(ModelContainer.Parse(valid, layers), layers));
        var width = new OcrLimits { MaxLayerWidth = 16 };
        Assert.Throws<OcrModelException>(() => DocExtractModel.FromContainer(ModelContainer.Parse(valid, width), width));
        var activations = new OcrLimits { MaxActivationElements = 1000 };
        Assert.Throws<OcrLimitException>(() => DocExtractModel.FromContainer(ModelContainer.Parse(valid, activations), activations));

        // A huge declared width with a tiny file fails on the missing tensor, not on an allocation.
        byte[] huge = TestDocModels.RandomContainer(shape, 1, editConfig: c => c.Replace("\"ffn\":48", "\"ffn\":4096", StringComparison.Ordinal));
        AssertRejected(huge);
        var many = new OcrLimits { MaxExtractionClasses = 4 };
        Assert.Throws<OcrModelException>(() => DocExtractModel.FromContainer(ModelContainer.Parse(valid, many), many));
    }

    // ---- extractor ----

    private static OcrPage PageOf(string[] texts, OcrBox[] boxes, double width, double height)
    {
        var words = texts.Select((t, i) => new OcrWord(t, boxes[i], 0.9f, 0.9f, OcrSource.Ocr, [])).ToArray();
        var lines = words.Select(w => new OcrLine(w.Text, [w], [0], w.Box, 0.9f, OcrSource.Ocr)).ToArray();
        var block = new OcrBlock(OcrBlockRole.Body, false, new OcrBox(0, 0, width, height), lines, 0.9f);
        return new OcrPage(0, width, height, 72, [block]);
    }

    [Fact]
    public void ExtractionIsDeterministicAcrossPathsAndDegreesAndNeverReadsOutsideThePage()
    {
        foreach (TestDocModels.Shape shape in new[] { TestDocModels.Small, TestDocModels.SmallWindow })
        {
            DocumentExtractor extractor = DocumentExtractor.Load(TestDocModels.RandomContainer(shape, 3));
            foreach (int words in new[] { 1, 2, 9, 30, 41 })
            {
                (string[] texts, OcrBox[] boxes, double w, double h) = TestDocModels.Page(words, words);
                OcrPage page = PageOf(texts, boxes, w, h);
                DocumentExtraction reference = extractor.WithPath(KernelPath.Scalar).Extract(page, 1);
                DocInvoice invoice = extractor.ExtractInvoice(page, 3);
                DocReceipt receipt = extractor.ExtractReceipt(page, 2);
                Assert.All(invoice.Extraction.Fields, f => Assert.StartsWith("BT-", f.Id, StringComparison.Ordinal));
                Assert.All(receipt.Extraction.Fields, f => Assert.StartsWith("receipt.", f.Id, StringComparison.Ordinal));
                foreach (KernelPath path in Dispatch.Available())
                {
                    foreach (int degree in new[] { 1, 4 })
                    {
                        DocumentExtraction got = extractor.WithPath(path).Extract(page, degree);
                        Assert.Equal(reference.DocumentType, got.DocumentType);
                        Assert.Equal(BitConverter.SingleToInt32Bits(reference.DocumentTypeConfidence), BitConverter.SingleToInt32Bits(got.DocumentTypeConfidence));
                        Assert.Equal(reference.Fields.Select(Describe), got.Fields.Select(Describe));
                        Assert.Equal(reference.LineItems.SelectMany(g => g.Fields).Select(Describe), got.LineItems.SelectMany(g => g.Fields).Select(Describe));
                        Assert.Equal(reference.KeyValues.Select(k => string.Join(',', k.KeyWordIndices) + ">" + string.Join(',', k.ValueWordIndices)), got.KeyValues.Select(k => string.Join(',', k.KeyWordIndices) + ">" + string.Join(',', k.ValueWordIndices)));
                    }
                }

                foreach (ExtractedField f in reference.Fields.Concat(reference.LineItems.SelectMany(g => g.Fields)))
                {
                    Assert.All(f.WordIndices, i => Assert.InRange(i, 0, words - 1));
                    Assert.Equal(string.Join(' ', f.WordIndices.Select(i => texts[i])), f.Text);
                    Assert.InRange(f.Confidence, 0f, 1f);
                }
            }
        }
    }

    private static string Describe(ExtractedField f) =>
        string.Create(CultureInfo.InvariantCulture, $"{f.Id}|{f.Text}|{string.Join(',', f.WordIndices)}|{BitConverter.SingleToInt32Bits(f.Confidence)}|{f.Value}");

    [Fact]
    public void PagesWithoutWordsAndOverTheLimitAreHandled()
    {
        DocumentExtractor extractor = DocumentExtractor.Load(TestDocModels.RandomContainer(TestDocModels.Small, 3));
        var empty = new OcrPage(0, 100, 100, 72, []);
        DocumentExtraction none = extractor.Extract(empty);
        Assert.Equal(DocumentType.Other, none.DocumentType);
        Assert.Empty(none.Fields);
        Assert.Single(none.Diagnostics);
        Assert.Throws<ArgumentNullException>(() => extractor.Extract(null!));

        (string[] texts, OcrBox[] boxes, double w, double h) = TestDocModels.Page(20, 1);
        var tight = DocumentExtractor.Load(TestDocModels.RandomContainer(TestDocModels.Small, 3), new OcrLimits { MaxWordsPerPage = 10 });
        Assert.Throws<OcrLimitException>(() => tight.Extract(PageOf(texts, boxes, w, h)));
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => extractor.Extract(PageOf(texts, boxes, w, h), 1, cts.Token));

        // Non-finite boxes are read as zeros, not propagated.
        OcrBox[] bad = [.. boxes];
        bad[3] = new OcrBox(double.NaN, double.PositiveInfinity, double.NegativeInfinity, 1e300);
        _ = extractor.Extract(PageOf(texts, bad, w, h));
    }

    [Fact]
    public void WindowsCoverTheWholePageAndEachWordHasOneOwner()
    {
        foreach (int n in new[] { 1, 12, 13, 21, 22, 30, 100 })
        {
            int window = 12, stride = 9;
            var starts = new List<int> { 0 };
            while (starts[^1] + window < n)
            {
                starts.Add(starts[^1] + stride);
            }

            for (int word = 0; word < n; word++)
            {
                int k = DocForward.Owner(word, starts, n, window);
                Assert.InRange(word, starts[k], Math.Min(starts[k] + window, n) - 1);
            }
        }
    }

    // ---- assembly on a hand-made page ----

    [Fact]
    public void AssemblyReadsFieldsLineItemsKeyValuesAndNormalizesValues()
    {
        TestDocModels.Shape shape = TestDocModels.Small;
        DocumentExtractor extractor = DocumentExtractor.Load(TestDocModels.RandomContainer(shape, 3));
        DocExtractModel m = extractor.Model;
        string[] texts = ["Invoice", "No", "INV-77", "Date", "13.10.2026", "Total", "EUR", "1.234,50", "Widget", "2", "Gadget", "3", "7,50", "9,00"];
        int n = texts.Length;
        OcrBox[] boxes = [.. texts.Select((_, i) => new OcrBox(10 + (i * 40), 10 + (i / 8 * 20), 45 + (i * 40), 25 + (i / 8 * 20)))];
        int tagCount = m.Tags;
        var tags = new float[n * tagCount];
        var roles = new float[n * 5];
        void Tag(int word, string cls, bool begin)
        {
            int c = shape.Classes.ToList().IndexOf(cls);
            tags[(word * tagCount) + 1 + (2 * c) + (begin ? 0 : 1)] = 9;
        }

        for (int i = 0; i < n; i++)
        {
            tags[i * tagCount] = 1; // O unless set below
            roles[i * 5] = 1;
        }

        void Set(int word, string cls, bool begin)
        {
            tags[word * tagCount] = 0;
            Tag(word, cls, begin);
        }

        Set(2, "BT-1", true);
        Set(4, "BT-2", true);
        Set(6, "BT-112", true);
        Set(7, "BT-112", false);
        Set(8, "BT-153", true);
        Set(9, "BT-129", true);
        Set(10, "BT-153", true);
        Set(11, "BT-129", true);
        Set(12, "BT-116", true);
        Set(13, "receipt.total", true); // not allowed for an invoice: masked

        void Role(int word, int role)
        {
            roles[word * 5] = 0;
            roles[(word * 5) + role] = 9;
        }

        Role(5, 1); // B-KEY "Total"
        Role(6, 3); // B-VAL "EUR 1.234,50"
        Role(7, 4);
        var group = Enumerable.Range(0, n).ToArray();
        group[9] = 8; // "2" belongs to Widget
        group[11] = 10; // "3" belongs to Gadget
        group[12] = 12;
        var kv = Enumerable.Range(0, n).ToArray();
        kv[6] = 5;
        var o = new DocOutputs { N = n, Tags = tags, Roles = roles, Doc = [9, 0, 0], GroupPointer = group, KeyValuePointer = kv };
        int[] index = [.. Enumerable.Range(0, n)];
        OcrSource[] sources = [.. index.Select(_ => OcrSource.Ocr)];
        DocumentExtraction x = DocAssembly.Assemble(m, o, texts, boxes, index, sources, null, NumberStyle.CommaDecimal, DateOrder.Auto);

        Assert.Equal(DocumentType.Invoice, x.DocumentType);
        Assert.Equal("INV-77", x.Get("BT-1")!.Text);
        ExtractedField date = x.Get("BT-2")!;
        Assert.Equal("2026-10-13", date.Value);
        Assert.Equal(new DateOnly(2026, 10, 13), date.Date);
        ExtractedField total = x.Get("BT-112")!;
        Assert.Equal("EUR 1.234,50", total.Text);
        Assert.Equal("1234.50", total.Value);
        Assert.Equal(1234.50m, total.Number);
        Assert.Equal("EUR", total.Currency);
        Assert.Equal([6, 7], total.WordIndices);
        Assert.Equal(new OcrBox(250, 10, 325, 25), total.Box);
        Assert.Null(x.Get("receipt.total"));

        Assert.Equal(2, x.LineItems.Count);
        Assert.Equal(8, x.LineItems[0].Anchor);
        Assert.Equal("Widget", x.LineItems[0].Get("BT-153")!.Text);
        Assert.Equal("2", x.LineItems[0].Get("BT-129")!.Text);
        Assert.Equal("3", x.LineItems[1].Get("BT-129")!.Text);
        Assert.Single(x.TaxRows);
        Assert.Equal("7.50", x.TaxRows[0].Get("BT-116")!.Value);

        OcrKeyValue pair = Assert.Single(x.KeyValues);
        Assert.Equal([5], pair.KeyWordIndices);
        Assert.Equal([6, 7], pair.ValueWordIndices);

        // Asked for a receipt, the same labels read with the receipt schema: only the receipt class survives.
        DocumentExtraction asReceipt = DocAssembly.Assemble(m, o, texts, boxes, index, sources, "receipt", NumberStyle.CommaDecimal, DateOrder.Auto);
        Assert.Equal(DocumentType.Invoice, asReceipt.DocumentType);
        Assert.Equal("receipt.total", Assert.Single(asReceipt.Fields).Id);
        Assert.Single(asReceipt.Diagnostics);
    }

    [Fact]
    public void ItemPointersFollowChainsAndEndCyclesAtTheSmallestIndex()
    {
        int[] p = [0, 0, 1, 2, 3, 4, 7, 6, 9, 8];
        Assert.Equal(0, DocAssembly.Anchor(p, 0));
        Assert.Equal(0, DocAssembly.Anchor(p, 2));
        Assert.Equal(0, DocAssembly.Anchor(p, 4)); // four steps reach the root
        Assert.Equal(1, DocAssembly.Anchor(p, 5)); // five steps are too many: stops where the fourth step lands
        Assert.Equal(6, DocAssembly.Anchor(p, 7)); // two-cycle: smallest index on it
        Assert.Equal(8, DocAssembly.Anchor(p, 9));
    }

    [Fact]
    public void WordsAreNormalizedToNfcAndCutToTheCharacterLimitByCodePoint()
    {
        DocExtractModel m = DocumentExtractor.Load(TestDocModels.RandomContainer(TestDocModels.Small, 1)).Model;
        var composed = new int[8];
        DocFeatures.EncodeWord(m, "\u0101\u0102", composed);
        Assert.Equal([1, 2, 0, 0, 0, 0, 0, 0], composed);

        // The platform's normalization does the composing. With invariant globalization it leaves non-ASCII text alone (the repo's own
        // builds run that way), so this part only runs where the platform composes.
        if ("a\u0304".Normalize(NormalizationForm.FormC) == "\u0101")
        {
            var decomposed = new int[8];
            DocFeatures.EncodeWord(m, "a\u0304A\u0306", decomposed); // the charset holds the composed letters only
            Assert.Equal(composed, decomposed);
        }

        var ids = new int[8];
        DocFeatures.EncodeWord(m, "\u4e2d\ud83d\ude00x", ids); // not in the charset; one astral code point is one character
        Assert.Equal([m.UnkChar, m.UnkChar, m.UnkChar, 0, 0, 0, 0, 0], ids);
        DocFeatures.EncodeWord(m, new string('\u0101', 20), ids);
        Assert.All(ids, v => Assert.Equal(1, v));
        DocFeatures.EncodeWord(m, "\ud800\u0101", ids); // an unpaired surrogate reads as the replacement character
        Assert.Equal([m.UnkChar, 1], ids[..2]);
    }

    [Fact]
    public void ThePackageWithoutWeightsSaysSoPlainly()
    {
        Assert.Null(DocModels.Extractor);
        Assert.False(DocModels.HasExtractor);
        Assert.NotNull(DocModels.TryReadExtractor(out _));
        Assert.Throws<OcrModelException>(() => DocModels.LoadExtractor());
    }

    [Fact]
    public void TheDefaultExtractorMustBeConfiguredAndIsSetOnce()
    {
        // Process-wide state: only the unconfigured failure is checked here, so other tests keep their own explicit extractors.
        if (!ExtractorConfigured())
        {
            Assert.Throws<OcrModelException>(() => _ = OneOcr.DefaultExtractor);
        }
    }

    private static bool ExtractorConfigured()
    {
        try
        {
            _ = OneOcr.DefaultExtractor;
            return true;
        }
        catch (OcrModelException)
        {
            return false;
        }
    }
}
