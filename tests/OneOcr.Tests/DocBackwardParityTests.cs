using System.Globalization;
using System.Text;
using System.Text.Json;
using OneOcr.Documents;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Training;

namespace OneOcr.Tests;

/// <summary>
/// Backward passes of the document-extraction model against PyTorch (<c>ONEOCR_DOCEXTRACT_BACKWARD_DIR</c>, a directory written by
/// <c>training/parity/docextract_backward.py</c>; the tests skip without it). Every comparison allows
/// <c>2 * Atol + 2 * Floor + 2 * Rtol * |reference|</c>: <c>Atol</c> and <c>Rtol</c> are this library's per-layer tolerances against float64
/// (docs/spec/docextract-backward.md), doubled because the reference carries float32 error too, and <c>Floor</c> is PyTorch's own measured
/// float32-versus-float64 maximum absolute error for that tensor, recorded in the fixture.
/// </summary>
public class DocBackwardParityTests
{
    // Tolerances against float64 per layer type (atol, rtol); see the table in docs/spec/docextract-backward.md.
    private static readonly (double Atol, double Rtol) LayerNormTol = (2e-7, 1e-6);
    private static readonly (double Atol, double Rtol) GeluTol = (2e-6, 1e-5);
    private static readonly (double Atol, double Rtol) AttentionTol = (3e-7, 2e-6);
    private static readonly (double Atol, double Rtol) PairTol = (1e-6, 5e-6);
    private static readonly (double Atol, double Rtol) PointerTol = (3e-7, 2e-6);
    private static readonly (double Atol, double Rtol) LossTol = (2e-7, 1e-6);
    private static readonly (double Atol, double Rtol) ModelTol = (1e-6, 1e-5);

    private sealed class Fixture
    {
        public required string Dir { get; init; }
        public required JsonElement Root { get; init; }

        public bool Has(string key) => Root.GetProperty("tensors").TryGetProperty(key, out _);

        public float[] F(string key)
        {
            JsonElement e = Root.GetProperty("tensors").GetProperty(key);
            byte[] raw = File.ReadAllBytes(Path.Combine(Dir, e.GetProperty("file").GetString()!));
            var v = new float[raw.Length / 4];
            Buffer.BlockCopy(raw, 0, v, 0, raw.Length);
            return v;
        }

        public int[] I(string key)
        {
            JsonElement e = Root.GetProperty("tensors").GetProperty(key);
            byte[] raw = File.ReadAllBytes(Path.Combine(Dir, e.GetProperty("file").GetString()!));
            var v = new int[raw.Length / 4];
            Buffer.BlockCopy(raw, 0, v, 0, raw.Length);
            return v;
        }

        public double Floor(string key) => Root.GetProperty("tensors").GetProperty(key).TryGetProperty("max_abs", out JsonElement m) ? m.GetDouble() : 0;
    }

    private static Fixture? Open()
    {
        string? dir = Environment.GetEnvironmentVariable("ONEOCR_DOCEXTRACT_BACKWARD_DIR");
        if (string.IsNullOrEmpty(dir))
        {
            return null;
        }

        return new Fixture { Dir = dir, Root = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "fixtures.json"))).RootElement };
    }

    private static Fixture Require()
    {
        Fixture? f = Open();
        if (f is null)
        {
            Assert.Skip("ONEOCR_DOCEXTRACT_BACKWARD_DIR is not set.");
        }

        return f;
    }

    /// <summary>Worst use of the allowance over all elements; asserts it is at most 1 and notes it.</summary>
    private static double Compare(string what, ReadOnlySpan<float> ours, ReadOnlySpan<float> reference, double floor, (double Atol, double Rtol) tol, StringBuilder report)
    {
        Assert.Equal(reference.Length, ours.Length);
        double worst = 0, maxErr = 0;
        int at = -1;
        for (int i = 0; i < ours.Length; i++)
        {
            double err = Math.Abs((double)ours[i] - reference[i]);
            double allowed = (2 * tol.Atol) + (2 * floor) + (2 * tol.Rtol * Math.Abs(reference[i]));
            if (!(err <= allowed))
            {
                worst = double.PositiveInfinity;
                at = i;
                break;
            }

            maxErr = Math.Max(maxErr, err);
            if (err / allowed > worst)
            {
                worst = err / allowed;
                at = i;
            }
        }

        report.AppendLine(CultureInfo.InvariantCulture, $"{what}: max error {maxErr:E2}, floor {floor:E2}, worst use of the allowance {worst:F3}");
        Assert.True(worst <= 1, $"{what}: element {at} is {(at >= 0 ? ours[at] : 0):E6}, reference {(at >= 0 ? reference[at] : 0):E6}, allowed {2 * tol.Atol:E1} + {2 * floor:E1} + {2 * tol.Rtol:E1}|ref|");
        return worst;
    }

    private static void Note(StringBuilder report, string title)
    {
        string text = report.ToString();
        TestNotes.Add($"## {title}\n{text}");
    }

    [Fact]
    public void WholePageGradientsMatchPyTorchForEveryParameter()
    {
        Fixture fx = Require();
        JsonElement page = fx.Root.GetProperty("page");
        DocExtractModel model = DocExtractModel.FromContainer(ModelContainer.Load(Path.Combine(fx.Dir, "model.onm")), OcrLimits.Default);
        JsonElement[] words = [.. page.GetProperty("words").EnumerateArray()];
        string[] texts = [.. words.Select(w => w.GetProperty("text").GetString()!)];
        OcrBox[] boxes = [.. words.Select(w =>
        {
            double[] b = [.. w.GetProperty("box").EnumerateArray().Select(e => e.GetDouble())];
            return new OcrBox(b[0], b[1], b[2], b[3]);
        })];
        DocFeatures f = DocFeatures.Build(model, texts, boxes, page.GetProperty("width").GetDouble(), page.GetProperty("height").GetDouble());
        int i0 = 0;
        foreach (JsonElement row in page.GetProperty("chars").EnumerateArray())
        {
            foreach (JsonElement c in row.EnumerateArray())
            {
                Assert.Equal(c.GetInt32(), f.Chars[i0++]);
            }
        }

        JsonElement lw = page.GetProperty("loss_weights");
        var settings = new DocTrainSettings
        {
            BackboneScale = 1.0,
            Weights = new DocLossWeights
            {
                Tag = (float)lw.GetProperty("tag").GetDouble(), Role = (float)lw.GetProperty("role").GetDouble(), Doc = (float)lw.GetProperty("doc").GetDouble(),
                Item = (float)lw.GetProperty("item").GetDouble(), KeyValue = (float)lw.GetProperty("kv").GetDouble(),
            },
            TagClassWeights = fx.F("model.tag_class_weights"),
        };
        var targets = new DocTargets(fx.I("model.target_tags"), fx.I("model.target_roles"), page.GetProperty("doc_target").GetInt32(), fx.I("model.target_item"), fx.I("model.target_kv"));
        var report = new StringBuilder();
        foreach (KernelPath path in Dispatch.Available())
        {
            var trainer = new DocTrainer(model, path, settings);
            double loss = trainer.ComputeGradients([new DocExample(f, targets)]);
            double expected = page.GetProperty("total_f64").GetDouble();
            Assert.True(Math.Abs(loss - expected) <= 2e-5 * Math.Abs(expected), $"{path}: loss {loss:R}, float64 reference {expected:R}");

            (float[] tags, float[] roles, float[] doc, float[] rel) = trainer.ForwardForTests(f);
            Compare($"{path} tags", tags, fx.F("model.out.tags"), fx.Floor("model.out.tags"), ModelTol, report);
            Compare($"{path} rel", rel, fx.F("model.out.rel"), fx.Floor("model.out.rel"), ModelTol, report);

            int checkedTensors = 0;
            for (int p = 0; p < trainer.Names.Count; p++)
            {
                string name = trainer.Names[p];
                string key = "model.grad." + name;
                float[] ours = trainer.Gradients[p];
                float[] expectedGrad = fx.F(key);
                ours = ToContainerLayout(model, name, ours);
                Compare($"{path} d {name}", ours, expectedGrad, fx.Floor(key), ModelTol, report);
                checkedTensors++;
            }

            Assert.Equal(DocParams.Count(model), checkedTensors);
        }

        Note(report, "docextract backward, whole page");
    }

    /// <summary>The runtime's weight layouts back to the container's: convolution taps (out, channel, k), unpadded feature and pair inputs.</summary>
    private static float[] ToContainerLayout(DocExtractModel m, string name, float[] a)
    {
        switch (name)
        {
            case "char_conv.weight":
            {
                var r = new float[a.Length];
                int cd = m.CharDim;
                for (int o = 0; o < m.CharConv; o++)
                {
                    for (int ch = 0; ch < cd; ch++)
                    {
                        for (int k = 0; k < 3; k++)
                        {
                            r[(((o * cd) + ch) * 3) + k] = a[(o * 3 * cd) + (k * cd) + ch];
                        }
                    }
                }

                return r;
            }

            case "feat_proj.weight":
                return Unpad(a, m.D, 8);
            case "pair1.weight":
                return Unpad(a, m.PairHidden, DocKernels.PairFeatures);
            default:
                return a;
        }

        static float[] Unpad(float[] a, int rows, int cols)
        {
            var r = new float[rows * cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    if (j < cols)
                    {
                        r[(i * cols) + j] = a[(i * 16) + j];
                    }
                    else
                    {
                        Assert.Equal(0f, a[(i * 16) + j]);
                    }
                }
            }

            return r;
        }
    }

    [Fact]
    public void LayerNormBackwardMatchesPyTorch()
    {
        Fixture fx = Require();
        var report = new StringBuilder();
        foreach (int d in new[] { 192, 50 })
        {
            string g = $"layers.layer_norm{d}";
            float[] x = fx.F(g + ".in.x"), w = fx.F(g + ".in.w"), b = fx.F(g + ".in.b"), gy = fx.F(g + ".gy");
            int rows = x.Length / d;
            foreach (KernelPath path in Dispatch.Available())
            {
                var y = new float[x.Length];
                var xhat = new float[x.Length];
                var inv = new float[rows];
                DocBackward.LayerNormForward(path, x, y, rows, d, w, b, xhat, inv);
                var reference = new float[x.Length];
                DocKernels.LayerNorm(path, x, reference, rows, d, w, b);
                Assert.Equal(reference.Select(BitConverter.SingleToInt32Bits), y.Select(BitConverter.SingleToInt32Bits));
                var dx = new float[x.Length];
                var dw = new float[d];
                var db = new float[d];
                DocBackward.LayerNormBackward(path, gy, xhat, inv, w, rows, d, dx, dw, db);
                Compare($"{path} layer norm {d} dx", dx, fx.F(g + ".d.x"), fx.Floor(g + ".d.x"), LayerNormTol, report);
                Compare($"{path} layer norm {d} dw", dw, fx.F(g + ".d.w"), fx.Floor(g + ".d.w"), LayerNormTol, report);
                Compare($"{path} layer norm {d} db", db, fx.F(g + ".d.b"), fx.Floor(g + ".d.b"), LayerNormTol, report);
            }
        }

        Note(report, "docextract backward, layer norm");
    }

    [Fact]
    public void GeluBackwardMatchesPyTorch()
    {
        Fixture fx = Require();
        var report = new StringBuilder();
        float[] x = fx.F("layers.gelu.in.x"), gy = fx.F("layers.gelu.gy");
        foreach (KernelPath path in Dispatch.Available())
        {
            var y = new float[x.Length];
            var th = new float[x.Length];
            DocBackward.GeluForward(path, x, y, th);
            var reference = (float[])x.Clone();
            DocKernels.Gelu(path, reference);
            Assert.Equal(reference.Select(BitConverter.SingleToInt32Bits), y.Select(BitConverter.SingleToInt32Bits));
            var dx = new float[x.Length];
            DocBackward.GeluBackward(gy, x, th, dx);
            Compare($"{path} gelu dx", dx, fx.F("layers.gelu.d.x"), fx.Floor("layers.gelu.d.x"), GeluTol, report);
        }

        Note(report, "docextract backward, gelu");
    }

    [Fact]
    public void AttentionBackwardMatchesPyTorch()
    {
        Fixture fx = Require();
        var report = new StringBuilder();
        foreach (int n in new[] { 37, 64 })
        {
            string g = $"layers.attention{n}";
            float[] q = fx.F(g + ".in.q"), k = fx.F(g + ".in.k"), v = fx.F(g + ".in.v"), bias = fx.F(g + ".in.bias"), gy = fx.F(g + ".gy");
            int dh = q.Length / n;
            float scale = MathF.Sqrt(dh);
            foreach (KernelPath path in Dispatch.Available())
            {
                var p = new float[n * n];
                Gemm.NT(path, q, dh, k, dh, n, n, dh, default, p, n, 1);
                DocKernels.SoftmaxRowsWithBias(path, p, bias, n, n, scale);
                var dP = new float[n * n];
                var dV = new float[n * dh];
                DocBackward.ContextBackward(path, p, gy, dh, v, dh, n, dh, dP, dV, dh);
                var dRaw = new float[n * n];
                var dBias = new float[n * n];
                DocBackward.SoftmaxRowsBackward(path, p, dP, n, n, scale, dRaw, dBias);
                var dQ = new float[n * dh];
                var dK = new float[n * dh];
                DocBackward.ScoresBackward(path, dRaw, q, dh, k, dh, n, dh, dQ, dh, dK, dh);
                Compare($"{path} attention {n} dq", dQ, fx.F(g + ".d.q"), fx.Floor(g + ".d.q"), AttentionTol, report);
                Compare($"{path} attention {n} dk", dK, fx.F(g + ".d.k"), fx.Floor(g + ".d.k"), AttentionTol, report);
                Compare($"{path} attention {n} dv", dV, fx.F(g + ".d.v"), fx.Floor(g + ".d.v"), AttentionTol, report);
                Compare($"{path} attention {n} dbias", dBias, fx.F(g + ".d.bias"), fx.Floor(g + ".d.bias"), AttentionTol, report);
            }
        }

        Note(report, "docextract backward, attention");
    }

    [Fact]
    public void PairMlpBackwardMatchesPyTorch()
    {
        Fixture fx = Require();
        var report = new StringBuilder();
        string g = "layers.pair_mlp";
        float[] geo = fx.F(g + ".in.geo"), w1 = fx.F(g + ".in.w1"), b1 = fx.F(g + ".in.b1"), w2 = fx.F(g + ".in.w2"), b2 = fx.F(g + ".in.b2"), gy = fx.F(g + ".gy");
        int n = geo.Length / 4, hid = b1.Length, channels = b2.Length, pairs = n * n;
        var w1p = new float[hid * 16];
        for (int i = 0; i < hid; i++)
        {
            Array.Copy(w1, i * 13, w1p, i * 16, 13);
        }

        var w2t = new float[hid * channels];
        Backward.Transpose(w2, hid, channels, hid, w2t);
        foreach (KernelPath path in Dispatch.Available())
        {
            var feat = new float[pairs * 16];
            DocKernels.PairFeaturesRows(geo, n, 0, n, feat);
            var h = new float[pairs * hid];
            DocKernels.Linear(path, feat, 16, pairs, w1p, b1, 16, hid, h, hid);
            DocKernels.Relu(h);
            var pb = new float[channels * pairs];
            Gemm.NT(path, h, hid, w2, hid, pairs, channels, hid, b2, pb, 1, pairs);
            Compare($"{path} pair mlp output", pb, fx.F(g + ".y"), fx.Floor(g + ".y"), PairTol, report);
            var dW1 = new float[hid * 16];
            var dB1 = new float[hid];
            var dW2 = new float[channels * hid];
            var dB2 = new float[channels];
            DocBackward.PairMlpBackward(path, gy, h, feat, w2t, pairs, hid, channels, dW1, dB1, dW2, dB2);
            var dW1Unpadded = new float[hid * 13];
            for (int i = 0; i < hid; i++)
            {
                Array.Copy(dW1, i * 16, dW1Unpadded, i * 13, 13);
                for (int j = 13; j < 16; j++)
                {
                    Assert.Equal(0f, dW1[(i * 16) + j]);
                }
            }

            Compare($"{path} pair mlp dw1", dW1Unpadded, fx.F(g + ".d.w1"), fx.Floor(g + ".d.w1"), PairTol, report);
            Compare($"{path} pair mlp db1", dB1, fx.F(g + ".d.b1"), fx.Floor(g + ".d.b1"), PairTol, report);
            Compare($"{path} pair mlp dw2", dW2, fx.F(g + ".d.w2"), fx.Floor(g + ".d.w2"), PairTol, report);
            Compare($"{path} pair mlp db2", dB2, fx.F(g + ".d.b2"), fx.Floor(g + ".d.b2"), PairTol, report);
        }

        Note(report, "docextract backward, pair MLP");
    }

    [Fact]
    public void PointerScoreBackwardMatchesPyTorch()
    {
        Fixture fx = Require();
        var report = new StringBuilder();
        string g = "layers.pointer";
        float[] q = fx.F(g + ".in.q"), k = fx.F(g + ".in.k"), gy = fx.F(g + ".gy");
        int n = (int)Math.Round(Math.Sqrt(gy.Length));
        int dim = q.Length / n;
        float scale = MathF.Sqrt(dim);
        foreach (KernelPath path in Dispatch.Available())
        {
            var dRaw = new float[n * n];
            for (int i = 0; i < dRaw.Length; i++)
            {
                dRaw[i] = gy[i] / scale;
            }

            var dQ = new float[n * dim];
            var dK = new float[n * dim];
            DocBackward.ScoresBackward(path, dRaw, q, dim, k, dim, n, dim, dQ, dim, dK, dim);
            Compare($"{path} pointer dq", dQ, fx.F(g + ".d.q"), fx.Floor(g + ".d.q"), PointerTol, report);
            Compare($"{path} pointer dk", dK, fx.F(g + ".d.k"), fx.Floor(g + ".d.k"), PointerTol, report);
            Compare($"{path} pointer dbias", gy, fx.F(g + ".d.bias"), fx.Floor(g + ".d.bias"), PointerTol, report);
        }

        Note(report, "docextract backward, pointer scores");
    }

    [Fact]
    public void CrossEntropyMatchesPyTorchWithIgnoredRowsAndClassWeights()
    {
        Fixture fx = Require();
        var report = new StringBuilder();
        foreach (string tag in new[] { "plain", "weighted" })
        {
            string g = $"layers.ce_{tag}";
            float[] z = fx.F(g + ".in.z");
            int[] target = fx.I(g + ".in.target");
            float[] weight = fx.Has(g + ".in.weight") ? fx.F(g + ".in.weight") : [];
            int rows = target.Length, classes = z.Length / rows;
            var d = new float[z.Length];
            double loss = DocLoss.CrossEntropy(z, classes, rows, classes, target, weight, 1.0, d);
            Compare($"{tag} loss", [(float)loss], fx.F(g + ".loss"), fx.Floor(g + ".loss"), LossTol, report);
            Compare($"{tag} dz", d, fx.F(g + ".d.z"), fx.Floor(g + ".d.z"), LossTol, report);
        }

        Note(report, "docextract backward, cross entropy");
    }
}
