using System.Text.Json;
using OneImage;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;
using OneOcr.Training;

namespace OneOcr.Tests;

/// <summary>
/// Compares the backward passes, the CTC loss and AdamW with PyTorch references (a directory named by ONEOCR_BACKWARD_PARITY_DIR; the
/// file format is described in training/parity/README.md). Every comparison allows
/// <c>2 * Atol + 2 * Floor + 2 * Rtol * |reference|</c>: <c>Atol</c> and <c>Rtol</c> are this library's documented tolerances against
/// float64 (docs/spec/backward-numerics.md), doubled because the reference carries float32 error too, and <c>Floor</c> is PyTorch's
/// own measured float32-versus-float64 maximum absolute error for that tensor, recorded in the fixture.
/// </summary>
public class BackwardParityTests
{
    private const double ConvAtol = 7e-6, ConvRtol = 3.5e-5;
    private const double LstmAtol = 1e-6, LstmRtol = 5e-6;
    private const double FcAtol = 1e-6, FcRtol = 5e-6;
    private const double CtcAtol = 1e-6, CtcRtol = 1e-6;

    private static string? Dir => Environment.GetEnvironmentVariable("ONEOCR_BACKWARD_PARITY_DIR");

    private sealed class Fixture
    {
        public required string Dir { get; init; }
        public required JsonElement Root { get; init; }
        public required CrnnModel Model { get; init; }

        public float[] Read(JsonElement tensor)
        {
            var all = new List<float>();
            foreach (JsonElement f in tensor.GetProperty("files").EnumerateArray())
            {
                byte[] raw = File.ReadAllBytes(Path.Combine(Dir, f.GetProperty("file").GetString()!));
                var part = new float[raw.Length / 4];
                Buffer.BlockCopy(raw, 0, part, 0, raw.Length);
                all.AddRange(part);
            }

            return [.. all];
        }

        public double Floor(string group, string layer, string tensor)
        {
            JsonElement floors = Root.GetProperty("fp32_vs_fp64_floor");
            return floors.GetProperty(group).TryGetProperty(layer, out JsonElement l) && l.TryGetProperty(tensor, out JsonElement t)
                ? t.GetProperty("max_abs").GetDouble() : 0;
        }
    }

    private static Fixture? Open()
    {
        string? dir = Dir;
        if (string.IsNullOrEmpty(dir))
        {
            return null;
        }

        var doc = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "fixtures.json")));
        Assert.Equal("oneocr-backward-parity-1", doc.RootElement.GetProperty("format").GetString());
        CrnnModel model = CrnnModel.FromContainer(ModelContainer.Load(Path.Combine(dir, "model.onm")), OcrLimits.Default);
        return new Fixture { Dir = dir, Root = doc.RootElement, Model = model };
    }

    private static float[] ChwToHwc(float[] v, int c, int h, int w)
    {
        var r = new float[v.Length];
        for (int ch = 0; ch < c; ch++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    r[(((y * w) + x) * c) + ch] = v[(((ch * h) + y) * w) + x];
                }
            }
        }

        return r;
    }

    private static float[] HwcToChw(float[] v, int c, int h, int w)
    {
        var r = new float[c * h * w];
        for (int ch = 0; ch < c; ch++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    r[(((ch * h) + y) * w) + x] = v[(((y * w) + x) * c) + ch];
                }
            }
        }

        return r;
    }

    // [Out][KPadded] with taps in (ky, kx, in) order  <->  [Out][In][kh][kw].
    private static float[] PaddedToTorch(ConvBlock b, float[] padded)
    {
        var r = new float[b.Out * b.In * b.KernelH * b.KernelW];
        for (int o = 0; o < b.Out; o++)
        {
            for (int ic = 0; ic < b.In; ic++)
            {
                for (int ky = 0; ky < b.KernelH; ky++)
                {
                    for (int kx = 0; kx < b.KernelW; kx++)
                    {
                        r[((((o * b.In) + ic) * b.KernelH) + ky) * b.KernelW + kx] = padded[(o * b.KPadded) + (((ky * b.KernelW) + kx) * b.In) + ic];
                    }
                }
            }
        }

        return r;
    }

    private static float[] TorchToPadded(ConvBlock b, float[] torch)
    {
        var r = new float[b.Out * b.KPadded];
        for (int o = 0; o < b.Out; o++)
        {
            for (int ic = 0; ic < b.In; ic++)
            {
                for (int ky = 0; ky < b.KernelH; ky++)
                {
                    for (int kx = 0; kx < b.KernelW; kx++)
                    {
                        r[(o * b.KPadded) + (((ky * b.KernelW) + kx) * b.In) + ic] = torch[((((o * b.In) + ic) * b.KernelH) + ky) * b.KernelW + kx];
                    }
                }
            }
        }

        return r;
    }

    // Returns the worst error as a multiple of the allowance, and writes a line to ONEOCR_TEST_LOG when set.
    private static void Compare(string what, ReadOnlySpan<float> got, float[] expected, double atol, double rtol, double floor, bool soft = false)
    {
        Assert.Equal(expected.Length, got.Length);
        double worst = 0, maxAbs = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(got[i] - (double)expected[i]);
            maxAbs = Math.Max(maxAbs, diff);
            double allowance = (2 * atol) + (2 * floor) + (2 * rtol * Math.Abs(expected[i]));
            worst = Math.Max(worst, allowance == 0 ? (diff == 0 ? 0 : double.PositiveInfinity) : diff / allowance);
        }

        string? log = Environment.GetEnvironmentVariable("ONEOCR_TEST_LOG");
        if (log is not null)
        {
            File.AppendAllText(log, $"PYTORCH {what}\tmaxAbs={maxAbs:E2}\tfloor={floor:E2}\tx allowance={worst:F3}\n");
        }

        if (soft)
        {
            return;
        }

        Assert.True(worst <= 1, $"{what}: error is {worst:F2}x the allowance (atol {atol}, rtol {rtol}, floor {floor:E2}); max abs difference {maxAbs:E2}");
    }

    [Fact]
    public void LayerGradientsMatchPyTorch()
    {
        Fixture? fx = Open();
        if (fx is null)
        {
            return;
        }

        CrnnModel model = fx.Model;
        int hidden = model.Hidden, gates = 4 * hidden;
        var pool = new BufferPool<float>(1 << 28, 4);
        var ipool = new BufferPool<int>(1 << 28, 4);
        foreach (KernelPath path in Dispatch.Available())
        {
            foreach (JsonElement line in fx.Root.GetProperty("layer_lines").EnumerateArray())
            {
                foreach (JsonElement layer in line.GetProperty("layers").EnumerateArray())
                {
                    string name = layer.GetProperty("name").GetString()!;
                    string kind = layer.GetProperty("kind").GetString()!;
                    string tag = $"{path} {line.GetProperty("id").GetString()} {name}";
                    using var scratch = new TrainScratch(pool, ipool);
                    float[] x = fx.Read(layer.GetProperty("input"));
                    float[] gy = fx.Read(layer.GetProperty("grad_output"));
                    int[] xs = layer.GetProperty("input").GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                    JsonElement expected = layer.GetProperty("expected");
                    float[] Expected(string key) => fx.Read(expected.GetProperty(key));
                    double Floor(string key) => fx.Floor("layers", name, key);

                    switch (kind)
                    {
                        case "conv":
                        {
                            int bi = int.Parse(name.Split('.')[1]);
                            ConvBlock b = model.Conv[bi];
                            int c = xs[0], h = xs[1], w = xs[2];
                            Assert.Equal(b.In, c);
                            float[] xh = ChwToHwc(x, c, h, w);
                            float[] col = new float[h * w * b.KPadded];
                            Backward.Im2Col(xh, h, w, c, b.KernelH, b.KernelW, b.PadH, b.PadW, b.KPadded, col);
                            var cache = new ConvCache { H = h, W = w, ConvH = h, ConvW = w, OutH = h, OutW = w, Col = col, Argmax = [] };
                            var wt = new float[b.KPadded * b.Out];
                            Backward.Transpose(b.Weights, b.KPadded, b.Out, b.KPadded, wt);
                            var dW = new float[b.Out * b.KPadded];
                            var dB = new float[b.Out];
                            var dIn = new float[h * w * c];
                            ConvTrain.BackwardConv(b, path, scratch, cache, wt, ChwToHwc(gy, b.Out, h, w), dW, dB, dIn);
                            Compare(tag + " dx", HwcToChw(dIn, c, h, w), Expected("x"), ConvAtol, ConvRtol, Floor("x"));
                            Compare(tag + " dW", PaddedToTorch(b, dW), Expected($"conv.{bi}.weight"), ConvAtol, ConvRtol, Floor($"conv.{bi}.weight"));
                            Compare(tag + " dBias", dB, Expected($"conv.{bi}.bias"), ConvAtol, ConvRtol, Floor($"conv.{bi}.bias"));
                            break;
                        }

                        case "relu":
                        case "maxpool":
                        {
                            int c = xs[0], h = xs[1], w = xs[2];
                            int ph = 1, pw = 1;
                            if (kind == "maxpool")
                            {
                                int[] k = layer.GetProperty("attrs").GetProperty("kernel").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                                ph = k[0];
                                pw = k[1];
                            }

                            int oh = h / ph, ow = w / pw;
                            float[] xh = ChwToHwc(x, c, h, w);
                            var pooled = new float[oh * ow * c];
                            var arg = new int[oh * ow * c];
                            Backward.ReluPool(xh, w, c, ph, pw, oh, ow, pooled, arg);
                            var dConv = new float[h * w * c];
                            Backward.ReluPoolBackward(ChwToHwc(gy, c, oh, ow), arg, oh * ow * c, dConv);
                            float[] want = Expected("x");
                            if (kind == "maxpool")
                            {
                                // Windows whose maximum is 0 send their gradient to a position the preceding ReLU masks anyway.
                                for (int i = 0; i < want.Length; i++)
                                {
                                    if (!(x[i] > 0))
                                    {
                                        want[i] = 0;
                                    }
                                }
                            }

                            Compare(tag + " dx", HwcToChw(dConv, c, h, w), want, 0, 0, 0);
                            break;
                        }

                        case "lstm":
                        {
                            int l = int.Parse(name[^1..]);
                            LstmLayer lay = model.Lstm[l];
                            int t = xs[0];
                            float[] y = LstmTrain.Forward(lay, hidden, path, scratch, x, t, out LstmCache cache);
                            Assert.True(y.Length >= t * 2 * hidden);
                            var whhT = new float[2][];
                            var wihCatT = new float[lay.In * 2 * gates];
                            for (int d = 0; d < 2; d++)
                            {
                                LstmDirection dir = d == 0 ? lay.Forward : lay.Reverse;
                                whhT[d] = new float[hidden * gates];
                                Backward.Transpose(dir.WeightHh, hidden, gates, hidden, whhT[d]);
                                var tmp = new float[lay.In * gates];
                                Backward.Transpose(dir.WeightIh, lay.In, gates, lay.In, tmp);
                                for (int i = 0; i < lay.In; i++)
                                {
                                    tmp.AsSpan(i * gates, gates).CopyTo(wihCatT.AsSpan((i * 2 * gates) + (d * gates), gates));
                                }
                            }

                            float[][] dWih = [new float[gates * lay.In], new float[gates * lay.In]];
                            float[][] dWhh = [new float[gates * hidden], new float[gates * hidden]];
                            float[][] dB = [new float[gates], new float[gates]];
                            var dX = new float[t * lay.In];
                            LstmTrain.BackwardPass(lay, hidden, path, scratch, cache, gy, whhT, wihCatT, dWih, dWhh, dB, dX);
                            Compare(tag + " dx", dX, Expected("x"), LstmAtol, LstmRtol, Floor("x"));
                            for (int d = 0; d < 2; d++)
                            {
                                string sfx = d == 0 ? string.Empty : "_reverse";
                                foreach (var (key, got) in new[] { ($"lstm.weight_ih_l{l}{sfx}", dWih[d]), ($"lstm.weight_hh_l{l}{sfx}", dWhh[d]), ($"lstm.bias_l{l}{sfx}", dB[d]) })
                                {
                                    Compare($"{tag} d[{key}]", got, Expected(key), LstmAtol, LstmRtol, Floor(key));
                                }
                            }

                            break;
                        }

                        case "linear":
                        {
                            int t = xs[0], inSize = xs[1], outSize = model.Classes;
                            var wt = new float[inSize * outSize];
                            Backward.Transpose(model.FcWeight, inSize, outSize, inSize, wt);
                            var dW = new float[outSize * inSize];
                            var dB = new float[outSize];
                            var dX = new float[t * inSize];
                            LinearTrain.BackwardPass(path, scratch, x, gy, wt, t, inSize, outSize, dW, dB, dX);
                            Compare(tag + " dx", dX, Expected("x"), FcAtol, FcRtol, Floor("x"));
                            Compare(tag + " dW", dW, Expected("fc.weight"), FcAtol, FcRtol, Floor("fc.weight"));
                            Compare(tag + " dBias", dB, Expected("fc.bias"), FcAtol, FcRtol, Floor("fc.bias"));
                            break;
                        }

                        default:
                            Assert.Fail("unknown layer kind " + kind);
                            break;
                    }
                }
            }
        }
    }

    [Theory]
    [InlineData("truth")]
    [InlineData("shifted")]
    public void CtcLossAndLogitGradientMatchPyTorch(string batchName)
    {
        Fixture? fx = Open();
        if (fx is null)
        {
            return;
        }

        JsonElement batch = fx.Root.GetProperty("model_level").GetProperty("batches").GetProperty(batchName).GetProperty("batch");
        int b = batch.GetArrayLength();
        JsonElement floors = fx.Root.GetProperty("fp32_vs_fp64_floor").GetProperty("model").GetProperty(batchName);
        double dFloor = floors.GetProperty("dlogits").GetProperty("max_abs").GetDouble();
        double nllFloor = floors.GetProperty("nll").GetProperty("max_abs").GetDouble();
        foreach (JsonElement s in batch.EnumerateArray())
        {
            int steps = s.GetProperty("steps").GetInt32();
            int[] target = s.GetProperty("target").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            float[] logits = fx.Read(s.GetProperty("logits"));
            var grad = new float[steps * fx.Model.Classes];
            double nll = CtcLoss.Compute(logits, steps, fx.Model.Classes, target, 1.0 / b, grad);
            string id = s.GetProperty("id").GetString()!;
            Compare($"ctc {batchName} {id} nll", [(float)nll], [(float)s.GetProperty("nll").GetDouble()], CtcAtol, CtcRtol, nllFloor);
            Compare($"ctc {batchName} {id} dlogits", grad, fx.Read(s.GetProperty("dlogits")), CtcAtol, CtcRtol, dFloor);
        }
    }

    [Fact]
    public void AdamWMatchesPyTorchStepsOnStoredGradients()
    {
        Fixture? fx = Open();
        if (fx is null)
        {
            return;
        }

        JsonElement opt = fx.Root.GetProperty("optimizer");
        var options = new AdamWOptions
        {
            LearningRate = opt.GetProperty("lr").GetDouble(),
            Beta1 = opt.GetProperty("betas")[0].GetDouble(),
            Beta2 = opt.GetProperty("betas")[1].GetDouble(),
            Epsilon = opt.GetProperty("eps").GetDouble(),
            WeightDecay = opt.GetProperty("weight_decay").GetDouble(),
        };
        var trainer = new CrnnTrainer(fx.Model, Dispatch.Best, options, decayBiases: opt.GetProperty("decay_applies_to_biases").GetBoolean());
        JsonElement sets = opt.GetProperty("sets");
        var names = trainer.ParameterNames;
        var adam = new AdamW([.. trainer.Parameters], [.. names.Select(_ => true)], options);
        float[][] Load(string set) => names.Select((n, i) =>
        {
            float[] v = fx.Read(sets.GetProperty(set).GetProperty(n));
            if (n.StartsWith("conv.", StringComparison.Ordinal) && n.EndsWith(".weight", StringComparison.Ordinal))
            {
                v = TorchToPadded(fx.Model.Conv[int.Parse(n.Split('.')[1])], v);
            }

            return v;
        }).ToArray();

        double replayFloor = opt.GetProperty("fp32_vs_fp64_replay_floor").EnumerateObject()
            .Select(p => p.Value.ValueKind == JsonValueKind.Object && p.Value.TryGetProperty("max_abs", out JsonElement m) ? m.GetDouble() : 0).DefaultIfEmpty(0).Max();
        for (int step = 1; step <= 3; step++)
        {
            adam.Step(Load($"grads_step{step}"), options.LearningRate);
            if (step is 1 or 3)
            {
                float[][] want = Load($"weights_after_step{step}");
                for (int i = 0; i < names.Count; i++)
                {
                    // Early steps move every weight by about lr * sign(g): where |g| is far below epsilon the move is sensitive to
                    // gradient noise, so allow a fraction of one step per step taken.
                    Compare($"adamw step {step} {names[i]}", trainer.Parameters[i], want[i], 1e-6 + (0.05 * options.LearningRate * step), 1e-6, replayFloor);
                }
            }
        }
    }

    [Fact]
    public void WholeModelGradientsAndThreeStepsMatchPyTorch()
    {
        Fixture? fx = Open();
        if (fx is null)
        {
            return;
        }

        JsonElement opt = fx.Root.GetProperty("optimizer");
        var options = new AdamWOptions
        {
            LearningRate = opt.GetProperty("lr").GetDouble(),
            Beta1 = opt.GetProperty("betas")[0].GetDouble(),
            Beta2 = opt.GetProperty("betas")[1].GetDouble(),
            Epsilon = opt.GetProperty("eps").GetDouble(),
            WeightDecay = opt.GetProperty("weight_decay").GetDouble(),
        };
        var batch = new List<TrainLine>();
        foreach (JsonElement s in fx.Root.GetProperty("model_level").GetProperty("batches").GetProperty("shifted").GetProperty("batch").EnumerateArray())
        {
            Image image = Image.Load(Path.Combine(fx.Dir, s.GetProperty("png").GetString()!));
            var px = new Gray8[image.Width * image.Height];
            image.CopyPixelsTo<Gray8>(px, new ImageRect(0, 0, image.Width, image.Height));
            batch.Add(new TrainLine(px.Select(p => p.V).ToArray(), image.Width, s.GetProperty("target").EnumerateArray().Select(e => e.GetInt32()).ToArray()));
        }

        JsonElement sets = opt.GetProperty("sets");
        JsonElement paramFloors = fx.Root.GetProperty("fp32_vs_fp64_floor").GetProperty("model").GetProperty("param_grads_step1");
        var trainer = new CrnnTrainer(fx.Model, Dispatch.Best, options, decayBiases: true);
        var names = trainer.ParameterNames;
        double loss1 = trainer.ComputeGradients(batch, degree: 2);
        for (int i = 0; i < names.Count; i++)
        {
            string n = names[i];
            float[] got = trainer.Gradients[i];
            float[] want = fx.Read(sets.GetProperty("grads_step1").GetProperty(n));
            double floor = paramFloors.GetProperty(n).GetProperty("max_abs").GetDouble();
            bool conv = n.StartsWith("conv.", StringComparison.Ordinal);
            if (conv && n.EndsWith(".weight", StringComparison.Ordinal))
            {
                got = PaddedToTorch(fx.Model.Conv[int.Parse(n.Split('.')[1])], got);
            }

            (double atol, double rtol) = conv ? (ConvAtol, ConvRtol) : n.StartsWith("lstm.", StringComparison.Ordinal) ? (LstmAtol, LstmRtol) : (FcAtol, FcRtol);
            if (n == "conv.0.weight")
            {
                // A ReLU or max-pool decision that lies within float rounding of a tie can go either way in two implementations and
                // moves one position's gradient to another; it shows up as a few taps of one output channel. Allow one such flip.
                ConvBlock c0 = fx.Model.Conv[0];
                var bad = Enumerable.Range(0, want.Length)
                    .Where(q => Math.Abs(got[q] - want[q]) > (2 * atol) + (2 * floor) + (2 * rtol * Math.Abs(want[q]))).ToArray();
                Assert.True(bad.Length <= c0.In * c0.KernelH * c0.KernelW && bad.Select(q => q / (c0.In * c0.KernelH * c0.KernelW)).Distinct().Count() <= 1,
                    $"{bad.Length} elements of conv.0.weight differ in more than one output channel");
                Compare($"model grad step 1 {n} (outside the flipped channel)", got.Where((_, q) => !bad.Contains(q)).ToArray(), want.Where((_, q) => !bad.Contains(q)).ToArray(), atol, rtol, floor);
            }
            else
            {
                Compare($"model grad step 1 {n}", got, want, atol, rtol, floor);
            }
        }

        // The trajectory: three real steps, gradients recomputed at each step's weights.
        for (int step = 1; step <= 3; step++)
        {
            trainer.Step(batch, degree: 2);
            if (step is 1 or 3)
            {
                for (int i = 0; i < names.Count; i++)
                {
                    string n = names[i];
                    float[] want = fx.Read(sets.GetProperty($"weights_after_step{step}").GetProperty(n));
                    float[] got = trainer.Parameters[i];
                    bool convWeight = n.StartsWith("conv.", StringComparison.Ordinal) && n.EndsWith(".weight", StringComparison.Ordinal);
                    if (convWeight)
                    {
                        got = PaddedToTorch(fx.Model.Conv[int.Parse(n.Split('.')[1])], got);
                    }

                    // Adam moves a weight by about lr * sign(g) once |g| is well above epsilon, but a gradient inside the noise of
                    // either implementation (PyTorch's float32 error is the recorded floor) can flip that sign, so only weights whose
                    // gradient is at least 50 floors in every step so far are compared.
                    double floor = paramFloors.GetProperty(n).GetProperty("max_abs").GetDouble();
                    var keep = new bool[want.Length];
                    for (int s2 = 1; s2 <= step; s2++)
                    {
                        float[] g = fx.Read(sets.GetProperty($"grads_step{s2}").GetProperty(n));
                        for (int q = 0; q < g.Length; q++)
                        {
                            keep[q] = (s2 == 1 || keep[q]) && Math.Abs(g[q]) > 50 * floor;
                        }
                    }

                    int[] idx = Enumerable.Range(0, want.Length).Where(q => keep[q]).ToArray();
                    Compare($"model weights after step {step} {n} ({idx.Length} of {want.Length} compared)", idx.Select(q => got[q]).ToArray(), idx.Select(q => want[q]).ToArray(), 0.01 * options.LearningRate * step, 1e-5, 0);
                }
            }
        }
    }
}
