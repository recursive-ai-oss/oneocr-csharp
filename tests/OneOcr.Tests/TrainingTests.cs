using System.Security.Cryptography;
using OneOcr.Kernels;
using OneOcr.Model;
using OneOcr.Recognition;
using OneOcr.Training;

namespace OneOcr.Tests;

/// <summary>Straightforward double-precision implementations of the backward passes, written from the definitions (no matrix products).</summary>
internal static class ReferenceBackward64
{
    public static (double[] DWeight, double[] DBias, double[] DInput) Conv(
        ConvBlock b, float[] input, int h, int w, ConvCache cache, float[] dPooled)
    {
        int ch = cache.ConvH, cw = cache.ConvW;
        var dConv = new double[ch * cw * b.Out];
        for (int i = 0; i < cache.OutH * cache.OutW * b.Out; i++)
        {
            if (cache.Argmax[i] >= 0)
            {
                dConv[cache.Argmax[i]] = dPooled[i];
            }
        }

        var dW = new double[b.Out * b.KPadded];
        var dB = new double[b.Out];
        var dIn = new double[h * w * b.In];
        for (int y = 0; y < ch; y++)
        {
            for (int x = 0; x < cw; x++)
            {
                for (int o = 0; o < b.Out; o++)
                {
                    double g = dConv[(((y * cw) + x) * b.Out) + o];
                    dB[o] += g;
                    for (int ky = 0; ky < b.KernelH; ky++)
                    {
                        for (int kx = 0; kx < b.KernelW; kx++)
                        {
                            int iy = y - b.PadH + ky, ix = x - b.PadW + kx;
                            if (iy < 0 || iy >= h || ix < 0 || ix >= w)
                            {
                                continue;
                            }

                            for (int c = 0; c < b.In; c++)
                            {
                                int tap = (((ky * b.KernelW) + kx) * b.In) + c;
                                dW[(o * b.KPadded) + tap] += g * input[(((iy * w) + ix) * b.In) + c];
                                dIn[(((iy * w) + ix) * b.In) + c] += g * b.Weights[(o * b.KPadded) + tap];
                            }
                        }
                    }
                }
            }
        }

        return (dW, dB, dIn);
    }

    private static double Sig(double v) => 1 / (1 + Math.Exp(-v));

    /// <summary>Backpropagation through time for one bidirectional layer with exact sigmoid and tanh.</summary>
    public static (double[][] DWih, double[][] DWhh, double[][] DBias, double[] DX) Lstm(LstmLayer layer, int hidden, float[] x, int t, float[] dOut)
    {
        int gates = 4 * hidden, inSize = layer.In;
        var dWih = new double[2][];
        var dWhh = new double[2][];
        var dBias = new double[2][];
        var dX = new double[t * inSize];
        for (int d = 0; d < 2; d++)
        {
            LstmDirection dir = d == 0 ? layer.Forward : layer.Reverse;
            bool reverse = d == 1;
            int column = reverse ? hidden : 0;
            var hs = new double[(t + 1) * hidden];
            var cs = new double[(t + 1) * hidden];
            var gi = new double[t * hidden];
            var gf = new double[t * hidden];
            var gg = new double[t * hidden];
            var go = new double[t * hidden];
            var hPrevOf = new int[t];
            // hs/cs slot 0 is the zero initial state; slot s+1 is the state after time s.
            int prevSlot = 0;
            for (int step = 0; step < t; step++)
            {
                int s = reverse ? t - 1 - step : step;
                hPrevOf[s] = prevSlot;
                for (int j = 0; j < hidden; j++)
                {
                    double[] z = new double[4];
                    for (int q = 0; q < 4; q++)
                    {
                        int row = (q * hidden) + j;
                        double v = dir.Bias[row];
                        for (int i = 0; i < inSize; i++)
                        {
                            v += (double)dir.WeightIh[(row * inSize) + i] * x[(s * inSize) + i];
                        }

                        for (int i = 0; i < hidden; i++)
                        {
                            v += (double)dir.WeightHh[(row * hidden) + i] * hs[(prevSlot * hidden) + i];
                        }

                        z[q] = v;
                    }

                    double i0 = Sig(z[0]), f0 = Sig(z[1]), g0 = Math.Tanh(z[2]), o0 = Sig(z[3]);
                    double c = (f0 * cs[(prevSlot * hidden) + j]) + (i0 * g0);
                    gi[(s * hidden) + j] = i0;
                    gf[(s * hidden) + j] = f0;
                    gg[(s * hidden) + j] = g0;
                    go[(s * hidden) + j] = o0;
                    cs[((s + 1) * hidden) + j] = c;
                    hs[((s + 1) * hidden) + j] = o0 * Math.Tanh(c);
                }

                // Slots are indexed by time, so give the next visited step this step's slot.
                prevSlot = s + 1;
            }

            var wih = new double[gates * inSize];
            var whh = new double[gates * hidden];
            var db = new double[gates];
            var dhNext = new double[hidden];
            var dcNext = new double[hidden];
            for (int step = t - 1; step >= 0; step--)
            {
                int s = reverse ? t - 1 - step : step;
                int pSlot = hPrevOf[s];
                var dz = new double[gates];
                var dhPrev = new double[hidden];
                for (int j = 0; j < hidden; j++)
                {
                    double dh = dOut[(s * 2 * hidden) + column + j] + dhNext[j];
                    double c = cs[((s + 1) * hidden) + j], tc = Math.Tanh(c);
                    double i0 = gi[(s * hidden) + j], f0 = gf[(s * hidden) + j], g0 = gg[(s * hidden) + j], o0 = go[(s * hidden) + j];
                    double dc = dcNext[j] + (dh * o0 * (1 - (tc * tc)));
                    dz[j] = dc * g0 * i0 * (1 - i0);
                    dz[hidden + j] = dc * cs[(pSlot * hidden) + j] * f0 * (1 - f0);
                    dz[(2 * hidden) + j] = dc * i0 * (1 - (g0 * g0));
                    dz[(3 * hidden) + j] = dh * tc * o0 * (1 - o0);
                    dcNext[j] = dc * f0;
                }

                for (int row = 0; row < gates; row++)
                {
                    db[row] += dz[row];
                    for (int i = 0; i < inSize; i++)
                    {
                        wih[(row * inSize) + i] += dz[row] * x[(s * inSize) + i];
                        dX[(s * inSize) + i] += dz[row] * dir.WeightIh[(row * inSize) + i];
                    }

                    for (int i = 0; i < hidden; i++)
                    {
                        whh[(row * hidden) + i] += dz[row] * hs[(pSlot * hidden) + i];
                        dhPrev[i] += dz[row] * dir.WeightHh[(row * hidden) + i];
                    }
                }

                dhNext = dhPrev;
            }

            dWih[d] = wih;
            dWhh[d] = whh;
            dBias[d] = db;
        }

        return (dWih, dWhh, dBias, dX);
    }
}

public class TrainingTests
{
    private static CrnnModel Model(TestModels.Shape shape, int seed) =>
        CrnnModel.FromContainer(ModelContainer.Parse(TestModels.RandomContainer(shape, seed)), OcrLimits.Default);

    private static float[] RandomFloats(int n, int seed, double scale = 1)
    {
        var rng = new Random(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++)
        {
            v[i] = (float)(((rng.NextDouble() * 2) - 1) * scale);
        }

        return v;
    }

    // Worst error as a multiple of the tolerance |got - ref| <= atol + rtol * |ref|; also logged when ONEOCR_TEST_LOG names a file.
    private static void Check(string name, ReadOnlySpan<float> got, double[] reference, double atol, double rtol)
    {
        Assert.True(got.Length >= reference.Length, $"{name}: too few values");
        double worst = 0, worstAbs = 0, worstRel = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double err = Math.Abs(got[i] - reference[i]);
            worst = Math.Max(worst, err / (atol + (rtol * Math.Abs(reference[i]))));
            worstAbs = Math.Max(worstAbs, err);
            if (Math.Abs(reference[i]) > 0.05)
            {
                worstRel = Math.Max(worstRel, err / Math.Abs(reference[i]));
            }
        }

        string? log = Environment.GetEnvironmentVariable("ONEOCR_TEST_LOG");
        if (log is not null)
        {
            File.AppendAllText(log, $"{name}\tmaxAbs={worstAbs:E2}\tmaxRelWhere>0.05={worstRel:E2}\tx tolerance={worst:F3}\n");
        }

        Assert.True(worst <= 1, $"{name}: error is {worst:F2}x the tolerance (atol {atol}, rtol {rtol}); max abs {worstAbs:E2}");
    }

    private static double[] Widen(float[] v) => v.Select(x => (double)x).ToArray();

    // Gradient tolerances against float64, measured on random-weight models (docs/spec/backward-numerics.md).
    private const double ConvAtol = 7e-6, ConvRtol = 3.5e-5;
    private const double LstmAtol = 1e-6, LstmRtol = 5e-6;
    private const double FcAtol = 1e-6, FcRtol = 5e-6;

    [Theory]
    [InlineData(0, 64)]
    [InlineData(1, 40)]
    public void ConvBackwardMatchesDoublePrecisionReference(int shapeIndex, int width)
    {
        var shape = shapeIndex == 0 ? TestModels.Small : TestModels.SmokeLike;
        CrnnModel model = Model(shape, seed: 21 + shapeIndex);
        var pool = new BufferPool<float>(1 << 28, 4);
        var ipool = new BufferPool<int>(1 << 28, 4);
        foreach (KernelPath path in Dispatch.Available())
        {
            using var scratch = new TrainScratch(pool, ipool);
            int h = model.Input.Height, w = width;
            float[] x = RandomFloats(h * w, 5);
            for (int i = 0; i < model.Conv.Length; i++)
            {
                ConvBlock b = model.Conv[i];
                float[] y = ConvTrain.Forward(b, path, scratch, x, h, w, out ConvCache cache);
                int n = cache.OutH * cache.OutW * b.Out;
                float[] dPooled = RandomFloats(n, 100 + i);
                var wt = new float[b.KPadded * b.Out];
                Backward.Transpose(b.Weights, b.KPadded, b.Out, b.KPadded, wt);
                var dW = new float[b.Out * b.KPadded];
                var dB = new float[b.Out];
                var dIn = new float[h * w * b.In];
                ConvTrain.BackwardPass(b, path, scratch, cache, wt, dPooled, dW, dB, dIn);
                var (rW, rB, rIn) = ReferenceBackward64.Conv(b, x, h, w, cache, dPooled);
                Check($"{path} conv.{i} dW", dW, rW, ConvAtol, ConvRtol);
                Check($"{path} conv.{i} dBias", dB, rB, ConvAtol, ConvRtol);
                Check($"{path} conv.{i} dInput", dIn, rIn, ConvAtol, ConvRtol);
                x = y.AsSpan(0, n).ToArray();
                h = cache.OutH;
                w = cache.OutW;
            }
        }
    }

    [Theory]
    [InlineData(0, 21)]
    [InlineData(0, 7)]
    [InlineData(1, 30)]
    public void LstmBackwardMatchesDoublePrecisionReference(int shapeIndex, int steps)
    {
        var shape = shapeIndex == 0 ? TestModels.Small : TestModels.SmokeLike;
        CrnnModel model = Model(shape, seed: 33 + shapeIndex);
        var pool = new BufferPool<float>(1 << 28, 4);
        var ipool = new BufferPool<int>(1 << 28, 4);
        int hidden = model.Hidden, gates = 4 * hidden;
        foreach (KernelPath path in Dispatch.Available())
        {
            using var scratch = new TrainScratch(pool, ipool);
            float[] x = RandomFloats(steps * model.FeatureSize, 8);
            for (int l = 0; l < model.Lstm.Length; l++)
            {
                LstmLayer layer = model.Lstm[l];
                float[] y = LstmTrain.Forward(layer, hidden, path, scratch, x, steps, out LstmCache cache);
                float[] dOut = RandomFloats(steps * 2 * hidden, 200 + l);
                var whhT = new float[2][];
                var wihCatT = new float[layer.In * 2 * gates];
                for (int d = 0; d < 2; d++)
                {
                    LstmDirection dir = d == 0 ? layer.Forward : layer.Reverse;
                    whhT[d] = new float[hidden * gates];
                    Backward.Transpose(dir.WeightHh, hidden, gates, hidden, whhT[d]);
                    var tmp = new float[layer.In * gates];
                    Backward.Transpose(dir.WeightIh, layer.In, gates, layer.In, tmp);
                    for (int i = 0; i < layer.In; i++)
                    {
                        tmp.AsSpan(i * gates, gates).CopyTo(wihCatT.AsSpan((i * 2 * gates) + (d * gates), gates));
                    }
                }

                float[][] dWih = [new float[gates * layer.In], new float[gates * layer.In]];
                float[][] dWhh = [new float[gates * hidden], new float[gates * hidden]];
                float[][] dB = [new float[gates], new float[gates]];
                var dX = new float[steps * layer.In];
                LstmTrain.BackwardPass(layer, hidden, path, scratch, cache, dOut, whhT, wihCatT, dWih, dWhh, dB, dX);
                var (rWih, rWhh, rB, rX) = ReferenceBackward64.Lstm(layer, hidden, x, steps, dOut);
                for (int d = 0; d < 2; d++)
                {
                    Check($"{path} lstm.l{l}.{d} dWih", dWih[d], rWih[d], LstmAtol, LstmRtol);
                    Check($"{path} lstm.l{l}.{d} dWhh", dWhh[d], rWhh[d], LstmAtol, LstmRtol);
                    Check($"{path} lstm.l{l}.{d} dBias", dB[d], rB[d], LstmAtol, LstmRtol);
                }

                Check($"{path} lstm.l{l} dInput", dX, rX, LstmAtol, LstmRtol);
                x = y.AsSpan(0, steps * 2 * hidden).ToArray();
            }
        }
    }

    [Fact]
    public void LinearBackwardMatchesDoublePrecisionReference()
    {
        var pool = new BufferPool<float>(1 << 26, 4);
        var ipool = new BufferPool<int>(1 << 26, 4);
        const int t = 37, inSize = 50, outSize = 41;
        float[] x = RandomFloats(t * inSize, 1), w = RandomFloats(outSize * inSize, 2), dy = RandomFloats(t * outSize, 3);
        var rW = new double[outSize * inSize];
        var rB = new double[outSize];
        var rX = new double[t * inSize];
        for (int s = 0; s < t; s++)
        {
            for (int o = 0; o < outSize; o++)
            {
                rB[o] += dy[(s * outSize) + o];
                for (int i = 0; i < inSize; i++)
                {
                    rW[(o * inSize) + i] += (double)dy[(s * outSize) + o] * x[(s * inSize) + i];
                    rX[(s * inSize) + i] += (double)dy[(s * outSize) + o] * w[(o * inSize) + i];
                }
            }
        }

        var wt = new float[inSize * outSize];
        Backward.Transpose(w, inSize, outSize, inSize, wt);
        foreach (KernelPath path in Dispatch.Available())
        {
            using var scratch = new TrainScratch(pool, ipool);
            var dW = new float[outSize * inSize];
            var dB = new float[outSize];
            var dX = new float[t * inSize];
            LinearTrain.BackwardPass(path, scratch, x, dy, wt, t, inSize, outSize, dW, dB, dX);
            Check($"{path} fc dW", dW, rW, FcAtol, FcRtol);
            Check($"{path} fc dBias", dB, rB, FcAtol, FcRtol);
            Check($"{path} fc dInput", dX, rX, FcAtol, FcRtol);
        }
    }

    [Fact]
    public void WindowScatterIsTheAdjointOfWindowGather()
    {
        // <Im2Col(x), y> == <x, Col2Im(y)> for a block with padding, which is what makes the input gradient correct.
        const int h = 7, w = 9, c = 3, kh = 3, kw = 3, pad = 1, kPadded = 32;
        int ch = h + (2 * pad) - kh + 1, cw = w + (2 * pad) - kw + 1;
        float[] x = RandomFloats(h * w * c, 1);
        float[] y = RandomFloats(ch * cw * kPadded, 2);
        for (int p = 0; p < ch * cw; p++)
        {
            Array.Clear(y, (p * kPadded) + (kh * kw * c), kPadded - (kh * kw * c));
        }

        var col = new float[ch * cw * kPadded];
        Backward.Im2Col(x, h, w, c, kh, kw, pad, pad, kPadded, col);
        var back = new float[h * w * c];
        Backward.Col2Im(y, h, w, c, kh, kw, pad, pad, kPadded, back);
        double left = 0, right = 0;
        for (int i = 0; i < col.Length; i++)
        {
            left += (double)col[i] * y[i];
        }

        for (int i = 0; i < back.Length; i++)
        {
            right += (double)x[i] * back[i];
        }

        Assert.True(Math.Abs(left - right) <= 1e-6 * Math.Abs(left), $"{left:R} vs {right:R}");
    }

    [Fact]
    public void ReluPoolPicksFirstMaximumAndDropsNonPositiveWindows()
    {
        // 2x4 map of one channel, pool 2x2: window one is [1,3;3,-1] (first 3 wins), window two is all non-positive.
        float[] conv = [1, 3, -1, 0, 3, -1, -2, -3];
        var pooled = new float[2];
        var arg = new int[2];
        Backward.ReluPool(conv, 4, 1, 2, 2, 1, 2, pooled, arg);
        Assert.Equal([3f, 0f], pooled);
        Assert.Equal([1, -1], arg);
        var dConv = new float[8];
        Backward.ReluPoolBackward([5f, 7f], arg, 2, dConv);
        Assert.Equal([0f, 5f, 0f, 0f, 0f, 0f, 0f, 0f], dConv);
    }

    // ---- CTC ----

    private static double[] Softmax(float[] logits, int t, int classes)
    {
        var y = new double[logits.Length];
        for (int s = 0; s < t; s++)
        {
            double sum = 0;
            for (int k = 0; k < classes; k++)
            {
                sum += Math.Exp(logits[(s * classes) + k]);
            }

            for (int k = 0; k < classes; k++)
            {
                y[(s * classes) + k] = Math.Exp(logits[(s * classes) + k]) / sum;
            }
        }

        return y;
    }

    private static bool Collapses(int[] path, int[] target)
    {
        var outp = new List<int>();
        int last = -1;
        foreach (int p in path)
        {
            if (p != last && p != 0)
            {
                outp.Add(p);
            }

            last = p;
        }

        return outp.SequenceEqual(target);
    }

    [Theory]
    [InlineData(new[] { 1, 2 })]
    [InlineData(new[] { 1, 1 })]
    [InlineData(new[] { 3 })]
    [InlineData(new int[0])]
    [InlineData(new[] { 2, 1, 2 })]
    public void CtcLossAndGradientMatchBruteForceOverAllAlignments(int[] target)
    {
        const int t = 6, classes = 4;
        float[] logits = RandomFloats(t * classes, 17 + target.Length, 2);
        double[] y = Softmax(logits, t, classes);
        double p = 0;
        var gradSum = new double[t * classes];
        var path = new int[t];
        long total = 1;
        for (int i = 0; i < t; i++)
        {
            total *= classes;
        }

        for (long code = 0; code < total; code++)
        {
            long c = code;
            double prob = 1;
            for (int s = 0; s < t; s++)
            {
                path[s] = (int)(c % classes);
                c /= classes;
                prob *= y[(s * classes) + path[s]];
            }

            if (Collapses(path, target))
            {
                p += prob;
                for (int s = 0; s < t; s++)
                {
                    gradSum[(s * classes) + path[s]] += prob / y[(s * classes) + path[s]];
                }
            }
        }

        var grad = new float[t * classes];
        double loss = CtcLoss.Compute(logits, t, classes, target, 1.0, grad);
        Assert.Equal(-Math.Log(p), loss, 1e-10);
        var expected = new double[t * classes];
        for (int i = 0; i < expected.Length; i++)
        {
            // Graves et al. 2006, eq. 16 with the softmax output layer folded in: y - (1/p) * (probability mass through (t, k)).
            expected[i] = y[i] - (gradSum[i] * y[i] / p);
        }

        Check("ctc grad", grad, expected, 1e-6, 1e-6);

        var scaled = new float[t * classes];
        CtcLoss.Compute(logits, t, classes, target, 0.25, scaled);
        for (int i = 0; i < scaled.Length; i++)
        {
            Assert.Equal(grad[i] * 0.25f, scaled[i], 1e-6f);
        }
    }

    [Fact]
    public void CtcWithNoPossibleAlignmentHasInfiniteLossAndZeroGradient()
    {
        // Two equal labels need a blank between them, so three steps are the minimum.
        float[] logits = RandomFloats(2 * 4, 3);
        var grad = Enumerable.Repeat(1f, 8).ToArray();
        Assert.True(double.IsPositiveInfinity(CtcLoss.Compute(logits, 2, 4, [1, 1], 1.0, grad)));
        Assert.All(grad, g => Assert.Equal(0f, g));
        Assert.Throws<ArgumentException>(() => CtcLoss.Compute(logits, 2, 4, [0], 1.0, grad));
        Assert.Throws<ArgumentException>(() => CtcLoss.Compute(logits, 2, 4, [4], 1.0, grad));
    }

    // ---- Optimizer ----

    [Fact]
    public void AdamWMatchesTheDoubleAlgorithm()
    {
        var options = new AdamWOptions { LearningRate = 1e-2, Beta1 = 0.9, Beta2 = 0.99, Epsilon = 1e-8, WeightDecay = 0.1 };
        float[] w = RandomFloats(50, 1), b = RandomFloats(7, 2);
        var rw = Widen(w);
        var rb = Widen(b);
        var opt = new AdamW([w, b], [true, false], options);
        var mw = new double[50];
        var vw = new double[50];
        var mb = new double[7];
        var vb = new double[7];
        for (int step = 1; step <= 20; step++)
        {
            float[] gw = RandomFloats(50, 10 + step), gb = RandomFloats(7, 50 + step);
            opt.Step([gw, gb], options.LearningRate);
            void Ref(double[] p, double[] m, double[] v, float[] g, double decay)
            {
                for (int i = 0; i < p.Length; i++)
                {
                    p[i] *= 1 - (options.LearningRate * decay);
                    m[i] = (options.Beta1 * m[i]) + ((1 - options.Beta1) * g[i]);
                    v[i] = (options.Beta2 * v[i]) + ((1 - options.Beta2) * g[i] * g[i]);
                    double mh = m[i] / (1 - Math.Pow(options.Beta1, step)), vh = v[i] / (1 - Math.Pow(options.Beta2, step));
                    p[i] -= options.LearningRate * mh / (Math.Sqrt(vh) + options.Epsilon);
                }
            }

            Ref(rw, mw, vw, gw, options.WeightDecay);
            Ref(rb, mb, vb, gb, 0);
        }

        // The reference adds epsilon after the bias correction, the implementation before it (PyTorch's order): equal to ~1e-8 relative.
        Check("adamw weight", w, rw, 1e-6, 1e-6);
        Check("adamw bias", b, rb, 1e-6, 1e-6);
        Assert.Equal(20, opt.Steps);
    }

    // ---- Whole network ----

    private static byte[] GlyphLine(int[] labels, int seed)
    {
        // Each class is a 12-column pattern of its own; the pattern is stable across lines so the task is learnable.
        int width = (12 * labels.Length) + 8;
        var px = new byte[32 * width];
        var rng = new Random(seed);
        for (int i = 0; i < px.Length; i++)
        {
            px[i] = (byte)(235 + rng.Next(0, 20));
        }

        for (int n = 0; n < labels.Length; n++)
        {
            var glyph = new Random(1000 + labels[n]);
            for (int x = 0; x < 10; x++)
            {
                for (int y = 4; y < 28; y++)
                {
                    if (glyph.Next(0, 2) == 1)
                    {
                        px[(y * width) + 4 + (n * 12) + x] = (byte)rng.Next(0, 30);
                    }
                }
            }
        }

        return px;
    }

    private static List<TrainLine> Batch(params int[][] labels) =>
        labels.Select((l, i) => new TrainLine(GlyphLine(l, i), (12 * l.Length) + 8, l)).ToList();

    private static string Hash(IEnumerable<float[]> tensors, double loss)
    {
        using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        h.AppendData(BitConverter.GetBytes(loss));
        foreach (float[] t in tensors)
        {
            var bytes = new byte[t.Length * 4];
            for (int i = 0; i < t.Length; i++)
            {
                BitConverter.TryWriteBytes(bytes.AsSpan(i * 4), t[i]);
            }

            h.AppendData(bytes);
        }

        return Convert.ToHexString(h.GetHashAndReset());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void TrainingForwardIsBitIdenticalToInference(int shapeIndex)
    {
        var shape = shapeIndex == 0 ? TestModels.Small : TestModels.SmokeLike;
        byte[] file = TestModels.RandomContainer(shape, seed: 77);
        foreach (KernelPath path in Dispatch.Available())
        {
            var recognizer = new LineRecognizer(CrnnModel.FromContainer(ModelContainer.Parse(file), OcrLimits.Default), OcrLimits.Default, path);
            var trainer = new CrnnTrainer(CrnnModel.FromContainer(ModelContainer.Parse(file), OcrLimits.Default), path, new AdamWOptions());
            foreach (int width in new[] { 33, 64, 97, 200 })
            {
                byte[] image = TestModels.RandomImage(32, width, seed: width);
                float[] expected = recognizer.Forward(image, width, 1, null, CancellationToken.None, out int steps);
                float[] got = trainer.Forward(image, width, out int gotSteps);
                Assert.Equal(steps, gotSteps);
                for (int i = 0; i < steps * shape.Classes; i++)
                {
                    Assert.True(
                        BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(got[i]),
                        $"{path} width {width} logit {i}: {expected[i]:R} vs {got[i]:R}");
                }
            }
        }
    }

    [Fact]
    public void GradientsAgreeWithDirectionalDerivativesOfTheLoss()
    {
        var shape = TestModels.Small;
        var batch = Batch([3, 9, 3], [14], [5, 6, 7, 5], [2, 2]);
        var trainer = new CrnnTrainer(Model(shape, seed: 5), Dispatch.Best, new AdamWOptions());
        double loss = trainer.ComputeGradients(batch);
        Assert.Equal(trainer.Evaluate(batch), loss, 1e-9);

        float[][] g = trainer.Gradients.Select(a => (float[])a.Clone()).ToArray();
        double norm = Math.Sqrt(g.Sum(a => a.Sum(v => (double)v * v)));
        Assert.True(norm > 1e-3, "the gradient is degenerate");

        // d = g / |g| gives a derivative of |g|; a random direction gives g . d.
        foreach (bool useGradient in new[] { true, false })
        {
            var rng = new Random(9);
            float[][] d = useGradient
                ? g.Select(a => a.Select(v => (float)(v / norm)).ToArray()).ToArray()
                : g.Select(a => a.Select(_ => (float)((rng.NextDouble() * 2) - 1)).ToArray()).ToArray();
            double expected = 0, dnorm = 0;
            for (int i = 0; i < g.Length; i++)
            {
                for (int j = 0; j < g[i].Length; j++)
                {
                    expected += (double)g[i][j] * d[i][j];
                    dnorm += (double)d[i][j] * d[i][j];
                }
            }

            double eps = useGradient ? 0.005 : 0.005 / Math.Sqrt(dnorm);
            float[][] original = trainer.Parameters.Select(a => (float[])a.Clone()).ToArray();
            double Loss(double sign)
            {
                for (int i = 0; i < original.Length; i++)
                {
                    for (int j = 0; j < original[i].Length; j++)
                    {
                        trainer.Parameters[i][j] = original[i][j] + (float)(sign * eps * d[i][j]);
                    }
                }

                return trainer.Evaluate(batch);
            }

            double numeric = (Loss(+1) - Loss(-1)) / (2 * eps);
            Loss(0);
            Assert.True(Math.Abs(numeric - expected) <= 0.03 * norm, $"gradient direction {useGradient}: numeric {numeric:G6} vs analytic {expected:G6} (|g| {norm:G6})");
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void LossGradientsAndWeightsAreBitIdenticalOnEveryPathAndDegree(int shapeIndex)
    {
        var shape = shapeIndex == 0 ? TestModels.Small : TestModels.SmokeLike;
        int classes = shape.Classes;
        var batch = Batch([3, 9, 3, 12], [14], [5, 6, 7, 5, 30], [2, 2], [1, 4, 8], [classes - 1, 3]);
        string? baseline = null;
        foreach (KernelPath path in Dispatch.Available())
        {
            foreach (int degree in new[] { 1, 4, 7 })
            {
                var trainer = new CrnnTrainer(Model(shape, seed: 12), path, new AdamWOptions { LearningRate = 2e-3 }, maxGradNorm: 3);
                var parts = new List<float[]>();
                var losses = new List<double>();
                for (int step = 0; step < 3; step++)
                {
                    losses.Add(trainer.Step(batch, degree));
                    if (step == 0)
                    {
                        parts.AddRange(trainer.Gradients);
                    }
                }

                parts.AddRange(trainer.Parameters);
                string hash = Hash(parts, losses.Sum());
                baseline ??= hash;
                if (Environment.GetEnvironmentVariable("ONEOCR_TEST_LOG") is { } log)
                {
                    File.AppendAllText(log, $"HASH shape {shapeIndex} {path} degree {degree} {hash}\n");
                }

                Assert.True(baseline == hash, $"{path} at degree {degree} differs from the first run");
            }
        }
    }

    [Fact]
    public void OverfitsAFewLinesAndReadsThemBack()
    {
        var shape = TestModels.Small;
        var lines = new[] { new[] { 3, 9, 3 }, new[] { 14, 2 }, new[] { 5, 6, 7 }, new[] { 2, 2, 8 } };
        var batch = Batch(lines);
        CrnnModel model = Model(shape, seed: 3);
        var trainer = new CrnnTrainer(model, Dispatch.Best, new AdamWOptions { LearningRate = 5e-3, WeightDecay = 0 }, maxGradNorm: 5);
        double first = trainer.Evaluate(batch);
        double last = first;
        for (int step = 0; step < 400; step++)
        {
            last = trainer.Step(batch, 2);
            if (last < 0.01)
            {
                break;
            }
        }

        Assert.True(last < first * 0.05, $"loss went from {first:F3} to {last:F3}");
        for (int i = 0; i < batch.Count; i++)
        {
            float[] logits = trainer.Forward(batch[i].Gray.Span, batch[i].Width, out int steps);
            string expected = string.Concat(lines[i].Select(c => model.Charset[c]));
            Assert.Equal(expected, trainer.DecodeGreedy(logits, steps));
        }
    }

    [Fact]
    public void ABatchLineThatIsTooNarrowOrImpossibleContributesNothing()
    {
        var shape = TestModels.Small;
        var good = Batch([3, 9, 3])[0];
        // Width 1 leaves no columns; a long target in a short line has no alignment.
        var narrow = new TrainLine(new byte[32], 1, [3]);
        var impossible = new TrainLine(good.Gray, good.Width, Enumerable.Repeat(4, 40).ToArray());
        var a = new CrnnTrainer(Model(shape, seed: 1), Dispatch.Best, new AdamWOptions());
        var b = new CrnnTrainer(Model(shape, seed: 1), Dispatch.Best, new AdamWOptions());
        double lossA = a.ComputeGradients([good, narrow, impossible]);
        double lossB = b.ComputeGradients([good]);
        // Same line, but the batch of three divides by three.
        Assert.Equal(lossB / 3, lossA, 1e-9);
        for (int i = 0; i < a.Gradients.Count; i++)
        {
            double scale = b.Gradients[i].Max(v => Math.Abs(v));
            for (int j = 0; j < a.Gradients[i].Length; j++)
            {
                Assert.Equal(b.Gradients[i][j] / 3, a.Gradients[i][j], (scale * 1e-4) + 1e-9);
            }
        }
    }
}
