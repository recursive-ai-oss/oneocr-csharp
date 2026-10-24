using OneOcr.Kernels;
using OneOcr.Recognition;

namespace OneOcr.Training;

/// <summary>Arrays rented for one line's forward and backward pass; disposing returns all of them, on every path.</summary>
internal sealed class TrainScratch : IDisposable
{
    private readonly BufferPool<float> _floats;
    private readonly BufferPool<int> _ints;
    private readonly List<float[]> _floatsOut = [];
    private readonly List<int[]> _intsOut = [];

    public TrainScratch(BufferPool<float> floats, BufferPool<int> ints)
    {
        _floats = floats;
        _ints = ints;
    }

    /// <summary>An array of at least <paramref name="n"/> floats with unspecified contents.</summary>
    public float[] Floats(long n)
    {
        float[] a = _floats.Rent(n);
        _floatsOut.Add(a);
        return a;
    }

    /// <summary>An array of at least <paramref name="n"/> zeroed floats.</summary>
    public float[] Zeros(long n)
    {
        float[] a = Floats(n);
        Array.Clear(a, 0, (int)n);
        return a;
    }

    public int[] Ints(long n)
    {
        int[] a = _ints.Rent(n);
        _intsOut.Add(a);
        return a;
    }

    public void Dispose()
    {
        foreach (float[] a in _floatsOut)
        {
            _floats.Return(a);
        }

        foreach (int[] a in _intsOut)
        {
            _ints.Return(a);
        }

        _floatsOut.Clear();
        _intsOut.Clear();
    }
}

/// <summary>Backward of a linear layer <c>y = x W^T + b</c> applied to <c>t</c> rows.</summary>
internal static class LinearTrain
{
    /// <summary>
    /// Writes <c>dW[o][i] = sum_t dY[t][o] x[t][i]</c>, <c>dB[o] = sum_t dY[t][o]</c> and <c>dX[t][i] = sum_o dY[t][o] W[o][i]</c>, each a
    /// library-order dot product; <paramref name="weightT"/> is W transposed to <c>[in][out]</c>.
    /// </summary>
    public static void BackwardPass(
        KernelPath path, TrainScratch scratch, ReadOnlySpan<float> x, ReadOnlySpan<float> dY, ReadOnlySpan<float> weightT, int t, int inSize, int outSize,
        Span<float> dWeight, Span<float> dBias, Span<float> dX)
    {
        float[] dyT = scratch.Floats((long)t * outSize);
        Backward.Transpose(dY, outSize, t, outSize, dyT);
        float[] xT = scratch.Floats((long)t * inSize);
        Backward.Transpose(x, inSize, t, inSize, xT);
        Gemm.NT(path, dyT, t, xT, t, outSize, inSize, t, default, dWeight, inSize, 1);
        Backward.RowSums(path, dyT, t, outSize, t, dBias);
        if (!dX.IsEmpty)
        {
            Gemm.NT(path, dY, outSize, weightT, outSize, t, inSize, outSize, default, dX, inSize, 1);
        }
    }
}

/// <summary>What the backward pass of one convolution block needs from its forward pass.</summary>
internal sealed class ConvCache
{
    public required int H { get; init; }
    public required int W { get; init; }
    public required int ConvH { get; init; }
    public required int ConvW { get; init; }
    public required int OutH { get; init; }
    public required int OutW { get; init; }

    /// <summary>Input windows, <c>[ConvH * ConvW][KPadded]</c>.</summary>
    public required float[] Col { get; init; }

    /// <summary>For each pooled element, the flat index of the convolution output that won the pool, or -1.</summary>
    public required int[] Argmax { get; init; }
}

/// <summary>Forward (keeping what backward needs) and backward of a convolution block: convolution, ReLU, max pooling.</summary>
internal static class ConvTrain
{
    /// <summary>
    /// Runs the block on a channels-last <c>h x w x In</c> input and returns the pooled <c>OutH x OutW x Out</c> output. The
    /// convolution output is the same set of dot products the inference path computes, so the values are bit-identical to it.
    /// </summary>
    public static float[] Forward(ConvBlock b, KernelPath path, TrainScratch scratch, ReadOnlySpan<float> input, int h, int w, out ConvCache cache)
    {
        int ch = h + (2 * b.PadH) - b.KernelH + 1, cw = w + (2 * b.PadW) - b.KernelW + 1;
        int oh = ch / b.PoolH, ow = cw / b.PoolW;
        float[] col = scratch.Floats((long)ch * cw * b.KPadded);
        Backward.Im2Col(input, h, w, b.In, b.KernelH, b.KernelW, b.PadH, b.PadW, b.KPadded, col);
        float[] convOut = scratch.Floats((long)ch * cw * b.Out);
        Gemm.NT(path, col, b.KPadded, b.Weights, b.KPadded, ch * cw, b.Out, b.KPadded, b.Bias, convOut, b.Out, 1);
        float[] pooled = scratch.Floats((long)oh * ow * b.Out);
        int[] argmax = scratch.Ints((long)oh * ow * b.Out);
        Backward.ReluPool(convOut, cw, b.Out, b.PoolH, b.PoolW, oh, ow, pooled, argmax);
        cache = new ConvCache { H = h, W = w, ConvH = ch, ConvW = cw, OutH = oh, OutW = ow, Col = col, Argmax = argmax };
        return pooled;
    }

    /// <summary>
    /// Gradients of the block. <paramref name="weightT"/> is the weights transposed to <c>[KPadded][Out]</c>. Writes the weight
    /// gradient (<c>[Out][KPadded]</c>, zero in the padding columns) and bias gradient, and, when <paramref name="dInput"/> is given,
    /// the gradient of the <c>h x w x In</c> input.
    /// </summary>
    public static void BackwardPass(
        ConvBlock b, KernelPath path, TrainScratch scratch, ConvCache cache, ReadOnlySpan<float> weightT, ReadOnlySpan<float> dPooled,
        Span<float> dWeight, Span<float> dBias, Span<float> dInput)
    {
        int positions = cache.ConvH * cache.ConvW;
        float[] dConv = scratch.Floats((long)positions * b.Out);
        Backward.ReluPoolBackward(dPooled, cache.Argmax, cache.OutH * cache.OutW * b.Out, dConv.AsSpan(0, positions * b.Out));
        BackwardConv(b, path, scratch, cache, weightT, dConv, dWeight, dBias, dInput);
    }

    /// <summary>The convolution part of <see cref="BackwardPass"/>, from the gradient of the convolution output <c>[ConvH * ConvW][Out]</c>.</summary>
    public static void BackwardConv(
        ConvBlock b, KernelPath path, TrainScratch scratch, ConvCache cache, ReadOnlySpan<float> weightT, ReadOnlySpan<float> dConv,
        Span<float> dWeight, Span<float> dBias, Span<float> dInput)
    {
        int positions = cache.ConvH * cache.ConvW;
        float[] dConvT = scratch.Floats((long)positions * b.Out);
        Backward.Transpose(dConv, b.Out, positions, b.Out, dConvT);
        Backward.RowSums(path, dConvT, positions, b.Out, positions, dBias);

        float[] colT = scratch.Floats((long)positions * b.KPadded);
        Backward.Transpose(cache.Col, b.KPadded, positions, b.KPadded, colT);
        Gemm.NT(path, dConvT, positions, colT, positions, b.Out, b.KPadded, positions, default, dWeight, b.KPadded, 1);

        if (!dInput.IsEmpty)
        {
            float[] dCol = scratch.Floats((long)positions * b.KPadded);
            Gemm.NT(path, dConv, b.Out, weightT, b.Out, positions, b.KPadded, b.Out, default, dCol, b.KPadded, 1);
            Backward.Col2Im(dCol, cache.H, cache.W, b.In, b.KernelH, b.KernelW, b.PadH, b.PadW, b.KPadded, dInput);
        }
    }
}

/// <summary>What the backward pass of one bidirectional LSTM layer needs from its forward pass. Arrays are indexed by time step.</summary>
internal sealed class LstmCache
{
    public required int T { get; init; }

    /// <summary>Layer input, <c>[T][In]</c>.</summary>
    public required float[] X { get; init; }

    /// <summary>Per direction: activated gates <c>[T][4H]</c> in the order input, forget, cell, output.</summary>
    public required float[][] Gates { get; init; }

    /// <summary>Per direction: the cell state after each step <c>[T][H]</c>.</summary>
    public required float[][] Cell { get; init; }

    /// <summary>Per direction: tanh of the cell state after each step <c>[T][H]</c>.</summary>
    public required float[][] CellTanh { get; init; }

    /// <summary>Per direction: the hidden state each step started from <c>[T][H]</c> (zeros for the first step visited).</summary>
    public required float[][] HPrev { get; init; }
}

/// <summary>Forward and backward of a bidirectional LSTM layer (backpropagation through time).</summary>
internal static class LstmTrain
{
    /// <summary>
    /// Runs the layer on <paramref name="x"/> (<c>[T][In]</c>) and returns the <c>[T][2H]</c> output (forward direction in the first
    /// H columns). The arithmetic is that of the inference path, written one element at a time.
    /// </summary>
    public static float[] Forward(LstmLayer layer, int hidden, KernelPath path, TrainScratch scratch, float[] x, int t, out LstmCache cache)
    {
        int gates = 4 * hidden;
        float[] output = scratch.Floats((long)t * 2 * hidden);
        cache = new LstmCache
        {
            T = t,
            X = x,
            Gates = [scratch.Floats((long)t * gates), scratch.Floats((long)t * gates)],
            Cell = [scratch.Floats((long)t * hidden), scratch.Floats((long)t * hidden)],
            CellTanh = [scratch.Floats((long)t * hidden), scratch.Floats((long)t * hidden)],
            HPrev = [scratch.Floats((long)t * hidden), scratch.Floats((long)t * hidden)],
        };
        for (int d = 0; d < 2; d++)
        {
            LstmDirection dir = d == 0 ? layer.Forward : layer.Reverse;
            bool reverse = d == 1;
            float[] proj = scratch.Floats((long)t * gates);
            Gemm.NT(path, x, layer.In, dir.WeightIh, layer.In, t, gates, layer.In, dir.Bias, proj, gates, 1);
            float[] z = scratch.Floats(gates);
            float[] cOld = scratch.Zeros(hidden);
            int column = reverse ? hidden : 0;
            for (int step = 0; step < t; step++)
            {
                int s = reverse ? t - 1 - step : step;
                int prev = reverse ? s + 1 : s - 1;
                Span<float> hp = cache.HPrev[d].AsSpan(s * hidden, hidden);
                if (step == 0)
                {
                    hp.Clear();
                }
                else
                {
                    output.AsSpan((prev * 2 * hidden) + column, hidden).CopyTo(hp);
                }

                Gemm.NT(path, hp, hidden, dir.WeightHh, hidden, 1, gates, hidden, proj.AsSpan(s * gates, gates), z, gates, 1);
                Span<float> g = cache.Gates[d].AsSpan(s * gates, gates);
                for (int j = 0; j < hidden; j++)
                {
                    float gi = FastMath.Sigmoid(z[j]);
                    float gf = FastMath.Sigmoid(z[hidden + j]);
                    float gg = FastMath.Tanh(z[(2 * hidden) + j]);
                    float go = FastMath.Sigmoid(z[(3 * hidden) + j]);
                    float cNew = MathF.FusedMultiplyAdd(gf, cOld[j], gi * gg);
                    float tc = FastMath.Tanh(cNew);
                    g[j] = gi;
                    g[hidden + j] = gf;
                    g[(2 * hidden) + j] = gg;
                    g[(3 * hidden) + j] = go;
                    cOld[j] = cNew;
                    cache.Cell[d][(s * hidden) + j] = cNew;
                    cache.CellTanh[d][(s * hidden) + j] = tc;
                    output[(s * 2 * hidden) + column + j] = go * tc;
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Gradients of the layer. <paramref name="whhT"/> holds each direction's recurrent weights transposed to <c>[H][4H]</c> and
    /// <paramref name="wihCatT"/> the two input-weight matrices joined and transposed to <c>[In][8H]</c> (forward gates first). Writes
    /// per direction the input-weight gradient <c>[4H][In]</c>, recurrent-weight gradient <c>[4H][H]</c> and bias gradient
    /// <c>[4H]</c> (the gradient of <c>b_ih + b_hh</c>), and <paramref name="dX"/> <c>[T][In]</c> when given.
    /// </summary>
    public static void BackwardPass(
        LstmLayer layer, int hidden, KernelPath path, TrainScratch scratch, LstmCache cache, ReadOnlySpan<float> dOut,
        float[][] whhT, ReadOnlySpan<float> wihCatT,
        float[][] dWeightIh, float[][] dWeightHh, float[][] dBias, Span<float> dX)
    {
        int t = cache.T, gates = 4 * hidden, inSize = layer.In;
        float[] dz = scratch.Floats((long)t * 2 * gates);
        float[] dhRec = scratch.Floats(hidden);
        float[] dcNext = scratch.Floats(hidden);
        for (int d = 0; d < 2; d++)
        {
            bool reverse = d == 1;
            int column = reverse ? hidden : 0;
            Array.Clear(dhRec, 0, hidden);
            Array.Clear(dcNext, 0, hidden);
            for (int step = t - 1; step >= 0; step--)
            {
                int s = reverse ? t - 1 - step : step;
                int prev = reverse ? s + 1 : s - 1;
                ReadOnlySpan<float> g = cache.Gates[d].AsSpan(s * gates, gates);
                Span<float> row = dz.AsSpan((s * 2 * gates) + (d * gates), gates);
                for (int j = 0; j < hidden; j++)
                {
                    float gi = g[j], gf = g[hidden + j], gg = g[(2 * hidden) + j], go = g[(3 * hidden) + j];
                    float tc = cache.CellTanh[d][(s * hidden) + j];
                    float cPrev = step == 0 ? 0f : cache.Cell[d][(prev * hidden) + j];
                    float dh = dOut[(s * 2 * hidden) + column + j] + dhRec[j];
                    float dtc = dh * go;
                    float dOutGate = dh * tc;
                    float dc = dcNext[j] + (dtc * (1f - (tc * tc)));
                    float di = dc * gg;
                    float dg = dc * gi;
                    float df = dc * cPrev;
                    dcNext[j] = dc * gf;
                    row[j] = di * (gi * (1f - gi));
                    row[hidden + j] = df * (gf * (1f - gf));
                    row[(2 * hidden) + j] = dg * (1f - (gg * gg));
                    row[(3 * hidden) + j] = dOutGate * (go * (1f - go));
                }

                if (step > 0)
                {
                    Gemm.NT(path, row, gates, whhT[d], gates, 1, hidden, gates, default, dhRec, hidden, 1);
                }
            }
        }

        float[] xT = scratch.Floats((long)t * inSize);
        Backward.Transpose(cache.X, inSize, t, inSize, xT);
        float[] dzT = scratch.Floats((long)t * gates);
        float[] hT = scratch.Floats((long)t * hidden);
        for (int d = 0; d < 2; d++)
        {
            Backward.Transpose(dz.AsSpan(d * gates), 2 * gates, t, gates, dzT);
            Backward.Transpose(cache.HPrev[d], hidden, t, hidden, hT);
            Gemm.NT(path, dzT, t, xT, t, gates, inSize, t, default, dWeightIh[d], inSize, 1);
            Gemm.NT(path, dzT, t, hT, t, gates, hidden, t, default, dWeightHh[d], hidden, 1);
            Backward.RowSums(path, dzT, t, gates, t, dBias[d]);
        }

        if (!dX.IsEmpty)
        {
            Gemm.NT(path, dz, 2 * gates, wihCatT, 2 * gates, t, inSize, 2 * gates, default, dX, inSize, 1);
        }
    }
}
