using System.Buffers;
using OneOcr.Kernels;

namespace OneOcr.Recognition;

/// <summary>Copies of intermediate tensors, for parity tests against a reference implementation.</summary>
internal sealed class ForwardTrace
{
    public Dictionary<string, (int[] Shape, float[] Data)> Layers { get; } = new(StringComparer.Ordinal);

    public void Add(string name, ReadOnlySpan<float> data, params int[] shape) =>
        Layers[name] = (shape, data.ToArray());
}

/// <summary>
/// The <c>crnn-v0</c> forward pass. Every output element is an independent dot product (or element-wise function) with the
/// accumulation order fixed by <see cref="Gemm"/>, and parallel work is split by output, never within a reduction, so the
/// result does not depend on the kernel path or the degree of parallelism.
/// </summary>
internal static class CrnnForward
{
    private const int ColumnChunk = 64;

    /// <summary>Normalizes 8-bit grayscale pixels with the model's input parameters (a 256-entry table).</summary>
    public static float[] Normalize(ReadOnlySpan<byte> gray, ModelInputView input)
    {
        var table = new float[256];
        for (int p = 0; p < 256; p++)
        {
            float x = p / 255f;
            if (input.Invert)
            {
                x = 1f - x;
            }

            table[p] = (x - input.Mean) / input.Std;
        }

        var result = new float[gray.Length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = table[gray[i]];
        }

        return result;
    }

    /// <summary>Runs the network on a normalized <c>[1, H, W]</c> image; returns logits <c>[T, classes]</c> and T (0 when the line is too narrow).</summary>
    public static float[] Run(
        CrnnModel model, KernelPath path, int degree, float[] input, int width, OcrLimits limits, ForwardTrace? trace,
        out int steps, CancellationToken cancellationToken)
    {
        int height = model.Input.Height;
        trace?.Add("input", input, 1, height, width);

        // Check every buffer size before allocating anything for the line.
        int h = height, w = width;
        foreach (ConvBlock b in model.Conv)
        {
            int ch = h + (2 * b.PadH) - b.KernelH + 1;
            int cw = w + (2 * b.PadW) - b.KernelW + 1;
            if (cw < 1)
            {
                steps = 0;
                return [];
            }

            CheckElements(limits, (long)b.Out * ch * cw);
            CheckElements(limits, (long)ColumnChunk * b.KPadded);
            h = ch / b.PoolH;
            w = cw / b.PoolW;
            if (w < 1)
            {
                steps = 0;
                return [];
            }

            CheckElements(limits, (long)b.Out * h * w);
        }

        CheckElements(limits, (long)w * 4 * model.Hidden);
        CheckElements(limits, (long)w * model.Classes);

        float[] current = input;
        int c = 1;
        h = height;
        w = width;
        for (int i = 0; i < model.Conv.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ConvBlock b = model.Conv[i];
            current = ConvReluPool(b, path, degree, current, h, w, out int oh, out int ow);
            h = oh;
            w = ow;
            c = b.Out;
            trace?.Add($"conv.{i}", current, c, h, w);
        }

        // [C, 1, T] to a sequence of T vectors of C values.
        int t = w;
        var seq = new float[checked(t * c)];
        for (int ch = 0; ch < c; ch++)
        {
            for (int x = 0; x < t; x++)
            {
                seq[(x * c) + ch] = current[(ch * t) + x];
            }
        }

        int hidden = model.Hidden;
        float[] layerIn = seq;
        for (int l = 0; l < model.Lstm.Length; l++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            layerIn = BiLstm(model.Lstm[l], path, degree, layerIn, t, hidden);
            trace?.Add($"lstm.l{l}", layerIn, t, 2 * hidden);
        }

        var logits = new float[checked(t * model.Classes)];
        ParallelRows(t, degree, 1, (start, count) => Gemm.NT(
            path, layerIn.AsSpan(start * 2 * hidden), 2 * hidden, model.FcWeight, 2 * hidden, count, model.Classes, 2 * hidden,
            model.FcBias, logits.AsSpan(start * model.Classes), model.Classes, 1));
        trace?.Add("fc", logits, t, model.Classes);
        steps = t;
        return logits;
    }

    private static void CheckElements(OcrLimits limits, long elements)
    {
        if (elements > limits.MaxActivationElements)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxActivationElements), limits.MaxActivationElements, elements);
        }
    }

    // Convolution as one dot product per output value over zero-padded im2col rows, then ReLU, then max pooling.
    private static float[] ConvReluPool(ConvBlock b, KernelPath path, int degree, float[] input, int h, int w, out int outH, out int outW)
    {
        int ch = h + (2 * b.PadH) - b.KernelH + 1;
        int cw = w + (2 * b.PadW) - b.KernelW + 1;
        var raw = new float[checked(b.Out * ch * cw)];
        int chunksPerRow = (cw + ColumnChunk - 1) / ColumnChunk;
        int tasks = ch * chunksPerRow;
        long plane = (long)ch * cw;

        void Work(int id)
        {
            int y = id / chunksPerRow;
            int x0 = id % chunksPerRow * ColumnChunk;
            int n = Math.Min(ColumnChunk, cw - x0);
            float[] col = ArrayPool<float>.Shared.Rent(ColumnChunk * b.KPadded);
            try
            {
                Array.Clear(col, 0, n * b.KPadded);
                for (int i = 0; i < n; i++)
                {
                    Im2ColRow(b, input, h, w, y, x0 + i, col.AsSpan(i * b.KPadded, b.K));
                }

                // out[oc, y, x0 + i] = dot(weights[oc], col[i]) + bias[oc]
                Gemm.NT(path, col, b.KPadded, b.Weights, b.KPadded, n, b.Out, b.KPadded, b.Bias,
                    raw.AsSpan((y * cw) + x0), 1, plane);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(col);
            }
        }

        if (degree > 1 && tasks > 1)
        {
            Parallel.For(0, tasks, new ParallelOptions { MaxDegreeOfParallelism = degree }, Work);
        }
        else
        {
            for (int id = 0; id < tasks; id++)
            {
                Work(id);
            }
        }

        outH = ch / b.PoolH;
        outW = cw / b.PoolW;
        var pooled = new float[checked(b.Out * outH * outW)];
        // ReLU then max pool: starting the maximum at 0 applies the ReLU; NaN never wins a comparison.
        for (int oc = 0; oc < b.Out; oc++)
        {
            ReadOnlySpan<float> src = raw.AsSpan(oc * ch * cw, ch * cw);
            Span<float> dst = pooled.AsSpan(oc * outH * outW, outH * outW);
            for (int y = 0; y < outH; y++)
            {
                for (int x = 0; x < outW; x++)
                {
                    float m = 0f;
                    for (int dy = 0; dy < b.PoolH; dy++)
                    {
                        int rowStart = (((y * b.PoolH) + dy) * cw) + (x * b.PoolW);
                        for (int dx = 0; dx < b.PoolW; dx++)
                        {
                            float v = src[rowStart + dx];
                            if (v > m)
                            {
                                m = v;
                            }
                        }
                    }

                    dst[(y * outW) + x] = m;
                }
            }
        }

        return pooled;
    }

    // Fills one im2col row (K values in (channel, ky, kx) order) for the output position (y, x); out-of-image taps are zero.
    private static void Im2ColRow(ConvBlock b, float[] input, int h, int w, int y, int x, Span<float> dst)
    {
        int xStart = x - b.PadW;
        int index = 0;
        for (int c = 0; c < b.In; c++)
        {
            for (int ky = 0; ky < b.KernelH; ky++)
            {
                int yy = y + ky - b.PadH;
                Span<float> run = dst.Slice(index, b.KernelW);
                index += b.KernelW;
                if (yy < 0 || yy >= h)
                {
                    run.Clear();
                    continue;
                }

                int rowBase = ((c * h) + yy) * w;
                if (xStart >= 0 && xStart + b.KernelW <= w)
                {
                    input.AsSpan(rowBase + xStart, b.KernelW).CopyTo(run);
                    continue;
                }

                for (int kx = 0; kx < b.KernelW; kx++)
                {
                    int xx = xStart + kx;
                    run[kx] = xx >= 0 && xx < w ? input[rowBase + xx] : 0f;
                }
            }
        }
    }

    // One bidirectional layer: input projections for all steps at once, then the two directions' recurrences.
    private static float[] BiLstm(LstmLayer layer, KernelPath path, int degree, float[] x, int t, int hidden)
    {
        var output = new float[checked(t * 2 * hidden)];
        void Direction(LstmDirection d, bool reverse)
        {
            int gates = 4 * hidden;
            float[] proj = ArrayPool<float>.Shared.Rent(checked(t * gates));
            float[] z = new float[gates];
            float[] cell = new float[hidden];
            float[] zeros = new float[hidden];
            try
            {
                // proj[s] = W_ih * x_s + (b_ih + b_hh), one dot product per entry.
                Gemm.NT(path, x, layer.In, d.WeightIh, layer.In, t, gates, layer.In, d.Bias, proj, gates, 1);
                int column = reverse ? hidden : 0;
                for (int step = 0; step < t; step++)
                {
                    int s = reverse ? t - 1 - step : step;
                    int prev = reverse ? s + 1 : s - 1;
                    ReadOnlySpan<float> hPrev = step == 0 ? zeros : output.AsSpan((prev * 2 * hidden) + column, hidden);
                    // z = proj[s] + W_hh * h_prev
                    Gemm.NT(path, hPrev, hidden, d.WeightHh, hidden, 1, gates, hidden, proj.AsSpan(s * gates, gates), z, gates, 1);
                    Elementwise.LstmCell(path, z, cell, output.AsSpan((s * 2 * hidden) + column, hidden));
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(proj);
            }
        }

        if (degree > 1)
        {
            Parallel.Invoke(() => Direction(layer.Forward, false), () => Direction(layer.Reverse, true));
        }
        else
        {
            Direction(layer.Forward, false);
            Direction(layer.Reverse, true);
        }

        return output;
    }

    private static void ParallelRows(int rows, int degree, int minChunk, Action<int, int> body)
    {
        if (rows == 0)
        {
            return;
        }

        if (degree <= 1 || rows <= minChunk)
        {
            body(0, rows);
            return;
        }

        int chunk = Math.Max(minChunk, (rows + (degree * 2) - 1) / (degree * 2));
        int chunks = (rows + chunk - 1) / chunk;
        Parallel.For(0, chunks, new ParallelOptions { MaxDegreeOfParallelism = degree }, i =>
        {
            int start = i * chunk;
            body(start, Math.Min(chunk, rows - start));
        });
    }
}

/// <summary>Input parameters that matter for normalization.</summary>
internal readonly record struct ModelInputView(float Mean, float Std, bool Invert);
