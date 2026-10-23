using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using OneOcr.Kernels;

namespace OneOcr.Recognition;

/// <summary>Copies of intermediate tensors, for parity tests against a reference implementation.</summary>
internal sealed class ForwardTrace
{
    public Dictionary<string, (int[] Shape, float[] Data)> Layers { get; } = new(StringComparer.Ordinal);

    /// <summary>Wall time of each stage in milliseconds, by layer name.</summary>
    public Dictionary<string, double> Milliseconds { get; } = new(StringComparer.Ordinal);

    public void Add(string name, ReadOnlySpan<float> data, params int[] shape) =>
        Layers[name] = (shape, data.ToArray());
}

/// <summary>
/// The <c>crnn-v0</c> forward pass. Every output element is an independent dot product (or element-wise function) with the
/// accumulation order fixed by <see cref="Gemm"/>, and parallel work is split by output, never within a reduction, so the
/// result does not depend on the kernel path or the degree of parallelism. Activations are channels-last
/// (<c>[row][column][channel]</c>) and stored with the zero border the next convolution needs, so a window is read in place.
/// </summary>
internal static class CrnnForward
{
    // Output positions per convolution task: a multiple of both tile heights (3 and 4).
    private const int PositionChunk = 96;

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

    private static float[] Rent(long elements)
    {
        var array = ArrayPool<float>.Shared.Rent(checked((int)elements));
        Array.Clear(array, 0, (int)elements);
        return array;
    }

    private static void Return(ref float[]? array)
    {
        if (array is not null)
        {
            ArrayPool<float>.Shared.Return(array);
            array = null;
        }
    }

    /// <summary>
    /// Runs the network on a normalized <c>[1, H, W]</c> image; returns logits <c>[T, classes]</c> in a pooled array that is at
    /// least that long (the caller may give it back to <see cref="ArrayPool{T}.Shared"/>), and T (0 when the line is too narrow).
    /// </summary>
    public static float[] Run(
        CrnnModel model, KernelPath path, int degree, float[] input, int width, OcrLimits limits, ForwardTrace? trace,
        out int steps, CancellationToken cancellationToken)
    {
        int height = model.Input.Height;
        trace?.Add("input", input, 1, height, width);

        // Check every buffer size before allocating anything for the line.
        int h = height, w = width;
        ConvBlock first = model.Conv[0];
        CheckElements(limits, (long)(h + (2 * first.PadH)) * (w + (2 * first.PadW)));
        foreach (ConvBlock b in model.Conv)
        {
            int ch = h + (2 * b.PadH) - b.KernelH + 1;
            int cw = w + (2 * b.PadW) - b.KernelW + 1;
            if (cw < 1)
            {
                steps = 0;
                return [];
            }

            CheckElements(limits, (long)PositionChunk * b.PoolH * b.Out);
            CheckElements(limits, (long)PositionChunk * b.KPadded);
            h = ch / b.PoolH;
            w = cw / b.PoolW;
            if (w < 1)
            {
                steps = 0;
                return [];
            }

            CheckElements(limits, (long)b.Out * (h + (2 * b.PadH)) * (w + (2 * b.PadW)));
        }

        CheckElements(limits, (long)w * 4 * model.Hidden);
        CheckElements(limits, (long)w * 2 * model.Hidden);
        CheckElements(limits, (long)w * model.Classes);

        long stamp = System.Diagnostics.Stopwatch.GetTimestamp();
        void Lap(string name)
        {
            if (trace is not null)
            {
                long now = System.Diagnostics.Stopwatch.GetTimestamp();
                trace.Milliseconds[name] = System.Diagnostics.Stopwatch.GetElapsedTime(stamp, now).TotalMilliseconds;
                stamp = System.Diagnostics.Stopwatch.GetTimestamp();
            }
        }

        float[]? current = null, lstmIn = null, lstmOut = null;
        try
        {
            // The input image with the first block's zero border, one channel.
            int c = 1;
            h = height;
            w = width;
            {
                int hp = h + (2 * first.PadH), wp = w + (2 * first.PadW);
                current = Rent((long)hp * wp);
                for (int y = 0; y < h; y++)
                {
                    input.AsSpan(y * w, w).CopyTo(current.AsSpan(((y + first.PadH) * wp) + first.PadW, w));
                }
            }

            for (int i = 0; i < model.Conv.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ConvBlock b = model.Conv[i];
                int hp = h + (2 * b.PadH), wp = w + (2 * b.PadW);
                int ch = hp - b.KernelH + 1, cw = wp - b.KernelW + 1;
                int oh = ch / b.PoolH, ow = cw / b.PoolW;
                int nextPadH = i + 1 < model.Conv.Length ? model.Conv[i + 1].PadH : 0;
                int nextPadW = i + 1 < model.Conv.Length ? model.Conv[i + 1].PadW : 0;
                int nextWp = ow + (2 * nextPadW);
                float[] next = Rent((long)(oh + (2 * nextPadH)) * nextWp * b.Out);
                try
                {
                    ConvReluPool(b, path, degree, current!, wp, oh, ow, next, nextPadH, nextPadW, nextWp);
                }
                catch
                {
                    ArrayPool<float>.Shared.Return(next);
                    throw;
                }

                Return(ref current);
                current = next;
                h = oh;
                w = ow;
                c = b.Out;
                Lap($"conv.{i}");
                if (trace is not null)
                {
                    // Channels-first, as the reference implementations lay it out.
                    var nchw = new float[c * h * w];
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            for (int ch2 = 0; ch2 < c; ch2++)
                            {
                                nchw[(((ch2 * h) + y) * w) + x] = current[((((y + nextPadH) * nextWp) + x + nextPadW) * c) + ch2];
                            }
                        }
                    }

                    trace.Add($"conv.{i}", nchw, c, h, w);
                    stamp = System.Diagnostics.Stopwatch.GetTimestamp();
                }
            }

            // [1, T, C] with no border is already a sequence of T vectors of C values.
            int t = w;
            lstmIn = current;
            current = null;
            int hidden = model.Hidden;
            for (int l = 0; l < model.Lstm.Length; l++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                lstmOut = BiLstm(model.Lstm[l], path, degree, lstmIn!, t, hidden);
                Return(ref lstmIn);
                lstmIn = lstmOut;
                lstmOut = null;
                Lap($"lstm.l{l}");
                trace?.Add($"lstm.l{l}", lstmIn.AsSpan(0, t * 2 * hidden), t, 2 * hidden);
                if (trace is not null)
                {
                    stamp = System.Diagnostics.Stopwatch.GetTimestamp();
                }
            }

            float[] logits = ArrayPool<float>.Shared.Rent(checked(t * model.Classes));
            float[] layerIn = lstmIn!;
            ParallelRows(t, degree, 1, (start, count) => Gemm.NT(
                path, layerIn.AsSpan(start * 2 * hidden), 2 * hidden, model.FcWeight, 2 * hidden, count, model.Classes, 2 * hidden,
                model.FcBias, logits.AsSpan(start * model.Classes), model.Classes, 1));
            Lap("fc");
            trace?.Add("fc", logits.AsSpan(0, t * model.Classes), t, model.Classes);
            steps = t;
            return logits;
        }
        finally
        {
            Return(ref current);
            Return(ref lstmIn);
            Return(ref lstmOut);
        }
    }

    private static void CheckElements(OcrLimits limits, long elements)
    {
        if (elements > limits.MaxActivationElements)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxActivationElements), limits.MaxActivationElements, elements);
        }
    }

    // Convolution, ReLU and max pooling in one pass over bands: a task takes one pooled row and a run of pooled columns, computes the
    // poolH x (m * poolW) convolution outputs they cover into a small band buffer, and pools that into the interior of dst (which
    // is zeroed, with its border). Starting the maximum at 0 applies the ReLU; a NaN never wins a comparison. The band stays in cache
    // and no full-size convolution output exists. Each output is still one dot product, so the order is unchanged.
    private static void ConvReluPool(
        ConvBlock b, KernelPath path, int degree, float[] input, int wp, int oh, int ow, float[] dst, int padH, int padW, int dstWidth)
    {
        int pooledChunk = Math.Max(1, PositionChunk / b.PoolW);
        int chunksPerRow = (ow + pooledChunk - 1) / pooledChunk;
        int tasks = oh * chunksPerRow;
        bool wide = path is KernelPath.Avx2 or KernelPath.Avx512;
        bool singleChannel = wide && b.WeightsT is not null;
        bool inPlace = wide && b.RowsAreBlocks;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        void Work(int id)
        {
            int py = id / chunksPerRow;
            int px0 = id % chunksPerRow * pooledChunk;
            int m = Math.Min(pooledChunk, ow - px0);
            int n = m * b.PoolW;
            int x0 = px0 * b.PoolW;
            float[] band = ArrayPool<float>.Shared.Rent(b.PoolH * n * b.Out);
            float[]? col = inPlace || singleChannel ? null : ArrayPool<float>.Shared.Rent(n * b.KPadded);
            try
            {
                for (int dy = 0; dy < b.PoolH; dy++)
                {
                    int y = (py * b.PoolH) + dy;
                    Span<float> dstRow = band.AsSpan(dy * n * b.Out, n * b.Out);
                    if (singleChannel)
                    {
                        Gemm.ConvSingleChannel(input, wp, b.KernelH, b.KernelW, y, x0, n, b.WeightsT, b.Out, b.Bias, dstRow);
                    }
                    else if (inPlace)
                    {
                        Gemm.ConvDirect(path, input, wp, b.In, b.KernelH, b.KernelW, y, x0, n, b.Weights, b.Out, b.Bias, dstRow);
                    }
                    else
                    {
                        // Windows copied into rows (one contiguous run per kernel row), then one dot product per output.
                        Array.Clear(col!, 0, n * b.KPadded);
                        int run = b.KernelW * b.In;
                        for (int i = 0; i < n; i++)
                        {
                            for (int ky = 0; ky < b.KernelH; ky++)
                            {
                                input.AsSpan(((((y + ky) * wp) + x0 + i) * b.In), run).CopyTo(col.AsSpan((i * b.KPadded) + (ky * run), run));
                            }
                        }

                        Gemm.NT(path, col, b.KPadded, b.Weights, b.KPadded, n, b.Out, b.KPadded, b.Bias, dstRow, b.Out, 1);
                    }
                }

                for (int xi = 0; xi < m; xi++)
                {
                    Span<float> best = dst.AsSpan((((py + padH) * dstWidth) + px0 + xi + padW) * b.Out, b.Out);
                    for (int dy = 0; dy < b.PoolH; dy++)
                    {
                        for (int dx = 0; dx < b.PoolW; dx++)
                        {
                            MaxInto(best, band.AsSpan((((dy * n) + (xi * b.PoolW) + dx) * b.Out), b.Out));
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(band);
                if (col is not null)
                {
                    ArrayPool<float>.Shared.Return(col);
                }
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
    }

    private static void MaxInto(Span<float> best, ReadOnlySpan<float> values)
    {
        int i = 0;
        int lanes = Vector<float>.Count;
        for (; i + lanes <= best.Length; i += lanes)
        {
            var v = new Vector<float>(values.Slice(i, lanes));
            var m = new Vector<float>(best.Slice(i, lanes));
            Vector.ConditionalSelect(Vector.GreaterThan(v, m), v, m).CopyTo(best.Slice(i, lanes));
        }

        for (; i < best.Length; i++)
        {
            if (values[i] > best[i])
            {
                best[i] = values[i];
            }
        }
    }

    // One bidirectional layer: input projections for all steps at once, then the two directions' recurrences.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float[] BiLstm(LstmLayer layer, KernelPath path, int degree, float[] x, int t, int hidden)
    {
        float[] output = ArrayPool<float>.Shared.Rent(checked(t * 2 * hidden));
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
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

        try
        {
            if (degree > 1)
            {
                Parallel.Invoke(() => Direction(layer.Forward, false), () => Direction(layer.Reverse, true));
            }
            else
            {
                Direction(layer.Forward, false);
                Direction(layer.Reverse, true);
            }
        }
        catch
        {
            ArrayPool<float>.Shared.Return(output);
            throw;
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
