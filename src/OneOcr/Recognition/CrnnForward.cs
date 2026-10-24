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
        var result = new float[gray.Length];
        NormalizeInto(gray, input, result);
        return result;
    }

    /// <summary>Normalizes into the start of <paramref name="result"/>, which must hold at least <c>gray.Length</c> values.</summary>
    public static void NormalizeInto(ReadOnlySpan<byte> gray, ModelInputView input, float[] result)
    {
        Span<float> table = stackalloc float[256];
        for (int p = 0; p < 256; p++)
        {
            float x = p / 255f;
            if (input.Invert)
            {
                x = 1f - x;
            }

            table[p] = (x - input.Mean) / input.Std;
        }

        for (int i = 0; i < gray.Length; i++)
        {
            result[i] = table[gray[i]];
        }
    }

    private static float[] Rent(BufferPool<float> pool, long elements)
    {
        float[] array = pool.Rent(elements);
        Array.Clear(array, 0, (int)elements);
        return array;
    }

    private static void Return(BufferPool<float> pool, ref float[]? array)
    {
        if (array is not null)
        {
            pool.Return(array);
            array = null;
        }
    }

    /// <summary>
    /// Runs the network on a normalized <c>[1, H, W]</c> image; returns logits <c>[T, classes]</c> in a pooled array that is at
    /// least that long (the caller may give it back to <paramref name="pool"/>), and T (0 when the line is too narrow).
    /// </summary>
    public static float[] Run(
        CrnnModel model, KernelPath path, int degree, BufferPool<float> pool, float[] input, int width, OcrLimits limits, ForwardTrace? trace,
        out int steps, CancellationToken cancellationToken)
    {
        float[]? lstmIn = ConvStack(model, path, degree, pool, input, width, limits, trace, out int t, cancellationToken);
        if (lstmIn is null)
        {
            steps = 0;
            return [];
        }

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

        float[]? lstmOut = null;
        try
        {
            int hidden = model.Hidden;
            for (int l = 0; l < model.Lstm.Length; l++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                lstmOut = BiLstm(model.Lstm[l], path, degree, pool, lstmIn!, t, hidden);
                Return(pool, ref lstmIn);
                lstmIn = lstmOut;
                lstmOut = null;
                Lap($"lstm.l{l}");
                trace?.Add($"lstm.l{l}", lstmIn.AsSpan(0, t * 2 * hidden), t, 2 * hidden);
                if (trace is not null)
                {
                    stamp = System.Diagnostics.Stopwatch.GetTimestamp();
                }
            }

            float[] logits = pool.Rent(checked(t * model.Classes));
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
            Return(pool, ref lstmIn);
            Return(pool, ref lstmOut);
        }
    }

    /// <summary>
    /// The convolution stack on a normalized <c>[1, H, W]</c> image. Returns the sequence <c>[T, C]</c> (T = <paramref name="steps"/>) in
    /// a pooled array the caller gives back, or null when the line is too narrow for the network (steps 0). Checks every buffer size
    /// against the limits before allocating anything for the line.
    /// </summary>
    public static float[]? ConvStack(
        CrnnModel model, KernelPath path, int degree, BufferPool<float> pool, float[] input, int width, OcrLimits limits, ForwardTrace? trace,
        out int steps, CancellationToken cancellationToken)
    {
        int height = model.Input.Height;
        trace?.Add("input", input.AsSpan(0, height * width), 1, height, width);

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
                return null;
            }

            CheckElements(limits, (long)PositionChunk * b.PoolH * b.Out);
            CheckElements(limits, (long)PositionChunk * b.KPadded);
            h = ch / b.PoolH;
            w = cw / b.PoolW;
            if (w < 1)
            {
                steps = 0;
                return null;
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

        float[]? current = null;
        try
        {
            // The input image with the first block's zero border, one channel.
            int c = 1;
            h = height;
            w = width;
            {
                int hp = h + (2 * first.PadH), wp = w + (2 * first.PadW);
                current = Rent(pool, (long)hp * wp);
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
                float[] next = Rent(pool, (long)(oh + (2 * nextPadH)) * nextWp * b.Out);
                try
                {
                    ConvReluPool(b, path, degree, pool, current!, wp, h, oh, ow, next, nextPadH, nextPadW, nextWp);
                }
                catch
                {
                    pool.Return(next);
                    throw;
                }

                Return(pool, ref current);
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
            float[] result = current!;
            current = null;
            steps = w;
            return result;
        }
        finally
        {
            Return(pool, ref current);
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
        ConvBlock b, KernelPath path, int degree, BufferPool<float> pool, float[] input, int wp, int inputRows, int oh, int ow, float[] dst, int padH, int padW, int dstWidth)
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
            float[] band = pool.Rent(b.PoolH * n * b.Out);
            float[]? col = inPlace || singleChannel ? null : pool.Rent(n * b.KPadded);
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
                        // Kernel rows that fall on the zero border (above or below the image rows) contribute nothing; leaving them out is exact.
                        int first = 0, rowsUsed = b.KernelH;
                        if (b.FiniteWeights)
                        {
                            first = Math.Min(b.KernelH, Math.Max(0, b.PadH - y));
                            rowsUsed = Math.Min(b.KernelH, inputRows + b.PadH - y) - first;
                        }

                        Gemm.ConvDirect(path, input, wp, b.In, b.KernelH, b.KernelW, y, x0, n, b.Weights, b.Out, b.Bias, dstRow, first, Math.Max(0, rowsUsed));
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
                pool.Return(band);
                if (col is not null)
                {
                    pool.Return(col);
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
    private static float[] BiLstm(LstmLayer layer, KernelPath path, int degree, BufferPool<float> pool, float[] x, int t, int hidden)
    {
        float[] output = pool.Rent(checked(t * 2 * hidden));
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        void Direction(LstmDirection d, bool reverse)
        {
            int gates = 4 * hidden;
            float[] proj = pool.Rent(checked(t * gates));
            // One pooled block holds the gate sums, the cell state and a zero vector for the first step.
            float[] scratch = pool.Rent(gates + (2 * hidden));
            scratch.AsSpan(gates, 2 * hidden).Clear();
            Span<float> z = scratch.AsSpan(0, gates), cell = scratch.AsSpan(gates, hidden), zeros = scratch.AsSpan(gates + hidden, hidden);
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
                pool.Return(proj);
                pool.Return(scratch);
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
            pool.Return(output);
            throw;
        }

        return output;
    }

    // Steps whose input projections are computed together in a batched layer: enough rows for the register tiles, few enough to stay small.
    private const int ProjectionChunk = 16;

    /// <summary>
    /// One bidirectional layer over several lines at once. The result for each line is bit for bit what <see cref="BiLstm"/> gives it
    /// alone: every gate value is the same dot product in the same order, the only change is that the recurrent product of all lines
    /// still running at a step is one matrix product, so each weight row is loaded once for all of them instead of once per line.
    /// A line's step is independent of every other line's, so which lines share a product never matters.
    /// </summary>
    /// <remarks>
    /// <c>x</c> holds per line the input sequence <c>[t, layer.In]</c>, <c>t</c> the number of steps per line (at least 1). <c>y</c>
    /// receives per line a pooled array <c>[t, 2 * hidden]</c>; the caller gives them back, also after a failure.
    /// </remarks>
    public static void BiLstmGroup(
        LstmLayer layer, KernelPath path, BufferPool<float> pool, float[]?[] x, int[] t, float[]?[] y, int hidden, CancellationToken cancellationToken)
    {
        int n = x.Length;
        var order = new int[n];
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
        }

        // Longest first, so the lines still running at any step are a prefix of the order.
        Array.Sort(order, (a, b) => t[a] != t[b] ? t[b].CompareTo(t[a]) : a.CompareTo(b));
        int maxT = t[order[0]];
        for (int b = 0; b < n; b++)
        {
            y[b] = pool.Rent(checked(t[b] * 2 * hidden));
        }

        int gates = 4 * hidden;
        int chunk = ProjectionChunk;
        float[] projection = pool.Rent(checked((long)n * chunk * gates));
        float[] scratch = pool.Rent(checked((long)n * ((2 * hidden) + gates)));
        try
        {
            Span<float> previous = scratch.AsSpan(0, n * hidden);
            Span<float> sums = scratch.AsSpan(n * hidden, n * gates);
            Span<float> cell = scratch.AsSpan((n * hidden) + (n * gates), n * hidden);
            for (int direction = 0; direction < 2; direction++)
            {
                bool reverse = direction == 1;
                LstmDirection d = reverse ? layer.Reverse : layer.Forward;
                int column = reverse ? hidden : 0;
                cell.Clear();
                previous.Clear();
                int active = n;
                for (int u0 = 0; u0 < maxT; u0 += chunk)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int u1 = Math.Min(u0 + chunk, maxT);
                    // W_ih * x_s + (b_ih + b_hh) for the steps of this chunk, one dot product per entry. The rows of a line
                    // are its steps s in increasing order, whichever way the recurrence runs.
                    for (int j = 0; j < n && t[order[j]] > u0; j++)
                    {
                        int b = order[j];
                        int count = Math.Min(u1, t[b]) - u0;
                        int low = reverse ? t[b] - u0 - count : u0;
                        Gemm.NT(
                            path, x[b].AsSpan(low * layer.In), layer.In, d.WeightIh, layer.In, count, gates, layer.In, d.Bias,
                            projection.AsSpan(j * chunk * gates), gates, 1);
                    }

                    for (int u = u0; u < u1; u++)
                    {
                        while (active > 0 && t[order[active - 1]] <= u)
                        {
                            active--;
                        }

                        if (u > 0)
                        {
                            for (int j = 0; j < active; j++)
                            {
                                int b = order[j];
                                int before = reverse ? t[b] - u : u - 1;
                                y[b].AsSpan((before * 2 * hidden) + column, hidden).CopyTo(previous.Slice(j * hidden, hidden));
                            }
                        }

                        // z = proj[s] + W_hh * h_prev: the product first, then the one addition.
                        Gemm.NT(path, previous, hidden, d.WeightHh, hidden, active, gates, hidden, default, sums, gates, 1);
                        for (int j = 0; j < active; j++)
                        {
                            int b = order[j];
                            int count = Math.Min(u1, t[b]) - u0;
                            int low = reverse ? t[b] - u0 - count : u0;
                            int s = reverse ? t[b] - 1 - u : u;
                            Span<float> z = sums.Slice(j * gates, gates);
                            AddInPlace(z, projection.AsSpan(((j * chunk) + (s - low)) * gates, gates));
                            Elementwise.LstmCell(path, z, cell.Slice(j * hidden, hidden), y[b].AsSpan((s * 2 * hidden) + column, hidden));
                        }
                    }
                }
            }
        }
        finally
        {
            pool.Return(projection);
            pool.Return(scratch);
        }
    }

    // Element-wise sum, exactly rounded whatever the vector width.
    private static void AddInPlace(Span<float> a, ReadOnlySpan<float> b)
    {
        int i = 0;
        int lanes = Vector<float>.Count;
        for (; i + lanes <= a.Length; i += lanes)
        {
            (new Vector<float>(a.Slice(i, lanes)) + new Vector<float>(b.Slice(i, lanes))).CopyTo(a.Slice(i, lanes));
        }

        for (; i < a.Length; i++)
        {
            a[i] += b[i];
        }
    }

    /// <summary>The per-step classifier: logits <c>[t, classes]</c> in a pooled array from the sequence <c>[t, 2 * hidden]</c>.</summary>
    public static float[] Classify(CrnnModel model, KernelPath path, BufferPool<float> pool, float[] sequence, int t)
    {
        int hidden = model.Hidden;
        float[] logits = pool.Rent(checked(t * model.Classes));
        Gemm.NT(path, sequence, 2 * hidden, model.FcWeight, 2 * hidden, t, model.Classes, 2 * hidden, model.FcBias, logits, model.Classes, 1);
        return logits;
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
