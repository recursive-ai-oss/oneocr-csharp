using OneOcr.Kernels;
using OneOcr.Recognition;

namespace OneOcr.Detection;

/// <summary>The detector's three output maps for one padded input.</summary>
internal sealed class DetectorMaps
{
    /// <summary>Text-line core logits, <c>[ProbHeight][ProbWidth]</c> (stride 2).</summary>
    public required float[] Prob { get; init; }

    public required int ProbHeight { get; init; }
    public required int ProbWidth { get; init; }

    /// <summary>Per-class region-core logits, <c>[LayoutHeight][LayoutWidth][LayoutStride]</c> (stride 8; channels beyond the classes are zero).</summary>
    public required float[] Layout { get; init; }

    public required int LayoutHeight { get; init; }
    public required int LayoutWidth { get; init; }
    public required int LayoutStride { get; init; }

    /// <summary>Logits for the page being rotated 0, 90, 180 and 270 degrees counter-clockwise from upright.</summary>
    public required float[] Orientation { get; init; }
}

/// <summary>
/// The <c>dbnet-lite-v0</c> forward pass (docs/spec/detector-v0.md). Every output value is an independent sum in the library's
/// fixed order (<see cref="Gemm"/> for the 1x1 convolutions, <see cref="DetectKernels"/> for the rest), and parallel work is
/// split by output rows, so the maps do not depend on the kernel path or the degree of parallelism.
/// </summary>
internal static class DbNetForward
{
    // Output positions per task of a block: enough rows that a task is a few thousand positions of work.
    private const int PositionsPerTask = 4096;

    /// <summary>Checks that an input of this size can run (multiples of 32, within the limits) before any buffer exists.</summary>
    public static void CheckInput(DbNetModel model, int height, int width, OcrLimits limits)
    {
        if (height < 32 || width < 32 || height % 32 != 0 || width % 32 != 0)
        {
            throw new ArgumentException("The detector input must be a multiple of 32 pixels in both directions.");
        }

        if (Math.Max(height, width) > limits.MaxDetectorSide)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxDetectorSide), limits.MaxDetectorSide, Math.Max(height, width));
        }

        // The widest buffers: the stem's output and the stride-2 head, the stride-4 maps, and the stride-8 head.
        int widest = Math.Max(Math.Max(model.StemPadded, DetectKernels.Padded(model.Text2.In)), DetectKernels.Padded(model.Text2.Out));
        CheckElements(limits, (long)(height / 2) * (width / 2) * widest);
        int fpn = DetectKernels.Padded(model.Smooth4.In);
        CheckElements(limits, (long)(height / 4) * (width / 4) * Math.Max(fpn, DetectKernels.Padded(model.Stages[0][^1].Out)));
        CheckElements(limits, (long)(height / 8) * (width / 8) * Math.Max(DetectKernels.Padded(model.LayoutBlocks[0].Out), DetectKernels.Padded(model.Stages[1][^1].Out)));
        CheckElements(limits, (long)(height / 2) * (width / 2));
    }

    /// <summary>
    /// Runs the network on a normalized <c>[height][width]</c> gray image whose sides are multiples of 32.
    /// </summary>
    /// <param name="model">The detector.</param>
    /// <param name="path">The kernel path.</param>
    /// <param name="degree">Most threads to use.</param>
    /// <param name="pool">Scratch for the activations; everything taken is given back, also when a task throws.</param>
    /// <param name="input">Normalized pixels, at least <paramref name="height"/> times <paramref name="width"/> of them.</param>
    /// <param name="height">Height, a multiple of 32.</param>
    /// <param name="width">Width, a multiple of 32.</param>
    /// <param name="limits">Bounds checked before the buffers are taken.</param>
    /// <param name="trace">Receives copies of intermediate tensors (channels-first) when not null, for tests.</param>
    /// <param name="cancellationToken">Cancels between layers.</param>
    /// <returns>The output maps.</returns>
    public static DetectorMaps Run(
        DbNetModel model, KernelPath path, int degree, BufferPool<float> pool, float[] input, int height, int width, OcrLimits limits,
        ForwardTrace? trace, CancellationToken cancellationToken)
    {
        CheckInput(model, height, width, limits);
        if (input.Length < (long)height * width)
        {
            throw new ArgumentException("The input is smaller than height times width.", nameof(input));
        }

        var arena = new Arena(pool);
        float[] Take(long elements) => arena.Take(elements, clear: true);
        void Give(float[]? a) => arena.Give(a);

        try
        {
            int h2 = height / 2, w2 = width / 2;
            int stemP = model.StemPadded;
            trace?.Add("input", input.AsSpan(0, height * width), 1, height, width);

            // Stem: 3x3 stride 2 on the single input channel.
            float[] stem = Take((long)h2 * w2 * stemP);
            Rows(h2, Math.Max(1, PositionsPerTask / w2), degree, (y0, y1) =>
                DetectKernels.Stem(path, input, height, width, model.StemWeights, model.StemBias, stemP, y0, y1, stem.AsSpan(y0 * w2 * stemP)), cancellationToken);
            TraceMap(trace, "stem", stem, h2, w2, model.StemChannels, stemP);

            // Backbone. Each stage output gets its lateral at once and is then dropped, except stage 3 (the orientation head's input).
            var lateral = new float[4][];
            float[] current = stem;
            int h = h2, w = w2;
            float[]? stage3 = null;
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < model.Stages[i].Length; j++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    float[] next = Block(model.Stages[i][j], path, degree, arena, current, h, w, j == 0 ? 2 : 1, limits, cancellationToken, out int nh, out int nw);
                    if (!ReferenceEquals(current, stem))
                    {
                        Give(current);
                    }

                    current = next;
                    h = nh;
                    w = nw;
                }

                DwBlock outBlock = model.Stages[i][^1];
                TraceMap(trace, $"stage{i}", current, h, w, outBlock.Out, outBlock.OutPadded);
                lateral[i] = Pointwise(model.Lateral[i], path, degree, arena, current, h * w, relu: false, model.Lateral[i].OutPadded, cancellationToken);
                if (i == 3)
                {
                    stage3 = current;
                }
            }

            int h3 = h, w3 = w;
            float[] orientation = Orientation(model, path, stage3!, h3 * w3);
            trace?.Add("orient", orientation, 4);
            Give(stage3);

            // Top-down: p16 = lat2 + up2(lat3); p8 = smooth8(lat1 + up2(p16)); p4 = smooth4(lat0 + up2(p8)). p16 is not smoothed.
            int fpnP = model.Lateral[0].OutPadded;
            AddUp(lateral[2], h3 * 2, w3 * 2, fpnP, lateral[3], path, degree, cancellationToken);
            Give(lateral[3]);
            AddUp(lateral[1], h3 * 4, w3 * 4, fpnP, lateral[2], path, degree, cancellationToken);
            Give(lateral[2]);
            float[] p8 = Block(model.Smooth8, path, degree, arena, lateral[1], h3 * 4, w3 * 4, 1, limits, cancellationToken, out int h8, out int w8);
            Give(lateral[1]);
            TraceMap(trace, "p8", p8, h8, w8, model.Smooth8.Out, model.Smooth8.OutPadded);
            AddUp(lateral[0], h3 * 8, w3 * 8, fpnP, p8, path, degree, cancellationToken);
            float[] p4 = Block(model.Smooth4, path, degree, arena, lateral[0], h3 * 8, w3 * 8, 1, limits, cancellationToken, out int h4, out int w4);
            Give(lateral[0]);
            TraceMap(trace, "p4", p4, h4, w4, model.Smooth4.Out, model.Smooth4.OutPadded);

            // Layout head on p8.
            float[] feature = p8;
            int fh = h8, fw = w8;
            for (int j = 0; j < model.LayoutBlocks.Length; j++)
            {
                float[] next = Block(model.LayoutBlocks[j], path, degree, arena, feature, fh, fw, 1, limits, cancellationToken, out _, out _);
                Give(feature);
                feature = next;
            }

            DwBlock lastLayout = model.LayoutBlocks[^1];
            TraceMap(trace, "layout_feat", feature, fh, fw, lastLayout.Out, lastLayout.OutPadded);
            int layoutStride = model.LayoutOut.OutPadded;
            var layout = new float[checked((long)fh * fw * layoutStride)];
            PointwiseInto(model.LayoutOut, path, degree, feature, fh * fw, relu: false, layout, layoutStride, cancellationToken);
            Give(feature);
            TraceMap(trace, "layout", layout, fh, fw, model.Classes, layoutStride);

            // Text head: stride 4 block, then the stride-2 block on up2(t4) + projection of the stem output.
            float[] t4 = Block(model.Text4, path, degree, arena, p4, h4, w4, 1, limits, cancellationToken, out _, out _);
            Give(p4);
            TraceMap(trace, "text4", t4, h4, w4, model.Text4.Out, model.Text4.OutPadded);
            float[] projection = Pointwise(model.StemProjection, path, degree, arena, stem, h2 * w2, relu: false, model.StemProjection.OutPadded, cancellationToken);
            Give(stem);
            AddUp(projection, h2, w2, model.StemProjection.OutPadded, t4, path, degree, cancellationToken);
            Give(t4);
            float[] t2 = Block(model.Text2, path, degree, arena, projection, h2, w2, 1, limits, cancellationToken, out _, out _);
            Give(projection);
            TraceMap(trace, "text2", t2, h2, w2, model.Text2.Out, model.Text2.OutPadded);
            var prob = new float[checked((long)h2 * w2)];
            PointwiseInto(model.ProbOut, path, degree, t2, h2 * w2, relu: false, prob, 1, cancellationToken);
            Give(t2);
            trace?.Add("prob", prob, 1, h2, w2);

            return new DetectorMaps
            {
                Prob = prob, ProbHeight = h2, ProbWidth = w2,
                Layout = layout, LayoutHeight = fh, LayoutWidth = fw, LayoutStride = layoutStride,
                Orientation = orientation,
            };
        }
        finally
        {
            arena.ReturnAll();
        }
    }

    private static float[] Orientation(DbNetModel model, KernelPath path, float[] stage3, int pixels)
    {
        int c = model.Stages[3][^1].Out;
        int cp = DetectKernels.Padded(c);
        var means = new float[cp];
        DetectKernels.ChannelMeans(path, stage3, pixels, cp, means);
        var logits = new float[4];
        for (int k = 0; k < 4; k++)
        {
            logits[k] = Gemm.Dot(path, model.OrientWeights.AsSpan(k * c, c), means.AsSpan(0, c)) + model.OrientBias[k];
        }

        return logits;
    }

    private static float[] Block(
        DwBlock block, KernelPath path, int degree, Arena arena, float[] source, int h, int w, int stride,
        OcrLimits limits, CancellationToken cancellationToken, out int oh, out int ow)
    {
        oh = h / stride;
        ow = w / stride;
        int cinP = block.InPadded, coutP = block.OutPadded;
        CheckElements(limits, (long)oh * ow * coutP);
        float[] destination = arena.Take((long)oh * ow * coutP, clear: true);
        int rowsPerTask = Math.Max(1, PositionsPerTask / ow);
        int outWidth = ow;
        CheckElements(limits, (long)rowsPerTask * ow * cinP);
        Rows(oh, rowsPerTask, degree, (y0, y1) =>
        {
            int rows = y1 - y0;
            float[] scratch = arena.Take((long)rows * outWidth * cinP, clear: false);
            try
            {
                DetectKernels.Depthwise(path, source, h, w, cinP, stride, block.DepthwiseWeights, block.DepthwiseBias, true, y0, y1, scratch);
                PointwiseRows(block.Pointwise, path, scratch, 0, destination, y0 * outWidth, rows * outWidth, true, coutP);
            }
            finally
            {
                arena.Give(scratch);
            }
        }, cancellationToken);
        return destination;
    }

    private static float[] Pointwise(
        PointwiseConv conv, KernelPath path, int degree, Arena arena, float[] source, int pixels, bool relu, int outStride,
        CancellationToken cancellationToken)
    {
        float[] destination = arena.Take((long)pixels * outStride, clear: true);
        PointwiseInto(conv, path, degree, source, pixels, relu, destination, outStride, cancellationToken);
        return destination;
    }

    private static void PointwiseInto(
        PointwiseConv conv, KernelPath path, int degree, float[] source, int pixels, bool relu, float[] destination, int outStride,
        CancellationToken cancellationToken)
    {
        int chunk = Math.Max(1, PositionsPerTask);
        Rows(pixels, chunk, degree, (p0, p1) =>
            PointwiseRows(conv, path, source, p0, destination, p0, p1 - p0, relu, outStride), cancellationToken);
    }

    // Positions [srcStart, srcStart + count) of `source` (channel stride InPadded) into positions [dstStart, dstStart + count) of
    // `destination` (channel stride outStride; channels beyond Out are left as they are, which is zero in cleared buffers).
    private static void PointwiseRows(
        PointwiseConv conv, KernelPath path, float[] source, int srcStart, float[] destination, int dstStart, int count, bool relu, int outStride)
    {
        int inP = conv.InPadded;
        Gemm.NT(
            path, source.AsSpan(srcStart * inP), inP, conv.Weights, inP, count, conv.Out, inP, conv.Bias,
            destination.AsSpan(dstStart * outStride), outStride, 1);
        if (relu)
        {
            DetectKernels.Relu(path, destination.AsSpan(dstStart * outStride, count * outStride));
        }
    }

    private static void AddUp(
        float[] destination, int h, int w, int c, float[] source, KernelPath path, int degree, CancellationToken cancellationToken) =>
        Rows(h, Math.Max(1, PositionsPerTask / w), degree, (y0, y1) =>
            DetectKernels.AddUpsampled(path, destination, h, w, c, source, y0, y1), cancellationToken);

    // Splits [0, count) into chunks of `chunk` and runs them on up to `degree` threads. Chunks write disjoint outputs.
    private static void Rows(int count, int chunk, int degree, Action<int, int> body, CancellationToken cancellationToken)
    {
        int tasks = (count + chunk - 1) / chunk;
        ParallelRun.For(tasks, degree, t => body(t * chunk, Math.Min(count, (t + 1) * chunk)), cancellationToken);
    }

    private static void CheckElements(OcrLimits limits, long elements)
    {
        if (elements > limits.MaxActivationElements)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxActivationElements), limits.MaxActivationElements, elements);
        }
    }

    // Channels-first copy of a channels-last map for the reference comparison.
    private static void TraceMap(ForwardTrace? trace, string name, float[] data, int h, int w, int channels, int stride)
    {
        if (trace is null)
        {
            return;
        }

        var nchw = new float[channels * h * w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    nchw[(((c * h) + y) * w) + x] = data[((((y * w) + x)) * stride) + c];
                }
            }
        }

        trace.Add(name, nchw, channels, h, w);
    }
}

/// <summary>Scratch arrays taken from a pool during one forward pass; whatever is still out when the pass ends goes back.</summary>
internal sealed class Arena(BufferPool<float> pool)
{
    private readonly List<float[]> _live = [];

    /// <summary>An array of at least <paramref name="elements"/> floats, with its first <paramref name="elements"/> zeroed when asked.</summary>
    public float[] Take(long elements, bool clear)
    {
        float[] a = pool.Rent(elements);
        if (clear)
        {
            Array.Clear(a, 0, (int)elements);
        }

        lock (_live)
        {
            _live.Add(a);
        }

        return a;
    }

    /// <summary>Gives an array back; unknown or null arrays are ignored.</summary>
    public void Give(float[]? a)
    {
        if (a is null)
        {
            return;
        }

        lock (_live)
        {
            if (!_live.Remove(a))
            {
                return;
            }
        }

        pool.Return(a);
    }

    /// <summary>Gives back everything still taken.</summary>
    public void ReturnAll()
    {
        float[][] rest;
        lock (_live)
        {
            rest = [.. _live];
            _live.Clear();
        }

        foreach (float[] a in rest)
        {
            pool.Return(a);
        }
    }
}
