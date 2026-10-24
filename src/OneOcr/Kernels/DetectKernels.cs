using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace OneOcr.Kernels;

/// <summary>
/// Kernels of the page detector (docs/spec/detector-v0.md): the first 3x3 convolution, depthwise 3x3 convolution, nearest 2x
/// upsampling fused with an addition, a ReLU pass and the global average. Activations are channels-last
/// (<c>[row][column][channel]</c>) with the channel count rounded up to a multiple of 16; the padding channels hold zeros and
/// stay zero. Every output is one independent value, so the vector widths are lanes of different channels, never parts of one
/// sum: a 3x3 tap sum is the library's 16-lane dot product with taps 0..8 in their own lanes and lanes 9..15 at +0
/// (docs/spec/numerics.md), evaluated for 1, 4, 8 or 16 channels at a time. All paths are therefore bit-identical, and so is any
/// split of the rows over threads. The wrappers validate every length; the cores read through pointers.
/// </summary>
internal static unsafe class DetectKernels
{
    /// <summary>Rounds a channel count up to the multiple of 16 activations are stored with.</summary>
    public static int Padded(int channels) => (channels + 15) & ~15;

    /// <summary>
    /// First convolution: 3x3, stride 2, zero padding 1, one input channel, ReLU. Output rows
    /// <paramref name="oy0"/>..<paramref name="oy1"/> of an output <c>h/2</c> by <c>w/2</c> with <paramref name="oc"/> channels.
    /// </summary>
    /// <param name="path">The kernel path.</param>
    /// <param name="input">Normalized pixels, <c>[h][w]</c>.</param>
    /// <param name="h">Height, even.</param>
    /// <param name="w">Width, even.</param>
    /// <param name="weightsT">Weights as <c>[tap][oc]</c>, taps in (ky, kx) order.</param>
    /// <param name="bias">One value per output channel.</param>
    /// <param name="oc">Output channels, a multiple of 16.</param>
    /// <param name="oy0">First output row.</param>
    /// <param name="oy1">One past the last output row.</param>
    /// <param name="output">Receives rows <paramref name="oy0"/>.. only: index 0 is row <paramref name="oy0"/>, column 0, channel 0.</param>
    public static void Stem(
        KernelPath path, ReadOnlySpan<float> input, int h, int w, ReadOnlySpan<float> weightsT, ReadOnlySpan<float> bias, int oc,
        int oy0, int oy1, Span<float> output)
    {
        if (h < 2 || w < 2 || (h & 1) != 0 || (w & 1) != 0 || oc < 16 || oc % 16 != 0 || oy0 < 0 || oy1 < oy0 || oy1 > h / 2)
        {
            throw new ArgumentOutOfRangeException(nameof(h), "Invalid stem shape.");
        }

        if (input.Length < (long)h * w || weightsT.Length < 9L * oc || bias.Length < oc || output.Length < (long)(oy1 - oy0) * (w / 2) * oc)
        {
            throw new ArgumentException("A buffer is smaller than the convolution shape needs.");
        }

        Dispatch.Require(path);
        if (oy1 == oy0)
        {
            return;
        }

        fixed (float* pin = input, pw = weightsT, pb = bias, po = output)
        {
            switch (path)
            {
                case KernelPath.Avx512: StemCore<Lanes16>(pin, h, w, pw, pb, oc, oy0, oy1, po); break;
                case KernelPath.Avx2: StemCore<Lanes8>(pin, h, w, pw, pb, oc, oy0, oy1, po); break;
                case KernelPath.Vector128: StemCore<Lanes4>(pin, h, w, pw, pb, oc, oy0, oy1, po); break;
                default: StemCore<Lanes1>(pin, h, w, pw, pb, oc, oy0, oy1, po); break;
            }
        }
    }

    /// <summary>
    /// Depthwise 3x3 convolution with zero padding 1 and stride 1 or 2, bias, and optionally ReLU. The input is <c>h</c> by
    /// <c>w</c> (even when the stride is 2) with <paramref name="c"/> channels; output rows <paramref name="oy0"/>..<paramref name="oy1"/>
    /// of the <c>h/stride</c> by <c>w/stride</c> result are written.
    /// </summary>
    /// <param name="path">The kernel path.</param>
    /// <param name="input">Activations, <c>[h][w][c]</c>.</param>
    /// <param name="h">Input height.</param>
    /// <param name="w">Input width.</param>
    /// <param name="c">Channels, a multiple of 16.</param>
    /// <param name="stride">1 or 2.</param>
    /// <param name="weightsT">Weights as <c>[tap][c]</c>, taps in (ky, kx) order.</param>
    /// <param name="bias">One value per channel.</param>
    /// <param name="relu">Whether to apply ReLU to each output.</param>
    /// <param name="oy0">First output row.</param>
    /// <param name="oy1">One past the last output row.</param>
    /// <param name="output">Receives rows <paramref name="oy0"/>.. only: index 0 is row <paramref name="oy0"/>, column 0, channel 0.</param>
    public static void Depthwise(
        KernelPath path, ReadOnlySpan<float> input, int h, int w, int c, int stride, ReadOnlySpan<float> weightsT, ReadOnlySpan<float> bias,
        bool relu, int oy0, int oy1, Span<float> output)
    {
        if (h < 1 || w < 1 || c < 16 || c % 16 != 0 || stride < 1 || stride > 2 || (stride == 2 && (((h | w) & 1) != 0)) || oy0 < 0 || oy1 < oy0 || oy1 > h / stride)
        {
            throw new ArgumentOutOfRangeException(nameof(h), "Invalid depthwise shape.");
        }

        int ow = w / stride;
        if (input.Length < (long)h * w * c || weightsT.Length < 9L * c || bias.Length < c || output.Length < (long)(oy1 - oy0) * ow * c)
        {
            throw new ArgumentException("A buffer is smaller than the convolution shape needs.");
        }

        Dispatch.Require(path);
        if (oy1 == oy0)
        {
            return;
        }

        fixed (float* pin = input, pw = weightsT, pb = bias, po = output)
        {
            switch (path)
            {
                case KernelPath.Avx512: DepthwiseCore<Lanes16>(pin, h, w, c, stride, pw, pb, relu, oy0, oy1, po); break;
                case KernelPath.Avx2: DepthwiseCore<Lanes8>(pin, h, w, c, stride, pw, pb, relu, oy0, oy1, po); break;
                case KernelPath.Vector128: DepthwiseCore<Lanes4>(pin, h, w, c, stride, pw, pb, relu, oy0, oy1, po); break;
                default: DepthwiseCore<Lanes1>(pin, h, w, c, stride, pw, pb, relu, oy0, oy1, po); break;
            }
        }
    }

    /// <summary>
    /// Adds a nearest-neighbour 2x upsampling of <paramref name="source"/> to rows <paramref name="y0"/>..<paramref name="y1"/> of
    /// <paramref name="destination"/>: <c>dst[y][x][k] += src[y/2][x/2][k]</c>. Addition is commutative, so this is also
    /// <c>up2(src) + dst</c> bit for bit.
    /// </summary>
    /// <param name="path">The kernel path.</param>
    /// <param name="destination">Activations, <c>[h][w][c]</c>, updated in place.</param>
    /// <param name="h">Destination height, even.</param>
    /// <param name="w">Destination width, even.</param>
    /// <param name="c">Channels, a multiple of 16.</param>
    /// <param name="source">Activations, <c>[h/2][w/2][c]</c>.</param>
    /// <param name="y0">First destination row.</param>
    /// <param name="y1">One past the last destination row.</param>
    public static void AddUpsampled(KernelPath path, Span<float> destination, int h, int w, int c, ReadOnlySpan<float> source, int y0, int y1)
    {
        if (h < 2 || w < 2 || (h & 1) != 0 || (w & 1) != 0 || c < 16 || c % 16 != 0 || y0 < 0 || y1 < y0 || y1 > h)
        {
            throw new ArgumentOutOfRangeException(nameof(h), "Invalid upsampling shape.");
        }

        if (destination.Length < (long)h * w * c || source.Length < (long)(h / 2) * (w / 2) * c)
        {
            throw new ArgumentException("A buffer is smaller than the shape needs.");
        }

        Dispatch.Require(path);
        fixed (float* pd = destination, ps = source)
        {
            switch (path)
            {
                case KernelPath.Avx512: AddUpsampledCore<Lanes16>(pd, w, c, ps, y0, y1); break;
                case KernelPath.Avx2: AddUpsampledCore<Lanes8>(pd, w, c, ps, y0, y1); break;
                case KernelPath.Vector128: AddUpsampledCore<Lanes4>(pd, w, c, ps, y0, y1); break;
                default: AddUpsampledCore<Lanes1>(pd, w, c, ps, y0, y1); break;
            }
        }
    }

    /// <summary>ReLU in place (<c>v &gt; 0 ? v : 0</c>) over a span whose length is a multiple of 16.</summary>
    /// <param name="path">The kernel path.</param>
    /// <param name="values">The values.</param>
    public static void Relu(KernelPath path, Span<float> values)
    {
        if (values.Length % 16 != 0)
        {
            throw new ArgumentException("The length must be a multiple of 16.", nameof(values));
        }

        Dispatch.Require(path);
        fixed (float* p = values)
        {
            switch (path)
            {
                case KernelPath.Avx512: ReluCore<Lanes16>(p, values.Length); break;
                case KernelPath.Avx2: ReluCore<Lanes8>(p, values.Length); break;
                case KernelPath.Vector128: ReluCore<Lanes4>(p, values.Length); break;
                default: ReluCore<Lanes1>(p, values.Length); break;
            }
        }
    }

    /// <summary>
    /// Per channel, the sum over the <paramref name="pixels"/> positions in row-major order, accumulated one position after the
    /// other in fp32, divided by the position count (IEEE division).
    /// </summary>
    /// <param name="path">The kernel path.</param>
    /// <param name="input">Activations, <c>[pixels][c]</c>.</param>
    /// <param name="pixels">Number of positions, at least 1.</param>
    /// <param name="c">Channels, a multiple of 16.</param>
    /// <param name="mean">Receives <paramref name="c"/> values.</param>
    public static void ChannelMeans(KernelPath path, ReadOnlySpan<float> input, int pixels, int c, Span<float> mean)
    {
        if (pixels < 1 || c < 16 || c % 16 != 0 || input.Length < (long)pixels * c || mean.Length < c)
        {
            throw new ArgumentException("Invalid shape or buffer for the channel means.");
        }

        Dispatch.Require(path);
        fixed (float* pin = input, pm = mean)
        {
            switch (path)
            {
                case KernelPath.Avx512: MeansCore<Lanes16>(pin, pixels, c, pm); break;
                case KernelPath.Avx2: MeansCore<Lanes8>(pin, pixels, c, pm); break;
                case KernelPath.Vector128: MeansCore<Lanes4>(pin, pixels, c, pm); break;
                default: MeansCore<Lanes1>(pin, pixels, c, pm); break;
            }
        }
    }

    // The 16-partial tree of the dot product rule (numerics.md) for the nine taps p0..p8; partials 9..15 are +0.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TV Tree9<TV>(TV p0, TV p1, TV p2, TV p3, TV p4, TV p5, TV p6, TV p7, TV p8)
        where TV : struct, ILanes<TV>
    {
        TV z = TV.Zero;
        TV s0 = TV.Add(p0, p8), s1 = TV.Add(p1, z), s2 = TV.Add(p2, z), s3 = TV.Add(p3, z);
        TV s4 = TV.Add(p4, z), s5 = TV.Add(p5, z), s6 = TV.Add(p6, z), s7 = TV.Add(p7, z);
        TV t0 = TV.Add(s0, s4), t1 = TV.Add(s1, s5), t2 = TV.Add(s2, s6), t3 = TV.Add(s3, s7);
        return TV.Add(TV.Add(t0, t2), TV.Add(t1, t3));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void StemCore<TV>(float* input, int h, int w, float* weightsT, float* bias, int oc, int oy0, int oy1, float* output)
        where TV : struct, ILanes<TV>
    {
        int oh = h / 2, ow = w / 2;
        Debug.Assert(oy1 <= oh);
        float* x = stackalloc float[9];
        float* o = output;
        for (int oy = oy0; oy < oy1; oy++)
        {
            for (int ox = 0; ox < ow; ox++)
            {
                // A tap outside the image multiplies a zero: fma(w, 0, +0) is +0 for either sign of w, so +0 stands in for it.
                for (int ky = 0; ky < 3; ky++)
                {
                    int iy = (oy * 2) - 1 + ky;
                    for (int kx = 0; kx < 3; kx++)
                    {
                        int ix = (ox * 2) - 1 + kx;
                        x[(ky * 3) + kx] = (uint)iy < (uint)h && (uint)ix < (uint)w ? input[((long)iy * w) + ix] : 0f;
                    }
                }

                TV zero = TV.Zero;
                TV x0 = TV.Broadcast(x[0]), x1 = TV.Broadcast(x[1]), x2 = TV.Broadcast(x[2]), x3 = TV.Broadcast(x[3]), x4 = TV.Broadcast(x[4]);
                TV x5 = TV.Broadcast(x[5]), x6 = TV.Broadcast(x[6]), x7 = TV.Broadcast(x[7]), x8 = TV.Broadcast(x[8]);
                for (int cb = 0; cb < oc; cb += TV.Width)
                {
                    TV sum = Tree9(
                        TV.Fma(TV.Load(weightsT + cb), x0, zero), TV.Fma(TV.Load(weightsT + oc + cb), x1, zero),
                        TV.Fma(TV.Load(weightsT + (2 * oc) + cb), x2, zero), TV.Fma(TV.Load(weightsT + (3 * oc) + cb), x3, zero),
                        TV.Fma(TV.Load(weightsT + (4 * oc) + cb), x4, zero), TV.Fma(TV.Load(weightsT + (5 * oc) + cb), x5, zero),
                        TV.Fma(TV.Load(weightsT + (6 * oc) + cb), x6, zero), TV.Fma(TV.Load(weightsT + (7 * oc) + cb), x7, zero),
                        TV.Fma(TV.Load(weightsT + (8 * oc) + cb), x8, zero));
                    TV.Relu(TV.Add(sum, TV.Load(bias + cb))).Store(o + cb);
                }

                o += oc;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DepthwiseCore<TV>(
        float* input, int h, int w, int c, int stride, float* weightsT, float* bias, bool relu, int oy0, int oy1, float* output)
        where TV : struct, ILanes<TV>
    {
        int ow = w / stride;
        float** t = stackalloc float*[9];
        float* o = output;
        long rowStride = (long)w * c;
        for (int oy = oy0; oy < oy1; oy++)
        {
            for (int ox = 0; ox < ow; ox++)
            {
                // Taps outside the image contribute +0 (see StemCore) and are skipped.
                for (int ky = 0; ky < 3; ky++)
                {
                    int iy = (oy * stride) - 1 + ky;
                    for (int kx = 0; kx < 3; kx++)
                    {
                        int ix = (ox * stride) - 1 + kx;
                        t[(ky * 3) + kx] = (uint)iy < (uint)h && (uint)ix < (uint)w ? input + (iy * rowStride) + ((long)ix * c) : null;
                    }
                }

                TV zero = TV.Zero;
                for (int cb = 0; cb < c; cb += TV.Width)
                {
                    TV p0 = t[0] == null ? zero : TV.Fma(TV.Load(weightsT + cb), TV.Load(t[0] + cb), zero);
                    TV p1 = t[1] == null ? zero : TV.Fma(TV.Load(weightsT + c + cb), TV.Load(t[1] + cb), zero);
                    TV p2 = t[2] == null ? zero : TV.Fma(TV.Load(weightsT + (2 * c) + cb), TV.Load(t[2] + cb), zero);
                    TV p3 = t[3] == null ? zero : TV.Fma(TV.Load(weightsT + (3 * c) + cb), TV.Load(t[3] + cb), zero);
                    TV p4 = t[4] == null ? zero : TV.Fma(TV.Load(weightsT + (4 * c) + cb), TV.Load(t[4] + cb), zero);
                    TV p5 = t[5] == null ? zero : TV.Fma(TV.Load(weightsT + (5 * c) + cb), TV.Load(t[5] + cb), zero);
                    TV p6 = t[6] == null ? zero : TV.Fma(TV.Load(weightsT + (6 * c) + cb), TV.Load(t[6] + cb), zero);
                    TV p7 = t[7] == null ? zero : TV.Fma(TV.Load(weightsT + (7 * c) + cb), TV.Load(t[7] + cb), zero);
                    TV p8 = t[8] == null ? zero : TV.Fma(TV.Load(weightsT + (8 * c) + cb), TV.Load(t[8] + cb), zero);
                    TV v = TV.Add(Tree9(p0, p1, p2, p3, p4, p5, p6, p7, p8), TV.Load(bias + cb));
                    (relu ? TV.Relu(v) : v).Store(o + cb);
                }

                o += c;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AddUpsampledCore<TV>(float* destination, int w, int c, float* source, int y0, int y1)
        where TV : struct, ILanes<TV>
    {
        int sw = w / 2;
        for (int y = y0; y < y1; y++)
        {
            float* srcRow = source + ((long)(y / 2) * sw * c);
            float* dstRow = destination + ((long)y * w * c);
            for (int x = 0; x < w; x++)
            {
                float* s = srcRow + ((long)(x / 2) * c);
                float* d = dstRow + ((long)x * c);
                for (int cb = 0; cb < c; cb += TV.Width)
                {
                    TV.Add(TV.Load(d + cb), TV.Load(s + cb)).Store(d + cb);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ReluCore<TV>(float* p, int n)
        where TV : struct, ILanes<TV>
    {
        for (int i = 0; i < n; i += TV.Width)
        {
            TV.Relu(TV.Load(p + i)).Store(p + i);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MeansCore<TV>(float* input, int pixels, int c, float* mean)
        where TV : struct, ILanes<TV>
    {
        for (int cb = 0; cb < c; cb += TV.Width)
        {
            TV acc = TV.Zero;
            for (int p = 0; p < pixels; p++)
            {
                acc = TV.Add(acc, TV.Load(input + ((long)p * c) + cb));
            }

            acc.Store(mean + cb);
        }

        float count = pixels;
        for (int k = 0; k < c; k++)
        {
            mean[k] /= count;
        }
    }
}
