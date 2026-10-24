using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OneOcr.Kernels;

// Convolution entry points on channels-last activations ([row][column][channel]) with the kernel taps ordered (ky, kx, channel):
// output (y, x) is the dot product, in the fixed accumulation order, of the weight row [ky][kx][c] with the window of the padded
// input whose top-left corner is (y, x). Reading the input rows in place needs no copy: for one ky the kw * c values of a window
// are adjacent in memory.
internal static unsafe partial class Gemm
{
    /// <summary>
    /// Convolution of one output row segment by tiles that read the input in place: output positions (y, x0 .. x0 + n - 1) for all
    /// <paramref name="oc"/> output channels, written to <paramref name="output"/> as <c>[position][channel]</c>. Needs
    /// <c>kw * c</c> to be a multiple of 16 and the AVX2 or AVX-512 path; other layers use <see cref="NT"/> over copied windows.
    /// </summary>
    /// <param name="path">The AVX2 or AVX-512 path.</param>
    /// <param name="input">Padded input, <c>[padded row][padded column][c]</c>.</param>
    /// <param name="wp">Width of a padded input row in pixels.</param>
    /// <param name="c">Input channels.</param>
    /// <param name="kh">Kernel height.</param>
    /// <param name="kw">Kernel width.</param>
    /// <param name="y">Output row.</param>
    /// <param name="x0">First output column.</param>
    /// <param name="n">Number of output positions.</param>
    /// <param name="weights"><c>[oc][kh][kw][c]</c>.</param>
    /// <param name="oc">Output channels.</param>
    /// <param name="bias">One value per output channel.</param>
    /// <param name="output">Receives <c>n * oc</c> values.</param>
    /// <param name="tapRowStart">First kernel row that is read (default 0).</param>
    /// <param name="tapRows">
    /// Number of kernel rows read from <paramref name="tapRowStart"/> on; negative means all the rest. The rows left out must be rows
    /// of the zero border: a product with an exact zero adds nothing to a finite sum (<c>fma(0, w, s) = s</c>, and a partial sum that
    /// starts at +0 is never -0), so skipping those taps leaves every bit of the result as it was. Needs finite weights.
    /// </param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ConvDirect(
        KernelPath path, ReadOnlySpan<float> input, int wp, int c, int kh, int kw, int y, int x0, int n,
        ReadOnlySpan<float> weights, int oc, ReadOnlySpan<float> bias, Span<float> output, int tapRowStart = 0, int tapRows = -1)
    {
        if (path != KernelPath.Avx2 && path != KernelPath.Avx512)
        {
            throw new ArgumentException("Direct convolution runs on the AVX2 or AVX-512 path.", nameof(path));
        }

        if (wp < 1 || c < 1 || kh < 1 || kw < 1 || y < 0 || x0 < 0 || n < 1 || oc < 1 || ((long)kw * c) % 16 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(n), "Invalid convolution shape.");
        }

        if (tapRows < 0)
        {
            tapRows = kh - tapRowStart;
        }

        if (tapRowStart < 0 || tapRows < 0 || tapRowStart + tapRows > kh)
        {
            throw new ArgumentOutOfRangeException(nameof(tapRows), "The tap rows must lie within the kernel.");
        }

        long kSeg = (long)kw * c;
        long rowStride = (long)wp * c;
        // The last float read: the last position of the last tap row read, last channel.
        long lastRead = tapRows == 0 ? 0 : ((y + tapRowStart + tapRows - 1) * rowStride) + ((long)(x0 + n - 1) * c) + kSeg;
        if (lastRead > input.Length || weights.Length < (long)oc * kh * kSeg || bias.Length < oc || output.Length < (long)n * oc)
        {
            throw new ArgumentException("A buffer is smaller than the convolution shape needs.");
        }

        Dispatch.Require(path);
        int mr = path == KernelPath.Avx512 ? 4 : 3;
        int nr = path == KernelPath.Avx512 ? 4 : 2;
        fixed (float* pin = input, pw = weights, pbias = bias, pout = output)
        {
            float** ap = stackalloc float*[4];
            float** bp = stackalloc float*[4];
            float* tile = stackalloc float[16];
            float* origin = pin + (((y + tapRowStart) * rowStride) + ((long)x0 * c));
            long total = kh * kSeg;
            long first = tapRowStart * kSeg;
            for (int i0 = 0; i0 < n; i0 += mr)
            {
                int rows = Math.Min(mr, n - i0);
                for (int r = 0; r < mr; r++)
                {
                    ap[r] = origin + ((long)(i0 + Math.Min(r, rows - 1)) * c);
                }

                for (int j0 = 0; j0 < oc; j0 += nr)
                {
                    int cols = Math.Min(nr, oc - j0);
                    for (int q = 0; q < nr; q++)
                    {
                        bp[q] = pw + ((j0 + Math.Min(q, cols - 1)) * total) + first;
                    }

                    if (path == KernelPath.Avx512)
                    {
                        Tile4x4Avx512(ap, bp, tapRows, (int)kSeg, rowStride, tile);
                    }
                    else
                    {
                        Tile3x2Avx2(ap, bp, tapRows, (int)kSeg, rowStride, tile);
                    }

                    StoreTile(tile, rows, cols, pbias + j0, pout + ((long)i0 * oc) + j0, oc, 1, path);
                }
            }
        }
    }

    /// <summary>
    /// Convolution of a single-channel input whose kernel has at most 16 taps (so the whole dot product is one block of 16 lanes,
    /// every lane holding a single product). Vectorizes over output channels instead of over taps: lane k of the dot product is
    /// <c>fma(w[k], x[k], 0)</c> for the taps and 0 beyond, then the fixed tree, all element-wise. Output as for <see cref="ConvDirect"/>.
    /// Equal to <see cref="NT"/> over copied windows, bit for bit.
    /// </summary>
    /// <param name="input">Padded single-channel input, <c>[padded row][padded column]</c>.</param>
    /// <param name="wp">Width of a padded input row.</param>
    /// <param name="kh">Kernel height.</param>
    /// <param name="kw">Kernel width; <c>kh * kw</c> is at most 16.</param>
    /// <param name="y">Output row.</param>
    /// <param name="x0">First output column.</param>
    /// <param name="n">Number of output positions.</param>
    /// <param name="weightsT">Weights as <c>[tap][oc]</c> with taps in (ky, kx) order.</param>
    /// <param name="oc">Output channels, a multiple of 8.</param>
    /// <param name="bias">One value per output channel.</param>
    /// <param name="output">Receives <c>n * oc</c> values.</param>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ConvSingleChannel(
        ReadOnlySpan<float> input, int wp, int kh, int kw, int y, int x0, int n,
        ReadOnlySpan<float> weightsT, int oc, ReadOnlySpan<float> bias, Span<float> output)
    {
        int taps = kh * kw;
        if (!Avx2.IsSupported || !Fma.IsSupported)
        {
            throw new PlatformNotSupportedException("Needs AVX2 and FMA.");
        }

        if (wp < 1 || kh < 1 || kw < 1 || taps > 16 || y < 0 || x0 < 0 || n < 1 || oc < 8 || oc % 8 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(n), "Invalid convolution shape.");
        }

        long lastRead = (((long)y + kh - 1) * wp) + x0 + n - 1 + kw;
        if (lastRead > input.Length || weightsT.Length < (long)taps * oc || bias.Length < oc || output.Length < (long)n * oc)
        {
            throw new ArgumentException("A buffer is smaller than the convolution shape needs.");
        }

        fixed (float* pin = input, pw = weightsT, pbias = bias, pout = output)
        {
            Vector256<float>* p = stackalloc Vector256<float>[16];
            float* xs = stackalloc float[16];
            for (int pos = 0; pos < n; pos++)
            {
                if (taps == 9 && kh == 3)
                {
                    Taps9(pin + (((long)y * wp) + x0 + pos), wp, pw, oc, pbias, pout + ((long)pos * oc));
                    continue;
                }

                int t = 0;
                for (int ky = 0; ky < kh; ky++)
                {
                    float* row = pin + (((long)(y + ky) * wp) + x0 + pos);
                    for (int kx = 0; kx < kw; kx++)
                    {
                        xs[t++] = row[kx];
                    }
                }

                float* o = pout + ((long)pos * oc);
                for (int blk = 0; blk < oc; blk += 8)
                {
                    Vector256<float> zero = Vector256<float>.Zero;
                    for (int k = 0; k < taps; k++)
                    {
                        p[k] = Fma.MultiplyAdd(Avx.LoadVector256(pw + ((long)k * oc) + blk), Vector256.Create(xs[k]), zero);
                    }

                    for (int k = taps; k < 16; k++)
                    {
                        p[k] = zero;
                    }

                    // The order rule's tree over the 16 partials, element-wise across the 8 output channels.
                    Vector256<float> s0 = Avx.Add(p[0], p[8]), s1 = Avx.Add(p[1], p[9]), s2 = Avx.Add(p[2], p[10]), s3 = Avx.Add(p[3], p[11]);
                    Vector256<float> s4 = Avx.Add(p[4], p[12]), s5 = Avx.Add(p[5], p[13]), s6 = Avx.Add(p[6], p[14]), s7 = Avx.Add(p[7], p[15]);
                    Vector256<float> t0 = Avx.Add(s0, s4), t1 = Avx.Add(s1, s5), t2 = Avx.Add(s2, s6), t3 = Avx.Add(s3, s7);
                    Vector256<float> u0 = Avx.Add(t0, t2), u1 = Avx.Add(t1, t3);
                    Avx.Store(o + blk, Avx.Add(Avx.Add(u0, u1), Avx.LoadVector256(pbias + blk)));
                }
            }
        }
    }

    // The 3 x 3 case of the loop above with the 16 partials held in registers. Lanes 9..15 are zero, and a partial is the product
    // fma(w, x, +0), which is never -0, so adding the zero of a lane in the tree leaves it unchanged; only lane 0 + lane 8 is a real
    // addition. The remaining additions are those of the tree, in its order.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Taps9(float* row0, int wp, float* weightsT, int oc, float* bias, float* o)
    {
        float* row1 = row0 + wp, row2 = row1 + wp;
        Vector256<float> x0 = Avx.BroadcastScalarToVector256(row0), x1 = Avx.BroadcastScalarToVector256(row0 + 1), x2 = Avx.BroadcastScalarToVector256(row0 + 2);
        Vector256<float> x3 = Avx.BroadcastScalarToVector256(row1), x4 = Avx.BroadcastScalarToVector256(row1 + 1), x5 = Avx.BroadcastScalarToVector256(row1 + 2);
        Vector256<float> x6 = Avx.BroadcastScalarToVector256(row2), x7 = Avx.BroadcastScalarToVector256(row2 + 1), x8 = Avx.BroadcastScalarToVector256(row2 + 2);
        Vector256<float> zero = Vector256<float>.Zero;
        for (int blk = 0; blk < oc; blk += 8)
        {
            float* w = weightsT + blk;
            Vector256<float> p0 = Fma.MultiplyAdd(Avx.LoadVector256(w), x0, zero);
            Vector256<float> p1 = Fma.MultiplyAdd(Avx.LoadVector256(w + oc), x1, zero);
            Vector256<float> p2 = Fma.MultiplyAdd(Avx.LoadVector256(w + (2 * oc)), x2, zero);
            Vector256<float> p3 = Fma.MultiplyAdd(Avx.LoadVector256(w + (3 * oc)), x3, zero);
            Vector256<float> p4 = Fma.MultiplyAdd(Avx.LoadVector256(w + (4 * oc)), x4, zero);
            Vector256<float> p5 = Fma.MultiplyAdd(Avx.LoadVector256(w + (5 * oc)), x5, zero);
            Vector256<float> p6 = Fma.MultiplyAdd(Avx.LoadVector256(w + (6 * oc)), x6, zero);
            Vector256<float> p7 = Fma.MultiplyAdd(Avx.LoadVector256(w + (7 * oc)), x7, zero);
            Vector256<float> p8 = Fma.MultiplyAdd(Avx.LoadVector256(w + (8 * oc)), x8, zero);
            Vector256<float> t0 = Avx.Add(Avx.Add(p0, p8), p4), t1 = Avx.Add(p1, p5), t2 = Avx.Add(p2, p6), t3 = Avx.Add(p3, p7);
            Vector256<float> u0 = Avx.Add(t0, t2), u1 = Avx.Add(t1, t3);
            Avx.Store(o + blk, Avx.Add(Avx.Add(u0, u1), Avx.LoadVector256(bias + blk)));
        }
    }
}
