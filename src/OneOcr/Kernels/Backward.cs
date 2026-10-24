using System.Diagnostics;

namespace OneOcr.Kernels;

/// <summary>
/// Building blocks of the backward pass (docs/spec/backward-numerics.md). Every sum over many terms is a dot product computed by
/// <see cref="Gemm"/> in its fixed 16-lane order, so a gradient is bit-identical on every kernel path and at every degree of
/// parallelism; the layout helpers below only move values or perform independent per-element work. This file contains no
/// pointers: the matrix products run in the checked <see cref="Gemm.NT"/> wrappers.
/// </summary>
internal static class Backward
{
    private static float[] s_ones = CreateOnes(4096);

    private static float[] CreateOnes(int n)
    {
        var ones = new float[n];
        Array.Fill(ones, 1f);
        return ones;
    }

    /// <summary>A read-only run of <paramref name="n"/> ones, used to sum a row as a dot product (so the sum has the library's order).</summary>
    public static ReadOnlySpan<float> Ones(int n)
    {
        float[] ones = Volatile.Read(ref s_ones);
        if (ones.Length < n)
        {
            ones = CreateOnes(Math.Max(n, ones.Length * 2));
            Volatile.Write(ref s_ones, ones);
        }

        return ones.AsSpan(0, n);
    }

    /// <summary>
    /// <c>dst[c * rows + r] = src[r * ld + c]</c> for r in [0, rows) and c in [0, cols): the transpose of a <c>rows x cols</c> block
    /// whose rows are <paramref name="ld"/> apart.
    /// </summary>
    public static void Transpose(ReadOnlySpan<float> src, int ld, int rows, int cols, Span<float> dst)
    {
        if (rows < 0 || cols < 0 || ld < cols)
        {
            throw new ArgumentOutOfRangeException(nameof(rows), "Dimensions must be non-negative and the stride at least the column count.");
        }

        if (rows == 0 || cols == 0)
        {
            return;
        }

        if ((((long)rows - 1) * ld) + cols > src.Length || (long)rows * cols > dst.Length)
        {
            throw new ArgumentException("A matrix is smaller than its declared shape.");
        }

        const int Block = 32;
        for (int r0 = 0; r0 < rows; r0 += Block)
        {
            int r1 = Math.Min(rows, r0 + Block);
            for (int c0 = 0; c0 < cols; c0 += Block)
            {
                int c1 = Math.Min(cols, c0 + Block);
                for (int r = r0; r < r1; r++)
                {
                    int s = (r * ld) + c0;
                    for (int c = c0; c < c1; c++, s++)
                    {
                        dst[(c * rows) + r] = src[s];
                    }
                }
            }
        }
    }

    /// <summary><c>dst[r] = sum of row r</c> (<c>rows</c> rows of <c>n</c> values, <paramref name="ld"/> apart) as a dot product with ones.</summary>
    public static void RowSums(KernelPath path, ReadOnlySpan<float> m, int ld, int rows, int n, Span<float> dst) =>
        Gemm.NT(path, m, ld, Ones(n), Math.Max(n, 1), rows, 1, n, default, dst, 1, 1);

    /// <summary>
    /// Convolution input windows as rows: row <c>y * cw + x</c> holds the zero-padded window of the <c>h x w x c</c> channels-last
    /// <paramref name="input"/> at output position (y, x), taps in (ky, kx, channel) order, followed by zeros up to
    /// <paramref name="kPadded"/>. Exactly the dot-product operands of the forward pass.
    /// </summary>
    public static void Im2Col(
        ReadOnlySpan<float> input, int h, int w, int c, int kh, int kw, int padH, int padW, int kPadded, Span<float> col)
    {
        int ch = h + (2 * padH) - kh + 1, cw = w + (2 * padW) - kw + 1;
        Debug.Assert(ch >= 1 && cw >= 1 && kPadded >= kh * kw * c);
        if ((long)h * w * c > input.Length || (long)ch * cw * kPadded > col.Length)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        col.Slice(0, ch * cw * kPadded).Clear();
        for (int y = 0; y < ch; y++)
        {
            for (int x = 0; x < cw; x++)
            {
                int row = ((y * cw) + x) * kPadded;
                for (int ky = 0; ky < kh; ky++)
                {
                    int iy = y - padH + ky;
                    if ((uint)iy >= (uint)h)
                    {
                        continue;
                    }

                    for (int kx = 0; kx < kw; kx++)
                    {
                        int ix = x - padW + kx;
                        if ((uint)ix >= (uint)w)
                        {
                            continue;
                        }

                        input.Slice(((iy * w) + ix) * c, c).CopyTo(col.Slice(row + (((ky * kw) + kx) * c), c));
                    }
                }
            }
        }
    }

    /// <summary>
    /// The inverse scatter of <see cref="Im2Col"/>, written as a gather so every input element is an independent output: element
    /// (iy, ix, channel) is the sum, starting from +0 and added in ascending (ky, kx) order with plain additions, of the window
    /// gradients of the positions whose window covers it.
    /// </summary>
    public static void Col2Im(
        ReadOnlySpan<float> dCol, int h, int w, int c, int kh, int kw, int padH, int padW, int kPadded, Span<float> dInput)
    {
        int ch = h + (2 * padH) - kh + 1, cw = w + (2 * padW) - kw + 1;
        if ((long)ch * cw * kPadded > dCol.Length || (long)h * w * c > dInput.Length)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        for (int iy = 0; iy < h; iy++)
        {
            for (int ix = 0; ix < w; ix++)
            {
                Span<float> dst = dInput.Slice(((iy * w) + ix) * c, c);
                dst.Clear();
                for (int ky = 0; ky < kh; ky++)
                {
                    int y = iy + padH - ky;
                    if ((uint)y >= (uint)ch)
                    {
                        continue;
                    }

                    for (int kx = 0; kx < kw; kx++)
                    {
                        int x = ix + padW - kx;
                        if ((uint)x >= (uint)cw)
                        {
                            continue;
                        }

                        ReadOnlySpan<float> src = dCol.Slice((((y * cw) + x) * kPadded) + (((ky * kw) + kx) * c), c);
                        for (int q = 0; q < dst.Length; q++)
                        {
                            dst[q] += src[q];
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// ReLU and max pooling of the convolution output <c>ch x cw x c</c> (channels-last) into <c>oh x ow x c</c>. Each pooled value
    /// is the maximum of its window starting from 0 (so the ReLU is applied), a value that is not strictly greater than the running
    /// maximum is ignored (the first of equal maxima wins, a NaN never wins). <paramref name="argmax"/> receives the flat index in the
    /// convolution output of the element that won, or -1 when none was positive.
    /// </summary>
    public static void ReluPool(
        ReadOnlySpan<float> convOut, int cw, int c, int poolH, int poolW, int oh, int ow, Span<float> pooled, Span<int> argmax)
    {
        int count = oh * ow * c;
        if (pooled.Length < count || argmax.Length < count || (((long)((oh * poolH) - 1) * cw) + (ow * poolW)) * c > convOut.Length)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        pooled.Slice(0, count).Clear();
        argmax.Slice(0, count).Fill(-1);
        for (int py = 0; py < oh; py++)
        {
            for (int px = 0; px < ow; px++)
            {
                int o = ((py * ow) + px) * c;
                for (int dy = 0; dy < poolH; dy++)
                {
                    for (int dx = 0; dx < poolW; dx++)
                    {
                        int pos = ((((py * poolH) + dy) * cw) + (px * poolW) + dx) * c;
                        for (int q = 0; q < c; q++)
                        {
                            float v = convOut[pos + q];
                            if (v > pooled[o + q])
                            {
                                pooled[o + q] = v;
                                argmax[o + q] = pos + q;
                            }
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Gradient of <see cref="ReluPool"/>: the upstream gradient of a pooled element goes to the convolution output element that won
    /// (nothing when none was positive, which is the ReLU's zero gradient). <paramref name="dConv"/> is cleared first.
    /// </summary>
    public static void ReluPoolBackward(ReadOnlySpan<float> dPooled, ReadOnlySpan<int> argmax, int count, Span<float> dConv)
    {
        dConv.Clear();
        if (dPooled.Length < count || argmax.Length < count)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        for (int i = 0; i < count; i++)
        {
            int target = argmax[i];
            if (target >= 0)
            {
                dConv[target] = dPooled[i];
            }
        }
    }
}
