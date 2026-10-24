using System.Buffers;

namespace OneOcr.Kernels;

/// <summary>
/// Forward kernels of the document-extraction transformer (docs/spec/numerics.md, "Document extraction kernels"). Every sum of
/// products goes through <see cref="Gemm"/>, so it follows the 16-lane fused-multiply-add order on every path; the exponentials and
/// hyperbolic tangents go through the library's own <see cref="FastMath"/> via <see cref="Elementwise"/>. What remains is element-wise
/// IEEE arithmetic (add, subtract, multiply, divide, square root), which gives the same bits whatever the vector width. Nothing here
/// reduces across threads: callers split rows or heads, and each output is computed by exactly one call.
/// </summary>
internal static class DocKernels
{
    /// <summary>Layer-norm epsilon, as in the reference graph.</summary>
    public const float LayerNormEps = 1e-5f;

    /// <summary>Number of pair features per ordered pair of words.</summary>
    public const int PairFeatures = 13;

    /// <summary>Pair features padded with zeros to one whole block of 16 lanes (zero products leave every partial sum unchanged).</summary>
    public const int PairFeaturesPadded = 16;

    private const float C0 = 0.79788456f; // sqrt(2 / pi) rounded to float
    private const float C1 = 0.044715f;

    private static readonly float[] s_ones = Enumerable.Repeat(1f, 8192).ToArray();

    /// <summary>A run of ones, for sums computed as a dot product with ones (fused with a factor of one, so exactly the sum).</summary>
    public static ReadOnlySpan<float> Ones(int n) =>
        n <= s_ones.Length ? s_ones.AsSpan(0, n) : throw new ArgumentOutOfRangeException(nameof(n), "Row too long for the sum helper.");

    /// <summary><c>y = x Wᵀ + b</c> for <paramref name="rows"/> rows; <c>W</c> is <c>[outDim, inDim]</c>.</summary>
    public static void Linear(
        KernelPath path, ReadOnlySpan<float> x, int ldx, int rows, ReadOnlySpan<float> w, ReadOnlySpan<float> b, int inDim, int outDim,
        Span<float> y, int ldy) =>
        Gemm.NT(path, x, ldx, w, inDim, rows, outDim, inDim, b, y, ldy, 1);

    /// <summary>
    /// Adds embedding rows: <c>dst[i, :] += table[ids[i], :]</c> for the rows <paramref name="ids"/> names (an id outside the table is a caller bug).
    /// </summary>
    public static void AddRows(Span<float> dst, int dim, ReadOnlySpan<float> table, int tableRows, ReadOnlySpan<int> ids)
    {
        if (dst.Length < (long)ids.Length * dim || table.Length < (long)tableRows * dim)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        for (int i = 0; i < ids.Length; i++)
        {
            int id = ids[i];
            if ((uint)id >= (uint)tableRows)
            {
                throw new ArgumentOutOfRangeException(nameof(ids), "An embedding index is outside the table.");
            }

            ReadOnlySpan<float> src = table.Slice(id * dim, dim);
            Span<float> d = dst.Slice(i * dim, dim);
            for (int k = 0; k < dim; k++)
            {
                d[k] += src[k];
            }
        }
    }

    /// <summary>
    /// Character-CNN word vector for one word: gathers the embeddings of its characters, applies a kernel-3 convolution with zero padding
    /// (taps ordered (k, channel), so the weight is <c>[out, 3, charDim]</c>), ReLU, and takes the maximum over the word's valid
    /// positions (characters with a non-zero id). A word without valid positions gives zeros. Positions after the last valid one are not
    /// computed: they would be multiplied by zero.
    /// </summary>
    public static void CharPool(
        KernelPath path, ReadOnlySpan<int> ids, ReadOnlySpan<float> table, int tableRows, int charDim, ReadOnlySpan<float> convWeight,
        ReadOnlySpan<float> convBias, int convOut, Span<float> pooled)
    {
        int len = ids.Length;
        if (pooled.Length < convOut || convWeight.Length < (long)convOut * 3 * charDim || table.Length < (long)tableRows * charDim)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        pooled[..convOut].Clear();
        int last = len;
        while (last > 0 && ids[last - 1] == 0)
        {
            last--;
        }

        if (last == 0)
        {
            return;
        }

        float[] pad = ArrayPool<float>.Shared.Rent((len + 2) * charDim);
        float[] conv = ArrayPool<float>.Shared.Rent(last * convOut);
        float[] windows = ArrayPool<float>.Shared.Rent(last * 3 * charDim);
        try
        {
            Span<float> p = pad.AsSpan(0, (len + 2) * charDim);
            p[..charDim].Clear();
            p[((len + 1) * charDim)..].Clear();
            for (int i = 0; i < len; i++)
            {
                if ((uint)ids[i] >= (uint)tableRows)
                {
                    throw new ArgumentOutOfRangeException(nameof(ids), "A character index is outside the embedding table.");
                }

                table.Slice(ids[i] * charDim, charDim).CopyTo(p.Slice((i + 1) * charDim, charDim));
            }

            // Row i of the padded buffer starts the window (char i-1, char i, char i+1): the three taps are adjacent, so copying the
            // windows out gives the matrix product its operand (the product does not take overlapping rows).
            for (int i = 0; i < last; i++)
            {
                p.Slice(i * charDim, 3 * charDim).CopyTo(windows.AsSpan(i * 3 * charDim, 3 * charDim));
            }

            Gemm.NT(path, windows, 3 * charDim, convWeight, 3 * charDim, last, convOut, 3 * charDim, convBias, conv, convOut, 1);
            for (int i = 0; i < last; i++)
            {
                if (ids[i] == 0)
                {
                    continue;
                }

                ReadOnlySpan<float> row = conv.AsSpan(i * convOut, convOut);
                for (int c = 0; c < convOut; c++)
                {
                    // ReLU and running maximum in one step; the running value starts at +0.
                    pooled[c] = row[c] > pooled[c] ? row[c] : pooled[c];
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(pad);
            ArrayPool<float>.Shared.Return(conv);
            ArrayPool<float>.Shared.Return(windows);
        }
    }

    /// <summary>
    /// Layer normalization over the last axis, in place or not: mean = (sum of x) / d and variance = (sum of squared deviations) / d, both
    /// sums in the 16-lane order; <c>y = fma((x - mean) * (1 / sqrt(variance + eps)), weight, bias)</c>.
    /// </summary>
    public static void LayerNorm(
        KernelPath path, ReadOnlySpan<float> x, Span<float> y, int rows, int d, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias)
    {
        if (x.Length < (long)rows * d || y.Length < (long)rows * d || weight.Length < d || bias.Length < d)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        ReadOnlySpan<float> ones = Ones(d);
        float[] scratch = ArrayPool<float>.Shared.Rent(d);
        try
        {
            Span<float> dx = scratch.AsSpan(0, d);
            for (int r = 0; r < rows; r++)
            {
                ReadOnlySpan<float> xr = x.Slice(r * d, d);
                float mean = Gemm.Dot(path, xr, ones) / d;
                for (int i = 0; i < d; i++)
                {
                    dx[i] = xr[i] - mean;
                }

                float variance = Gemm.Dot(path, dx, dx) / d;
                float inv = 1f / MathF.Sqrt(variance + LayerNormEps);
                Span<float> yr = y.Slice(r * d, d);
                for (int i = 0; i < d; i++)
                {
                    yr[i] = MathF.FusedMultiplyAdd(dx[i] * inv, weight[i], bias[i]);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
    }

    /// <summary>
    /// GELU with the tanh form, in place: <c>0.5 x (1 + tanh(c0 * fma(c1, (x * x) * x, x)))</c>, tanh being the library's own.
    /// </summary>
    public static void Gelu(KernelPath path, Span<float> x)
    {
        float[] a = ArrayPool<float>.Shared.Rent(x.Length);
        float[] t = ArrayPool<float>.Shared.Rent(x.Length);
        try
        {
            Span<float> inner = a.AsSpan(0, x.Length);
            Span<float> th = t.AsSpan(0, x.Length);
            for (int i = 0; i < x.Length; i++)
            {
                float v = x[i];
                inner[i] = C0 * MathF.FusedMultiplyAdd(C1, (v * v) * v, v);
            }

            Elementwise.Apply(path, 2, inner, th);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = (0.5f * x[i]) * (1f + th[i]);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(a);
            ArrayPool<float>.Shared.Return(t);
        }
    }

    /// <summary>In-place ReLU (<c>v &gt; 0 ? v : 0</c>).</summary>
    public static void Relu(Span<float> x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = x[i] > 0f ? x[i] : 0f;
        }
    }

    /// <summary>
    /// Pair features of the ordered pairs (i, j) for i in [i0, i0 + rows) and all j, row (i - i0) * n + j of <paramref name="f"/> (16 columns,
    /// the last three zero). With <c>d(a) = a_j - a_i</c> on the box (x0, y0, x1, y1) and its centre, and <c>squash(v, s) = v / (s + |v|)</c>:
    /// the differences of centre x, centre y, x0, x1, y0, y1 each at s = 1 and s = 16, then the height difference at s = 1.
    /// </summary>
    public static void PairFeaturesRows(ReadOnlySpan<float> geo, int n, int i0, int rows, Span<float> f)
    {
        if (geo.Length < (long)n * 4 || f.Length < (long)rows * n * PairFeaturesPadded || i0 < 0 || i0 + rows > n)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        Span<float> a = stackalloc float[7];
        Span<float> b = stackalloc float[7];
        for (int i = i0; i < i0 + rows; i++)
        {
            Describe(geo.Slice(i * 4, 4), a);
            for (int j = 0; j < n; j++)
            {
                Describe(geo.Slice(j * 4, 4), b);
                Span<float> o = f.Slice(((i - i0) * n + j) * PairFeaturesPadded, PairFeaturesPadded);
                for (int q = 0; q < 6; q++)
                {
                    float dq = b[q] - a[q];
                    float ad = MathF.Abs(dq);
                    o[2 * q] = dq / (1f + ad);
                    o[(2 * q) + 1] = dq / (16f + ad);
                }

                float dh = b[6] - a[6];
                o[12] = dh / (1f + MathF.Abs(dh));
                o[13] = 0f;
                o[14] = 0f;
                o[15] = 0f;
            }
        }

        static void Describe(ReadOnlySpan<float> g, Span<float> v)
        {
            v[0] = (g[0] + g[2]) * 0.5f; // centre x
            v[1] = (g[1] + g[3]) * 0.5f; // centre y
            v[2] = g[0];
            v[3] = g[2];
            v[4] = g[1];
            v[5] = g[3];
            v[6] = g[3] - g[1];
        }
    }

    /// <summary>
    /// Rows [r0, r0 + count) of the attention probabilities of one head, in place on <paramref name="s"/> (<c>[count, n]</c>, holding the raw
    /// query-key dot products): <c>s = s / scale + bias</c> (two roundings), then the row's softmax: shift by the row maximum, our own exp,
    /// sum in the 16-lane order, divide.
    /// </summary>
    public static void SoftmaxRowsWithBias(KernelPath path, Span<float> s, ReadOnlySpan<float> bias, int count, int n, float scale)
    {
        if (s.Length < (long)count * n || bias.Length < (long)count * n)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        ReadOnlySpan<float> ones = Ones(n);
        float[] scratch = ArrayPool<float>.Shared.Rent(n);
        try
        {
            Span<float> shifted = scratch.AsSpan(0, n);
            for (int i = 0; i < count; i++)
            {
                Span<float> row = s.Slice(i * n, n);
                ReadOnlySpan<float> br = bias.Slice(i * n, n);
                float max = float.NegativeInfinity;
                for (int j = 0; j < n; j++)
                {
                    float v = (row[j] / scale) + br[j];
                    row[j] = v;
                    max = v > max ? v : max;
                }

                for (int j = 0; j < n; j++)
                {
                    shifted[j] = row[j] - max;
                }

                Elementwise.Apply(path, 0, shifted, row);
                float sum = Gemm.Dot(path, row, ones);
                for (int j = 0; j < n; j++)
                {
                    row[j] /= sum;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
    }

    /// <summary>Pointer logits of one relation: <c>rel = dot(q_i, k_j) / scale + bias</c> for rows [0, count), the dot product in the 16-lane order.</summary>
    public static void PointerScores(
        KernelPath path, ReadOnlySpan<float> q, int ldq, ReadOnlySpan<float> k, int ldk, int count, int n, int dim, float scale,
        ReadOnlySpan<float> bias, Span<float> rel)
    {
        Gemm.NT(path, q, ldq, k, ldk, count, n, dim, default, rel, n, 1);
        for (int i = 0; i < count * n; i++)
        {
            rel[i] = (rel[i] / scale) + bias[i];
        }
    }
}
