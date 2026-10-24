using System.Buffers;

namespace OneOcr.Kernels;

/// <summary>
/// Backward kernels of the document-extraction transformer (docs/spec/docextract-backward.md), the adjoints of <see cref="DocKernels"/>.
/// Like every backward kernel of the library, each sum over many terms is a 16-lane dot product (<see cref="Gemm"/>), and everything
/// else is element-wise IEEE arithmetic with every operation rounded on its own (no fused multiply-add outside a dot product), so a
/// gradient is bit-identical on every kernel path and at every degree of parallelism. No pointers here: the products run in the
/// checked <see cref="Gemm.NT"/> wrappers.
/// </summary>
internal static class DocBackward
{
    private const float C0 = 0.79788456f; // sqrt(2 / pi) rounded to float, as in the forward kernel
    private const float C1 = 0.044715f;

    /// <summary>
    /// Layer normalization forward, saving what its backward pass needs: the normalized rows <paramref name="xhat"/> (<c>(x - mean) * inv</c>,
    /// rounded as in <see cref="DocKernels.LayerNorm"/> before the scale) and <paramref name="inv"/> per row. <c>y</c> is bit-identical to
    /// <see cref="DocKernels.LayerNorm"/>. <paramref name="y"/> may alias <paramref name="x"/>.
    /// </summary>
    public static void LayerNormForward(
        KernelPath path, ReadOnlySpan<float> x, Span<float> y, int rows, int d, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
        Span<float> xhat, Span<float> inv)
    {
        if (x.Length < (long)rows * d || y.Length < (long)rows * d || xhat.Length < (long)rows * d || inv.Length < rows || weight.Length < d || bias.Length < d)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        ReadOnlySpan<float> ones = DocKernels.Ones(d);
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> xr = x.Slice(r * d, d);
            float mean = Gemm.Dot(path, xr, ones) / d;
            Span<float> xh = xhat.Slice(r * d, d);
            for (int i = 0; i < d; i++)
            {
                xh[i] = xr[i] - mean;
            }

            float variance = Gemm.Dot(path, xh, xh) / d;
            float iv = 1f / MathF.Sqrt(variance + DocKernels.LayerNormEps);
            inv[r] = iv;
            Span<float> yr = y.Slice(r * d, d);
            for (int i = 0; i < d; i++)
            {
                float h = xh[i] * iv;
                xh[i] = h;
                yr[i] = MathF.FusedMultiplyAdd(h, weight[i], bias[i]);
            }
        }
    }

    /// <summary>
    /// Layer normalization backward (<c>y = xhat * w + b</c>, <c>xhat = (x - mean) * inv</c>). Per row, with <c>g = dY * w</c>:
    /// <c>dX = inv * ((g - mean(g)) - xhat * mean(g * xhat))</c>, the two means 16-lane dot products divided by <c>d</c>; the scale and shift
    /// gradients are the column sums of <c>dY * xhat</c> and of <c>dY</c> over the rows, as dot products with ones along the row axis.
    /// </summary>
    public static void LayerNormBackward(
        KernelPath path, ReadOnlySpan<float> dY, ReadOnlySpan<float> xhat, ReadOnlySpan<float> inv, ReadOnlySpan<float> weight, int rows, int d,
        Span<float> dX, Span<float> dWeight, Span<float> dBias)
    {
        if (dY.Length < (long)rows * d || xhat.Length < (long)rows * d || inv.Length < rows || weight.Length < d || dWeight.Length < d || dBias.Length < d
            || (!dX.IsEmpty && dX.Length < (long)rows * d))
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        ReadOnlySpan<float> ones = DocKernels.Ones(d);
        float[] gBuf = ArrayPool<float>.Shared.Rent(d);
        float[] prod = ArrayPool<float>.Shared.Rent(rows * d);
        float[] prodT = ArrayPool<float>.Shared.Rent(rows * d);
        float[] dyT = ArrayPool<float>.Shared.Rent(rows * d);
        try
        {
            Span<float> g = gBuf.AsSpan(0, d);
            for (int r = 0; r < rows; r++)
            {
                ReadOnlySpan<float> dy = dY.Slice(r * d, d);
                ReadOnlySpan<float> xh = xhat.Slice(r * d, d);
                Span<float> p = prod.AsSpan(r * d, d);
                for (int i = 0; i < d; i++)
                {
                    g[i] = dy[i] * weight[i];
                    p[i] = dy[i] * xh[i];
                }

                if (!dX.IsEmpty)
                {
                    float a = Gemm.Dot(path, g, ones) / d;
                    float b = Gemm.Dot(path, g, xh) / d;
                    float iv = inv[r];
                    Span<float> dx = dX.Slice(r * d, d);
                    for (int i = 0; i < d; i++)
                    {
                        dx[i] = iv * ((g[i] - a) - (xh[i] * b));
                    }
                }
            }

            Backward.Transpose(prod.AsSpan(0, rows * d), d, rows, d, prodT);
            Backward.RowSums(path, prodT, rows, d, rows, dWeight);
            Backward.Transpose(dY, d, rows, d, dyT);
            Backward.RowSums(path, dyT, rows, d, rows, dBias);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(gBuf);
            ArrayPool<float>.Shared.Return(prod);
            ArrayPool<float>.Shared.Return(prodT);
            ArrayPool<float>.Shared.Return(dyT);
        }
    }

    /// <summary>
    /// GELU with the tanh form out of place, keeping tanh of the inner term for the backward pass. <paramref name="y"/> is bit-identical to
    /// what <see cref="DocKernels.Gelu"/> leaves in place of <paramref name="x"/>.
    /// </summary>
    public static void GeluForward(KernelPath path, ReadOnlySpan<float> x, Span<float> y, Span<float> th)
    {
        if (y.Length < x.Length || th.Length < x.Length)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        float[] a = ArrayPool<float>.Shared.Rent(x.Length);
        try
        {
            Span<float> inner = a.AsSpan(0, x.Length);
            for (int i = 0; i < x.Length; i++)
            {
                float v = x[i];
                inner[i] = C0 * MathF.FusedMultiplyAdd(C1, (v * v) * v, v);
            }

            Elementwise.Apply(path, 2, inner, th[..x.Length]);
            for (int i = 0; i < x.Length; i++)
            {
                y[i] = (0.5f * x[i]) * (1f + th[i]);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(a);
        }
    }

    /// <summary>
    /// GELU backward: <c>dX = dY * (0.5 (1 + t) + 0.5 x (1 - t²) c0 (1 + 3 c1 x²))</c> for <c>t = tanh(c0 (x + c1 x³))</c> saved by the
    /// forward pass; the derivative of the function as the forward pass computes it (with the library's tanh), written as separately
    /// rounded operations.
    /// </summary>
    public static void GeluBackward(ReadOnlySpan<float> dY, ReadOnlySpan<float> x, ReadOnlySpan<float> th, Span<float> dX)
    {
        if (dX.Length < dY.Length || x.Length < dY.Length || th.Length < dY.Length)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        const float ThreeC1 = 3f * C1;
        for (int i = 0; i < dY.Length; i++)
        {
            float v = x[i], t = th[i];
            float sech2 = 1f - (t * t);
            float dInner = C0 * (1f + (ThreeC1 * (v * v)));
            float slope = (0.5f * (1f + t)) + ((0.5f * v) * (sech2 * dInner));
            dX[i] = dY[i] * slope;
        }
    }

    /// <summary>
    /// Softmax backward for the rows of <see cref="DocKernels.SoftmaxRowsWithBias"/> (<c>s = raw / scale + bias</c>, <c>p = softmax(s)</c>):
    /// per row <c>ds = p * (dP - dot(p, dP))</c> (the dot a 16-lane product); <paramref name="dBias"/> receives <c>ds</c> (the bias is added
    /// once) and <paramref name="dRaw"/> receives <c>ds / scale</c>, the gradient of the raw query-key products.
    /// </summary>
    public static void SoftmaxRowsBackward(
        KernelPath path, ReadOnlySpan<float> p, ReadOnlySpan<float> dP, int rows, int n, float scale, Span<float> dRaw, Span<float> dBias)
    {
        if (p.Length < (long)rows * n || dP.Length < (long)rows * n || dRaw.Length < (long)rows * n || dBias.Length < (long)rows * n)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        for (int i = 0; i < rows; i++)
        {
            ReadOnlySpan<float> pr = p.Slice(i * n, n);
            ReadOnlySpan<float> dpr = dP.Slice(i * n, n);
            float dot = Gemm.Dot(path, pr, dpr);
            Span<float> raw = dRaw.Slice(i * n, n);
            Span<float> bias = dBias.Slice(i * n, n);
            for (int j = 0; j < n; j++)
            {
                float ds = pr[j] * (dpr[j] - dot);
                bias[j] = ds;
                raw[j] = ds / scale;
            }
        }
    }

    /// <summary>
    /// Adjoint of <c>scores = q kᵀ</c> for <paramref name="n"/> rows of <paramref name="dim"/>-wide <c>q</c> and <c>k</c> (row strides
    /// <paramref name="ldq"/> and <paramref name="ldk"/>): <c>dQ[i][c] = Σ_j dS[i][j] k[j][c]</c> and <c>dK[j][c] = Σ_i dS[i][j] q[i][c]</c>,
    /// 16-lane dot products over j and over i. The results go to <paramref name="dQ"/> and <paramref name="dK"/> at the given strides
    /// (overwritten).
    /// </summary>
    public static void ScoresBackward(
        KernelPath path, ReadOnlySpan<float> dS, ReadOnlySpan<float> q, int ldq, ReadOnlySpan<float> k, int ldk, int n, int dim,
        Span<float> dQ, int lddq, Span<float> dK, int lddk)
    {
        if (dS.Length < (long)n * n)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        float[] kT = ArrayPool<float>.Shared.Rent(n * dim);
        float[] qT = ArrayPool<float>.Shared.Rent(n * dim);
        float[] dST = ArrayPool<float>.Shared.Rent(n * n);
        try
        {
            Backward.Transpose(k, ldk, n, dim, kT);
            Backward.Transpose(q, ldq, n, dim, qT);
            Backward.Transpose(dS, n, n, n, dST);
            Gemm.NT(path, dS, n, kT, n, n, dim, n, default, dQ, lddq, 1);
            Gemm.NT(path, dST, n, qT, n, n, dim, n, default, dK, lddk, 1);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(kT);
            ArrayPool<float>.Shared.Return(qT);
            ArrayPool<float>.Shared.Return(dST);
        }
    }

    /// <summary>
    /// Adjoint of the probability-weighted sum <c>A = P V</c> of one head: <c>dP[i][j] = dot(dA[i], V[j])</c> over the head width and
    /// <c>dV[j][c] = Σ_i P[i][j] dA[i][c]</c> (a 16-lane dot over i). All matrices are <paramref name="n"/> rows; <paramref name="v"/>
    /// and <paramref name="dA"/> are <paramref name="dim"/> wide at strides <paramref name="ldv"/> and <paramref name="ldda"/>.
    /// </summary>
    public static void ContextBackward(
        KernelPath path, ReadOnlySpan<float> p, ReadOnlySpan<float> dA, int ldda, ReadOnlySpan<float> v, int ldv, int n, int dim,
        Span<float> dP, Span<float> dV, int lddv)
    {
        if (p.Length < (long)n * n || dP.Length < (long)n * n)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        float[] pT = ArrayPool<float>.Shared.Rent(n * n);
        float[] dAT = ArrayPool<float>.Shared.Rent(n * dim);
        try
        {
            Backward.Transpose(p, n, n, n, pT);
            Backward.Transpose(dA, ldda, n, dim, dAT);
            Gemm.NT(path, dA, ldda, v, ldv, n, n, dim, default, dP, n, 1);
            Gemm.NT(path, pT, n, dAT, n, n, dim, n, default, dV, lddv, 1);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(pT);
            ArrayPool<float>.Shared.Return(dAT);
        }
    }

    /// <summary>
    /// Adjoint of the pair-bias MLP <c>pb = pair2(relu(pair1(features)))</c> over <c>pairs = n²</c> ordered pairs, for the weights only (the
    /// features are data). <paramref name="dPb"/> is the gradient of the bias in its forward layout <c>[channels][pairs]</c>;
    /// <paramref name="hidden"/> the post-ReLU hidden activations <c>[pairs][hiddenWidth]</c>; <paramref name="features"/> the padded
    /// features <c>[pairs][16]</c>; <paramref name="pair2WeightT"/> the second layer's weight transposed to <c>[hiddenWidth][channels]</c>.
    /// Writes <c>dW2 [channels][hiddenWidth]</c>, <c>db2</c>, <c>dW1 [hiddenWidth][16]</c> and <c>db1</c>, every sum over pairs a 16-lane
    /// dot product in pair order (row-major over (i, j)).
    /// </summary>
    public static void PairMlpBackward(
        KernelPath path, ReadOnlySpan<float> dPb, ReadOnlySpan<float> hidden, ReadOnlySpan<float> features, ReadOnlySpan<float> pair2WeightT,
        int pairs, int hiddenWidth, int channels, Span<float> dW1, Span<float> db1, Span<float> dW2, Span<float> db2)
    {
        const int F = DocKernels.PairFeaturesPadded;
        if (dPb.Length < (long)channels * pairs || hidden.Length < (long)pairs * hiddenWidth || features.Length < (long)pairs * F
            || dW1.Length < hiddenWidth * F || dW2.Length < channels * hiddenWidth)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        float[] hT = ArrayPool<float>.Shared.Rent(pairs * hiddenWidth);
        float[] dPbT = ArrayPool<float>.Shared.Rent(pairs * channels);
        float[] dHid = ArrayPool<float>.Shared.Rent(pairs * hiddenWidth);
        float[] dHidT = ArrayPool<float>.Shared.Rent(pairs * hiddenWidth);
        float[] fT = ArrayPool<float>.Shared.Rent(pairs * F);
        try
        {
            Backward.RowSums(path, dPb, pairs, channels, pairs, db2);
            Backward.Transpose(hidden, hiddenWidth, pairs, hiddenWidth, hT);
            Gemm.NT(path, dPb, pairs, hT, pairs, channels, hiddenWidth, pairs, default, dW2, hiddenWidth, 1);

            Backward.Transpose(dPb, pairs, channels, pairs, dPbT);
            Gemm.NT(path, dPbT, channels, pair2WeightT, channels, pairs, hiddenWidth, channels, default, dHid, hiddenWidth, 1);
            for (int i = 0; i < pairs * hiddenWidth; i++)
            {
                // The ReLU passes the gradient where its output is positive.
                if (!(hidden[i] > 0f))
                {
                    dHid[i] = 0f;
                }
            }

            Backward.Transpose(dHid, hiddenWidth, pairs, hiddenWidth, dHidT);
            Backward.RowSums(path, dHidT, pairs, hiddenWidth, pairs, db1);
            Backward.Transpose(features, F, pairs, F, fT);
            Gemm.NT(path, dHidT, pairs, fT, pairs, hiddenWidth, F, pairs, default, dW1, F, 1);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(hT);
            ArrayPool<float>.Shared.Return(dPbT);
            ArrayPool<float>.Shared.Return(dHid);
            ArrayPool<float>.Shared.Return(dHidT);
            ArrayPool<float>.Shared.Return(fT);
        }
    }

    /// <summary>
    /// Scatter-add of row gradients into an embedding table gradient: <c>dTable[ids[i]] += dRows[i]</c>, rows visited in order with plain
    /// additions on one thread, so the order of the additions into a table row is the order of the rows. <paramref name="dTable"/> is cleared
    /// first. Rows whose id is <paramref name="skipId"/> (or any negative id) are not added.
    /// </summary>
    public static void EmbeddingBackward(ReadOnlySpan<float> dRows, int ldRows, ReadOnlySpan<int> ids, int dim, int tableRows, Span<float> dTable, int skipId = -1)
    {
        if (dTable.Length < (long)tableRows * dim)
        {
            throw new ArgumentException("A buffer is smaller than its declared shape.");
        }

        dTable[..(tableRows * dim)].Clear();
        for (int i = 0; i < ids.Length; i++)
        {
            int id = ids[i];
            if (id < 0 || id == skipId)
            {
                continue;
            }

            if ((uint)id >= (uint)tableRows)
            {
                throw new ArgumentOutOfRangeException(nameof(ids), "An embedding index is outside the table.");
            }

            ReadOnlySpan<float> src = dRows.Slice(i * ldRows, dim);
            Span<float> dst = dTable.Slice(id * dim, dim);
            for (int k = 0; k < dim; k++)
            {
                dst[k] += src[k];
            }
        }
    }
}
