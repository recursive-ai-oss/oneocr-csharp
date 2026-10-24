using System.Buffers;
using OneOcr.Kernels;

namespace OneOcr.Documents;

/// <summary>Raw model outputs for one page after the windows are merged: logits per word and the pointer targets.</summary>
internal sealed class DocOutputs
{
    public required int N { get; init; }

    /// <summary>Tag logits <c>[N, Tags]</c>.</summary>
    public required float[] Tags { get; init; }

    /// <summary>Role logits <c>[N, Roles]</c>.</summary>
    public required float[] Roles { get; init; }

    /// <summary>Document-type logits.</summary>
    public required float[] Doc { get; init; }

    /// <summary>Per word, the page index of the word its item pointer picks (the lowest index on a tie).</summary>
    public required int[] GroupPointer { get; init; }

    /// <summary>Per word, the page index of the word its key/value pointer picks.</summary>
    public required int[] KeyValuePointer { get; init; }
}

/// <summary>Raw outputs of one window, with the full pointer logits.</summary>
internal sealed class DocWindowOutputs
{
    public required int N { get; init; }
    public required float[] Tags { get; init; }
    public required float[] Roles { get; init; }
    public required float[] Doc { get; init; }

    /// <summary>Pointer logits <c>[2, N, N]</c>: relation 0 (item) then relation 1 (key/value).</summary>
    public required float[] Rel { get; init; }
}

/// <summary>
/// Forward pass of the <c>docextract-v0</c> model (docs/spec/docextract-v0.md). Rows, heads and words are split between threads; no sum
/// is split, so the result does not depend on the degree of parallelism, and every sum of products follows the library's 16-lane order,
/// so it does not depend on the kernel path either.
/// </summary>
internal static class DocForward
{
    private const int AttentionRows = 64;

    /// <summary>
    /// Runs a page of any length: up to <see cref="DocExtractModel.MaxWords"/> words in one window, longer pages in windows of that many
    /// words starting every three quarters of a window. Each word takes its tags, roles and pointer row from the window whose centre is
    /// nearest to it (the earlier window on a tie); pointers become page indices; the document logits are the mean of the windows' logits
    /// weighted by the number of words each window owns.
    /// </summary>
    public static DocOutputs Run(DocExtractModel m, DocFeatures f, KernelPath path, int degree, CancellationToken cancellationToken)
    {
        int n = f.Count;
        int window = m.MaxWords;
        int stride = window - (window / 4);
        var starts = new List<int> { 0 };
        while (starts[^1] + window < n)
        {
            starts.Add(starts[^1] + stride);
        }

        var tags = new float[checked(n * m.Tags)];
        var roles = new float[checked(n * m.Roles)];
        var groupPtr = new int[n];
        var kvPtr = new int[n];
        var windowDocs = new float[starts.Count][];
        var owned = new int[starts.Count];

        for (int k = 0; k < starts.Count; k++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int s = starts[k];
            int count = Math.Min(window, n - s);
            DocWindowOutputs w = RunWindow(m, f, s, count, path, degree, null, cancellationToken);
            windowDocs[k] = w.Doc;
            for (int i = 0; i < count; i++)
            {
                int word = s + i;
                if (Owner(word, starts, n, window) != k)
                {
                    continue;
                }

                owned[k]++;
                Array.Copy(w.Tags, i * m.Tags, tags, word * m.Tags, m.Tags);
                Array.Copy(w.Roles, i * m.Roles, roles, word * m.Roles, m.Roles);
                groupPtr[word] = s + ArgMax(w.Rel.AsSpan(i * count, count));
                kvPtr[word] = s + ArgMax(w.Rel.AsSpan(((count + i) * count), count));
            }
        }

        var doc = new float[m.DocTypes];
        if (starts.Count == 1)
        {
            windowDocs[0].CopyTo(doc, 0);
        }
        else
        {
            for (int c = 0; c < doc.Length; c++)
            {
                double acc = 0;
                for (int k = 0; k < starts.Count; k++)
                {
                    acc += windowDocs[k][c] * (double)owned[k];
                }

                doc[c] = (float)(acc / n);
            }
        }

        return new DocOutputs { N = n, Tags = tags, Roles = roles, Doc = doc, GroupPointer = groupPtr, KeyValuePointer = kvPtr };
    }

    /// <summary>Index of the window that owns a word: the one whose centre is nearest to the word's centre, the earlier on a tie.</summary>
    internal static int Owner(int word, List<int> starts, int n, int window)
    {
        int best = 0;
        double bestDistance = double.MaxValue;
        for (int k = 0; k < starts.Count; k++)
        {
            int end = Math.Min(starts[k] + window, n);
            double centre = (starts[k] + end) / 2.0;
            double d = Math.Abs(word + 0.5 - centre);
            if (d < bestDistance)
            {
                best = k;
                bestDistance = d;
            }
        }

        return best;
    }

    /// <summary>Index of the largest value; the lowest index on a tie. NaN never wins.</summary>
    internal static int ArgMax(ReadOnlySpan<float> v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[best] || float.IsNaN(v[best]))
            {
                best = i;
            }
        }

        return best;
    }

    /// <summary>Runs words [start, start + count) of the page as one window.</summary>
    /// <param name="m">The model.</param>
    /// <param name="f">The page's inputs.</param>
    /// <param name="start">First word.</param>
    /// <param name="count">Number of words, at most the model's window.</param>
    /// <param name="path">Kernel path.</param>
    /// <param name="degree">Threads.</param>
    /// <param name="trace">Receives named intermediates when not null: <c>char_pool</c>, <c>emb</c>, <c>pair_bias</c>, <c>block.i</c>, <c>final</c>.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    public static DocWindowOutputs RunWindow(
        DocExtractModel m, DocFeatures f, int start, int count, KernelPath path, int degree, Dictionary<string, float[]>? trace,
        CancellationToken cancellationToken)
    {
        int n = count;
        if (n < 1 || n > m.MaxWords || start < 0 || start + n > f.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "A window holds 1 to MaxWords words of the page.");
        }

        int d = m.D;
        int heads = m.Heads;
        int dh = d / heads;
        int mc = m.MaxChars;
        var rented = new List<float[]>();
        float[] Rent(long elements)
        {
            float[] a = ArrayPool<float>.Shared.Rent(checked((int)elements));
            lock (rented)
            {
                rented.Add(a);
            }

            return a;
        }

        try
        {
            // ---- word embedding ----
            float[] pool = Rent((long)n * m.CharConv);
            Rows(n, degree, 8, (r0, cnt) =>
            {
                for (int i = r0; i < r0 + cnt; i++)
                {
                    DocKernels.CharPool(
                        path, f.Chars.AsSpan((start + i) * mc, mc), m.CharEmb, m.CharEmb.Length / m.CharDim, m.CharDim, m.CharConvWeight,
                        m.CharConvBias, m.CharConv, pool.AsSpan(i * m.CharConv, m.CharConv));
                }
            });
            trace?.Add("char_pool", pool.AsSpan(0, n * m.CharConv).ToArray());
            cancellationToken.ThrowIfCancellationRequested();

            float[] x = Rent((long)n * d);
            float[] u = Rent((long)n * d);
            float[] qkv = Rent((long)n * 3 * d);
            float[] attn = Rent((long)n * d);
            DocKernels.Linear(path, pool, m.CharConv, n, m.CharProjWeight, m.CharProjBias, m.CharConv, d, x, d);

            // Position rows are added in the order of the reference graph: x0, x1, y0, y1; then the projected numeric features.
            var ids = new int[n];
            int binRows = m.PosX0.Length / d;
            foreach ((int column, float[] table) in new[] { (0, m.PosX0), (2, m.PosX1), (1, m.PosY0), (3, m.PosY1) })
            {
                for (int i = 0; i < n; i++)
                {
                    ids[i] = f.Bins[((start + i) * 4) + column];
                }

                DocKernels.AddRows(x, d, table, binRows, ids);
            }

            float[] featPad = Rent((long)n * 16);
            for (int i = 0; i < n; i++)
            {
                f.Feats.AsSpan((start + i) * 8, 8).CopyTo(featPad.AsSpan(i * 16, 8));
                featPad.AsSpan((i * 16) + 8, 8).Clear();
            }

            Rows(n, degree, 16, (r0, cnt) =>
            {
                float[] t = ArrayPool<float>.Shared.Rent(cnt * d);
                try
                {
                    DocKernels.Linear(path, featPad.AsSpan(r0 * 16), 16, cnt, m.FeatWeight, m.FeatBias, 16, d, t, d);
                    Span<float> xr = x.AsSpan(r0 * d, cnt * d);
                    for (int i = 0; i < xr.Length; i++)
                    {
                        xr[i] += t[i];
                    }

                    DocKernels.LayerNorm(path, xr, xr, cnt, d, m.EmbLnWeight, m.EmbLnBias);
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(t);
                }
            });
            trace?.Add("emb", x.AsSpan(0, n * d).ToArray());
            cancellationToken.ThrowIfCancellationRequested();

            // ---- pair bias: [pair channels, n, n] ----
            int nn = n * n; // at most MaxWordsPerWindow squared, bounded by MaxActivationElements at load
            float[] pb = Rent(m.PairChannels * nn);
            int pairRows = Math.Max(1, 4096 / n);
            Rows(n, degree, Math.Min(pairRows, n), (r0, cnt) =>
            {
                for (int a = r0; a < r0 + cnt; a += pairRows)
                {
                    int rows = Math.Min(pairRows, r0 + cnt - a);
                    float[] feat = ArrayPool<float>.Shared.Rent(rows * n * 16);
                    float[] hid = ArrayPool<float>.Shared.Rent(rows * n * m.PairHidden);
                    try
                    {
                        DocKernels.PairFeaturesRows(f.Geo.AsSpan(start * 4, n * 4), n, a, rows, feat);
                        DocKernels.Linear(path, feat, 16, rows * n, m.Pair1Weight, m.Pair1Bias, 16, m.PairHidden, hid, m.PairHidden);
                        DocKernels.Relu(hid.AsSpan(0, rows * n * m.PairHidden));
                        // Channel c of row i, column j lands at c * n * n + i * n + j.
                        Gemm.NT(
                            path, hid, m.PairHidden, m.Pair2Weight, m.PairHidden, rows * n, m.PairChannels, m.PairHidden, m.Pair2Bias,
                            pb.AsSpan(a * n), 1, nn);
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(feat);
                        ArrayPool<float>.Shared.Return(hid);
                    }
                }
            });
            if (trace is not null)
            {
                trace.Add("pair_bias", pb.AsSpan(0, (int)(m.PairChannels * nn)).ToArray());
            }

            cancellationToken.ThrowIfCancellationRequested();

            // ---- transformer blocks ----
            float scale = MathF.Sqrt(dh);
            float[] vt = Rent((long)heads * dh * n);
            int attentionChunks = (n + AttentionRows - 1) / AttentionRows;
            for (int l = 0; l < m.Layers; l++)
            {
                DocExtractModel.Block b = m.Blocks[l];
                int layer = l;
                Rows(n, degree, 16, (r0, cnt) =>
                {
                    DocKernels.LayerNorm(path, x.AsSpan(r0 * d, cnt * d), u.AsSpan(r0 * d, cnt * d), cnt, d, b.Ln1Weight, b.Ln1Bias);
                    DocKernels.Linear(path, u.AsSpan(r0 * d), d, cnt, b.QkvWeight, b.QkvBias, d, 3 * d, qkv.AsSpan(r0 * 3 * d), 3 * d);
                });

                // V transposed per head, so that the probability-weighted sum is a dot product over contiguous rows.
                for (int h = 0; h < heads; h++)
                {
                    for (int c = 0; c < dh; c++)
                    {
                        Span<float> dst = vt.AsSpan(((h * dh) + c) * n, n);
                        int src = (2 * d) + (h * dh) + c;
                        for (int j = 0; j < n; j++)
                        {
                            dst[j] = qkv[(j * 3 * d) + src];
                        }
                    }
                }

                ParallelRun.For(heads * attentionChunks, degree, task =>
                {
                    int h = task / attentionChunks;
                    int r0 = task % attentionChunks * AttentionRows;
                    int cnt = Math.Min(AttentionRows, n - r0);
                    float[] s = ArrayPool<float>.Shared.Rent(cnt * n);
                    try
                    {
                        Gemm.NT(path, qkv.AsSpan((r0 * 3 * d) + (h * dh)), 3 * d, qkv.AsSpan(d + (h * dh)), 3 * d, cnt, n, dh, default, s, n, 1);
                        DocKernels.SoftmaxRowsWithBias(path, s.AsSpan(0, cnt * n), pb.AsSpan((((layer * heads) + h) * nn) + (r0 * n), cnt * n), cnt, n, scale);
                        Gemm.NT(path, s, n, vt.AsSpan(h * dh * n), n, cnt, dh, n, default, attn.AsSpan((r0 * d) + (h * dh)), d, 1);
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(s);
                    }
                }, cancellationToken);

                Rows(n, degree, 16, (r0, cnt) =>
                {
                    float[] t = ArrayPool<float>.Shared.Rent(cnt * d);
                    float[] hid = ArrayPool<float>.Shared.Rent(cnt * m.Ffn);
                    try
                    {
                        DocKernels.Linear(path, attn.AsSpan(r0 * d), d, cnt, b.ProjWeight, b.ProjBias, d, d, t, d);
                        Span<float> xr = x.AsSpan(r0 * d, cnt * d);
                        for (int i = 0; i < xr.Length; i++)
                        {
                            xr[i] += t[i];
                        }

                        DocKernels.LayerNorm(path, xr, u.AsSpan(r0 * d, cnt * d), cnt, d, b.Ln2Weight, b.Ln2Bias);
                        DocKernels.Linear(path, u.AsSpan(r0 * d), d, cnt, b.Fc1Weight, b.Fc1Bias, d, m.Ffn, hid, m.Ffn);
                        DocKernels.Gelu(path, hid.AsSpan(0, cnt * m.Ffn));
                        DocKernels.Linear(path, hid, m.Ffn, cnt, b.Fc2Weight, b.Fc2Bias, m.Ffn, d, t, d);
                        for (int i = 0; i < xr.Length; i++)
                        {
                            xr[i] += t[i];
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(t);
                        ArrayPool<float>.Shared.Return(hid);
                    }
                });
                trace?.Add($"block.{l}", x.AsSpan(0, n * d).ToArray());
                cancellationToken.ThrowIfCancellationRequested();
            }

            // ---- heads ----
            DocKernels.LayerNorm(path, x.AsSpan(0, n * d), x.AsSpan(0, n * d), n, d, m.FinalLnWeight, m.FinalLnBias);
            trace?.Add("final", x.AsSpan(0, n * d).ToArray());
            var tags = new float[n * m.Tags];
            var roles = new float[n * m.Roles];
            int r2 = 2 * m.RelDim;
            float[] relQ = Rent((long)n * r2);
            float[] relK = Rent((long)n * r2);
            Rows(n, degree, 16, (r0, cnt) =>
            {
                DocKernels.Linear(path, x.AsSpan(r0 * d), d, cnt, m.TagWeight, m.TagBias, d, m.Tags, tags.AsSpan(r0 * m.Tags), m.Tags);
                DocKernels.Linear(path, x.AsSpan(r0 * d), d, cnt, m.RoleWeight, m.RoleBias, d, m.Roles, roles.AsSpan(r0 * m.Roles), m.Roles);
                DocKernels.Linear(path, x.AsSpan(r0 * d), d, cnt, m.RelQWeight, m.RelQBias, d, r2, relQ.AsSpan(r0 * r2), r2);
                DocKernels.Linear(path, x.AsSpan(r0 * d), d, cnt, m.RelKWeight, m.RelKBias, d, r2, relK.AsSpan(r0 * r2), r2);
            });

            // Document type: the words' mean (summed in word order in double precision, divided by the word count), through the head.
            var mean = new float[d];
            for (int c = 0; c < d; c++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    sum += x[(i * d) + c];
                }

                mean[c] = (float)(sum / n);
            }

            var doc = new float[m.DocTypes];
            DocKernels.Linear(path, mean, d, 1, m.DocWeight, m.DocBias, d, m.DocTypes, doc, m.DocTypes);

            // Pointers: rel[r, i, j] = dot(q[i, r], k[j, r]) / sqrt(rel_dim) + pair_bias[layers * heads + r, i, j].
            var rel = new float[2 * nn];
            float relScale = MathF.Sqrt(m.RelDim);
            for (int r = 0; r < 2; r++)
            {
                int rr = r;
                Rows(n, degree, 16, (r0, cnt) =>
                {
                    DocKernels.PointerScores(
                        path, relQ.AsSpan((r0 * r2) + (rr * m.RelDim)), r2, relK.AsSpan(rr * m.RelDim), r2, cnt, n, m.RelDim, relScale,
                        pb.AsSpan(((m.Layers * heads) + rr) * nn + (r0 * n), cnt * n), rel.AsSpan((rr * nn) + (r0 * n), cnt * n));
                });
            }

            return new DocWindowOutputs { N = n, Tags = tags, Roles = roles, Doc = doc, Rel = rel };
        }
        finally
        {
            foreach (float[] a in rented)
            {
                ArrayPool<float>.Shared.Return(a);
            }
        }
    }

    private static void Rows(int rows, int degree, int minChunk, Action<int, int> body)
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
        ParallelRun.For(chunks, degree, i =>
        {
            int start = i * chunk;
            body(start, Math.Min(chunk, rows - start));
        }, CancellationToken.None);
    }
}
