using OneOcr.Documents;
using OneOcr.Kernels;
using OneOcr.Recognition;

namespace OneOcr.Training;

/// <summary>What the model should predict for one page. A negative entry ignores that word (masked out of the loss).</summary>
/// <param name="Tags">BIO tag per word (0 = O, <c>1 + 2c</c> = B-c, <c>2 + 2c</c> = I-c); null for no tag loss.</param>
/// <param name="Roles">Key/value role per word (0 = O, 1 = B-KEY, 2 = I-KEY, 3 = B-VAL, 4 = I-VAL); null for no role loss.</param>
/// <param name="Doc">Document type index, or a negative number for no document loss.</param>
/// <param name="Item">Per word, the page index of the anchor of its line item or tax row; null for no item-pointer loss.</param>
/// <param name="KeyValue">Per word, the index of the first word of its key (for value words); null for no key/value pointer loss.</param>
internal sealed record DocTargets(int[]? Tags, int[]? Roles = null, int Doc = -1, int[]? Item = null, int[]? KeyValue = null);

/// <summary>Weights of the loss terms; a term with weight 0 is not computed.</summary>
internal sealed record DocLossWeights
{
    public float Tag { get; init; } = 1f;
    public float Role { get; init; }
    public float Doc { get; init; }
    public float Item { get; init; }
    public float KeyValue { get; init; }
}

internal sealed record DocExample(DocFeatures Features, DocTargets Targets);

/// <summary>Settings of a <see cref="DocTrainer"/>.</summary>
internal sealed record DocTrainSettings
{
    public AdamWOptions Adam { get; init; } = new();

    /// <summary>
    /// Learning-rate multiplier of every tensor except the tag head (the head of a new schema trains at the full rate). 0 freezes the
    /// backbone, the other heads and the pointer layers: only the tag head learns, and the other loss terms are switched off.
    /// </summary>
    public double BackboneScale { get; init; }

    public DocLossWeights Weights { get; init; } = new();

    /// <summary>Optional weight per tag class (PyTorch's <c>weight</c> of cross-entropy), for example to make the rare tags count more than O.</summary>
    public float[]? TagClassWeights { get; init; }

    /// <summary>Clip the global gradient norm to this value; 0 turns clipping off.</summary>
    public double MaxGradNorm { get; init; }

    /// <summary>Whether biases and layer-norm parameters receive weight decay (weight matrices and tables always do).</summary>
    public bool DecayBiases { get; init; }
}

/// <summary>The trainable tensors of a <c>docextract-v0</c> model in their fixed order, with the names of the container.</summary>
internal static class DocParams
{
    public const int CharEmb = 0, CharConvW = 1, CharConvB = 2, CharProjW = 3, CharProjB = 4, PosX0 = 5, PosX1 = 6, PosY0 = 7, PosY1 = 8;
    public const int FeatW = 9, FeatB = 10, EmbLnW = 11, EmbLnB = 12, BlockBase = 13, BlockSize = 12;

    // Offsets inside a block.
    public const int Ln1W = 0, Ln1B = 1, QkvW = 2, QkvB = 3, ProjW = 4, ProjB = 5, Ln2W = 6, Ln2B = 7, Fc1W = 8, Fc1B = 9, Fc2W = 10, Fc2B = 11;

    // Offsets after the blocks.
    public const int Pair1W = 0, Pair1B = 1, Pair2W = 2, Pair2B = 3, FinalLnW = 4, FinalLnB = 5, TagW = 6, TagB = 7, RoleW = 8, RoleB = 9, DocW = 10, DocB = 11,
        RelQW = 12, RelQB = 13, RelKW = 14, RelKB = 15, TailSize = 16;

    public static int Block(int layer, int offset) => BlockBase + (layer * BlockSize) + offset;

    public static int Tail(DocExtractModel m, int offset) => BlockBase + (m.Layers * BlockSize) + offset;

    public static int Count(DocExtractModel m) => BlockBase + (m.Layers * BlockSize) + TailSize;

    /// <summary>The model's weight arrays (the model's own, so updates land in the model) with their container names.</summary>
    public static (string Name, float[] Value)[] List(DocExtractModel m)
    {
        var l = new List<(string, float[])>
        {
            ("char_emb.weight", m.CharEmb), ("char_conv.weight", m.CharConvWeight), ("char_conv.bias", m.CharConvBias),
            ("char_proj.weight", m.CharProjWeight), ("char_proj.bias", m.CharProjBias),
            ("pos_x0.weight", m.PosX0), ("pos_x1.weight", m.PosX1), ("pos_y0.weight", m.PosY0), ("pos_y1.weight", m.PosY1),
            ("feat_proj.weight", m.FeatWeight), ("feat_proj.bias", m.FeatBias), ("emb_ln.weight", m.EmbLnWeight), ("emb_ln.bias", m.EmbLnBias),
        };
        for (int i = 0; i < m.Blocks.Length; i++)
        {
            DocExtractModel.Block b = m.Blocks[i];
            string p = $"blocks.{i}.";
            l.Add((p + "ln1.weight", b.Ln1Weight)); l.Add((p + "ln1.bias", b.Ln1Bias));
            l.Add((p + "qkv.weight", b.QkvWeight)); l.Add((p + "qkv.bias", b.QkvBias));
            l.Add((p + "proj.weight", b.ProjWeight)); l.Add((p + "proj.bias", b.ProjBias));
            l.Add((p + "ln2.weight", b.Ln2Weight)); l.Add((p + "ln2.bias", b.Ln2Bias));
            l.Add((p + "fc1.weight", b.Fc1Weight)); l.Add((p + "fc1.bias", b.Fc1Bias));
            l.Add((p + "fc2.weight", b.Fc2Weight)); l.Add((p + "fc2.bias", b.Fc2Bias));
        }

        l.Add(("pair1.weight", m.Pair1Weight)); l.Add(("pair1.bias", m.Pair1Bias));
        l.Add(("pair2.weight", m.Pair2Weight)); l.Add(("pair2.bias", m.Pair2Bias));
        l.Add(("final_ln.weight", m.FinalLnWeight)); l.Add(("final_ln.bias", m.FinalLnBias));
        l.Add(("tag_head.weight", m.TagWeight)); l.Add(("tag_head.bias", m.TagBias));
        l.Add(("role_head.weight", m.RoleWeight)); l.Add(("role_head.bias", m.RoleBias));
        l.Add(("doc_head.weight", m.DocWeight)); l.Add(("doc_head.bias", m.DocBias));
        l.Add(("rel_q.weight", m.RelQWeight)); l.Add(("rel_q.bias", m.RelQBias));
        l.Add(("rel_k.weight", m.RelKWeight)); l.Add(("rel_k.bias", m.RelKBias));
        return [.. l];
    }

    /// <summary>Whether weight decay applies to a tensor by default: weight matrices and tables, not biases or layer-norm parameters.</summary>
    public static bool DecaysByDefault(string name) => name.EndsWith(".weight", StringComparison.Ordinal) && !name.Contains("ln", StringComparison.Ordinal);
}

/// <summary>
/// One optimization step of the <c>docextract-v0</c> model on whole pages: forward with every activation kept, the losses, backward, AdamW.
/// A page is one example and one window (at most <see cref="DocExtractModel.MaxWords"/> words). The model's weight arrays are the fp32
/// master weights and are updated in place, so the model must not be shared with an extractor in use. Examples of a batch run in
/// parallel, each with its own gradient set, and the sets are added in example order, so loss, gradients and weights do not depend on the
/// degree of parallelism or the kernel path (docs/spec/docextract-backward.md). The forward values are bit-identical to those of
/// <c>DocForward</c>.
/// </summary>
internal sealed class DocTrainer
{
    private readonly DocExtractModel _m;
    private readonly KernelPath _path;
    private readonly DocTrainSettings _s;
    private readonly BufferPool<float> _floats;
    private readonly BufferPool<int> _ints;
    private readonly (string Name, float[] Value)[] _all;
    private readonly bool[] _trainable;
    private readonly int[] _trainIndex;
    private readonly float[][] _grad; // by trainable position
    private readonly List<float[][]> _exampleGrads = [];
    private readonly AdamW _optimizer;
    private readonly bool _backbone, _tagOn, _roleOn, _docOn, _itemOn, _kvOn;

    public DocTrainer(DocExtractModel model, KernelPath path, DocTrainSettings settings, long poolBytes = 1L << 30)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(settings);
        if (!(settings.BackboneScale >= 0) || !(settings.MaxGradNorm >= 0) || settings.Weights.Tag < 0 || settings.Weights.Role < 0
            || settings.Weights.Doc < 0 || settings.Weights.Item < 0 || settings.Weights.KeyValue < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(settings), "Scales, weights and the clipping norm cannot be negative.");
        }

        if (settings.TagClassWeights is { } cw && (cw.Length != model.Tags || cw.Any(static w => !(w >= 0))))
        {
            throw new ArgumentException("One non-negative weight per tag is required.", nameof(settings));
        }

        _m = model;
        _path = Dispatch.Require(path);
        _s = settings;
        _floats = new BufferPool<float>(poolBytes, sizeof(float));
        _ints = new BufferPool<int>(poolBytes, sizeof(int));
        _backbone = settings.BackboneScale > 0;
        DocLossWeights w = settings.Weights;
        _tagOn = w.Tag > 0;
        _roleOn = _backbone && w.Role > 0;
        _docOn = _backbone && w.Doc > 0;
        _itemOn = _backbone && w.Item > 0;
        _kvOn = _backbone && w.KeyValue > 0;
        _all = DocParams.List(model);
        _trainable = new bool[_all.Length];
        int tagW = DocParams.Tail(model, DocParams.TagW);
        for (int i = 0; i < _all.Length; i++)
        {
            _trainable[i] = _backbone;
        }

        // The other heads train only when their own loss is on; the pointer layers need either pointer loss.
        SetTrainable(DocParams.Tail(model, DocParams.RoleW), _roleOn);
        SetTrainable(DocParams.Tail(model, DocParams.DocW), _docOn);
        SetTrainable(DocParams.Tail(model, DocParams.RelQW), _itemOn || _kvOn);
        SetTrainable(DocParams.Tail(model, DocParams.RelKW), _itemOn || _kvOn);
        SetTrainable(tagW, _tagOn);
        if (!_trainable.Contains(true))
        {
            throw new ArgumentException("Nothing would train: the tag loss weight is 0 and the backbone is frozen.", nameof(settings));
        }

        _trainIndex = new int[_all.Length];
        var values = new List<float[]>();
        var decay = new List<bool>();
        var scales = new List<double>();
        var names = new List<string>();
        for (int i = 0; i < _all.Length; i++)
        {
            _trainIndex[i] = -1;
            if (!_trainable[i])
            {
                continue;
            }

            _trainIndex[i] = values.Count;
            values.Add(_all[i].Value);
            names.Add(_all[i].Name);
            bool isBias = !_all[i].Name.EndsWith(".weight", StringComparison.Ordinal) || _all[i].Name.Contains("ln", StringComparison.Ordinal);
            decay.Add(DocParams.DecaysByDefault(_all[i].Name) || (settings.DecayBiases && isBias));
            scales.Add(i == tagW || i == tagW + 1 ? 1.0 : settings.BackboneScale);
        }

        Names = names;
        Parameters = values;
        _grad = [.. values.Select(static v => new float[v.Length])];
        _optimizer = new AdamW([.. values], [.. decay], settings.Adam, [.. scales]);
    }

    private void SetTrainable(int firstOfPair, bool on)
    {
        _trainable[firstOfPair] = _trainable[firstOfPair + 1] = on;
    }

    /// <summary>Names of the trainable tensors (container names), in the order of <see cref="Parameters"/>.</summary>
    public IReadOnlyList<string> Names { get; private set; }

    /// <summary>The trainable tensors, updated in place (the model's own arrays, in the runtime's layouts).</summary>
    public IReadOnlyList<float[]> Parameters { get; private set; } = [];

    /// <summary>The gradients of the last <see cref="ComputeGradients"/> or <see cref="Step"/> (after clipping), in the order of <see cref="Parameters"/>.</summary>
    public IReadOnlyList<float[]> Gradients => _grad;

    public AdamW Optimizer => _optimizer;

    public double LastGradNorm { get; private set; }

    /// <summary>
    /// Runs the optimizer step on the gradients <see cref="ComputeGradients"/> left (clipped to the settings' norm). Returns nothing; the
    /// weights are the model's arrays.
    /// </summary>
    public void Apply(double learningRate)
    {
        if (_s.MaxGradNorm > 0 && LastGradNorm > _s.MaxGradNorm)
        {
            float scale = (float)(_s.MaxGradNorm / (LastGradNorm + 1e-6));
            foreach (float[] g in _grad)
            {
                for (int j = 0; j < g.Length; j++)
                {
                    g[j] *= scale;
                }
            }
        }

        _optimizer.Step(_grad, learningRate);
    }

    /// <summary>One step: gradients of the batch's mean loss, clipping, AdamW. Returns the loss before the update.</summary>
    public double Step(IReadOnlyList<DocExample> batch, int degree = 1, double? learningRate = null, CancellationToken cancellationToken = default)
    {
        double loss = ComputeGradients(batch, degree, cancellationToken);
        Apply(learningRate ?? _optimizer.Options.LearningRate);
        return loss;
    }

    /// <summary>The mean loss of a batch, without gradients or changes.</summary>
    public double Evaluate(IReadOnlyList<DocExample> batch)
    {
        ArgumentNullException.ThrowIfNull(batch);
        double total = 0;
        foreach (DocExample ex in batch)
        {
            using var scratch = new TrainScratch(_floats, _ints);
            Activations a = Forward(scratch, ex.Features);
            total += Losses(scratch, a, ex.Targets, 0.0, null, null);
        }

        return total / batch.Count;
    }

    /// <summary>Forward and backward over the batch: fills <see cref="Gradients"/> (unclipped) and <see cref="LastGradNorm"/>; returns the mean loss.</summary>
    public double ComputeGradients(IReadOnlyList<DocExample> batch, int degree = 1, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(batch);
        if (batch.Count == 0)
        {
            throw new ArgumentException("A batch needs at least one example.", nameof(batch));
        }

        // Examples run in waves of at most `degree`, each into its own gradient set; a wave is added into the running sum in example
        // order right away, so the sum is ((g0 + g1) + g2) + ... whatever the wave size.
        int wave = Math.Max(1, Math.Min(degree, batch.Count));
        while (_exampleGrads.Count < wave)
        {
            _exampleGrads.Add([.. _all.Select((p, i) => _trainable[i] ? new float[p.Value.Length] : null!)]);
        }

        Views views = BuildViews();
        var losses = new double[batch.Count];
        for (int first = 0; first < batch.Count; first += wave)
        {
            int count = Math.Min(wave, batch.Count - first);
            int start = first;
            ParallelRun.For(count, degree, s => losses[start + s] = ExampleStep(batch[start + s], batch.Count, views, _exampleGrads[s]), cancellationToken);
            Accumulate(start, count, degree, cancellationToken);
        }

        double total = 0;
        for (int b = 0; b < batch.Count; b++)
        {
            total += losses[b];
        }

        double sum = 0;
        foreach (float[] g in _grad)
        {
            sum += Gemm.Dot(_path, g, g);
        }

        LastGradNorm = Math.Sqrt(sum);
        return total;
    }

    private void Accumulate(int first, int count, int degree, CancellationToken cancellationToken)
    {
        const int Chunk = 1 << 16;
        var work = new List<(int Tensor, int Start)>();
        for (int p = 0; p < _grad.Length; p++)
        {
            for (int s = 0; s < _grad[p].Length; s += Chunk)
            {
                work.Add((p, s));
            }
        }

        int[] global = [.. Enumerable.Range(0, _all.Length).Where(i => _trainable[i])];
        ParallelRun.For(work.Count, degree, i =>
        {
            (int p, int s) = work[i];
            int n = Math.Min(Chunk, _grad[p].Length - s);
            Span<float> dst = _grad[p].AsSpan(s, n);
            for (int k = 0; k < count; k++)
            {
                ReadOnlySpan<float> src = _exampleGrads[k][global[p]].AsSpan(s, n);
                if (first + k == 0)
                {
                    src.CopyTo(dst);
                    continue;
                }

                for (int j = 0; j < n; j++)
                {
                    dst[j] += src[j];
                }
            }
        }, cancellationToken);
    }

    // ---- Transposed weights, rebuilt once per step ----

    private sealed class Views
    {
        public required float[] CharProj { get; init; }
        public required float[] Conv { get; init; }
        public required float[][] Qkv { get; init; }
        public required float[][] Proj { get; init; }
        public required float[][] Fc1 { get; init; }
        public required float[][] Fc2 { get; init; }
        public required float[] Pair2 { get; init; }
        public required float[] Tag { get; init; }
        public required float[] Role { get; init; }
        public required float[] Doc { get; init; }
        public required float[] RelQ { get; init; }
        public required float[] RelK { get; init; }
    }

    private Views? _views;

    private Views BuildViews()
    {
        DocExtractModel m = _m;
        int d = m.D, cd3 = 3 * m.CharDim, r2 = 2 * m.RelDim;
        _views ??= new Views
        {
            CharProj = new float[d * m.CharConv],
            Conv = new float[cd3 * m.CharConv],
            Qkv = [.. m.Blocks.Select(_ => new float[d * 3 * d])],
            Proj = [.. m.Blocks.Select(_ => new float[d * d])],
            Fc1 = [.. m.Blocks.Select(_ => new float[d * m.Ffn])],
            Fc2 = [.. m.Blocks.Select(_ => new float[m.Ffn * d])],
            Pair2 = new float[m.PairHidden * m.PairChannels],
            Tag = new float[d * m.Tags],
            Role = new float[d * m.Roles],
            Doc = new float[d * m.DocTypes],
            RelQ = new float[d * r2],
            RelK = new float[d * r2],
        };
        Views v = _views;
        void T(float[] w, int outDim, int inDim, float[] dst) => Backward.Transpose(w, inDim, outDim, inDim, dst);
        T(m.TagWeight, m.Tags, d, v.Tag);
        if (!_backbone)
        {
            return v;
        }

        T(m.CharProjWeight, d, m.CharConv, v.CharProj);
        T(m.CharConvWeight, m.CharConv, cd3, v.Conv);
        for (int l = 0; l < m.Layers; l++)
        {
            DocExtractModel.Block b = m.Blocks[l];
            T(b.QkvWeight, 3 * d, d, v.Qkv[l]);
            T(b.ProjWeight, d, d, v.Proj[l]);
            T(b.Fc1Weight, m.Ffn, d, v.Fc1[l]);
            T(b.Fc2Weight, d, m.Ffn, v.Fc2[l]);
        }

        T(m.Pair2Weight, m.PairChannels, m.PairHidden, v.Pair2);
        T(m.RoleWeight, m.Roles, d, v.Role);
        T(m.DocWeight, m.DocTypes, d, v.Doc);
        T(m.RelQWeight, r2, d, v.RelQ);
        T(m.RelKWeight, r2, d, v.RelK);
        return v;
    }

    // ---- Forward ----

    private sealed class Layer
    {
        public required float[] U1 { get; init; }
        public required float[] Xhat1 { get; init; }
        public required float[] Inv1 { get; init; }
        public required float[] Qkv { get; init; }
        public required float[] Probs { get; init; } // [heads][n][n]
        public required float[] Attn { get; init; }
        public required float[] U2 { get; init; }
        public required float[] Xhat2 { get; init; }
        public required float[] Inv2 { get; init; }
        public required float[] Hid { get; init; }
        public required float[] Th { get; init; }
        public required float[] G { get; init; }
    }

    private sealed class Activations
    {
        public required int N { get; init; }
        public required DocFeatures F { get; init; }
        public int Positions { get; set; }
        public int[] Last { get; set; } = [];
        public int[] Start { get; set; } = [];
        public float[] Windows { get; set; } = [];
        public float[] Conv { get; set; } = [];
        public int[] Argmax { get; set; } = [];
        public float[] Pool { get; set; } = [];
        public float[] FeatPad { get; set; } = [];
        public float[] EmbXhat { get; set; } = [];
        public float[] EmbInv { get; set; } = [];
        public float[] Pb { get; set; } = [];
        public float[] PairFeat { get; set; } = [];
        public float[] PairHid { get; set; } = [];
        public Layer[] Layers { get; set; } = [];
        public float[] FinalXhat { get; set; } = [];
        public float[] FinalInv { get; set; } = [];
        public float[] Final { get; set; } = [];
        public float[] Tags { get; set; } = [];
        public float[] Roles { get; set; } = [];
        public float[] Doc { get; set; } = [];
        public float[] Mean { get; set; } = [];
        public float[] RelQ { get; set; } = [];
        public float[] RelK { get; set; } = [];
        public float[] Rel { get; set; } = [];
    }

    /// <summary>The logits of a page, for tests: tags, roles, document type and the two pointer matrices of one window.</summary>
    internal (float[] Tags, float[] Roles, float[] Doc, float[] Rel) ForwardForTests(DocFeatures f)
    {
        using var scratch = new TrainScratch(_floats, _ints);
        Activations a = Forward(scratch, f);
        int n = a.N;
        return (a.Tags.AsSpan(0, n * _m.Tags).ToArray(), a.Roles.AsSpan(0, n * _m.Roles).ToArray(), a.Doc.AsSpan(0, _m.DocTypes).ToArray(), a.Rel.AsSpan(0, 2 * n * n).ToArray());
    }

    private Activations Forward(TrainScratch scratch, DocFeatures f)
    {
        DocExtractModel m = _m;
        int n = f.Count;
        if (n < 1 || n > m.MaxWords)
        {
            throw new ArgumentOutOfRangeException(nameof(f), $"A training page holds 1 to {m.MaxWords} words.");
        }

        long nn = (long)n * n;
        if (nn * (m.PairChannels + (2 * m.PairHidden) + 16) > int.MaxValue)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxActivationElements), int.MaxValue, nn * (m.PairChannels + (2 * m.PairHidden) + 16));
        }

        int d = m.D, heads = m.Heads, dh = d / heads, mc = m.MaxChars, cd = m.CharDim, cc = m.CharConv, nn32 = (int)nn;
        var a = new Activations { N = n, F = f };

        // ---- character CNN over all positions of all words ----
        a.Last = new int[n];
        a.Start = new int[n + 1];
        int p = 0;
        for (int i = 0; i < n; i++)
        {
            int last = mc;
            while (last > 0 && f.Chars[(i * mc) + last - 1] == 0)
            {
                last--;
            }

            a.Last[i] = last;
            a.Start[i] = p;
            p += last;
        }

        a.Start[n] = p;
        a.Positions = p;
        a.Windows = scratch.Zeros((long)p * 3 * cd);
        a.Conv = scratch.Floats((long)p * cc);
        a.Pool = scratch.Zeros((long)n * cc);
        a.Argmax = scratch.Ints((long)n * cc);
        Array.Fill(a.Argmax, -1, 0, n * cc);
        for (int i = 0; i < n; i++)
        {
            for (int q = 0; q < a.Last[i]; q++)
            {
                Span<float> row = a.Windows.AsSpan((a.Start[i] + q) * 3 * cd, 3 * cd);
                for (int k = 0; k < 3; k++)
                {
                    int idx = q - 1 + k;
                    if ((uint)idx < (uint)mc)
                    {
                        int id = f.Chars[(i * mc) + idx];
                        if ((uint)id >= (uint)(m.CharEmb.Length / cd))
                        {
                            throw new ArgumentOutOfRangeException(nameof(f), "A character index is outside the embedding table.");
                        }

                        m.CharEmb.AsSpan(id * cd, cd).CopyTo(row.Slice(k * cd, cd));
                    }
                }
            }
        }

        if (p > 0)
        {
            Gemm.NT(_path, a.Windows, 3 * cd, m.CharConvWeight, 3 * cd, p, cc, 3 * cd, m.CharConvBias, a.Conv, cc, 1);
        }

        for (int i = 0; i < n; i++)
        {
            for (int q = 0; q < a.Last[i]; q++)
            {
                if (f.Chars[(i * mc) + q] == 0)
                {
                    continue;
                }

                int r = a.Start[i] + q;
                for (int c = 0; c < cc; c++)
                {
                    float v = a.Conv[(r * cc) + c];
                    if (v > a.Pool[(i * cc) + c])
                    {
                        a.Pool[(i * cc) + c] = v;
                        a.Argmax[(i * cc) + c] = r;
                    }
                }
            }
        }

        // ---- embedding ----
        float[] x = scratch.Floats((long)n * d);
        DocKernels.Linear(_path, a.Pool, cc, n, m.CharProjWeight, m.CharProjBias, cc, d, x, d);
        var ids = new int[n];
        int binRows = m.PosX0.Length / d;
        foreach ((int column, float[] table) in new[] { (0, m.PosX0), (2, m.PosX1), (1, m.PosY0), (3, m.PosY1) })
        {
            for (int i = 0; i < n; i++)
            {
                ids[i] = f.Bins[(i * 4) + column];
            }

            DocKernels.AddRows(x, d, table, binRows, ids);
        }

        a.FeatPad = scratch.Zeros((long)n * 16);
        for (int i = 0; i < n; i++)
        {
            f.Feats.AsSpan(i * 8, 8).CopyTo(a.FeatPad.AsSpan(i * 16, 8));
        }

        float[] t = scratch.Floats((long)n * d);
        DocKernels.Linear(_path, a.FeatPad, 16, n, m.FeatWeight, m.FeatBias, 16, d, t, d);
        for (int i = 0; i < n * d; i++)
        {
            x[i] += t[i];
        }

        a.EmbXhat = scratch.Floats((long)n * d);
        a.EmbInv = scratch.Floats(n);
        DocBackward.LayerNormForward(_path, x.AsSpan(0, n * d), x.AsSpan(0, n * d), n, d, m.EmbLnWeight, m.EmbLnBias, a.EmbXhat, a.EmbInv);

        // ---- pair bias ----
        a.PairFeat = scratch.Floats(nn * 16);
        a.PairHid = scratch.Floats(nn * m.PairHidden);
        a.Pb = scratch.Floats(nn * m.PairChannels);
        DocKernels.PairFeaturesRows(f.Geo, n, 0, n, a.PairFeat);
        DocKernels.Linear(_path, a.PairFeat, 16, nn32, m.Pair1Weight, m.Pair1Bias, 16, m.PairHidden, a.PairHid, m.PairHidden);
        DocKernels.Relu(a.PairHid.AsSpan(0, nn32 * m.PairHidden));
        Gemm.NT(_path, a.PairHid, m.PairHidden, m.Pair2Weight, m.PairHidden, nn32, m.PairChannels, m.PairHidden, m.Pair2Bias, a.Pb, 1, nn32);

        // ---- blocks ----
        float scale = MathF.Sqrt(dh);
        float[] vt = scratch.Floats((long)heads * dh * n);
        a.Layers = new Layer[m.Layers];
        for (int l = 0; l < m.Layers; l++)
        {
            DocExtractModel.Block b = m.Blocks[l];
            var ly = new Layer
            {
                U1 = scratch.Floats((long)n * d), Xhat1 = scratch.Floats((long)n * d), Inv1 = scratch.Floats(n), Qkv = scratch.Floats((long)n * 3 * d),
                Probs = scratch.Floats((long)heads * nn), Attn = scratch.Floats((long)n * d), U2 = scratch.Floats((long)n * d),
                Xhat2 = scratch.Floats((long)n * d), Inv2 = scratch.Floats(n), Hid = scratch.Floats((long)n * m.Ffn), Th = scratch.Floats((long)n * m.Ffn),
                G = scratch.Floats((long)n * m.Ffn),
            };
            a.Layers[l] = ly;
            DocBackward.LayerNormForward(_path, x.AsSpan(0, n * d), ly.U1, n, d, b.Ln1Weight, b.Ln1Bias, ly.Xhat1, ly.Inv1);
            DocKernels.Linear(_path, ly.U1, d, n, b.QkvWeight, b.QkvBias, d, 3 * d, ly.Qkv, 3 * d);
            for (int h = 0; h < heads; h++)
            {
                for (int c = 0; c < dh; c++)
                {
                    int src = (2 * d) + (h * dh) + c;
                    Span<float> dst = vt.AsSpan(((h * dh) + c) * n, n);
                    for (int j = 0; j < n; j++)
                    {
                        dst[j] = ly.Qkv[(j * 3 * d) + src];
                    }
                }
            }

            for (int h = 0; h < heads; h++)
            {
                Span<float> s = ly.Probs.AsSpan(h * nn32, nn32);
                Gemm.NT(_path, ly.Qkv.AsSpan(h * dh), 3 * d, ly.Qkv.AsSpan(d + (h * dh)), 3 * d, n, n, dh, default, s, n, 1);
                DocKernels.SoftmaxRowsWithBias(_path, s, a.Pb.AsSpan(((l * heads) + h) * nn32, nn32), n, n, scale);
                Gemm.NT(_path, s, n, vt.AsSpan(h * dh * n), n, n, dh, n, default, ly.Attn.AsSpan(h * dh), d, 1);
            }

            DocKernels.Linear(_path, ly.Attn, d, n, b.ProjWeight, b.ProjBias, d, d, t, d);
            for (int i = 0; i < n * d; i++)
            {
                x[i] += t[i];
            }

            DocBackward.LayerNormForward(_path, x.AsSpan(0, n * d), ly.U2, n, d, b.Ln2Weight, b.Ln2Bias, ly.Xhat2, ly.Inv2);
            DocKernels.Linear(_path, ly.U2, d, n, b.Fc1Weight, b.Fc1Bias, d, m.Ffn, ly.Hid, m.Ffn);
            DocBackward.GeluForward(_path, ly.Hid.AsSpan(0, n * m.Ffn), ly.G, ly.Th);
            DocKernels.Linear(_path, ly.G, m.Ffn, n, b.Fc2Weight, b.Fc2Bias, m.Ffn, d, t, d);
            for (int i = 0; i < n * d; i++)
            {
                x[i] += t[i];
            }
        }

        // ---- heads ----
        a.Final = scratch.Floats((long)n * d);
        a.FinalXhat = scratch.Floats((long)n * d);
        a.FinalInv = scratch.Floats(n);
        DocBackward.LayerNormForward(_path, x.AsSpan(0, n * d), a.Final, n, d, m.FinalLnWeight, m.FinalLnBias, a.FinalXhat, a.FinalInv);
        a.Tags = scratch.Floats((long)n * m.Tags);
        a.Roles = scratch.Floats((long)n * m.Roles);
        int r2 = 2 * m.RelDim;
        a.RelQ = scratch.Floats((long)n * r2);
        a.RelK = scratch.Floats((long)n * r2);
        DocKernels.Linear(_path, a.Final, d, n, m.TagWeight, m.TagBias, d, m.Tags, a.Tags, m.Tags);
        DocKernels.Linear(_path, a.Final, d, n, m.RoleWeight, m.RoleBias, d, m.Roles, a.Roles, m.Roles);
        DocKernels.Linear(_path, a.Final, d, n, m.RelQWeight, m.RelQBias, d, r2, a.RelQ, r2);
        DocKernels.Linear(_path, a.Final, d, n, m.RelKWeight, m.RelKBias, d, r2, a.RelK, r2);

        a.Mean = scratch.Floats(d);
        for (int c = 0; c < d; c++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += a.Final[(i * d) + c];
            }

            a.Mean[c] = (float)(sum / n);
        }

        a.Doc = scratch.Floats(m.DocTypes);
        DocKernels.Linear(_path, a.Mean, d, 1, m.DocWeight, m.DocBias, d, m.DocTypes, a.Doc, m.DocTypes);

        a.Rel = scratch.Floats(2 * nn);
        float relScale = MathF.Sqrt(m.RelDim);
        for (int r = 0; r < 2; r++)
        {
            DocKernels.PointerScores(
                _path, a.RelQ.AsSpan(r * m.RelDim), r2, a.RelK.AsSpan(r * m.RelDim), r2, n, n, m.RelDim, relScale,
                a.Pb.AsSpan(((m.Layers * heads) + r) * nn32, nn32), a.Rel.AsSpan(r * nn32, nn32));
        }

        return a;
    }

    // ---- Losses and backward ----

    // Computes the loss terms of a page; when `views`/`grads` are given also runs backward into `grads` with the gradient scaled by `scale`.
    private double Losses(TrainScratch scratch, Activations a, DocTargets t, double scale, Views? views, float[][]? grads)
    {
        DocExtractModel m = _m;
        int n = a.N, d = m.D, nn = n * n, r2 = 2 * m.RelDim;
        DocLossWeights w = _s.Weights;
        bool back = grads is not null;
        double loss = 0;

        float[]? dFinal = back ? scratch.Zeros((long)n * d) : null;
        float[]? dPb = back && _backbone ? scratch.Zeros((long)m.PairChannels * nn) : null;
        bool needDx = back && _backbone;

        void AddInto(float[] src)
        {
            for (int i = 0; i < n * d; i++)
            {
                dFinal![i] += src[i];
            }
        }

        if (_tagOn && t.Tags is { } tags)
        {
            CheckTargets(tags, n, m.Tags, "Tags");
            float[] dTags = scratch.Floats((long)n * m.Tags);
            loss += w.Tag * DocLoss.CrossEntropy(a.Tags, m.Tags, n, m.Tags, tags, _s.TagClassWeights, scale * w.Tag, dTags);
            if (back)
            {
                float[]? dx = needDx ? scratch.Floats((long)n * d) : null;
                LinearTrain.BackwardPass(
                    _path, scratch, a.Final, dTags, views!.Tag, n, d, m.Tags, grads![DocParams.Tail(m, DocParams.TagW)], grads[DocParams.Tail(m, DocParams.TagB)],
                    dx is null ? default : dx.AsSpan(0, n * d));
                if (dx is not null)
                {
                    AddInto(dx);
                }
            }
        }

        if (_roleOn && t.Roles is { } roles)
        {
            CheckTargets(roles, n, m.Roles, "Roles");
            float[] dRoles = scratch.Floats((long)n * m.Roles);
            loss += w.Role * DocLoss.CrossEntropy(a.Roles, m.Roles, n, m.Roles, roles, default, scale * w.Role, dRoles);
            if (back)
            {
                float[] dx = scratch.Floats((long)n * d);
                LinearTrain.BackwardPass(
                    _path, scratch, a.Final, dRoles, views!.Role, n, d, m.Roles, grads![DocParams.Tail(m, DocParams.RoleW)], grads[DocParams.Tail(m, DocParams.RoleB)],
                    dx.AsSpan(0, n * d));
                AddInto(dx);
            }
        }

        if (_docOn && t.Doc >= 0)
        {
            if (t.Doc >= m.DocTypes)
            {
                throw new ArgumentOutOfRangeException(nameof(t), "The document type is outside the model's types.");
            }

            float[] dDoc = scratch.Floats(m.DocTypes);
            loss += w.Doc * DocLoss.CrossEntropy(a.Doc, m.DocTypes, 1, m.DocTypes, [t.Doc], default, scale * w.Doc, dDoc);
            if (back)
            {
                float[] dMean = scratch.Floats(d);
                LinearTrain.BackwardPass(
                    _path, scratch, a.Mean, dDoc, views!.Doc, 1, d, m.DocTypes, grads![DocParams.Tail(m, DocParams.DocW)], grads[DocParams.Tail(m, DocParams.DocB)],
                    dMean.AsSpan(0, d));

                // The mean over the n words: every word receives dMean / n.
                float[] dx = scratch.Floats((long)n * d);
                for (int i = 0; i < n; i++)
                {
                    for (int c = 0; c < d; c++)
                    {
                        dx[(i * d) + c] = dMean[c] / n;
                    }
                }

                AddInto(dx);
            }
        }

        float[]? dRelQ = null, dRelK = null;
        float relScale = MathF.Sqrt(m.RelDim);
        for (int r = 0; r < 2; r++)
        {
            int[]? target = r == 0 ? t.Item : t.KeyValue;
            if (!(r == 0 ? _itemOn : _kvOn) || target is null)
            {
                continue;
            }

            CheckTargets(target, n, n, r == 0 ? "Item" : "KeyValue");
            float weight = r == 0 ? w.Item : w.KeyValue;
            float[] dRel = scratch.Floats((long)nn);
            loss += weight * DocLoss.CrossEntropy(a.Rel.AsSpan(r * nn, nn), n, n, n, target, default, scale * weight, dRel);
            if (!back)
            {
                continue;
            }

            dRelQ ??= scratch.Zeros((long)n * r2);
            dRelK ??= scratch.Zeros((long)n * r2);
            dRel.AsSpan(0, nn).CopyTo(dPb!.AsSpan(((m.Layers * m.Heads) + r) * nn, nn));
            float[] dRaw = scratch.Floats((long)nn);
            for (int i = 0; i < nn; i++)
            {
                dRaw[i] = dRel[i] / relScale;
            }

            DocBackward.ScoresBackward(
                _path, dRaw, a.RelQ.AsSpan(r * m.RelDim), r2, a.RelK.AsSpan(r * m.RelDim), r2, n, m.RelDim,
                dRelQ.AsSpan(r * m.RelDim), r2, dRelK.AsSpan(r * m.RelDim), r2);
        }

        if (back && dRelQ is not null && dRelK is not null)
        {
            float[] dxq = scratch.Floats((long)n * d), dxk = scratch.Floats((long)n * d);
            LinearTrain.BackwardPass(
                _path, scratch, a.Final, dRelQ, views!.RelQ, n, d, r2, grads![DocParams.Tail(m, DocParams.RelQW)], grads[DocParams.Tail(m, DocParams.RelQB)],
                dxq.AsSpan(0, n * d));
            LinearTrain.BackwardPass(
                _path, scratch, a.Final, dRelK, views.RelK, n, d, r2, grads[DocParams.Tail(m, DocParams.RelKW)], grads[DocParams.Tail(m, DocParams.RelKB)],
                dxk.AsSpan(0, n * d));
            AddInto(dxq);
            AddInto(dxk);
        }

        if (back && _backbone)
        {
            Backbone(scratch, a, views!, grads!, dFinal!, dPb!);
        }

        return loss;
    }

    private static void CheckTargets(int[] targets, int n, int classes, string what)
    {
        if (targets.Length < n)
        {
            throw new ArgumentException($"{what} needs one target per word.", nameof(targets));
        }

        for (int i = 0; i < n; i++)
        {
            if (targets[i] >= classes)
            {
                throw new ArgumentOutOfRangeException(nameof(targets), $"{what} target {targets[i]} is outside 0..{classes - 1}.");
            }
        }
    }

    private void Backbone(TrainScratch scratch, Activations a, Views v, float[][] g, float[] dFinal, float[] dPb)
    {
        DocExtractModel m = _m;
        int n = a.N, d = m.D, heads = m.Heads, dh = d / heads, nn = n * n, cc = m.CharConv, cd = m.CharDim;
        float scale = MathF.Sqrt(dh);

        float[] dx = scratch.Floats((long)n * d);
        DocBackward.LayerNormBackward(
            _path, dFinal.AsSpan(0, n * d), a.FinalXhat, a.FinalInv, m.FinalLnWeight, n, d, dx, g[DocParams.Tail(m, DocParams.FinalLnW)], g[DocParams.Tail(m, DocParams.FinalLnB)]);

        for (int l = m.Layers - 1; l >= 0; l--)
        {
            DocExtractModel.Block b = m.Blocks[l];
            Layer ly = a.Layers[l];
            int Gi(int off) => DocParams.Block(l, off);

            // x_out = x1 + fc2(gelu(fc1(ln2(x1))))
            float[] dG = scratch.Floats((long)n * m.Ffn);
            LinearTrain.BackwardPass(_path, scratch, ly.G, dx, v.Fc2[l], n, m.Ffn, d, g[Gi(DocParams.Fc2W)], g[Gi(DocParams.Fc2B)], dG.AsSpan(0, n * m.Ffn));
            float[] dHid = scratch.Floats((long)n * m.Ffn);
            DocBackward.GeluBackward(dG.AsSpan(0, n * m.Ffn), ly.Hid, ly.Th, dHid);
            float[] dU2 = scratch.Floats((long)n * d);
            LinearTrain.BackwardPass(_path, scratch, ly.U2, dHid, v.Fc1[l], n, d, m.Ffn, g[Gi(DocParams.Fc1W)], g[Gi(DocParams.Fc1B)], dU2.AsSpan(0, n * d));
            float[] dLn = scratch.Floats((long)n * d);
            DocBackward.LayerNormBackward(_path, dU2, ly.Xhat2, ly.Inv2, b.Ln2Weight, n, d, dLn, g[Gi(DocParams.Ln2W)], g[Gi(DocParams.Ln2B)]);
            float[] dX1 = scratch.Floats((long)n * d);
            for (int i = 0; i < n * d; i++)
            {
                dX1[i] = dx[i] + dLn[i];
            }

            // x1 = x_in + proj(attn)
            float[] dAttn = scratch.Floats((long)n * d);
            LinearTrain.BackwardPass(_path, scratch, ly.Attn, dX1, v.Proj[l], n, d, d, g[Gi(DocParams.ProjW)], g[Gi(DocParams.ProjB)], dAttn.AsSpan(0, n * d));

            float[] dQkv = scratch.Floats((long)n * 3 * d);
            float[] dP = scratch.Floats((long)nn);
            float[] dRaw = scratch.Floats((long)nn);
            for (int h = 0; h < heads; h++)
            {
                ReadOnlySpan<float> probs = ly.Probs.AsSpan(h * nn, nn);
                DocBackward.ContextBackward(
                    _path, probs, dAttn.AsSpan(h * dh), d, ly.Qkv.AsSpan((2 * d) + (h * dh)), 3 * d, n, dh, dP, dQkv.AsSpan((2 * d) + (h * dh)), 3 * d);
                DocBackward.SoftmaxRowsBackward(_path, probs, dP, n, n, scale, dRaw, dPb.AsSpan(((l * heads) + h) * nn, nn));
                DocBackward.ScoresBackward(
                    _path, dRaw, ly.Qkv.AsSpan(h * dh), 3 * d, ly.Qkv.AsSpan(d + (h * dh)), 3 * d, n, dh,
                    dQkv.AsSpan(h * dh), 3 * d, dQkv.AsSpan(d + (h * dh)), 3 * d);
            }

            float[] dU1 = scratch.Floats((long)n * d);
            LinearTrain.BackwardPass(_path, scratch, ly.U1, dQkv, v.Qkv[l], n, d, 3 * d, g[Gi(DocParams.QkvW)], g[Gi(DocParams.QkvB)], dU1.AsSpan(0, n * d));
            DocBackward.LayerNormBackward(_path, dU1, ly.Xhat1, ly.Inv1, b.Ln1Weight, n, d, dLn, g[Gi(DocParams.Ln1W)], g[Gi(DocParams.Ln1B)]);
            float[] next = scratch.Floats((long)n * d);
            for (int i = 0; i < n * d; i++)
            {
                next[i] = dX1[i] + dLn[i];
            }

            dx = next;
        }

        // The embedding: layer norm, then the sum of the projected character vector, four position rows and the projected features.
        DocBackward.PairMlpBackward(
            _path, dPb, a.PairHid, a.PairFeat, v.Pair2, nn, m.PairHidden, m.PairChannels, g[DocParams.Tail(m, DocParams.Pair1W)], g[DocParams.Tail(m, DocParams.Pair1B)],
            g[DocParams.Tail(m, DocParams.Pair2W)], g[DocParams.Tail(m, DocParams.Pair2B)]);

        float[] dx0 = scratch.Floats((long)n * d);
        DocBackward.LayerNormBackward(_path, dx.AsSpan(0, n * d), a.EmbXhat, a.EmbInv, m.EmbLnWeight, n, d, dx0, g[DocParams.EmbLnW], g[DocParams.EmbLnB]);
        LinearTrain.BackwardPass(_path, scratch, a.FeatPad, dx0, default, n, 16, d, g[DocParams.FeatW], g[DocParams.FeatB], default);
        int[] ids = scratch.Ints(n);
        foreach ((int column, int slot) in new[] { (0, DocParams.PosX0), (2, DocParams.PosX1), (1, DocParams.PosY0), (3, DocParams.PosY1) })
        {
            for (int i = 0; i < n; i++)
            {
                ids[i] = a.F.Bins[(i * 4) + column];
            }

            DocBackward.EmbeddingBackward(dx0, d, ids.AsSpan(0, n), d, m.Bins, g[slot]);
        }

        float[] dPool = scratch.Floats((long)n * cc);
        LinearTrain.BackwardPass(_path, scratch, a.Pool, dx0, v.CharProj, n, cc, d, g[DocParams.CharProjW], g[DocParams.CharProjB], dPool.AsSpan(0, n * cc));
        int p = a.Positions;
        if (p == 0)
        {
            Array.Clear(g[DocParams.CharConvW]);
            Array.Clear(g[DocParams.CharConvB]);
            Array.Clear(g[DocParams.CharEmb]);
            return;
        }

        float[] dConv = scratch.Zeros((long)p * cc);
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < cc; c++)
            {
                int r = a.Argmax[(i * cc) + c];
                if (r >= 0)
                {
                    dConv[(r * cc) + c] = dPool[(i * cc) + c];
                }
            }
        }

        float[] dWin = scratch.Floats((long)p * 3 * cd);
        LinearTrain.BackwardPass(_path, scratch, a.Windows, dConv, v.Conv, p, 3 * cd, cc, g[DocParams.CharConvW], g[DocParams.CharConvB], dWin.AsSpan(0, p * 3 * cd));

        // Window row r, tap k holds character (q - 1 + k) of its word; its gradient goes to that character's table row (row 0, the
        // padding row, stays at zero like PyTorch's padding_idx).
        int mc = m.MaxChars;
        int[] tapIds = scratch.Ints((long)p * 3);
        for (int i = 0; i < n; i++)
        {
            for (int q = 0; q < a.Last[i]; q++)
            {
                for (int k = 0; k < 3; k++)
                {
                    int idx = q - 1 + k;
                    tapIds[((a.Start[i] + q) * 3) + k] = (uint)idx < (uint)mc ? a.F.Chars[(i * mc) + idx] : -1;
                }
            }
        }

        DocBackward.EmbeddingBackward(dWin, cd, tapIds.AsSpan(0, p * 3), cd, m.CharEmb.Length / cd, g[DocParams.CharEmb], skipId: 0);
    }

    private double ExampleStep(DocExample ex, int batchSize, Views views, float[][] grads)
    {
        using var scratch = new TrainScratch(_floats, _ints);
        Activations a = Forward(scratch, ex.Features);
        double loss = Losses(scratch, a, ex.Targets, 1.0 / batchSize, views, grads);
        return loss / batchSize;
    }
}
