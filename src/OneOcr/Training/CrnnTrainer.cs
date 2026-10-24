using System.Text;
using OneOcr.Kernels;
using OneOcr.Recognition;

namespace OneOcr.Training;

/// <summary>One training example: a line image already scaled to the model's input height, and its target as charset indices (never 0, the blank).</summary>
internal readonly record struct TrainLine(ReadOnlyMemory<byte> Gray, int Width, int[] Target);

/// <summary>
/// One optimization step of the <c>crnn-v0</c> recognizer: forward, CTC loss, backward, AdamW. The model's weight arrays are the
/// fp32 master weights and are updated in place, so the model must not be shared with a recognizer that is still in use (a
/// recognizer keeps derived copies of some weights). Lines of a batch run in parallel, each producing its own gradient; the
/// gradients are then added in line order, so the result does not depend on the degree of parallelism or the kernel path
/// (docs/spec/backward-numerics.md).
/// </summary>
internal sealed class CrnnTrainer
{
    private readonly CrnnModel _model;
    private readonly KernelPath _path;
    private readonly OcrLimits _limits;
    private readonly double _maxGradNorm;
    private readonly BufferPool<float> _floats;
    private readonly BufferPool<int> _ints;
    private readonly Dictionary<string, int> _classOf = new(StringComparer.Ordinal);
    private readonly float[][] _parameters;
    private readonly string[] _names;
    private readonly float[][] _grad;
    private readonly List<float[][]> _lineGrads = [];
    private readonly AdamW _optimizer;

    /// <summary>Creates a trainer that updates <paramref name="model"/>'s weights in place.</summary>
    /// <param name="model">The model; the trainer takes exclusive ownership of its weight arrays.</param>
    /// <param name="path">The kernel path; all paths give identical results.</param>
    /// <param name="options">The AdamW settings.</param>
    /// <param name="limits">Size limits; null means the defaults.</param>
    /// <param name="maxGradNorm">Clip the global gradient norm to this value; 0 turns clipping off.</param>
    /// <param name="decayBiases">Whether biases receive weight decay too; the default leaves them out.</param>
    public CrnnTrainer(CrnnModel model, KernelPath path, AdamWOptions options, OcrLimits? limits = null, double maxGradNorm = 0, bool decayBiases = false)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(options);
        if (!(maxGradNorm >= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(maxGradNorm), "The clipping norm must be zero or positive.");
        }

        _model = model;
        _path = Dispatch.Require(path);
        _limits = limits ?? OcrLimits.Default;
        _maxGradNorm = maxGradNorm;
        _floats = new BufferPool<float>(_limits.MaxPooledBufferBytes, sizeof(float));
        _ints = new BufferPool<int>(_limits.MaxPooledBufferBytes, sizeof(int));
        for (int i = 0; i < model.Charset.Count; i++)
        {
            _classOf.TryAdd(model.Charset[i], i);
        }

        // Fixed parameter order: conv (weight, bias) per block; per LSTM layer and direction (W_ih, W_hh, bias); classifier (weight, bias).
        var values = new List<float[]>();
        var names = new List<string>();
        var decay = new List<bool>();
        void Add(string name, float[] v, bool weightDecay)
        {
            names.Add(name);
            values.Add(v);
            decay.Add(weightDecay);
        }

        for (int i = 0; i < model.Conv.Length; i++)
        {
            Add($"conv.{i}.weight", model.Conv[i].Weights, true);
            Add($"conv.{i}.bias", model.Conv[i].Bias, decayBiases);
        }

        for (int l = 0; l < model.Lstm.Length; l++)
        {
            for (int d = 0; d < 2; d++)
            {
                LstmDirection dir = d == 0 ? model.Lstm[l].Forward : model.Lstm[l].Reverse;
                string sfx = d == 0 ? string.Empty : "_reverse";
                Add($"lstm.weight_ih_l{l}{sfx}", dir.WeightIh, true);
                Add($"lstm.weight_hh_l{l}{sfx}", dir.WeightHh, true);
                Add($"lstm.bias_l{l}{sfx}", dir.Bias, decayBiases);
            }
        }

        Add("fc.weight", model.FcWeight, true);
        Add("fc.bias", model.FcBias, decayBiases);
        _parameters = [.. values];
        _names = [.. names];
        _grad = _parameters.Select(p => new float[p.Length]).ToArray();
        _optimizer = new AdamW(_parameters, [.. decay], options);
    }

    /// <summary>The trainable tensors in their fixed order (conv weight and bias, LSTM W_ih, W_hh and combined bias, classifier), updated in place.</summary>
    public IReadOnlyList<float[]> Parameters => _parameters;

    public IReadOnlyList<string> ParameterNames => _names;

    /// <summary>The gradients of the last <see cref="Step"/> (after clipping), in the order of <see cref="Parameters"/>.</summary>
    public IReadOnlyList<float[]> Gradients => _grad;

    public AdamW Optimizer => _optimizer;

    /// <summary>The global gradient norm of the last step before clipping.</summary>
    public double LastGradNorm { get; private set; }

    /// <summary>Maps text to charset indices, one per character. Unknown characters are an error: a training set must be inside the charset.</summary>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var result = new List<int>();
        foreach (Rune r in text.EnumerateRunes())
        {
            if (!_classOf.TryGetValue(r.ToString(), out int cls) || cls == 0)
            {
                throw new ArgumentException($"The character U+{r.Value:X4} is not in the model's charset.", nameof(text));
            }

            result.Add(cls);
        }

        return [.. result];
    }

    /// <summary>Greedy CTC decoding of logits <c>[steps][classes]</c> into text: best class per step, repeats collapsed, then blanks removed.</summary>
    public string DecodeGreedy(ReadOnlySpan<float> logits, int steps)
    {
        int classes = _model.Classes;
        var sb = new StringBuilder();
        int last = 0;
        for (int t = 0; t < steps; t++)
        {
            int best = 0;
            for (int k = 1; k < classes; k++)
            {
                if (logits[(t * classes) + k] > logits[(t * classes) + best])
                {
                    best = k;
                }
            }

            if (best != 0 && best != last)
            {
                sb.Append(_model.Charset[best]);
            }

            last = best;
        }

        return sb.ToString();
    }

    /// <summary>Runs the network on a line (no gradients kept) and returns its logits <c>[steps][classes]</c>; steps is 0 for a line too narrow for the stack.</summary>
    public float[] Forward(ReadOnlySpan<byte> gray, int width, out int steps)
    {
        using var scratch = new TrainScratch(_floats, _ints);
        LineRun run = RunForward(scratch, gray, width);
        steps = run.Steps;
        return run.Steps == 0 ? [] : run.Logits.AsSpan(0, run.Steps * _model.Classes).ToArray();
    }

    /// <summary>
    /// One training step on <paramref name="batch"/>. The loss is the mean over the batch of each line's CTC negative log likelihood (not
    /// divided by target length, unlike PyTorch's <c>reduction='mean'</c>); a line with no possible alignment contributes 0 and no
    /// gradient, as with <c>zero_infinity=True</c>. Returns that loss, computed before the update.
    /// </summary>
    /// <param name="batch">The examples; at least one.</param>
    /// <param name="degree">Lines processed in parallel; 1 or less runs serially.</param>
    /// <param name="learningRate">Overrides the optimizer's learning rate for this step (for schedules).</param>
    /// <param name="cancellationToken">Observed between lines.</param>
    public double Step(IReadOnlyList<TrainLine> batch, int degree = 1, double? learningRate = null, CancellationToken cancellationToken = default)
    {
        double loss = ComputeGradients(batch, degree, cancellationToken);
        if (_maxGradNorm > 0 && LastGradNorm > _maxGradNorm)
        {
            float scale = (float)(_maxGradNorm / (LastGradNorm + 1e-6));
            foreach (float[] g in _grad)
            {
                for (int j = 0; j < g.Length; j++)
                {
                    g[j] *= scale;
                }
            }
        }

        _optimizer.Step(_grad, learningRate ?? _optimizer.Options.LearningRate);
        return loss;
    }

    /// <summary>The loss of <see cref="Step"/> for a batch, without computing gradients or changing anything.</summary>
    public double Evaluate(IReadOnlyList<TrainLine> batch)
    {
        ArgumentNullException.ThrowIfNull(batch);
        double total = 0;
        foreach (TrainLine line in batch)
        {
            using var scratch = new TrainScratch(_floats, _ints);
            LineRun run = RunForward(scratch, line.Gray.Span, line.Width);
            if (run.Steps > 0)
            {
                float[] sink = scratch.Floats((long)run.Steps * _model.Classes);
                double loss = CtcLoss.Compute(run.Logits, run.Steps, _model.Classes, line.Target, 0, sink);
                if (!double.IsInfinity(loss) && !double.IsNaN(loss))
                {
                    total += loss;
                }
            }
        }

        return total / batch.Count;
    }

    /// <summary>
    /// Forward and backward over the batch: fills <see cref="Gradients"/> (unclipped) and <see cref="LastGradNorm"/> and returns the
    /// loss, without touching the weights.
    /// </summary>
    public double ComputeGradients(IReadOnlyList<TrainLine> batch, int degree = 1, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(batch);
        if (batch.Count == 0)
        {
            throw new ArgumentException("A batch needs at least one line.", nameof(batch));
        }

        while (_lineGrads.Count < batch.Count)
        {
            _lineGrads.Add(_parameters.Select(p => new float[p.Length]).ToArray());
        }

        WeightViews views = BuildViews();
        var losses = new double[batch.Count];
        ParallelRun.For(batch.Count, degree, b => losses[b] = LineStep(batch[b], batch.Count, views, _lineGrads[b]), cancellationToken);

        double total = 0;
        for (int b = 0; b < batch.Count; b++)
        {
            total += losses[b];
        }

        ReduceGradients(batch.Count, degree, cancellationToken);
        return total;
    }

    // ---- Parameter indexing ----

    private static int ConvWeight(int block) => 2 * block;

    private int LstmBase(int layer, int direction) => (2 * _model.Conv.Length) + (((layer * 2) + direction) * 3);

    private int FcBase => (2 * _model.Conv.Length) + (6 * _model.Lstm.Length);

    /// <summary>The weights in the transposed layouts the backward products read, rebuilt once per step.</summary>
    private sealed class WeightViews
    {
        public required float[][] ConvT { get; init; }
        public required float[][][] WhhT { get; init; }
        public required float[][] WihCatT { get; init; }
        public required float[] FcT { get; init; }
    }

    private WeightViews BuildViews()
    {
        int hidden = _model.Hidden, gates = 4 * hidden;
        var convT = new float[_model.Conv.Length][];
        for (int i = 0; i < convT.Length; i++)
        {
            ConvBlock b = _model.Conv[i];
            convT[i] = new float[b.KPadded * b.Out];
            Backward.Transpose(b.Weights, b.KPadded, b.Out, b.KPadded, convT[i]);
        }

        var whhT = new float[_model.Lstm.Length][][];
        var wihCatT = new float[_model.Lstm.Length][];
        for (int l = 0; l < whhT.Length; l++)
        {
            LstmLayer layer = _model.Lstm[l];
            whhT[l] = new float[2][];
            wihCatT[l] = new float[layer.In * 2 * gates];
            for (int d = 0; d < 2; d++)
            {
                LstmDirection dir = d == 0 ? layer.Forward : layer.Reverse;
                whhT[l][d] = new float[hidden * gates];
                Backward.Transpose(dir.WeightHh, hidden, gates, hidden, whhT[l][d]);
                var tmp = new float[layer.In * gates];
                Backward.Transpose(dir.WeightIh, layer.In, gates, layer.In, tmp);
                for (int i = 0; i < layer.In; i++)
                {
                    tmp.AsSpan(i * gates, gates).CopyTo(wihCatT[l].AsSpan((i * 2 * gates) + (d * gates), gates));
                }
            }
        }

        var fcT = new float[_model.FcWeight.Length];
        Backward.Transpose(_model.FcWeight, 2 * hidden, _model.Classes, 2 * hidden, fcT);
        return new WeightViews { ConvT = convT, WhhT = whhT, WihCatT = wihCatT, FcT = fcT };
    }

    // ---- Forward ----

    private sealed class LineRun
    {
        public int Steps { get; init; }
        public float[] Logits { get; init; } = [];
        public ConvCache[] Conv { get; init; } = [];

        /// <summary>The input of each convolution block, then of the first LSTM layer.</summary>
        public float[][] ConvIn { get; init; } = [];
        public LstmCache[] Lstm { get; init; } = [];

        /// <summary>The output of each LSTM layer.</summary>
        public float[][] LstmOut { get; init; } = [];
    }

    private LineRun RunForward(TrainScratch scratch, ReadOnlySpan<byte> gray, int width)
    {
        int height = _model.Input.Height;
        if (width < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "The width must be positive.");
        }

        if (width > _limits.MaxLineWidth)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLineWidth), _limits.MaxLineWidth, width);
        }

        if (gray.Length != (long)width * height)
        {
            throw new ArgumentException($"Expected {height} rows of {width} pixels ({(long)width * height} bytes), got {gray.Length}.", nameof(gray));
        }

        // Check every buffer size before renting anything.
        int h = height, w = width;
        foreach (ConvBlock b in _model.Conv)
        {
            int ch = h + (2 * b.PadH) - b.KernelH + 1, cw = w + (2 * b.PadW) - b.KernelW + 1;
            if (cw < 1)
            {
                return new LineRun();
            }

            CheckElements((long)ch * cw * b.KPadded);
            CheckElements((long)ch * cw * b.Out);
            h = ch / b.PoolH;
            w = cw / b.PoolW;
            if (w < 1)
            {
                return new LineRun();
            }
        }

        CheckElements((long)w * 8 * _model.Hidden);
        CheckElements((long)w * _model.Classes);

        float[] input = scratch.Floats(gray.Length);
        CrnnForward.NormalizeInto(gray, new ModelInputView(_model.Input.Mean, _model.Input.Std, _model.Input.Invert), input);
        var convCaches = new ConvCache[_model.Conv.Length];
        var convIn = new float[_model.Conv.Length + 1][];
        convIn[0] = input;
        h = height;
        w = width;
        for (int i = 0; i < _model.Conv.Length; i++)
        {
            convIn[i + 1] = ConvTrain.Forward(_model.Conv[i], _path, scratch, convIn[i], h, w, out convCaches[i]);
            h = convCaches[i].OutH;
            w = convCaches[i].OutW;
        }

        int t = w;
        int hidden = _model.Hidden;
        var lstmCaches = new LstmCache[_model.Lstm.Length];
        var lstmOut = new float[_model.Lstm.Length][];
        float[] x = convIn[^1];
        for (int l = 0; l < _model.Lstm.Length; l++)
        {
            lstmOut[l] = LstmTrain.Forward(_model.Lstm[l], hidden, _path, scratch, x, t, out lstmCaches[l]);
            x = lstmOut[l];
        }

        float[] logits = scratch.Floats((long)t * _model.Classes);
        Gemm.NT(_path, x, 2 * hidden, _model.FcWeight, 2 * hidden, t, _model.Classes, 2 * hidden, _model.FcBias, logits, _model.Classes, 1);
        return new LineRun { Steps = t, Logits = logits, Conv = convCaches, ConvIn = convIn, Lstm = lstmCaches, LstmOut = lstmOut };
    }

    private void CheckElements(long elements)
    {
        if (elements > _limits.MaxActivationElements)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxActivationElements), _limits.MaxActivationElements, elements);
        }
    }

    // ---- One line: forward, loss, backward into its own gradient set ----

    private double LineStep(TrainLine line, int batchSize, WeightViews views, float[][] grads)
    {
        using var scratch = new TrainScratch(_floats, _ints);
        LineRun run = RunForward(scratch, line.Gray.Span, line.Width);
        int classes = _model.Classes, hidden = _model.Hidden;
        double loss = double.PositiveInfinity;
        float[]? dLogits = null;
        if (run.Steps > 0)
        {
            double scale = 1.0 / batchSize;
            dLogits = scratch.Floats((long)run.Steps * classes);
            loss = CtcLoss.Compute(run.Logits, run.Steps, classes, line.Target, scale, dLogits);
        }

        if (double.IsInfinity(loss) || double.IsNaN(loss))
        {
            foreach (float[] g in grads)
            {
                Array.Clear(g);
            }

            return 0;
        }

        int t = run.Steps;

        // Classifier.
        float[] top = run.LstmOut[^1];
        int fc = FcBase;
        float[] dX = scratch.Floats((long)t * 2 * hidden);
        LinearTrain.BackwardPass(_path, scratch, top, dLogits!, views.FcT, t, 2 * hidden, classes, grads[fc], grads[fc + 1], dX);

        // LSTM layers, top to bottom.
        for (int l = _model.Lstm.Length - 1; l >= 0; l--)
        {
            LstmLayer layer = _model.Lstm[l];
            float[] dIn = scratch.Floats((long)t * layer.In);
            int f = LstmBase(l, 0), r = LstmBase(l, 1);
            LstmTrain.BackwardPass(
                layer, hidden, _path, scratch, run.Lstm[l], dX, views.WhhT[l], views.WihCatT[l],
                [grads[f], grads[r]], [grads[f + 1], grads[r + 1]], [grads[f + 2], grads[r + 2]], dIn.AsSpan(0, t * layer.In));
            dX = dIn;
        }

        // Convolution blocks, top to bottom; the first block's input gradient is not needed.
        for (int i = _model.Conv.Length - 1; i >= 0; i--)
        {
            ConvBlock b = _model.Conv[i];
            ConvCache cache = run.Conv[i];
            float[]? dIn = i > 0 ? scratch.Floats((long)cache.H * cache.W * b.In) : null;
            ConvTrain.BackwardPass(
                b, _path, scratch, cache, views.ConvT[i], dX, grads[ConvWeight(i)], grads[ConvWeight(i) + 1],
                dIn is null ? default : dIn.AsSpan(0, cache.H * cache.W * b.In));
            if (dIn is not null)
            {
                dX = dIn;
            }
        }

        return loss / batchSize;
    }

    // Sums the per-line gradients in line order (plain additions starting from line 0), splitting the work by tensor range.
    private void ReduceGradients(int lines, int degree, CancellationToken cancellationToken)
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

        ParallelRun.For(work.Count, degree, i =>
        {
            (int p, int s) = work[i];
            int n = Math.Min(Chunk, _grad[p].Length - s);
            Span<float> dst = _grad[p].AsSpan(s, n);
            _lineGrads[0][p].AsSpan(s, n).CopyTo(dst);
            for (int b = 1; b < lines; b++)
            {
                ReadOnlySpan<float> src = _lineGrads[b][p].AsSpan(s, n);
                for (int j = 0; j < n; j++)
                {
                    dst[j] += src[j];
                }
            }
        }, cancellationToken);

        // Global norm: per-tensor sum of squares as a library-order dot product, tensors added in order in double precision.
        double sum = 0;
        foreach (float[] g in _grad)
        {
            sum += Gemm.Dot(_path, g, g);
        }

        LastGradNorm = Math.Sqrt(sum);
    }
}
