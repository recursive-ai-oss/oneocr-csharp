using System.Text;
using OneOcr.Decoding;
using OneOcr.Kernels;
using OneOcr.Model;

namespace OneOcr.Recognition;

/// <summary>The text read from one line image.</summary>
/// <param name="Text">The decoded text; empty when nothing was read.</param>
/// <param name="Confidence">
/// A score in 0..1: the mean, over the emitted characters, of the highest softmax probability seen while that character was
/// the winning class. 0 for an empty line. Not yet calibrated.
/// </param>
public readonly record struct RecognizedLine(string Text, float Confidence);

/// <summary>
/// Reads a single line of text from a grayscale image with a <c>crnn-v0</c> model: convolutional features, a bidirectional LSTM,
/// a per-step classifier and greedy CTC decoding. Instances are immutable and thread-safe.
/// Output is bit-identical on every supported CPU path and at every degree of parallelism.
/// </summary>
public sealed class LineRecognizer
{
    private readonly CrnnModel _model;
    private readonly KernelPath _path;
    private readonly OcrLimits _limits;
    private readonly BufferPool<float> _pool;
    private readonly bool[] _isSpace;

    internal LineRecognizer(CrnnModel model, OcrLimits limits, KernelPath path)
    {
        _model = model;
        _limits = limits;
        _path = Dispatch.Require(path);
        _pool = new BufferPool<float>(limits.MaxPooledBufferBytes, sizeof(float));
        _isSpace = new bool[model.Charset.Count];
        for (int i = 1; i < _isSpace.Length; i++)
        {
            _isSpace[i] = IsWhitespaceEntry(model.Charset[i]);
        }
    }

    // A class whose text is only whitespace separates words.
    private static bool IsWhitespaceEntry(string entry)
    {
        foreach (char ch in entry)
        {
            if (!char.IsWhiteSpace(ch))
            {
                return false;
            }
        }

        return entry.Length > 0;
    }

    internal bool HasCalibration => _model.Calibration is not null;

    /// <summary>The height, in pixels, line images must have.</summary>
    public int InputHeight => _model.Input.Height;

    /// <summary>Loads a recognizer from a model file.</summary>
    /// <param name="path">A container file with architecture <c>crnn-v0</c>.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The recognizer.</returns>
    /// <exception cref="OcrModelException">The file is malformed or describes another architecture.</exception>
    /// <exception cref="OcrLimitException">The file or model exceeds <paramref name="limits"/>.</exception>
    public static LineRecognizer Load(string path, OcrLimits? limits = null) =>
        FromContainer(ModelContainer.Load(path, limits), limits);

    /// <summary>Builds a recognizer from a parsed container.</summary>
    /// <param name="container">A container with architecture <c>crnn-v0</c>.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The recognizer.</returns>
    /// <exception cref="OcrModelException">The container describes another architecture or inconsistent shapes.</exception>
    /// <exception cref="OcrLimitException">The model exceeds <paramref name="limits"/>.</exception>
    public static LineRecognizer FromContainer(ModelContainer container, OcrLimits? limits = null)
    {
        limits ??= OcrLimits.Default;
        return new LineRecognizer(CrnnModel.FromContainer(container, limits), limits, Dispatch.Best);
    }

    /// <summary>Reads one line.</summary>
    /// <param name="gray">8-bit grayscale pixels, row-major, <see cref="InputHeight"/> rows of <paramref name="width"/> pixels.</param>
    /// <param name="width">Width in pixels, at most <see cref="OcrLimits.MaxLineWidth"/>.</param>
    /// <param name="maxDegreeOfParallelism">Most threads to use; 0 means min(4, cores). The result does not depend on it.</param>
    /// <param name="cancellationToken">Cancels between layers.</param>
    /// <returns>The text and its confidence. A line too narrow for the network (fewer columns than its receptive field) reads as empty.</returns>
    /// <exception cref="ArgumentException">The pixel count is not height times width, or the width is not positive.</exception>
    /// <exception cref="OcrLimitException">The width or the activations it implies exceed the limits.</exception>
    public RecognizedLine Recognize(ReadOnlySpan<byte> gray, int width, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default) =>
        Recognize(gray, width, null, maxDegreeOfParallelism, cancellationToken);

    /// <summary>Reads one line with the given decoding.</summary>
    /// <param name="gray">8-bit grayscale pixels, row-major, <see cref="InputHeight"/> rows of <paramref name="width"/> pixels.</param>
    /// <param name="width">Width in pixels, at most <see cref="OcrLimits.MaxLineWidth"/>.</param>
    /// <param name="decoding">How to turn the network output into text; null or <see cref="DecodeOptions.Greedy"/> for greedy decoding.</param>
    /// <param name="maxDegreeOfParallelism">Most threads to use; 0 means min(4, cores). The result does not depend on it.</param>
    /// <param name="cancellationToken">Cancels between layers.</param>
    /// <returns>The text and its confidence. With beam search the confidence is the per-character geometric mean of the probability of the text summed over alignments.</returns>
    /// <exception cref="ArgumentException">The pixel count is not height times width, or the width is not positive.</exception>
    /// <exception cref="ArgumentException">A decoding setting is out of range.</exception>
    /// <exception cref="OcrLimitException">The width, the activations it implies or the beam search exceed the limits.</exception>
    public RecognizedLine Recognize(
        ReadOnlySpan<byte> gray, int width, DecodeOptions? decoding, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default)
    {
        decoding?.Validate(_limits);
        float[] logits = Forward(gray, width, maxDegreeOfParallelism, null, cancellationToken, out int steps);
        try
        {
            return decoding is { Mode: CtcDecoding.Beam } ? DecodeBeam(logits, steps, decoding) : Decode(logits, steps);
        }
        finally
        {
            if (logits.Length > 0)
            {
                _pool.Return(logits);
            }
        }
    }

    internal float[] Forward(
        ReadOnlySpan<byte> gray, int width, int maxDegreeOfParallelism, ForwardTrace? trace, CancellationToken cancellationToken, out int steps)
    {
        if (width < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "The width must be positive.");
        }

        if (width > _limits.MaxLineWidth)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxLineWidth), _limits.MaxLineWidth, width);
        }

        if (gray.Length != (long)width * InputHeight)
        {
            throw new ArgumentException($"Expected {InputHeight} rows of {width} pixels ({(long)width * InputHeight} bytes), got {gray.Length}.", nameof(gray));
        }

        int degree = maxDegreeOfParallelism > 0 ? maxDegreeOfParallelism : Math.Min(4, Environment.ProcessorCount);
        float[] input = _pool.Rent(gray.Length);
        try
        {
            CrnnForward.NormalizeInto(gray, new ModelInputView(_model.Input.Mean, _model.Input.Std, _model.Input.Invert), input);
            return CrnnForward.Run(_model, _path, degree, _pool, input, width, _limits, trace, out steps, cancellationToken);
        }
        finally
        {
            _pool.Return(input);
        }
    }

    internal RecognizedLine DecodeBeam(float[] logits, int steps, DecodeOptions options)
    {
        long nodes = BeamDecoder.MaxNodes(options, steps);
        if (nodes > _limits.MaxBeamNodes)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxBeamNodes), _limits.MaxBeamNodes, nodes);
        }

        return BeamDecoder.Decode(options, options.Prepare(_model.Charset), logits, steps, _model.Classes, _path, Decode(logits, steps));
    }

    internal RecognizedLine Decode(float[] logits, int steps)
    {
        List<CharRun> runs = RentRuns(steps);
        CollectRuns(logits, steps, runs);
        return Decode(runs);
    }

    // Each thread keeps one list of runs and two string builders for the lines it reads, so that a line leaves no scratch behind.
    // Characters are about two steps apiece, which is what the list is sized to.
    [ThreadStatic]
    private static List<CharRun>? t_runs;

    [ThreadStatic]
    private static StringBuilder? t_lineText;

    [ThreadStatic]
    private static StringBuilder? t_wordText;

    private static List<CharRun> RentRuns(int steps)
    {
        List<CharRun> runs = t_runs ??= [];
        runs.Clear();
        runs.EnsureCapacity(Math.Max(4, (steps / 2) + 1));
        return runs;
    }

    private RecognizedLine Decode(List<CharRun> runs)
    {
        StringBuilder text = t_lineText ??= new StringBuilder();
        text.Clear();
        float total = 0f;
        foreach (CharRun run in runs)
        {
            text.Append(_model.Charset[run.Class]);
            total += run.MaxProbability;
        }

        return new RecognizedLine(text.ToString(), runs.Count == 0 ? 0f : total / runs.Count);
    }

    /// <summary>Reads a line and splits it into words with boxes (in pixels of the line image) and calibrated confidences. Greedy decoding.</summary>
    internal LineWords RecognizeWords(ReadOnlySpan<byte> gray, int width, int maxDegreeOfParallelism, CancellationToken cancellationToken, bool withChars = true)
    {
        float[] logits = Forward(gray, width, maxDegreeOfParallelism, null, cancellationToken, out int steps);
        try
        {
            return SplitWords(logits, steps, width, withChars);
        }
        finally
        {
            if (logits.Length > 0)
            {
                _pool.Return(logits);
            }
        }
    }

    // The number of groups of non-space characters.
    private int CountWords(List<CharRun> runs)
    {
        int words = 0;
        bool inWord = false;
        foreach (CharRun run in runs)
        {
            bool space = _isSpace[run.Class];
            if (!space && !inWord)
            {
                words++;
            }

            inWord = !space;
        }

        return words;
    }

    internal LineWords SplitWords(float[] logits, int steps, int width, bool withChars = true)
    {
        List<CharRun> runs = RentRuns(steps);
        CollectRuns(logits, steps, runs);
        RecognizedLine line = Decode(runs);
        var words = new List<RecognizedWord>(CountWords(runs));
        StringBuilder text = t_wordText ??= new StringBuilder();
        text.Clear();
        float total = 0f;
        double stepWidth = steps == 0 ? 0 : (double)width / steps;
        int first = -1;
        for (int i = 0; i <= runs.Count; i++)
        {
            bool boundary = i == runs.Count || _isSpace[runs[i].Class];
            if (!boundary)
            {
                if (first < 0)
                {
                    first = i;
                    text.Clear();
                    total = 0f;
                }

                text.Append(_model.Charset[runs[i].Class]);
                total += runs[i].MaxProbability;
                continue;
            }

            if (first < 0)
            {
                continue;
            }

            int last = i - 1;
            // A character is a run of steps; its box edge toward a neighbour is halfway across the gap between the two runs, so
            // neighbouring words tile the line. The ends of the line stop at the first and last step that fired.
            double left = first == 0 ? runs[first].First : (runs[first - 1].Last + 1 + runs[first].First) / 2.0;
            double right = last == runs.Count - 1 ? runs[last].Last + 1 : (runs[last].Last + 1 + runs[last + 1].First) / 2.0;
            string word = text.ToString();
            float raw = total / (last - first + 1);
            OcrScript script = ScriptDetector.Of(word);
            float calibrated = _model.Calibration is { } c ? c.Apply(script, raw) : raw;
            // Callers that only want the text and the word boxes skip the per-character list.
            IReadOnlyList<RecognizedChar> chars = [];
            if (withChars)
            {
                var list = new List<RecognizedChar>(last - first + 1);
                for (int k = first; k <= last; k++)
                {
                    list.Add(new RecognizedChar(_model.Charset[runs[k].Class], runs[k].First * stepWidth, (runs[k].Last + 1) * stepWidth, runs[k].MaxProbability));
                }

                chars = list;
            }

            words.Add(new RecognizedWord(word, (float)(left * stepWidth), (float)(right * stepWidth), calibrated, raw, script, last - first + 1, chars));
            first = -1;
        }

        return new LineWords(line.Text, line.Confidence, words);
    }

    // Greedy CTC (Graves et al. 2006, sec. 3.1): arg-max per step (lowest index on ties), merge repeats, drop blanks. A run is
    // the steps of one emitted character; its probability is the largest softmax probability of its class over the run.
    private void CollectRuns(float[] logits, int steps, List<CharRun> runs)
    {
        int classes = _model.Classes;
        int last = 0;
        int runFirst = 0;
        float runMax = 0f;
        for (int t = 0; t < steps; t++)
        {
            ReadOnlySpan<float> frame = logits.AsSpan(t * classes, classes);
            int best = 0;
            float bestValue = frame[0];
            for (int j = 1; j < frame.Length; j++)
            {
                if (frame[j] > bestValue)
                {
                    bestValue = frame[j];
                    best = j;
                }
            }

            if (best != last && last != 0)
            {
                runs.Add(new CharRun(last, runFirst, t - 1, runMax));
            }

            if (best != 0)
            {
                // Softmax probability of the winning class: exp(0) / sum exp(z - max), with exp(0) exactly 1.
                float p = 1f / Elementwise.SumExp(_path, frame, bestValue);
                if (best != last)
                {
                    runFirst = t;
                    runMax = p;
                }
                else if (p > runMax)
                {
                    runMax = p;
                }
            }

            last = best;
        }

        if (last != 0)
        {
            runs.Add(new CharRun(last, runFirst, steps - 1, runMax));
        }
    }
}

/// <summary>One emitted character of a greedy decoding: its class, the first and last step of its run, and the run's largest probability.</summary>
internal readonly record struct CharRun(int Class, int First, int Last, float MaxProbability);

/// <summary>A word of a line: x edges in pixels of the line image, calibrated and raw confidence, script, and the number of characters it was built from.</summary>
internal readonly record struct RecognizedWord(
    string Text, float X0, float X1, float Confidence, float RawConfidence, OcrScript Script, int Characters, IReadOnlyList<RecognizedChar> Chars);

/// <summary>One character of a recognized word: its text, the extent of its run of steps in pixels of the line image, and the run's largest probability.</summary>
internal readonly record struct RecognizedChar(string Text, double X0, double X1, float Probability);

/// <summary>A decoded line with its words. <c>RawConfidence</c> is the whole-line rule of <see cref="RecognizedLine"/>.</summary>
internal sealed record LineWords(string Text, float RawConfidence, IReadOnlyList<RecognizedWord> Words);
