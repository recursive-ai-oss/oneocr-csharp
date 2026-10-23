using System.Text;
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

    internal LineRecognizer(CrnnModel model, OcrLimits limits, KernelPath path)
    {
        _model = model;
        _limits = limits;
        _path = Dispatch.Require(path);
        _pool = new BufferPool<float>(limits.MaxPooledBufferBytes, sizeof(float));
    }

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
    public RecognizedLine Recognize(ReadOnlySpan<byte> gray, int width, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default)
    {
        float[] logits = Forward(gray, width, maxDegreeOfParallelism, null, cancellationToken, out int steps);
        try
        {
            return Decode(logits, steps);
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

    internal RecognizedLine Decode(float[] logits, int steps)
    {
        int classes = _model.Classes;
        var text = new StringBuilder();
        int last = 0;
        float runMax = 0f;
        float total = 0f;
        int count = 0;
        for (int t = 0; t < steps; t++)
        {
            ReadOnlySpan<float> frame = logits.AsSpan(t * classes, classes);
            // Greedy CTC (Graves et al. 2006, sec. 3.1): arg-max per step (lowest index on ties), merge repeats, drop blanks.
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
                total += runMax;
                count++;
            }

            if (best != 0)
            {
                // Softmax probability of the winning class: exp(0) / sum exp(z - max), with exp(0) exactly 1.
                float p = 1f / Elementwise.SumExp(_path, frame, bestValue);
                if (best != last)
                {
                    text.Append(_model.Charset[best]);
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
            total += runMax;
            count++;
        }

        return new RecognizedLine(text.ToString(), count == 0 ? 0f : total / count);
    }
}
