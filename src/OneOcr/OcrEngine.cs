using System.Text;
using OneImage;
using OneOcr.Decoding;
using OneOcr.Kernels;
using OneOcr.Layout;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr;

/// <summary>
/// Reads text from images with a loaded recognition model. Immutable and thread-safe; create one and share it.
/// Output is bit-identical on every supported CPU path and at every degree of parallelism.
/// </summary>
public sealed class OcrEngine
{
    private readonly LineRecognizer _recognizer;
    private readonly OcrLimits _limits;
    private readonly BufferPool<Gray8> _pixels;
    private readonly DecodeOptions _decoding;

    private OcrEngine(LineRecognizer recognizer, OcrLimits limits, BufferPool<Gray8> pixels, DecodeOptions decoding)
    {
        _recognizer = recognizer;
        _limits = limits;
        _pixels = pixels;
        _decoding = decoding;
    }

    /// <summary>The decoding this engine uses unless a call passes its own. Greedy unless <see cref="WithDecoding"/> changed it.</summary>
    public DecodeOptions Decoding => _decoding;

    /// <summary>An engine that shares this one's loaded model but decodes with <paramref name="decoding"/> by default.</summary>
    /// <param name="decoding">The decoding to use when a call does not pass one.</param>
    /// <returns>The new engine; this one is unchanged.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="decoding"/> is null.</exception>
    /// <exception cref="ArgumentException">A decoding setting is out of range.</exception>
    /// <exception cref="OcrLimitException">The beam is wider than <see cref="OcrLimits.MaxBeamWidth"/>.</exception>
    public OcrEngine WithDecoding(DecodeOptions decoding)
    {
        ArgumentNullException.ThrowIfNull(decoding);
        decoding.Validate(_limits);
        return new OcrEngine(_recognizer, _limits, _pixels, decoding);
    }

    /// <summary>Loads a recognition model from a container file.</summary>
    /// <param name="modelPath">A model container (architecture <c>crnn-v0</c>).</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The engine.</returns>
    /// <exception cref="OcrModelException">The file is malformed or describes another architecture.</exception>
    /// <exception cref="OcrLimitException">The model exceeds <paramref name="limits"/>.</exception>
    public static OcrEngine Load(string modelPath, OcrLimits? limits = null) =>
        Create(ModelContainer.Load(modelPath, limits), limits, Dispatch.Best);

    /// <summary>Loads a recognition model from a stream, reading at most <see cref="OcrLimits.MaxModelFileBytes"/> bytes.</summary>
    /// <param name="model">A stream positioned at the start of a model container.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The engine.</returns>
    /// <exception cref="OcrModelException">The data is malformed or describes another architecture.</exception>
    /// <exception cref="OcrLimitException">The model exceeds <paramref name="limits"/>.</exception>
    public static OcrEngine Load(Stream model, OcrLimits? limits = null) =>
        Create(ModelContainer.Load(model, limits), limits, Dispatch.Best);

    internal static OcrEngine Create(ModelContainer container, OcrLimits? limits, KernelPath path)
    {
        limits ??= OcrLimits.Default;
        return new OcrEngine(
            new LineRecognizer(CrnnModel.FromContainer(container, limits), limits, path), limits,
            new BufferPool<Gray8>(limits.MaxPooledBufferBytes, 1), DecodeOptions.Greedy);
    }

    /// <summary>
    /// Reads the text of a clean, upright, single-column page: finds the text lines, reads each, and joins the non-empty results
    /// top to bottom with <c>\n</c>. Within a line, spaces are whatever the recognizer produced.
    /// Images are converted to 8-bit grayscale first (transparency is not composited; flatten such images yourself).
    /// </summary>
    /// <param name="image">The page.</param>
    /// <param name="maxDegreeOfParallelism">Most threads to use; 0 means min(4, cores). The text does not depend on it.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The text; empty when no text lines were found.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="OcrLimitException">The image, the number of lines or a line's width exceeds the limits.</exception>
    public string ToText(Image image, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default) =>
        ToText(image, _decoding, maxDegreeOfParallelism, cancellationToken);

    /// <summary>
    /// Like <see cref="ToText(Image, int, CancellationToken)"/> with the given decoding for this call instead of the engine's.
    /// </summary>
    /// <param name="image">The page.</param>
    /// <param name="decoding">How each line's network output becomes text.</param>
    /// <param name="maxDegreeOfParallelism">Most threads to use; 0 means min(4, cores). The text does not depend on it.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The text; empty when no text lines were found.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> or <paramref name="decoding"/> is null.</exception>
    /// <exception cref="ArgumentException">A decoding setting is out of range.</exception>
    /// <exception cref="OcrLimitException">The image, the number of lines, a line's width or the beam search exceed the limits.</exception>
    public string ToText(Image image, DecodeOptions decoding, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(decoding);
        decoding.Validate(_limits);
        string[] texts = [];
        ForEachLine(
            image, maxDegreeOfParallelism, count => texts = new string[count],
            (index, _, _, pixels, cropWidth) =>
                // Lines are the unit of parallelism, so each is recognized on one thread.
                texts[index] = _recognizer.Recognize(pixels, cropWidth, decoding, 1, cancellationToken).Text,
            cancellationToken);

        var result = new StringBuilder();
        foreach (string text in texts)
        {
            if (text.Length == 0)
            {
                continue;
            }

            if (result.Length > 0)
            {
                result.Append('\n');
            }

            result.Append(text);
        }

        return result.ToString();
    }

    /// <summary>
    /// Reads a clean, upright, single-column page like <see cref="ToText(Image, int, CancellationToken)"/>, but returns the lines
    /// with their words, approximate boxes and confidences. Words are always read with greedy decoding, whatever
    /// <see cref="Decoding"/> says, because beam search does not keep the per-character alignment the boxes come from;
    /// the text equals <c>ToText</c> with greedy decoding.
    /// </summary>
    /// <param name="image">The page.</param>
    /// <param name="maxDegreeOfParallelism">Most threads to use; 0 means min(4, cores). The result does not depend on it.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The lines; <see cref="OcrRecognition.Lines"/> is empty when no text lines were found.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="OcrLimitException">The image, the number of lines or a line's width exceeds the limits.</exception>
    public OcrRecognition Recognize(Image image, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        OcrLine[] lines = [];
        ForEachLine(
            image, maxDegreeOfParallelism, count => lines = new OcrLine[count],
            (index, ink, crop, pixels, cropWidth) =>
            {
                LineWords found = _recognizer.RecognizeWords(pixels, cropWidth, 1, cancellationToken);
                var words = new OcrWord[found.Words.Count];
                double chars = 0, weighted = 0;
                for (int i = 0; i < words.Length; i++)
                {
                    RecognizedWord w = found.Words[i];
                    // The crop was scaled from crop.Width pixels to cropWidth; map back with the same ratio.
                    double scale = (double)crop.Width / cropWidth;
                    float left = (float)Math.Clamp(crop.X + (w.X0 * scale), crop.X, crop.Right);
                    float right = (float)Math.Clamp(crop.X + (w.X1 * scale), crop.X, crop.Right);
                    words[i] = new OcrWord(w.Text, left, ink.Y, right, ink.Bottom, w.Confidence, w.RawConfidence);
                    chars += w.Characters;
                    weighted += w.Confidence * (double)w.Characters;
                }

                lines[index] = new OcrLine(found.Text, words, ink.X, ink.Y, ink.Right, ink.Bottom, chars == 0 ? 0f : (float)(weighted / chars));
            },
            cancellationToken);

        return new OcrRecognition(lines, _recognizer.HasCalibration ? [] : [ModelContainer.NoCalibrationNote]);
    }

    private delegate void LineVisitor(int index, LineBox ink, LineBox crop, ReadOnlySpan<byte> pixels, int cropWidth);

    // Finds the page's lines and calls visit once per line, in parallel at the given degree, with the line's ink box, its crop box
    // in page pixels and the crop scaled to the model's input height.
    private void ForEachLine(Image image, int maxDegreeOfParallelism, Action<int> start, LineVisitor visit, CancellationToken cancellationToken)
    {
        long pixels = (long)image.Width * image.Height;
        if (pixels > _limits.MaxImagePixels)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxImagePixels), _limits.MaxImagePixels, pixels);
        }

        int degree = maxDegreeOfParallelism > 0 ? maxDegreeOfParallelism : Math.Min(4, Environment.ProcessorCount);
        int width = image.Width, height = image.Height;
        // One pooled copy of the page as 8-bit gray; lines are cut from it, so no line triggers a second decode.
        Gray8[] grayPixels = _pixels.Rent(pixels);
        try
        {
            int pixelCount = checked(width * height);
            image.CopyPixelsTo<Gray8>(grayPixels.AsSpan(0, pixelCount), new ImageRect(0, 0, width, height), degree, cancellationToken);
            ReadOnlySpan<byte> gray0 = PixelSpans.AsBytes(grayPixels.AsSpan(0, pixelCount));
            List<LineBox> lines = LineFinder.Find(gray0, width, height, _limits.MaxLinesPerPage);
            start(lines.Count);
            if (lines.Count == 0)
            {
                return;
            }

            int typical = LinePreprocessor.TypicalHeight(lines);
            int modelHeight = _recognizer.InputHeight;

            void Read(int index)
            {
                cancellationToken.ThrowIfCancellationRequested();
                LineBox box = LinePreprocessor.CropBox(lines[index], typical, width, height);
                Gray8[] crop = LinePreprocessor.Prepare(PixelSpans.AsBytes(grayPixels.AsSpan(0, pixelCount)), width, box, modelHeight, _pixels, out int cropWidth);
                try
                {
                    visit(index, lines[index], box, PixelSpans.AsBytes(crop.AsSpan(0, cropWidth * modelHeight)), cropWidth);
                }
                finally
                {
                    _pixels.Return(crop);
                }
            }

            if (degree > 1 && lines.Count > 1)
            {
                Parallel.For(0, lines.Count, new ParallelOptions { MaxDegreeOfParallelism = degree, CancellationToken = cancellationToken }, Read);
            }
            else
            {
                for (int i = 0; i < lines.Count; i++)
                {
                    Read(i);
                }
            }
        }
        finally
        {
            _pixels.Return(grayPixels);
        }
    }
}
