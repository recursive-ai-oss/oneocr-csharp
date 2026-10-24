using System.Text;
using OneImage;
using OneOcr.Decoding;
using OneOcr.Detection;
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
    private readonly BufferPool<byte> _masks;
    private readonly ResizerPool _resizers;
    private readonly DecodeOptions _decoding;
    private readonly IOrientationDetector _orientation;
    private readonly IPageDetector? _detector;

    private OcrEngine(
        LineRecognizer recognizer, OcrLimits limits, BufferPool<Gray8> pixels, BufferPool<byte> masks, ResizerPool resizers, DecodeOptions decoding,
        IPageDetector? detector)
    {
        _detector = detector;
        _resizers = resizers;
        _masks = masks;
        _recognizer = recognizer;
        _limits = limits;
        _pixels = pixels;
        _decoding = decoding;
        _orientation = new ClassicalOrientation(recognizer, pixels, masks, resizers, limits);
    }

    internal IOrientationDetector OrientationDetector => _orientation;

    /// <summary>The bounds this engine enforces.</summary>
    public OcrLimits Limits => _limits;

    /// <summary>Whether this engine finds lines, regions and the page's orientation with a page detector (see <see cref="WithDetector(string, OcrLimits?)"/>).</summary>
    public bool HasDetector => _detector is not null;

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
        return new OcrEngine(_recognizer, _limits, _pixels, _masks, _resizers, decoding, _detector);
    }

    /// <summary>
    /// An engine that shares this one's recognizer and finds text lines, layout regions and the page's orientation with a page detector
    /// model (architecture <c>dbnet-lite-v0</c>) instead of the built-in classical line finding. Page methods (<see cref="Ocr"/>,
    /// <see cref="ToText(Image, int, CancellationToken)"/>) use it: lines come from the detector, regions fill
    /// <see cref="OcrPage.Regions"/> and set header and footer blocks, and a page the detector is sure is rotated by 90, 180 or 270
    /// degrees is turned upright first (a diagnostic says so, and positions refer to the turned page). Layout classes the model was not
    /// trained on (<c>form_field</c> and <c>checkbox</c> unless the model's <c>layout_trained</c> lists them) are never reported.
    /// </summary>
    /// <param name="detectorModelPath">A model container with architecture <c>dbnet-lite-v0</c>.</param>
    /// <param name="limits">Bounds to enforce when reading the file; this engine's own limits when null.</param>
    /// <returns>The new engine; this one is unchanged.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="detectorModelPath"/> is null.</exception>
    /// <exception cref="OcrModelException">The file is malformed or describes another architecture.</exception>
    /// <exception cref="OcrLimitException">The model exceeds the limits.</exception>
    public OcrEngine WithDetector(string detectorModelPath, OcrLimits? limits = null) =>
        WithDetector(ModelContainer.Load(detectorModelPath, limits ?? _limits), Dispatch.Best);

    /// <summary>Like <see cref="WithDetector(string, OcrLimits?)"/> with the model read from a stream, at most <see cref="OcrLimits.MaxModelFileBytes"/> bytes.</summary>
    /// <param name="detectorModel">A stream positioned at the start of a model container.</param>
    /// <param name="limits">Bounds to enforce when reading; this engine's own limits when null.</param>
    /// <returns>The new engine; this one is unchanged.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="detectorModel"/> is null.</exception>
    /// <exception cref="OcrModelException">The data is malformed or describes another architecture.</exception>
    /// <exception cref="OcrLimitException">The model exceeds the limits.</exception>
    public OcrEngine WithDetector(Stream detectorModel, OcrLimits? limits = null) =>
        WithDetector(ModelContainer.Load(detectorModel, limits ?? _limits), Dispatch.Best);

    /// <summary>An engine that shares this one's recognizer and has no page detector: lines come from the built-in classical finder.</summary>
    /// <returns>The new engine; this one is unchanged.</returns>
    public OcrEngine WithoutDetector() => new(_recognizer, _limits, _pixels, _masks, _resizers, _decoding, null);

    internal OcrEngine WithDetector(ModelContainer container, KernelPath path) =>
        WithDetector(new PageDetector(DbNetModel.FromContainer(container, _limits), _limits, path));

    internal OcrEngine WithDetector(IPageDetector detector) =>
        new(_recognizer, _limits, _pixels, _masks, _resizers, _decoding, detector);

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
            new BufferPool<Gray8>(limits.MaxPooledBufferBytes, 1), new BufferPool<byte>(limits.MaxPooledBufferBytes, 1), new ResizerPool(), DecodeOptions.Greedy, null);
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
        if (decoding.Mode != CtcDecoding.Beam)
        {
            return BuildPage(image, 0, null, maxDegreeOfParallelism, includeChars: false, orient: true, deskew: true, cancellationToken).ToText();
        }

        string[] texts = [];
        ForEachLine(
            image, maxDegreeOfParallelism, count => texts = new string[count],
            (index, _, _, pixels, cropWidth) =>
                // Lines are the unit of parallelism, so each is recognized on one thread.
                texts[index] = _recognizer.Recognize(pixels, cropWidth, decoding, 1, cancellationToken).Text,
            cancellationToken, orient: true);

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
    /// Reads a page into the page model: finds the text lines (column by column, with ruling lines left out), reads each, and
    /// puts words, lines and blocks into reading order, with tables where a block is a grid of short cells. Words are always read
    /// with greedy decoding, whatever <see cref="Decoding"/> says, because beam search keeps no alignment and the boxes come from it.
    /// Images are converted to 8-bit grayscale first (transparency is not composited; flatten such images yourself).
    /// Word boxes are the ink extent of each word; character boxes are approximate and say so. Output is bit-identical on every
    /// supported CPU path and at every degree of parallelism.
    /// </summary>
    /// <param name="image">The page image.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The page; with no blocks when no text was found.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="options"/> holds a value out of range.</exception>
    /// <exception cref="OcrLimitException">The image, the number of lines or a line's width exceeds the limits.</exception>
    public OcrPage Ocr(Image image, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        options ??= new OcrOptions();
        options.Validate();
        return BuildPage(image, options.PageIndex, options.Dpi, options.MaxDegreeOfParallelism, options.IncludeChars, options.DetectOrientation, options.Deskew, cancellationToken);
    }

    private OcrPage BuildPage(Image image, int pageIndex, double? dpi, int maxDegreeOfParallelism, bool includeChars, bool orient, bool deskew, CancellationToken cancellationToken)
    {
        long pixels = (long)image.Width * image.Height;
        if (pixels > _limits.MaxImagePixels)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxImagePixels), _limits.MaxImagePixels, pixels);
        }

        int degree = maxDegreeOfParallelism > 0 ? maxDegreeOfParallelism : Math.Min(4, Environment.ProcessorCount);
        int width = image.Width, height = image.Height;
        Gray8[] gray = _pixels.Rent(pixels);
        try
        {
            int pixelCount = checked(width * height);
            image.CopyPixelsTo<Gray8>(gray.AsSpan(0, pixelCount), new ImageRect(0, 0, width, height), degree, cancellationToken);
            return ReadPrepared(gray, width, height, pageIndex, dpi, degree, includeChars, orient, deskew, cancellationToken);
        }
        finally
        {
            _pixels.Return(gray);
        }
    }

    // The orientation decision in use: the detector's head when a detector is loaded, the classical one otherwise.
    private IOrientationDetector Seam => _detector is null ? _orientation : new HeadOrientation(_detector, _pixels, _resizers);

    // Turns and straightens the page (see PagePreparation), reads it, and puts the result in the frame the caller asked about.
    private OcrPage ReadPrepared(
        Gray8[] gray, int width, int height, int pageIndex, double? dpi, int degree, bool includeChars, bool orient, bool deskew, CancellationToken cancellationToken)
    {
        var evidence = new OrientationEvidence();
        PreparedPage prepared = PagePreparation.Prepare(
            gray, width, height, orient, deskew, Seam, _limits, _pixels, degree, evidence, quarterInInputFrame: _detector is null, cancellationToken);
        try
        {
            // The head's first look is reusable when the pixels the reader gets are the ones it looked at.
            PageDetection? first = ReferenceEquals(prepared.Pixels, gray) ? evidence.Detection : null;
            OcrPage page = PageReader.Read(
                _recognizer, _limits, _pixels, _masks, _resizers, _detector, prepared.Pixels, prepared.Width, prepared.Height, pageIndex, dpi, degree, includeChars,
                first, turn: false, cancellationToken);
            return prepared.Finish(page);
        }
        finally
        {
            prepared.Release();
        }
    }

    /// <summary>One text line of a page as the line finder sees it, with the crop the recognizer reads and its greedy reading.</summary>
    internal sealed record LineSample(int X, int Y, int Width, int Height, string Text, float Confidence, byte[] Gray, int CropWidth);

    /// <summary>
    /// Finds the page's lines and reads each with greedy decoding, keeping the crop (scaled to the model's input height) so a caller
    /// can train on it. Lines come back in the finder's order (top to bottom).
    /// </summary>
    internal List<LineSample> ReadLineSamples(Image image, int maxDegreeOfParallelism, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(image);
        LineSample?[] samples = [];
        ForEachLine(
            image, maxDegreeOfParallelism, count => samples = new LineSample?[count],
            (index, ink, _, pixels, cropWidth) =>
            {
                RecognizedLine read = _recognizer.Recognize(pixels, cropWidth, null, 1, cancellationToken);
                samples[index] = new LineSample(ink.X, ink.Y, ink.Width, ink.Height, read.Text, read.Confidence, pixels.ToArray(), cropWidth);
            },
            cancellationToken);
        return [.. samples.Select(static s => s!)];
    }

    private delegate void LineVisitor(int index, LineBox ink, LineBox crop, ReadOnlySpan<byte> pixels, int cropWidth);

    // Finds the page's lines and calls visit once per line, in parallel at the given degree, with the line's ink box, its crop box
    // in page pixels and the crop scaled to the model's input height.
    private void ForEachLine(Image image, int maxDegreeOfParallelism, Action<int> start, LineVisitor visit, CancellationToken cancellationToken, bool orient = false)
    {
        long pixels = (long)image.Width * image.Height;
        if (pixels > _limits.MaxImagePixels)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxImagePixels), _limits.MaxImagePixels, pixels);
        }

        int degree = maxDegreeOfParallelism > 0 ? maxDegreeOfParallelism : Math.Min(4, Environment.ProcessorCount);
        int width = image.Width, height = image.Height;
        // One pooled copy of the page as 8-bit gray; lines are cut from it, so no line triggers a second decode.
        Gray8[] rented = _pixels.Rent(pixels);
        Gray8[] grayPixels = rented;
        PreparedPage? prepared = null;
        try
        {
            int pixelCount = checked(width * height);
            image.CopyPixelsTo<Gray8>(rented.AsSpan(0, pixelCount), new ImageRect(0, 0, width, height), degree, cancellationToken);
            var evidence = new OrientationEvidence();
            prepared = orient
                ? PagePreparation.Prepare(grayPixels, width, height, true, true, Seam, _limits, _pixels, degree, evidence, quarterInInputFrame: _detector is null, cancellationToken)
                : null;
            if (prepared is not null)
            {
                grayPixels = prepared.Pixels;
                (width, height) = (prepared.Width, prepared.Height);
                pixelCount = width * height;
            }

            LocatedPage? located = null;
            Gray8[] work = grayPixels;
            int workWidth = width, workHeight = height;
            List<LineBox> lines;
            if (_detector is null)
            {
                lines = LineFinder.Find(PixelSpans.AsBytes(grayPixels.AsSpan(0, pixelCount)), width, height, _limits.MaxLinesPerPage);
            }
            else
            {
                located = DetectorLines.Locate(
                    _detector, grayPixels, width, height, _pixels, _resizers, degree, cancellationToken,
                    first: ReferenceEquals(grayPixels, rented) ? evidence.Detection : null, turn: !orient);
                work = located.Gray;
                workWidth = located.Width;
                workHeight = located.Height;
                lines = [.. located.Strips.Select(static s => s.Box)];
            }

            try
            {
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
                    LineBox box = LinePreprocessor.CropBox(lines[index], typical, workWidth, workHeight);
                    Gray8[] crop = LinePreprocessor.Prepare(PixelSpans.AsBytes(work.AsSpan(0, pixelCount)), workWidth, box, modelHeight, _pixels, _resizers, out int cropWidth);
                    try
                    {
                        visit(index, lines[index], box, PixelSpans.AsBytes(crop.AsSpan(0, cropWidth * modelHeight)), cropWidth);
                    }
                    finally
                    {
                        _pixels.Return(crop);
                    }
                }

                ParallelRun.For(lines.Count, degree, Read, cancellationToken);
            }
            finally
            {
                located?.Release(_pixels);
            }
        }
        finally
        {
            prepared?.Release();
            _pixels.Return(rented);
        }
    }
}
