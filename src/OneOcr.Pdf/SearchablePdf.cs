using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using OneImage;
using OnePdf;

namespace OneOcr.Pdf;

/// <summary>The conformance level a searchable PDF is written to.</summary>
public enum SearchablePdfConformance
{
    /// <summary>A plain PDF.</summary>
    None = 0,

    /// <summary>PDF/A-2b (ISO 19005-2, level B), written by OnePDF's PDF/A save or converter.</summary>
    PdfA2b = 1,
}

/// <summary>Options for <see cref="SearchablePdf"/> and <see cref="SearchablePdfExtensions"/>.</summary>
public sealed class SearchablePdfOptions
{
    /// <summary>
    /// Options for reading images (resolution, orientation, deskew and the rest). Null uses the engine's own options. The resolution
    /// sets the size of the page the image is placed on: one pixel is 1/<see cref="OcrOptions.Dpi"/> inch, or 1/(the file's declared
    /// resolution) inch, or an estimate from the text size when the file declares none.
    /// </summary>
    public OcrOptions? Ocr { get; init; }

    /// <summary>
    /// How the pages of a PDF given to <c>MakeSearchable</c> are read: <see cref="PdfOcrOptions.TrustExistingOcrLayer"/>,
    /// <see cref="PdfOcrOptions.RenderDpi"/>, <see cref="PdfOcrOptions.Password"/>, <see cref="PdfOcrOptions.FirstPage"/> and
    /// <see cref="PdfOcrOptions.PageCount"/> (pages outside the range are copied unchanged). Null for the defaults.
    /// </summary>
    public PdfOcrOptions? Pdf { get; init; }

    /// <summary>The conformance level of the output. The default is a plain PDF.</summary>
    public SearchablePdfConformance Conformance { get; init; }

    /// <summary>The document title (<c>/Info</c> and, for PDF/A, XMP) of a PDF made from images; null writes none. Ignored by <c>MakeSearchable</c>, which keeps the file's own.</summary>
    public string? Title { get; init; }

    /// <summary>
    /// Whether an image file that is a JPEG (one frame, not turned by an EXIF orientation) is placed as the original bytes. The default is
    /// true; false decodes it and stores the pixels losslessly, which makes the file larger.
    /// </summary>
    public bool EmbedOriginalJpeg { get; init; } = true;

    /// <summary>
    /// The 16-byte file identifier of a PDF made from images. Null derives it from the page images and the recognized text, so the same input
    /// and options always give the same file. A PDF given to <c>MakeSearchable</c> keeps its own identifier.
    /// </summary>
    public byte[]? DocumentId { get; init; }

    internal void Validate()
    {
        Pdf?.Validate();
        if (!Enum.IsDefined(Conformance))
        {
            throw new ArgumentOutOfRangeException(nameof(Conformance), Conformance, "Unknown conformance level.");
        }

        if (DocumentId is { Length: not 16 })
        {
            throw new ArgumentException("DocumentId must be 16 bytes.", nameof(DocumentId));
        }
    }
}

/// <summary>What a searchable-PDF call did.</summary>
public sealed class SearchablePdfResult
{
    internal SearchablePdfResult(IReadOnlyList<OcrPage> pages, int pagesWritten, int wordsWritten, IReadOnlyList<string> diagnostics)
    {
        Pages = pages;
        PagesWithTextLayer = pagesWritten;
        WordsWritten = wordsWritten;
        Diagnostics = diagnostics;
    }

    /// <summary>The page models the text layer was made from: one per image page, or one per page of the PDF's selected range.</summary>
    public IReadOnlyList<OcrPage> Pages { get; }

    /// <summary>The number of pages that received a text layer. <c>MakeSearchable</c> leaves other pages exactly as they were.</summary>
    public int PagesWithTextLayer { get; }

    /// <summary>The number of words written into text layers.</summary>
    public int WordsWritten { get; }

    /// <summary>Things worth knowing, such as an invisible layer that could not be removed. Empty when nothing needs attention.</summary>
    public IReadOnlyList<string> Diagnostics { get; }
}

/// <summary>
/// Searchable PDFs: a page image (or an existing PDF page) keeps its look and gets an invisible text layer from local OCR, so that
/// selection, search and copy match the words on the page. Everything runs locally. The layout of the layer and the rules for what
/// is trusted, replaced and left alone are in <c>docs/spec/searchable-pdf.md</c>.
/// </summary>
/// <remarks>
/// Output is deterministic: the same input, options and engine give the same bytes, whatever the degree of parallelism or kernel path.
/// No wall-clock time is written. Methods that take no engine use <see cref="OcrEngine.Default"/>.
/// </remarks>
public static class SearchablePdf
{
    /// <summary>Reads each image (every frame of a multi-frame image is a page) and writes one PDF with the images as pages and the text as an invisible layer.</summary>
    /// <param name="images">The images, in page order.</param>
    /// <param name="output">Receives the PDF.</param>
    /// <param name="engine">The recognizer; null for <see cref="OcrEngine.Default"/>.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines.</param>
    /// <returns>What was written.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="images"/> or <paramref name="output"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="options"/> holds a value out of range, or there are no images.</exception>
    /// <exception cref="OcrLimitException">A page or the page count exceeds a limit of the engine.</exception>
    /// <exception cref="OcrFormatException">An image cannot be decoded.</exception>
    public static SearchablePdfResult FromImages(
        IEnumerable<Image> images, Stream output, OcrEngine? engine = null, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(output);
        options ??= new SearchablePdfOptions();
        options.Validate();
        using var builder = new ImagePdfBuilder(engine ?? OcrEngine.Default, options);
        foreach (Image image in images)
        {
            ArgumentNullException.ThrowIfNull(image, nameof(images));
            builder.Add(image, cancellationToken);
        }

        if (builder.PageCount == 0)
        {
            throw new ArgumentException("There are no images to write.", nameof(images));
        }

        return builder.Save(output);
    }

    /// <summary>
    /// Like <see cref="FromImages"/> for image files, which are placed without being re-encoded where possible: a JPEG as its own bytes
    /// (see <see cref="SearchablePdfOptions.EmbedOriginalJpeg"/>) and a PNG through OnePDF's PNG pass-through. TIFF and other formats are decoded and stored
    /// losslessly: bilevel pages as 1-bit, gray as 8-bit gray, color as 8-bit RGB.
    /// </summary>
    /// <param name="paths">The image files, in page order.</param>
    /// <param name="output">Receives the PDF.</param>
    /// <param name="engine">The recognizer; null for <see cref="OcrEngine.Default"/>.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines.</param>
    /// <returns>What was written.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="paths"/> or <paramref name="output"/> is null.</exception>
    /// <exception cref="OcrFormatException">A file cannot be decoded.</exception>
    /// <exception cref="IOException">A file cannot be read.</exception>
    public static SearchablePdfResult FromFiles(
        IEnumerable<string> paths, Stream output, OcrEngine? engine = null, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(paths);
        return FromData(paths.Select(static p => (ReadOnlyMemory<byte>)File.ReadAllBytes(p)), output, engine, options, cancellationToken);
    }

    /// <summary>Like <see cref="FromFiles(IEnumerable{string}, Stream, OcrEngine?, SearchablePdfOptions?, CancellationToken)"/>, writing to a file. The file is replaced only when the whole PDF was written.</summary>
    /// <param name="paths">The image files, in page order.</param>
    /// <param name="outputPath">The PDF to write.</param>
    /// <param name="engine">The recognizer; null for <see cref="OcrEngine.Default"/>.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines.</param>
    /// <returns>What was written.</returns>
    public static SearchablePdfResult FromFiles(
        IEnumerable<string> paths, string outputPath, OcrEngine? engine = null, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default) =>
        WriteFile(outputPath, s => FromFiles(paths, s, engine, options, cancellationToken));

    /// <summary>Like <see cref="FromFiles(IEnumerable{string}, Stream, OcrEngine?, SearchablePdfOptions?, CancellationToken)"/> for encoded images already in memory.</summary>
    /// <param name="images">The encoded images (PNG, JPEG, TIFF and the other formats the image library reads), in page order.</param>
    /// <param name="output">Receives the PDF.</param>
    /// <param name="engine">The recognizer; null for <see cref="OcrEngine.Default"/>.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines.</param>
    /// <returns>What was written.</returns>
    public static SearchablePdfResult FromData(
        IEnumerable<ReadOnlyMemory<byte>> images, Stream output, OcrEngine? engine = null, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(output);
        options ??= new SearchablePdfOptions();
        options.Validate();
        using var builder = new ImagePdfBuilder(engine ?? OcrEngine.Default, options);
        foreach (ReadOnlyMemory<byte> data in images)
        {
            builder.Add(data, cancellationToken);
        }

        if (builder.PageCount == 0)
        {
            throw new ArgumentException("There are no images to write.", nameof(images));
        }

        return builder.Save(output);
    }

    /// <summary>
    /// Adds a text layer to the pages of a PDF that need one and leaves the rest alone. A page is read as <see cref="PdfOcr"/> reads it
    /// (text layer where there is one, local OCR for scanned regions, third-party invisible layers and garbled text not trusted).
    /// Words recognized from pixels are written as an invisible layer in a new content stream appended to the page; a page with no such
    /// words is not touched at all. The page's own content is kept byte for byte, except that text a third-party OCR step left
    /// invisible (render mode 3) is cut out when our layer replaces it; see the spec for what that covers.
    /// </summary>
    /// <param name="pdf">The PDF file.</param>
    /// <param name="output">Receives the new PDF.</param>
    /// <param name="engine">The recognizer; null for <see cref="OcrEngine.Default"/>.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>What was written.</returns>
    /// <exception cref="OcrFormatException">The file is damaged, or cannot be converted to the requested PDF/A level.</exception>
    /// <exception cref="OcrLimitException">The file or a page exceeds a limit.</exception>
    public static SearchablePdfResult MakeSearchable(
        ReadOnlyMemory<byte> pdf, Stream output, OcrEngine? engine = null, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(output);
        options ??= new SearchablePdfOptions();
        options.Validate();
        return PdfSearchable.Run(pdf, output, engine ?? OcrEngine.Default, options, cancellationToken);
    }

    /// <summary>Like <see cref="MakeSearchable(ReadOnlyMemory{byte}, Stream, OcrEngine?, SearchablePdfOptions?, CancellationToken)"/>, reading a stream to its end.</summary>
    /// <param name="input">The PDF file.</param>
    /// <param name="output">Receives the new PDF.</param>
    /// <param name="engine">The recognizer; null for <see cref="OcrEngine.Default"/>.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>What was written.</returns>
    public static SearchablePdfResult MakeSearchable(
        Stream input, Stream output, OcrEngine? engine = null, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        long max = (options?.Pdf ?? new PdfOcrOptions()).MaxFileBytes;
        using var buffer = new MemoryStream();
        input.CopyTo(buffer);
        if (buffer.Length > max)
        {
            throw new OcrLimitException(nameof(PdfOcrOptions.MaxFileBytes), max, buffer.Length);
        }

        return MakeSearchable(buffer.GetBuffer().AsMemory(0, (int)buffer.Length), output, engine, options, cancellationToken);
    }

    /// <summary>Like <see cref="MakeSearchable(ReadOnlyMemory{byte}, Stream, OcrEngine?, SearchablePdfOptions?, CancellationToken)"/> for files. The output file is replaced only when the whole PDF was written.</summary>
    /// <param name="inputPath">The PDF file.</param>
    /// <param name="outputPath">The PDF to write; it may be the same path as the input.</param>
    /// <param name="engine">The recognizer; null for <see cref="OcrEngine.Default"/>.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>What was written.</returns>
    public static SearchablePdfResult MakeSearchable(
        string inputPath, string outputPath, OcrEngine? engine = null, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputPath);
        long max = (options?.Pdf ?? new PdfOcrOptions()).MaxFileBytes;
        long length = new FileInfo(inputPath).Length;
        if (length > max)
        {
            throw new OcrLimitException(nameof(PdfOcrOptions.MaxFileBytes), max, length);
        }

        byte[] bytes = File.ReadAllBytes(inputPath);
        return WriteFile(outputPath, s => MakeSearchable(bytes, s, engine, options, cancellationToken));
    }

    /// <summary>Like <see cref="FromImages"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="images">The images, in page order.</param>
    /// <param name="output">Receives the PDF.</param>
    /// <param name="engine">The recognizer; null for <see cref="OcrEngine.Default"/>.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines; the task then ends as cancelled.</param>
    /// <returns>What was written.</returns>
    public static Task<SearchablePdfResult> FromImagesAsync(
        IEnumerable<Image> images, Stream output, OcrEngine? engine = null, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => FromImages(images, output, engine, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="FromFiles(IEnumerable{string}, Stream, OcrEngine?, SearchablePdfOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="paths">The image files, in page order.</param>
    /// <param name="output">Receives the PDF.</param>
    /// <param name="engine">The recognizer; null for <see cref="OcrEngine.Default"/>.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines; the task then ends as cancelled.</param>
    /// <returns>What was written.</returns>
    public static Task<SearchablePdfResult> FromFilesAsync(
        IEnumerable<string> paths, Stream output, OcrEngine? engine = null, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => FromFiles(paths, output, engine, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="MakeSearchable(ReadOnlyMemory{byte}, Stream, OcrEngine?, SearchablePdfOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="pdf">The PDF file.</param>
    /// <param name="output">Receives the new PDF.</param>
    /// <param name="engine">The recognizer; null for <see cref="OcrEngine.Default"/>.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions; the task then ends as cancelled.</param>
    /// <returns>What was written.</returns>
    public static Task<SearchablePdfResult> MakeSearchableAsync(
        ReadOnlyMemory<byte> pdf, Stream output, OcrEngine? engine = null, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => MakeSearchable(pdf, output, engine, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="MakeSearchable(string, string, OcrEngine?, SearchablePdfOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="inputPath">The PDF file.</param>
    /// <param name="outputPath">The PDF to write.</param>
    /// <param name="engine">The recognizer; null for <see cref="OcrEngine.Default"/>.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions; the task then ends as cancelled.</param>
    /// <returns>What was written.</returns>
    public static Task<SearchablePdfResult> MakeSearchableAsync(
        string inputPath, string outputPath, OcrEngine? engine = null, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => MakeSearchable(inputPath, outputPath, engine, options, cancellationToken), cancellationToken);

    // Writes to a temporary file next to the target and moves it into place, so a failure leaves an existing file as it was.
    internal static SearchablePdfResult WriteFile(string path, Func<Stream, SearchablePdfResult> write)
    {
        ArgumentNullException.ThrowIfNull(path);
        string temp = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            SearchablePdfResult result;
            using (var file = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                result = write(file);
            }

            File.Move(temp, path, overwrite: true);
            return result;
        }
        finally
        {
            if (File.Exists(temp))
            {
                File.Delete(temp);
            }
        }
    }
}

/// <summary>Searchable-PDF output for a decoded image, as extension methods.</summary>
public static class SearchablePdfExtensions
{
    /// <summary>Reads the image with <see cref="OcrEngine.Default"/> and writes a one-page PDF (one page per frame of a multi-frame image) with the image and an invisible text layer.</summary>
    /// <param name="image">The image.</param>
    /// <param name="output">Receives the PDF.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines.</param>
    /// <returns>What was written.</returns>
    /// <remarks>A decoded <see cref="Image"/> has lost its original bytes, so its pixels are stored losslessly (see <see cref="SearchablePdf.FromFiles(IEnumerable{string}, Stream, OcrEngine?, SearchablePdfOptions?, CancellationToken)"/> to keep a JPEG as it is).</remarks>
    public static SearchablePdfResult ToSearchablePdf(this Image image, Stream output, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default) =>
        ToSearchablePdf(image, OcrEngine.Default, output, options, cancellationToken);

    /// <summary>Like <see cref="ToSearchablePdf(Image, Stream, SearchablePdfOptions?, CancellationToken)"/> with the given engine.</summary>
    /// <param name="image">The image.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="output">Receives the PDF.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines.</param>
    /// <returns>What was written.</returns>
    public static SearchablePdfResult ToSearchablePdf(this Image image, OcrEngine engine, Stream output, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(engine);
        return SearchablePdf.FromImages([image], output, engine, options, cancellationToken);
    }

    /// <summary>Like <see cref="ToSearchablePdf(Image, Stream, SearchablePdfOptions?, CancellationToken)"/>, writing a file that is replaced only when the whole PDF was written.</summary>
    /// <param name="image">The image.</param>
    /// <param name="path">The PDF to write.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines.</param>
    /// <returns>What was written.</returns>
    public static SearchablePdfResult ToSearchablePdf(this Image image, string path, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default) =>
        ToSearchablePdf(image, OcrEngine.Default, path, options, cancellationToken);

    /// <summary>Like <see cref="ToSearchablePdf(Image, string, SearchablePdfOptions?, CancellationToken)"/> with the given engine.</summary>
    /// <param name="image">The image.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="path">The PDF to write.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines.</param>
    /// <returns>What was written.</returns>
    public static SearchablePdfResult ToSearchablePdf(this Image image, OcrEngine engine, string path, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(engine);
        return SearchablePdf.WriteFile(path, s => ToSearchablePdf(image, engine, s, options, cancellationToken));
    }

    /// <summary>Like <see cref="ToSearchablePdf(Image, Stream, SearchablePdfOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The image.</param>
    /// <param name="output">Receives the PDF.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines; the task then ends as cancelled.</param>
    /// <returns>What was written.</returns>
    public static Task<SearchablePdfResult> ToSearchablePdfAsync(this Image image, Stream output, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToSearchablePdf(image, output, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToSearchablePdf(Image, OcrEngine, Stream, SearchablePdfOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The image.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="output">Receives the PDF.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines; the task then ends as cancelled.</param>
    /// <returns>What was written.</returns>
    public static Task<SearchablePdfResult> ToSearchablePdfAsync(this Image image, OcrEngine engine, Stream output, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToSearchablePdf(image, engine, output, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToSearchablePdf(Image, string, SearchablePdfOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The image.</param>
    /// <param name="path">The PDF to write.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines; the task then ends as cancelled.</param>
    /// <returns>What was written.</returns>
    public static Task<SearchablePdfResult> ToSearchablePdfAsync(this Image image, string path, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToSearchablePdf(image, path, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToSearchablePdf(Image, OcrEngine, string, SearchablePdfOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The image.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="path">The PDF to write.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and lines; the task then ends as cancelled.</param>
    /// <returns>What was written.</returns>
    public static Task<SearchablePdfResult> ToSearchablePdfAsync(this Image image, OcrEngine engine, string path, SearchablePdfOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToSearchablePdf(image, engine, path, options, cancellationToken), cancellationToken);
}

/// <summary>Builds a PDF whose pages are images with a text layer.</summary>
internal sealed class ImagePdfBuilder : IDisposable
{
    private readonly OcrEngine _engine;
    private readonly SearchablePdfOptions _options;
    private readonly PdfDocument _document = PdfDocument.CreateEmpty();
    private readonly PdfTextLayerWriter _writer;
    private readonly IncrementalHash _hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    private readonly List<OcrPage> _pages = [];
    private readonly List<string> _diagnostics = [];
    private int _words;

    public ImagePdfBuilder(OcrEngine engine, SearchablePdfOptions options)
    {
        _engine = engine;
        _options = options;
        _writer = new PdfTextLayerWriter(_document);
    }

    public void Dispose()
    {
        _document.Dispose();
        _hash.Dispose();
    }

    /// <summary>A decoded image: every frame is a page.</summary>
    public void Add(Image image, CancellationToken cancellationToken)
    {
        IReadOnlyList<OcrPage> pages = _engine.OcrPages(image, _options.Ocr, cancellationToken);
        for (int i = 0; i < pages.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ImageFrame frame = image.Frames[i];
            PdfCreatedImage embedded = EmbedFrame(frame, cancellationToken);
            AddPage(pages[i], embedded);
        }
    }

    /// <summary>An encoded image file: a JPEG or PNG is placed as it is where possible.</summary>
    public void Add(ReadOnlyMemory<byte> data, CancellationToken cancellationToken)
    {
        IReadOnlyList<OcrPage> pages = _engine.OcrPages(data, _options.Ocr, cancellationToken);
        LoadOptions load = LoadOptionsFor(_engine.Limits);
        ImageInfo? info = null;
        try
        {
            if (Image.TryIdentify(data, out ImageInfo? identified, load))
            {
                info = identified;
            }
        }
        catch (ImageException)
        {
            // The recognizer has already decoded the file; identification only chooses how to embed it.
        }

        bool oneFrame = pages.Count == 1 && info is { Frames.Count: 1 } && info.Metadata.Orientation == 1;
        ReadOnlySpan<byte> head = data.Span;
        if (oneFrame && _options.EmbedOriginalJpeg && head.Length > 3 && head[0] == 0xFF && head[1] == 0xD8)
        {
            PdfCreatedImage? jpeg = Try(() => _document.AddJpegImage(data));
            if (jpeg is not null)
            {
                Feed(data.Span);
                AddPage(pages[0], jpeg);
                return;
            }
        }
        else if (oneFrame && head.Length > 8 && head[0] == 0x89 && head[1] == (byte)'P' && head[2] == (byte)'N' && head[3] == (byte)'G')
        {
            PdfCreatedImage? png = Try(() => _document.AddPngImage(data));
            if (png is not null)
            {
                Feed(data.Span);
                AddPage(pages[0], png);
                return;
            }
        }

        Image decoded = Guard(() => Image.Load(data, load, cancellationToken));
        for (int i = 0; i < pages.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ImageFrame frame = decoded.Frames[i];
            PdfCreatedImage embedded = EmbedFrame(frame, cancellationToken);
            AddPage(pages[i], embedded);
            frame.ReleaseDecodedPixels();
        }
    }

    private PdfCreatedImage EmbedFrame(ImageFrame frame, CancellationToken cancellationToken)
    {
        byte[] encoded = [];
        PdfCreatedImage image = Guard(() => ImageEmbedding.Embed(_document, frame, out _, out encoded, cancellationToken));
        Feed(encoded);
        return image;
    }

    public int PageCount => _pages.Count;

    public SearchablePdfResult Save(Stream output)
    {
        if (_options.Title is { } title)
        {
            _document.SetInfo("Title", title);
        }

        byte[] id = _options.DocumentId ?? _hash.GetHashAndReset().AsSpan(0, 16).ToArray();
        var save = new PdfSaveOptions
        {
            DocumentId = id,
            CompressStreams = true,
            PdfA = _options.Conformance == SearchablePdfConformance.PdfA2b ? new PdfAOptions { Part = PdfAPart.Part2, Conformance = PdfAConformance.B } : null,
        };
        try
        {
            _document.Save(output, save);
        }
        catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
        {
            throw new OcrFormatException("The PDF cannot be written: " + e.Message, e);
        }

        return new SearchablePdfResult(_pages, _pages.Count, _words, _diagnostics);
    }

    private void AddPage(OcrPage page, PdfCreatedImage image)
    {
        double s = 72.0 / page.UnitsPerInch, sy = 72.0 / page.VerticalUnitsPerInch;
        double width = page.Width * s, height = page.Height * sy;
        PdfPage pdfPage = _document.InsertBlankPage(_document.PageCount, width, height);
        using (PdfPageContentBuilder b = pdfPage.BeginContent())
        {
            b.DrawImage(image, 0, 0, width, height);
            _words += _writer.Draw(b, page);
            b.Commit();
        }

        _pages.Add(page);
        string tail = page.VerticalUnitsPerInch == page.UnitsPerInch ? string.Empty : string.Create(CultureInfo.InvariantCulture, $":{page.VerticalUnitsPerInch:R}");
        Feed(Encoding.UTF8.GetBytes(string.Create(CultureInfo.InvariantCulture, $"|{page.Width:R}x{page.Height:R}@{page.UnitsPerInch:R}") + tail + "|" + page.ToText()));
    }

    private void Feed(ReadOnlySpan<byte> data) => _hash.AppendData(data);

    private static LoadOptions LoadOptionsFor(OcrLimits limits) => new()
    {
        Limits = new ImageLimits { MaxPixelsPerFrame = limits.MaxImagePixels, MaxFrames = limits.MaxPagesPerCall },
    };

    private static PdfCreatedImage? Try(Func<PdfCreatedImage> add)
    {
        try
        {
            return add();
        }
        catch (PdfException)
        {
            return null; // not a file OnePDF can pass through; the decoded pixels are stored instead
        }
    }

    private static T Guard<T>(Func<T> f)
    {
        try
        {
            return f();
        }
        catch (ImageLimitException e)
        {
            throw new OcrLimitException("ImageLimits", "The image exceeds a decoding limit: " + e.Message, e);
        }
        catch (ImageException e)
        {
            throw new OcrFormatException("The image cannot be decoded: " + e.Message, e);
        }
        catch (PdfLimitException e)
        {
            throw new OcrLimitException("OnePdf", "The PDF exceeds a limit: " + e.Message, e);
        }
        catch (PdfException e)
        {
            throw new OcrFormatException("The image cannot be placed in a PDF: " + e.Message, e);
        }
    }
}
