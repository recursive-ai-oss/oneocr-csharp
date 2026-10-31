using OnePdf;

namespace OneOcr.Pdf;

/// <summary>
/// Reads PDF files into the page model. Pages with a text layer give its words (<see cref="OcrSource.TextLayer"/>) in points, y down,
/// with <see cref="OcrPage.ToPdf"/> mapping back to PDF user space; pages with scanned or garbled regions give the local recognizer's
/// reading of those regions (<see cref="OcrSource.Ocr"/>), merged into the same page. See <see cref="PdfOcrOptions"/> for what is trusted.
/// </summary>
/// <remarks>
/// All parsing goes through OnePDF and rendering through the OneImage PDF bridge; both are safe code with their own limits.
/// Local reads never touch the network. Output is bit-identical at every degree of parallelism.
/// </remarks>
public static class PdfOcr
{
    /// <summary>Reads the pages of a PDF held in memory.</summary>
    /// <param name="pdf">The file's bytes.</param>
    /// <param name="engine">The recognizer; <see cref="OcrEngine.Default"/> when null.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>One page per PDF page in the selected range (<see cref="PdfOcrOptions.FirstPage"/>, <see cref="PdfOcrOptions.PageCount"/>). When the file opens but no pages can be found (a damaged page tree), a lenient read returns one empty 0 by 0 page whose diagnostics say so, never an empty list; <see cref="PdfOcrOptions.Strict"/> throws.</returns>
    /// <exception cref="ArgumentException">An option is out of range.</exception>
    /// <exception cref="OcrModelException">No engine was given and none is configured.</exception>
    /// <exception cref="OcrFormatException">The bytes are not a PDF that OnePDF can open (or the password is wrong); with <see cref="PdfOcrOptions.Strict"/>, also a file whose page tree yields no pages.</exception>
    /// <exception cref="OcrLimitException">The file or the page count exceeds a limit; with <see cref="PdfOcrOptions.Strict"/>, so can a page.</exception>
    public static IReadOnlyList<OcrPage> Read(
        ReadOnlyMemory<byte> pdf, OcrEngine? engine = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new PdfOcrOptions();
        options.Validate();
        engine ??= OcrEngine.Default;
        if (pdf.Length > options.MaxFileBytes)
        {
            throw new OcrLimitException(nameof(PdfOcrOptions.MaxFileBytes), options.MaxFileBytes, pdf.Length);
        }

        PdfDocument document = Open(pdf, options);
        using (document)
        {
            return ReadPages(document, () => pdf, engine, options, cancellationToken);
        }
    }

    /// <summary>Reads the pages of a PDF file.</summary>
    /// <param name="path">The file.</param>
    /// <param name="engine">The recognizer; <see cref="OcrEngine.Default"/> when null.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>One page per PDF page in the selected range.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is null.</exception>
    /// <exception cref="OcrFormatException">The file is not a PDF that OnePDF can open.</exception>
    /// <exception cref="OcrLimitException">The file is larger than <see cref="PdfOcrOptions.MaxFileBytes"/>, or a limit is exceeded.</exception>
    /// <exception cref="IOException">The file cannot be read.</exception>
    public static IReadOnlyList<OcrPage> Read(
        string path, OcrEngine? engine = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(path);
        options ??= new PdfOcrOptions();
        options.Validate();
        long length = new FileInfo(path).Length;
        if (length > options.MaxFileBytes)
        {
            throw new OcrLimitException(nameof(PdfOcrOptions.MaxFileBytes), options.MaxFileBytes, length);
        }

        return Read((ReadOnlyMemory<byte>)File.ReadAllBytes(path), engine, options, cancellationToken);
    }

    /// <summary>Reads the pages of a PDF from a stream, reading at most <see cref="PdfOcrOptions.MaxFileBytes"/> bytes.</summary>
    /// <param name="stream">A readable stream positioned at the start of the file.</param>
    /// <param name="engine">The recognizer; <see cref="OcrEngine.Default"/> when null.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>One page per PDF page in the selected range.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
    /// <exception cref="OcrFormatException">The data is not a PDF that OnePDF can open.</exception>
    /// <exception cref="OcrLimitException">The stream holds more than <see cref="PdfOcrOptions.MaxFileBytes"/> bytes, or a limit is exceeded.</exception>
    public static IReadOnlyList<OcrPage> Read(
        Stream stream, OcrEngine? engine = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        options ??= new PdfOcrOptions();
        options.Validate();
        using var buffer = new MemoryStream();
        var chunk = new byte[81920];
        long total = 0;
        int read;
        while ((read = stream.Read(chunk, 0, chunk.Length)) > 0)
        {
            total += read;
            if (total > options.MaxFileBytes)
            {
                throw new OcrLimitException(nameof(PdfOcrOptions.MaxFileBytes), options.MaxFileBytes, total);
            }

            buffer.Write(chunk, 0, read);
        }

        return Read((ReadOnlyMemory<byte>)buffer.ToArray(), engine, options, cancellationToken);
    }

    /// <summary>Like <see cref="Read(ReadOnlyMemory{byte}, OcrEngine?, PdfOcrOptions?, CancellationToken)"/>, on a worker thread.</summary>
    /// <param name="pdf">The file's bytes.</param>
    /// <param name="engine">The recognizer; <see cref="OcrEngine.Default"/> when null.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>The pages.</returns>
    public static Task<IReadOnlyList<OcrPage>> ReadAsync(
        ReadOnlyMemory<byte> pdf, OcrEngine? engine = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Read(pdf, engine, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="Read(string, OcrEngine?, PdfOcrOptions?, CancellationToken)"/>, on a worker thread.</summary>
    /// <param name="path">The file.</param>
    /// <param name="engine">The recognizer; <see cref="OcrEngine.Default"/> when null.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>The pages.</returns>
    public static Task<IReadOnlyList<OcrPage>> ReadAsync(
        string path, OcrEngine? engine = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Read(path, engine, options, cancellationToken), cancellationToken);

    internal static PdfDocument Open(ReadOnlyMemory<byte> pdf, PdfOcrOptions options)
    {
        try
        {
            return PdfDocument.Open(pdf, new PdfOpenOptions { Password = options.Password });
        }
        catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
        {
            throw PdfErrors.Wrap(e);
        }
    }

    /// <summary>Reads a range of pages of an open document; <paramref name="bytes"/> yields the file when a page needs rendering.</summary>
    internal static IReadOnlyList<OcrPage> ReadPages(
        PdfDocument document, Func<ReadOnlyMemory<byte>> bytes, OcrEngine engine, PdfOcrOptions options, CancellationToken cancellationToken)
    {
        int total;
        try
        {
            total = document.PageCount;
        }
        catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
        {
            // The page tree is read lazily, so a damaged file can fail here and not in Open.
            throw PdfErrors.Wrap(e);
        }
        if (total == 0)
        {
            // A PDF always has a page; none found means the catalog or page tree is damaged. Never return an empty list silently.
            const string Why = "The PDF has no readable pages: its page tree is missing or damaged.";
            if (options.Strict)
            {
                throw new OcrFormatException(Why);
            }

            return [new OcrPage { Index = 0, Width = 0, Height = 0, UnitsPerInch = 72, Blocks = [], Diagnostics = [new OcrDiagnostic(OcrDiagnosticCodes.Pdf, OcrDiagnosticSeverity.Warning, Why)] }];
        }

        int first = options.FirstPage;
        int count = first >= total ? 0 : (int)Math.Min((long)options.PageCount, (long)total - first);
        if (count > engine.Limits.MaxPagesPerCall)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxPagesPerCall), engine.Limits.MaxPagesPerCall, count);
        }

        if (count == 0)
        {
            return [];
        }

        int degree = options.MaxDegreeOfParallelism > 0 ? options.MaxDegreeOfParallelism : Math.Min(4, Environment.ProcessorCount);
        int outer = Math.Min(degree, count);
        var ctx = new ReadContext
        {
            Engine = engine,
            Options = options,
            Renderer = new PdfRenderer(bytes, options, engine.Limits, first, count),
            RegionDegree = outer > 1 ? 1 : degree,
        };
        var pages = new OcrPage[count];
        try
        {
            if (outer == 1)
            {
                for (int i = 0; i < count; i++)
                {
                    pages[i] = ReadOne(document, first + i, ctx, cancellationToken);
                }
            }
            else
            {
                Parallel.For(
                    0, count, new ParallelOptions { MaxDegreeOfParallelism = outer, CancellationToken = cancellationToken },
                    i => pages[i] = ReadOne(document, first + i, ctx, cancellationToken));
            }
        }
        catch (AggregateException e) when (e.InnerExceptions.Count > 0)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
        }

        return pages;
    }

    private static OcrPage ReadOne(PdfDocument document, int index, ReadContext ctx, CancellationToken cancellationToken)
    {
        try
        {
            return PdfPageReader.Read(document, index, ctx, cancellationToken);
        }
        catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
        {
            // A page whose dictionary cannot even be read: nothing of it can be used.
            throw PdfErrors.Wrap(e);
        }
    }
}

/// <summary>OCR for pages and documents that OnePDF has already opened, as extension methods.</summary>
/// <remarks>
/// Rendering needs the file's bytes and an open <see cref="PdfDocument"/> does not keep them, so a read that has to render
/// (a scanned region, garbled text, or the text-layer check, which <see cref="PdfOcrOptions.CrossCheckLines"/> = 0 turns off) saves the
/// document to memory first. Reading from bytes, a path or a stream with <see cref="PdfOcr"/> avoids that copy.
/// </remarks>
public static class PdfOcrExtensions
{
    /// <summary>Reads one page with <see cref="OcrEngine.Default"/>.</summary>
    /// <param name="page">The page.</param>
    /// <param name="options">Per-call options; null for the defaults. <see cref="PdfOcrOptions.FirstPage"/> and <see cref="PdfOcrOptions.PageCount"/> are ignored.</param>
    /// <param name="cancellationToken">Cancels between regions.</param>
    /// <returns>The page model.</returns>
    /// <exception cref="InvalidOperationException">No default engine is configured.</exception>
    public static OcrPage Ocr(this PdfPage page, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Ocr(page, OcrEngine.Default, options, cancellationToken);

    /// <summary>Reads one page with the given engine.</summary>
    /// <param name="page">The page.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="options">Per-call options; null for the defaults. <see cref="PdfOcrOptions.FirstPage"/> and <see cref="PdfOcrOptions.PageCount"/> are ignored.</param>
    /// <param name="cancellationToken">Cancels between regions.</param>
    /// <returns>The page model.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="page"/> or <paramref name="engine"/> is null.</exception>
    /// <exception cref="OcrFormatException">The page cannot be read at all, or (with <see cref="PdfOcrOptions.Strict"/>) only in part.</exception>
    /// <exception cref="OcrLimitException">A limit is exceeded.</exception>
    public static OcrPage Ocr(this PdfPage page, OcrEngine engine, PdfOcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(page);
        ArgumentNullException.ThrowIfNull(engine);
        options ??= new PdfOcrOptions();
        options.Validate();
        var single = new PdfOcrOptions
        {
            TrustExistingOcrLayer = options.TrustExistingOcrLayer,
            ReadImages = options.ReadImages,
            RenderDpi = options.RenderDpi,
            CrossCheckLines = options.CrossCheckLines,
            MinImagePoints = options.MinImagePoints,
            IncludeChars = options.IncludeChars,
            MaxDegreeOfParallelism = options.MaxDegreeOfParallelism,
            FirstPage = page.Index,
            PageCount = 1,
            Password = options.Password,
            Strict = options.Strict,
            MaxFileBytes = options.MaxFileBytes,
        };
        return PdfOcr.ReadPages(page.Document, () => Serialize(page.Document), engine, single, cancellationToken)[0];
    }

    /// <summary>Reads the text of one page with <see cref="OcrEngine.Default"/>: lines joined by <c>\n</c>.</summary>
    /// <param name="page">The page.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between regions.</param>
    /// <returns>The text.</returns>
    /// <exception cref="InvalidOperationException">No default engine is configured.</exception>
    public static string ToText(this PdfPage page, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Ocr(page, options, cancellationToken).ToText();

    /// <summary>Reads the text of one page with the given engine: lines joined by <c>\n</c>.</summary>
    /// <param name="page">The page.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between regions.</param>
    /// <returns>The text.</returns>
    public static string ToText(this PdfPage page, OcrEngine engine, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Ocr(page, engine, options, cancellationToken).ToText();

    /// <summary>Reads the pages of an open document with <see cref="OcrEngine.Default"/>.</summary>
    /// <param name="document">The document.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>One page model per page in the selected range.</returns>
    /// <exception cref="InvalidOperationException">No default engine is configured.</exception>
    public static IReadOnlyList<OcrPage> Ocr(this PdfDocument document, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Ocr(document, OcrEngine.Default, options, cancellationToken);

    /// <summary>Reads the pages of an open document with the given engine.</summary>
    /// <param name="document">The document.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>One page model per page in the selected range.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="document"/> or <paramref name="engine"/> is null.</exception>
    /// <exception cref="OcrLimitException">The page count or a page exceeds a limit.</exception>
    public static IReadOnlyList<OcrPage> Ocr(this PdfDocument document, OcrEngine engine, PdfOcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(document);
        ArgumentNullException.ThrowIfNull(engine);
        options ??= new PdfOcrOptions();
        options.Validate();
        ReadOnlyMemory<byte>? saved = null;
        return PdfOcr.ReadPages(document, () => saved ??= Serialize(document), engine, options, cancellationToken);
    }

    /// <summary>Reads the text of an open document with <see cref="OcrEngine.Default"/>: pages joined by a form feed (<c>\f</c>).</summary>
    /// <param name="document">The document.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>The text.</returns>
    public static string ToText(this PdfDocument document, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        string.Join('\f', Ocr(document, options, cancellationToken).Select(p => p.ToText()));

    /// <summary>Reads the text of an open document with the given engine: pages joined by a form feed (<c>\f</c>).</summary>
    /// <param name="document">The document.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>The text.</returns>
    public static string ToText(this PdfDocument document, OcrEngine engine, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        string.Join('\f', Ocr(document, engine, options, cancellationToken).Select(p => p.ToText()));

    /// <summary>Reads one page with <see cref="OcrEngine.Default"/> and writes it as Markdown (see <see cref="OcrMarkdown"/>). Header and footer roles from the text layer are honored.</summary>
    /// <param name="page">The page.</param>
    /// <param name="markdown">Markdown options; null for the defaults.</param>
    /// <param name="options">Per-call read options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between regions.</param>
    /// <returns>The Markdown.</returns>
    /// <exception cref="InvalidOperationException">No default engine is configured.</exception>
    public static string ToMarkdown(this PdfPage page, MarkdownOptions? markdown = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Ocr(page, options, cancellationToken).ToMarkdown(markdown);

    /// <summary>Reads one page with the given engine and writes it as Markdown.</summary>
    /// <param name="page">The page.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="markdown">Markdown options; null for the defaults.</param>
    /// <param name="options">Per-call read options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between regions.</param>
    /// <returns>The Markdown.</returns>
    public static string ToMarkdown(this PdfPage page, OcrEngine engine, MarkdownOptions? markdown = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Ocr(page, engine, options, cancellationToken).ToMarkdown(markdown);

    /// <summary>Reads an open document with <see cref="OcrEngine.Default"/> and writes it as Markdown, pages separated by a thematic break.</summary>
    /// <param name="document">The document.</param>
    /// <param name="markdown">Markdown options; null for the defaults.</param>
    /// <param name="options">Per-call read options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>The Markdown.</returns>
    public static string ToMarkdown(this PdfDocument document, MarkdownOptions? markdown = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Ocr(document, options, cancellationToken).ToMarkdown(markdown, cancellationToken);

    /// <summary>Reads an open document with the given engine and writes it as Markdown, pages separated by a thematic break.</summary>
    /// <param name="document">The document.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="markdown">Markdown options; null for the defaults.</param>
    /// <param name="options">Per-call read options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>The Markdown.</returns>
    public static string ToMarkdown(this PdfDocument document, OcrEngine engine, MarkdownOptions? markdown = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Ocr(document, engine, options, cancellationToken).ToMarkdown(markdown, cancellationToken);

    /// <summary>Like <see cref="Ocr(PdfDocument, OcrEngine, PdfOcrOptions?, CancellationToken)"/>, on a worker thread.</summary>
    /// <param name="document">The document.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages and regions.</param>
    /// <returns>The pages.</returns>
    public static Task<IReadOnlyList<OcrPage>> OcrAsync(
        this PdfDocument document, OcrEngine engine, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Ocr(document, engine, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="Ocr(PdfPage, OcrEngine, PdfOcrOptions?, CancellationToken)"/>, on a worker thread.</summary>
    /// <param name="page">The page.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between regions.</param>
    /// <returns>The page model.</returns>
    public static Task<OcrPage> OcrAsync(
        this PdfPage page, OcrEngine engine, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Ocr(page, engine, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="Ocr(PdfPage, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="page">The page.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The page model.</returns>
    public static Task<OcrPage> OcrAsync(this PdfPage page, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Ocr(page, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToText(PdfPage, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="page">The page.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The text.</returns>
    public static Task<string> ToTextAsync(this PdfPage page, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToText(page, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToText(PdfPage, OcrEngine, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="page">The page.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The text.</returns>
    public static Task<string> ToTextAsync(this PdfPage page, OcrEngine engine, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToText(page, engine, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToMarkdown(PdfPage, MarkdownOptions?, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="page">The page.</param>
    /// <param name="markdown">Markdown options; null for the defaults.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The Markdown.</returns>
    public static Task<string> ToMarkdownAsync(this PdfPage page, MarkdownOptions? markdown = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToMarkdown(page, markdown, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToMarkdown(PdfPage, OcrEngine, MarkdownOptions?, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="page">The page.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="markdown">Markdown options; null for the defaults.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The Markdown.</returns>
    public static Task<string> ToMarkdownAsync(this PdfPage page, OcrEngine engine, MarkdownOptions? markdown = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToMarkdown(page, engine, markdown, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="Ocr(PdfDocument, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="document">The document.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The pages.</returns>
    public static Task<IReadOnlyList<OcrPage>> OcrAsync(this PdfDocument document, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => Ocr(document, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToText(PdfDocument, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="document">The document.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The text.</returns>
    public static Task<string> ToTextAsync(this PdfDocument document, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToText(document, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToText(PdfDocument, OcrEngine, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="document">The document.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The text.</returns>
    public static Task<string> ToTextAsync(this PdfDocument document, OcrEngine engine, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToText(document, engine, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToMarkdown(PdfDocument, MarkdownOptions?, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="document">The document.</param>
    /// <param name="markdown">Markdown options; null for the defaults.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The Markdown.</returns>
    public static Task<string> ToMarkdownAsync(this PdfDocument document, MarkdownOptions? markdown = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToMarkdown(document, markdown, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToMarkdown(PdfDocument, OcrEngine, MarkdownOptions?, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="document">The document.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="markdown">Markdown options; null for the defaults.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The Markdown.</returns>
    public static Task<string> ToMarkdownAsync(this PdfDocument document, OcrEngine engine, MarkdownOptions? markdown = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ToMarkdown(document, engine, markdown, options, cancellationToken), cancellationToken);

    private static ReadOnlyMemory<byte> Serialize(PdfDocument document)
    {
        try
        {
            using var ms = new MemoryStream();
            document.Save(ms, new PdfSaveOptions());
            return ms.ToArray();
        }
        catch (Exception e) when (PdfErrors.IsLibraryFailure(e))
        {
            throw PdfErrors.Wrap(e);
        }
    }
}
