using OneImage;
using OneOcr.Documents;

namespace OneOcr;

/// <summary>
/// The process-wide default engine used by the extension methods that take no engine. Call <see cref="Configure(OcrEngine)"/>
/// once at startup; everything else can pass an <see cref="OcrEngine"/> explicitly.
/// </summary>
public static class OneOcr
{
    private static OcrEngine? s_default;
    private static DocumentExtractor? s_extractor;

    /// <summary>The configured default engine.</summary>
    /// <exception cref="OcrModelException">No engine has been configured.</exception>
    public static OcrEngine Default =>
        Volatile.Read(ref s_default) ?? throw new OcrModelException(
            "No OCR model is configured. Call OneOcr.Configure() (the bundled recognizer) or OneOcr.Configure(OcrEngine.Load(modelPath)) once at startup, or pass an engine explicitly.");

    /// <summary>Sets <see cref="Default"/>. May be called once.</summary>
    /// <param name="engine">The engine to use by default.</param>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is null.</exception>
    /// <exception cref="InvalidOperationException">A default engine was already configured.</exception>
    public static void Configure(OcrEngine engine)
    {
        ArgumentNullException.ThrowIfNull(engine);
        if (Interlocked.CompareExchange(ref s_default, engine, null) is not null)
        {
            throw new InvalidOperationException("OneOcr.Default is already configured.");
        }
    }

    /// <summary>Loads the recognizer and page detector that ship in the models package and sets it as <see cref="Default"/>. May be called once.</summary>
    /// <exception cref="OcrModelException">The package carries no weights, or they do not match their pin.</exception>
    /// <exception cref="InvalidOperationException">A default engine was already configured.</exception>
    public static void Configure() => Configure(OcrEngine.LoadBundled());

    /// <summary>The configured default document extractor, used by the extraction extension methods that take none.</summary>
    /// <exception cref="OcrModelException">No extractor has been configured.</exception>
    public static DocumentExtractor DefaultExtractor =>
        Volatile.Read(ref s_extractor) ?? throw new OcrModelException(
            "No document extraction model is configured. Call OneOcr.ConfigureExtractor(DocumentExtractor.Load(modelPath)) once at startup (the RecursiveAI.OneOcr.Models.Documents package offers the bundled model), or pass an extractor explicitly.");

    /// <summary>Sets <see cref="DefaultExtractor"/>. May be called once.</summary>
    /// <param name="extractor">The extractor to use by default.</param>
    /// <exception cref="ArgumentNullException"><paramref name="extractor"/> is null.</exception>
    /// <exception cref="InvalidOperationException">A default extractor was already configured.</exception>
    public static void ConfigureExtractor(DocumentExtractor extractor)
    {
        ArgumentNullException.ThrowIfNull(extractor);
        if (Interlocked.CompareExchange(ref s_extractor, extractor, null) is not null)
        {
            throw new InvalidOperationException("OneOcr.DefaultExtractor is already configured.");
        }
    }

    /// <summary>Loads a model file and sets it as <see cref="Default"/>. May be called once.</summary>
    /// <param name="modelPath">A model container file.</param>
    /// <exception cref="OcrModelException">The model is malformed.</exception>
    /// <exception cref="InvalidOperationException">A default engine was already configured.</exception>
    public static void Configure(string modelPath) => Configure(OcrEngine.Load(modelPath));
}

/// <summary>OCR as extension methods on images. Each has an <c>Async</c> twin that runs the same code on a thread-pool thread and returns the same result.</summary>
public static class OcrImageExtensions
{
    /// <summary>Reads the text of a page with <see cref="OneOcr.Default"/>. See <see cref="OcrEngine.ToText(Image, int, CancellationToken)"/>.</summary>
    /// <param name="image">The page.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The text, lines joined with <c>\n</c>.</returns>
    /// <exception cref="OcrModelException">No default engine is configured.</exception>
    public static string ToText(this Image image, CancellationToken cancellationToken = default) =>
        OneOcr.Default.ToText(image, 0, cancellationToken);

    /// <summary>Reads the text of a page with the given engine. See <see cref="OcrEngine.ToText(Image, int, CancellationToken)"/>.</summary>
    /// <param name="image">The page.</param>
    /// <param name="engine">The engine to use.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The text, lines joined with <c>\n</c>.</returns>
    public static string ToText(this Image image, OcrEngine engine, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.ToText(image, 0, cancellationToken);
    }

    /// <summary>Like <see cref="ToText(Image, CancellationToken)"/>, on a thread-pool thread, with the same text.</summary>
    /// <param name="image">The page; keep it alive until the task completes.</param>
    /// <param name="cancellationToken">Cancels between lines; the task then ends as cancelled.</param>
    /// <returns>The text.</returns>
    /// <exception cref="OcrModelException">No default engine is configured.</exception>
    public static Task<string> ToTextAsync(this Image image, CancellationToken cancellationToken = default) =>
        OneOcr.Default.ToTextAsync(image, 0, cancellationToken);

    /// <summary>Like <see cref="ToText(Image, OcrEngine, CancellationToken)"/>, on a thread-pool thread, with the same text.</summary>
    /// <param name="image">The page; keep it alive until the task completes.</param>
    /// <param name="engine">The engine to use.</param>
    /// <param name="cancellationToken">Cancels between lines; the task then ends as cancelled.</param>
    /// <returns>The text.</returns>
    public static Task<string> ToTextAsync(this Image image, OcrEngine engine, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.ToTextAsync(image, 0, cancellationToken);
    }

    /// <summary>Reads a page image with <see cref="OneOcr.Default"/> and writes it as Markdown (see <see cref="OcrMarkdown"/>). Everything is local.</summary>
    /// <param name="image">The page image.</param>
    /// <param name="options">Markdown options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The Markdown; empty when no text was found.</returns>
    /// <exception cref="OcrModelException">No default engine is configured.</exception>
    public static string ToMarkdown(this Image image, MarkdownOptions? options = null, CancellationToken cancellationToken = default) =>
        OneOcr.Default.Ocr(image, null, cancellationToken).ToMarkdown(options);

    /// <summary>Reads a page image with the given engine and writes it as Markdown.</summary>
    /// <param name="image">The page image.</param>
    /// <param name="engine">The engine to read with.</param>
    /// <param name="options">Markdown options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The Markdown; empty when no text was found.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is null.</exception>
    public static string ToMarkdown(this Image image, OcrEngine engine, MarkdownOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.Ocr(image, null, cancellationToken).ToMarkdown(options);
    }

    /// <summary>Like <see cref="ToMarkdown(Image, MarkdownOptions?, CancellationToken)"/>, on a thread-pool thread, with the same Markdown.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="options">Markdown options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines; the task then ends as cancelled.</param>
    /// <returns>The Markdown.</returns>
    /// <exception cref="OcrModelException">No default engine is configured.</exception>
    public static Task<string> ToMarkdownAsync(this Image image, MarkdownOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.ToMarkdown(options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ToMarkdown(Image, OcrEngine, MarkdownOptions?, CancellationToken)"/>, on a thread-pool thread, with the same Markdown.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="engine">The engine to read with.</param>
    /// <param name="options">Markdown options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines; the task then ends as cancelled.</param>
    /// <returns>The Markdown.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is null.</exception>
    public static Task<string> ToMarkdownAsync(this Image image, OcrEngine engine, MarkdownOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return Task.Run(() => image.ToMarkdown(engine, options, cancellationToken), cancellationToken);
    }

    /// <summary>Reads a page image into the page model with the default engine (see <see cref="OneOcr.Default"/>).</summary>
    /// <param name="image">The page image.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The page.</returns>
    /// <exception cref="OcrModelException">No default engine is configured.</exception>
    public static OcrPage Ocr(this Image image, OcrOptions? options = null, CancellationToken cancellationToken = default) =>
        OneOcr.Default.Ocr(image, options, cancellationToken);

    /// <summary>Reads a page image into the page model with the given engine.</summary>
    /// <param name="image">The page image.</param>
    /// <param name="engine">The engine to read with.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The page.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is null.</exception>
    public static OcrPage Ocr(this Image image, OcrEngine engine, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.Ocr(image, options, cancellationToken);
    }

    /// <summary>Like <see cref="Ocr(Image, OcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same page.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines; the task then ends as cancelled.</param>
    /// <returns>The page.</returns>
    /// <exception cref="OcrModelException">No default engine is configured.</exception>
    public static Task<OcrPage> OcrAsync(this Image image, OcrOptions? options = null, CancellationToken cancellationToken = default) =>
        OneOcr.Default.OcrAsync(image, options, cancellationToken);

    /// <summary>Like <see cref="Ocr(Image, OcrEngine, OcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same page.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="engine">The engine to read with.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines; the task then ends as cancelled.</param>
    /// <returns>The page.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is null.</exception>
    public static Task<OcrPage> OcrAsync(this Image image, OcrEngine engine, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.OcrAsync(image, options, cancellationToken);
    }
}
