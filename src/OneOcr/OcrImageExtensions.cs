using OneImage;

namespace OneOcr;

/// <summary>
/// OCR as extension methods on images. The overloads without an engine use <see cref="OcrEngine.Default"/>. Calls that return one
/// result for an image (<c>ToText</c>, <c>ToMarkdown</c>) read every frame of a multi-page file such as a TIFF; <see cref="Ocr(Image, OcrOptions?, CancellationToken)"/>
/// returns one page and refuses a multi-frame image, and <c>OcrPages</c> returns one page per frame. Each has an <c>Async</c> twin
/// that runs the same code on a thread-pool thread and returns the same result.
/// </summary>
public static class OcrImageExtensions
{
    /// <summary>Reads the text of a page with <see cref="OcrEngine.Default"/>. See <see cref="OcrEngine.ToText(Image, OcrOptions?, CancellationToken)"/>.</summary>
    /// <param name="image">The page, or a multi-frame image (its frames' texts are joined with a blank line).</param>
    /// <param name="options">Options for this call; null for the engine's <see cref="OcrEngine.Options"/>.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The text, lines joined with <c>\n</c>.</returns>
    /// <exception cref="InvalidOperationException">No default engine is configured.</exception>
    public static string ToText(this Image image, OcrOptions? options = null, CancellationToken cancellationToken = default) =>
        OcrEngine.Default.ToText(image, options, cancellationToken);

    /// <summary>Reads the text of a page with the given engine. See <see cref="OcrEngine.ToText(Image, OcrOptions?, CancellationToken)"/>.</summary>
    /// <param name="image">The page, or a multi-frame image (its frames' texts are joined with a blank line).</param>
    /// <param name="engine">The engine to use.</param>
    /// <param name="options">Options for this call; null for the engine's <see cref="OcrEngine.Options"/>.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The text, lines joined with <c>\n</c>.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is null.</exception>
    public static string ToText(this Image image, OcrEngine engine, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.ToText(image, options, cancellationToken);
    }

    /// <summary>Like <see cref="ToText(Image, OcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same text.</summary>
    /// <param name="image">The page; keep it alive until the task completes.</param>
    /// <param name="options">Options for this call; null for the engine's <see cref="OcrEngine.Options"/>.</param>
    /// <param name="cancellationToken">Cancels between lines; the task then ends as cancelled.</param>
    /// <returns>The text.</returns>
    /// <exception cref="InvalidOperationException">No default engine is configured.</exception>
    public static Task<string> ToTextAsync(this Image image, OcrOptions? options = null, CancellationToken cancellationToken = default) =>
        OcrEngine.Default.ToTextAsync(image, options, cancellationToken);

    /// <summary>Like <see cref="ToText(Image, OcrEngine, OcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same text.</summary>
    /// <param name="image">The page; keep it alive until the task completes.</param>
    /// <param name="engine">The engine to use.</param>
    /// <param name="options">Options for this call; null for the engine's <see cref="OcrEngine.Options"/>.</param>
    /// <param name="cancellationToken">Cancels between lines; the task then ends as cancelled.</param>
    /// <returns>The text.</returns>
    public static Task<string> ToTextAsync(this Image image, OcrEngine engine, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.ToTextAsync(image, options, cancellationToken);
    }

    /// <summary>Reads a page image with <see cref="OcrEngine.Default"/> and writes it as Markdown (see <see cref="OcrMarkdown"/>). Everything is local. Every frame of a multi-page file is read.</summary>
    /// <param name="image">The page image, or a multi-frame image.</param>
    /// <param name="options">Markdown options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The Markdown; empty when no text was found.</returns>
    /// <exception cref="InvalidOperationException">No default engine is configured.</exception>
    public static string ToMarkdown(this Image image, MarkdownOptions? options = null, CancellationToken cancellationToken = default) =>
        ToMarkdown(image, OcrEngine.Default, options, cancellationToken);

    /// <summary>Reads a page image with the given engine and writes it as Markdown. Every frame of a multi-page file is read.</summary>
    /// <param name="image">The page image, or a multi-frame image.</param>
    /// <param name="engine">The engine to read with.</param>
    /// <param name="options">Markdown options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The Markdown; empty when no text was found.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is null.</exception>
    public static string ToMarkdown(this Image image, OcrEngine engine, MarkdownOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        IReadOnlyList<OcrPage> pages = engine.OcrPages(image, null, cancellationToken);
        return pages.Count == 1 ? pages[0].ToMarkdown(options) : pages.ToMarkdown(options, cancellationToken);
    }

    /// <summary>Like <see cref="ToMarkdown(Image, MarkdownOptions?, CancellationToken)"/>, on a thread-pool thread, with the same Markdown.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="options">Markdown options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between lines; the task then ends as cancelled.</param>
    /// <returns>The Markdown.</returns>
    /// <exception cref="InvalidOperationException">No default engine is configured.</exception>
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

    /// <summary>Reads a single-frame page image into the page model with the default engine (see <see cref="OcrEngine.Default"/>).</summary>
    /// <param name="image">The page image. For a multi-page file use <see cref="OcrPages(Image, OcrOptions?, CancellationToken)"/>.</param>
    /// <param name="options">Options for this call; null for the engine's <see cref="OcrEngine.Options"/>.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The page.</returns>
    /// <exception cref="InvalidOperationException">No default engine is configured.</exception>
    /// <exception cref="ArgumentException"><paramref name="image"/> has more than one frame.</exception>
    public static OcrPage Ocr(this Image image, OcrOptions? options = null, CancellationToken cancellationToken = default) =>
        OcrEngine.Default.Ocr(image, options, cancellationToken);

    /// <summary>Reads a single-frame page image into the page model with the given engine.</summary>
    /// <param name="image">The page image. For a multi-page file use <see cref="OcrPages(Image, OcrEngine, OcrOptions?, CancellationToken)"/>.</param>
    /// <param name="engine">The engine to read with.</param>
    /// <param name="options">Options for this call; null for the engine's <see cref="OcrEngine.Options"/>.</param>
    /// <param name="cancellationToken">Cancels between lines.</param>
    /// <returns>The page.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="image"/> has more than one frame.</exception>
    public static OcrPage Ocr(this Image image, OcrEngine engine, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.Ocr(image, options, cancellationToken);
    }

    /// <summary>Reads every frame of an image into the page model with the default engine: one page per frame.</summary>
    /// <param name="image">The image, for example a multi-page TIFF.</param>
    /// <param name="options">Options for every page; null for the engine's <see cref="OcrEngine.Options"/>.</param>
    /// <param name="cancellationToken">Cancels between pages and lines.</param>
    /// <returns>The pages in order.</returns>
    /// <exception cref="InvalidOperationException">No default engine is configured.</exception>
    public static IReadOnlyList<OcrPage> OcrPages(this Image image, OcrOptions? options = null, CancellationToken cancellationToken = default) =>
        OcrEngine.Default.OcrPages(image, options, cancellationToken);

    /// <summary>Reads every frame of an image into the page model with the given engine: one page per frame.</summary>
    /// <param name="image">The image, for example a multi-page TIFF.</param>
    /// <param name="engine">The engine to read with.</param>
    /// <param name="options">Options for every page; null for the engine's <see cref="OcrEngine.Options"/>.</param>
    /// <param name="cancellationToken">Cancels between pages and lines.</param>
    /// <returns>The pages in order.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is null.</exception>
    public static IReadOnlyList<OcrPage> OcrPages(this Image image, OcrEngine engine, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.OcrPages(image, options, cancellationToken);
    }

    /// <summary>Like <see cref="Ocr(Image, OcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same page.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="options">Options for this call; null for the engine's <see cref="OcrEngine.Options"/>.</param>
    /// <param name="cancellationToken">Cancels between lines; the task then ends as cancelled.</param>
    /// <returns>The page.</returns>
    /// <exception cref="InvalidOperationException">No default engine is configured.</exception>
    public static Task<OcrPage> OcrAsync(this Image image, OcrOptions? options = null, CancellationToken cancellationToken = default) =>
        OcrEngine.Default.OcrAsync(image, options, cancellationToken);

    /// <summary>Like <see cref="Ocr(Image, OcrEngine, OcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same page.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="engine">The engine to read with.</param>
    /// <param name="options">Options for this call; null for the engine's <see cref="OcrEngine.Options"/>.</param>
    /// <param name="cancellationToken">Cancels between lines; the task then ends as cancelled.</param>
    /// <returns>The page.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="engine"/> is null.</exception>
    public static Task<OcrPage> OcrAsync(this Image image, OcrEngine engine, OcrOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return engine.OcrAsync(image, options, cancellationToken);
    }
}
