using OneImage;

namespace OneOcr.Documents;

/// <summary>
/// Document extraction as extension methods on pages and images. The extractor defaults to <see cref="DocumentExtractor.Default"/>;
/// image overloads read the image with <see cref="OcrEngine.Default"/> (or the engine given) first. All of it runs locally.
/// </summary>
public static class DocumentExtensions
{
    /// <summary>Reads a page as an invoice with <see cref="DocumentExtractor.Default"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>The invoice; fields not found are null.</returns>
    /// <exception cref="InvalidOperationException">No default extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Invoice ExtractInvoice(this OcrPage page, CancellationToken cancellationToken = default) => DocumentExtractor.Default.ExtractInvoice(page, null, cancellationToken);

    /// <summary>Reads a page as an invoice.</summary>
    /// <param name="page">The page.</param>
    /// <param name="extractor">The extractor to use.</param>
    /// <returns>The invoice; fields not found are null.</returns>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Invoice ExtractInvoice(this OcrPage page, DocumentExtractor extractor, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(extractor);
        return extractor.ExtractInvoice(page, null, cancellationToken);
    }

    /// <summary>Reads a page as a receipt with <see cref="DocumentExtractor.Default"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>The receipt; fields not found are null.</returns>
    /// <exception cref="InvalidOperationException">No default extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Receipt ExtractReceipt(this OcrPage page, CancellationToken cancellationToken = default) => DocumentExtractor.Default.ExtractReceipt(page, null, cancellationToken);

    /// <summary>Reads a page as a receipt.</summary>
    /// <param name="page">The page.</param>
    /// <param name="extractor">The extractor to use.</param>
    /// <returns>The receipt; fields not found are null.</returns>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Receipt ExtractReceipt(this OcrPage page, DocumentExtractor extractor, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(extractor);
        return extractor.ExtractReceipt(page, null, cancellationToken);
    }

    /// <summary>Reads a page with the schema the model predicts for it, using <see cref="DocumentExtractor.Default"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>Fields, line items, tax rows and key–value pairs.</returns>
    /// <exception cref="InvalidOperationException">No default extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static DocumentExtraction Extract(this OcrPage page, CancellationToken cancellationToken = default) => DocumentExtractor.Default.Extract(page, null, cancellationToken);

    /// <summary>Finds generic key–value pairs on a page, using <see cref="DocumentExtractor.Default"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>The pairs.</returns>
    /// <exception cref="InvalidOperationException">No default extractor is configured.</exception>
    public static IReadOnlyList<OcrKeyValue> FindKeyValues(this OcrPage page) => DocumentExtractor.Default.FindKeyValues(page);

    /// <summary>A copy of the page whose <see cref="OcrPage.KeyValues"/> also hold the pairs found by <see cref="DocumentExtractor.Default"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>The new page.</returns>
    /// <exception cref="InvalidOperationException">No default extractor is configured.</exception>
    public static OcrPage WithKeyValues(this OcrPage page) => DocumentExtractor.Default.WithKeyValues(page);

    /// <summary>Reads an image with <see cref="OcrEngine.Default"/> and reads the page as an invoice with <see cref="DocumentExtractor.Default"/>.</summary>
    /// <param name="image">The page image.</param>
    /// <returns>The invoice.</returns>
    /// <exception cref="InvalidOperationException">No default engine or extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Invoice ExtractInvoice(this Image image, CancellationToken cancellationToken = default) =>
        DocumentExtractor.Default.ExtractInvoice(OcrEngine.Default.Ocr(image, null, cancellationToken), null, cancellationToken);

    /// <summary>Like <see cref="ExtractInvoice(Image, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The invoice.</returns>
    /// <exception cref="InvalidOperationException">No default engine or extractor is configured.</exception>
    public static Task<Invoice> ExtractInvoiceAsync(this Image image, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.ExtractInvoice(cancellationToken), cancellationToken);

    /// <summary>Reads an image with the given engine and reads the page as an invoice.</summary>
    /// <param name="image">The page image.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="DocumentExtractor.Default"/> when null.</param>
    /// <returns>The invoice.</returns>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Invoice ExtractInvoice(this Image image, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return (extractor ?? DocumentExtractor.Default).ExtractInvoice(engine.Ocr(image, null, cancellationToken), null, cancellationToken);
    }

    /// <summary>Like <see cref="ExtractInvoice(Image, OcrEngine, DocumentExtractor?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="DocumentExtractor.Default"/> when null.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The invoice.</returns>
    public static Task<Invoice> ExtractInvoiceAsync(this Image image, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.ExtractInvoice(engine, extractor, cancellationToken), cancellationToken);

    /// <summary>Reads an image with <see cref="OcrEngine.Default"/> and reads the page as a receipt with <see cref="DocumentExtractor.Default"/>.</summary>
    /// <param name="image">The page image.</param>
    /// <returns>The receipt.</returns>
    /// <exception cref="InvalidOperationException">No default engine or extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Receipt ExtractReceipt(this Image image, CancellationToken cancellationToken = default) =>
        DocumentExtractor.Default.ExtractReceipt(OcrEngine.Default.Ocr(image, null, cancellationToken), null, cancellationToken);

    /// <summary>Like <see cref="ExtractReceipt(Image, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The receipt.</returns>
    /// <exception cref="InvalidOperationException">No default engine or extractor is configured.</exception>
    public static Task<Receipt> ExtractReceiptAsync(this Image image, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.ExtractReceipt(cancellationToken), cancellationToken);

    /// <summary>Reads an image with the given engine and reads the page as a receipt.</summary>
    /// <param name="image">The page image.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="DocumentExtractor.Default"/> when null.</param>
    /// <returns>The receipt.</returns>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Receipt ExtractReceipt(this Image image, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return (extractor ?? DocumentExtractor.Default).ExtractReceipt(engine.Ocr(image, null, cancellationToken), null, cancellationToken);
    }

    /// <summary>Like <see cref="ExtractReceipt(Image, OcrEngine, DocumentExtractor?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="DocumentExtractor.Default"/> when null.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The receipt.</returns>
    public static Task<Receipt> ExtractReceiptAsync(this Image image, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.ExtractReceipt(engine, extractor, cancellationToken), cancellationToken);
    /// <summary>Reads a page with <see cref="DocumentExtractor.Default"/> (a custom model) and returns its fields as <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">The caller's type.</typeparam>
    /// <param name="page">The page.</param>
    /// <param name="typeInfo">Source-generated type information for <typeparamref name="T"/>.</param>
    /// <returns>The value with per-field words, boxes and confidences; see <see cref="DocumentExtractor.Extract{T}"/>.</returns>
    /// <exception cref="InvalidOperationException">No default extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static ExtractionResult<T> Extract<T>(this OcrPage page, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, CancellationToken cancellationToken = default) =>
        DocumentExtractor.Default.Extract(page, typeInfo, null, cancellationToken);

    /// <summary>Reads a page with the given custom extractor and returns its fields as <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">The caller's type.</typeparam>
    /// <param name="page">The page.</param>
    /// <param name="typeInfo">Source-generated type information for <typeparamref name="T"/>.</param>
    /// <param name="extractor">The extractor to use.</param>
    /// <returns>The value with per-field words, boxes and confidences.</returns>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static ExtractionResult<T> Extract<T>(this OcrPage page, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, DocumentExtractor extractor, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(extractor);
        return extractor.Extract(page, typeInfo, null, cancellationToken);
    }

    /// <summary>Reads an image with <see cref="OcrEngine.Default"/>, then reads the page with <see cref="DocumentExtractor.Default"/> (a custom model) into <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">The caller's type.</typeparam>
    /// <param name="image">The page image.</param>
    /// <param name="typeInfo">Source-generated type information for <typeparamref name="T"/>.</param>
    /// <returns>The value with per-field words, boxes and confidences.</returns>
    /// <exception cref="InvalidOperationException">No default engine or extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static ExtractionResult<T> Extract<T>(this Image image, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, CancellationToken cancellationToken = default) =>
        DocumentExtractor.Default.Extract(OcrEngine.Default.Ocr(image, null, cancellationToken), typeInfo, null, cancellationToken);

    /// <summary>Like <see cref="Extract{T}(Image, System.Text.Json.Serialization.Metadata.JsonTypeInfo{T}, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <typeparam name="T">The caller's type.</typeparam>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="typeInfo">Source-generated type information for <typeparamref name="T"/>.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The value with per-field words, boxes and confidences.</returns>
    public static Task<ExtractionResult<T>> ExtractAsync<T>(this Image image, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.Extract(typeInfo, cancellationToken), cancellationToken);

    /// <summary>Reads an image with the given engine, then reads the page with a custom extractor into <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">The caller's type.</typeparam>
    /// <param name="image">The page image.</param>
    /// <param name="typeInfo">Source-generated type information for <typeparamref name="T"/>.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="DocumentExtractor.Default"/> when null.</param>
    /// <returns>The value with per-field words, boxes and confidences.</returns>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static ExtractionResult<T> Extract<T>(this Image image, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return (extractor ?? DocumentExtractor.Default).Extract(engine.Ocr(image, null, cancellationToken), typeInfo, null, cancellationToken);
    }

    /// <summary>Like <see cref="Extract{T}(Image, System.Text.Json.Serialization.Metadata.JsonTypeInfo{T}, OcrEngine, DocumentExtractor?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <typeparam name="T">The caller's type.</typeparam>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="typeInfo">Source-generated type information for <typeparamref name="T"/>.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="DocumentExtractor.Default"/> when null.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The value with per-field words, boxes and confidences.</returns>
    public static Task<ExtractionResult<T>> ExtractAsync<T>(this Image image, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.Extract(typeInfo, engine, extractor, cancellationToken), cancellationToken);
}
