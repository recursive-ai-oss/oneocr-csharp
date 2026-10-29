using OneImage;

namespace OneOcr.Documents;

/// <summary>
/// Document extraction as extension methods on pages and images. The extractor defaults to <see cref="OneOcr.DefaultExtractor"/>;
/// image overloads read the image with <see cref="OneOcr.Default"/> (or the engine given) first. All of it runs locally.
/// </summary>
public static class DocumentExtensions
{
    /// <summary>Reads a page as an invoice with <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>The invoice; fields not found are null.</returns>
    /// <exception cref="OcrModelException">No default extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Invoice ExtractInvoice(this OcrPage page, CancellationToken cancellationToken = default) => OneOcr.DefaultExtractor.ExtractInvoice(page, 0, cancellationToken);

    /// <summary>Reads a page as an invoice.</summary>
    /// <param name="page">The page.</param>
    /// <param name="extractor">The extractor to use.</param>
    /// <returns>The invoice; fields not found are null.</returns>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Invoice ExtractInvoice(this OcrPage page, DocumentExtractor extractor, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(extractor);
        return extractor.ExtractInvoice(page, 0, cancellationToken);
    }

    /// <summary>Reads a page as a receipt with <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>The receipt; fields not found are null.</returns>
    /// <exception cref="OcrModelException">No default extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Receipt ExtractReceipt(this OcrPage page, CancellationToken cancellationToken = default) => OneOcr.DefaultExtractor.ExtractReceipt(page, 0, cancellationToken);

    /// <summary>Reads a page as a receipt.</summary>
    /// <param name="page">The page.</param>
    /// <param name="extractor">The extractor to use.</param>
    /// <returns>The receipt; fields not found are null.</returns>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Receipt ExtractReceipt(this OcrPage page, DocumentExtractor extractor, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(extractor);
        return extractor.ExtractReceipt(page, 0, cancellationToken);
    }

    /// <summary>Reads a page with the schema the model predicts for it, using <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>Fields, line items, tax rows and key–value pairs.</returns>
    /// <exception cref="OcrModelException">No default extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static DocumentExtraction Extract(this OcrPage page, CancellationToken cancellationToken = default) => OneOcr.DefaultExtractor.Extract(page, 0, cancellationToken);

    /// <summary>Finds generic key–value pairs on a page, using <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>The pairs.</returns>
    /// <exception cref="OcrModelException">No default extractor is configured.</exception>
    public static IReadOnlyList<OcrKeyValue> FindKeyValues(this OcrPage page) => OneOcr.DefaultExtractor.FindKeyValues(page);

    /// <summary>A copy of the page whose <see cref="OcrPage.KeyValues"/> also hold the pairs found by <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>The new page.</returns>
    /// <exception cref="OcrModelException">No default extractor is configured.</exception>
    public static OcrPage WithKeyValues(this OcrPage page) => OneOcr.DefaultExtractor.WithKeyValues(page);

    /// <summary>Reads an image with <see cref="OneOcr.Default"/> and reads the page as an invoice with <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="image">The page image.</param>
    /// <returns>The invoice.</returns>
    /// <exception cref="OcrModelException">No default engine or extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Invoice ExtractInvoice(this Image image, CancellationToken cancellationToken = default) =>
        OneOcr.DefaultExtractor.ExtractInvoice(OneOcr.Default.Ocr(image, null, cancellationToken), 0, cancellationToken);

    /// <summary>Like <see cref="ExtractInvoice(Image, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The invoice.</returns>
    /// <exception cref="OcrModelException">No default engine or extractor is configured.</exception>
    public static Task<Invoice> ExtractInvoiceAsync(this Image image, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.ExtractInvoice(cancellationToken), cancellationToken);

    /// <summary>Reads an image with the given engine and reads the page as an invoice.</summary>
    /// <param name="image">The page image.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <returns>The invoice.</returns>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Invoice ExtractInvoice(this Image image, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return (extractor ?? OneOcr.DefaultExtractor).ExtractInvoice(engine.Ocr(image, null, cancellationToken), 0, cancellationToken);
    }

    /// <summary>Like <see cref="ExtractInvoice(Image, OcrEngine, DocumentExtractor?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The invoice.</returns>
    public static Task<Invoice> ExtractInvoiceAsync(this Image image, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.ExtractInvoice(engine, extractor, cancellationToken), cancellationToken);

    /// <summary>Reads an image with <see cref="OneOcr.Default"/> and reads the page as a receipt with <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="image">The page image.</param>
    /// <returns>The receipt.</returns>
    /// <exception cref="OcrModelException">No default engine or extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Receipt ExtractReceipt(this Image image, CancellationToken cancellationToken = default) =>
        OneOcr.DefaultExtractor.ExtractReceipt(OneOcr.Default.Ocr(image, null, cancellationToken), 0, cancellationToken);

    /// <summary>Like <see cref="ExtractReceipt(Image, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The receipt.</returns>
    /// <exception cref="OcrModelException">No default engine or extractor is configured.</exception>
    public static Task<Receipt> ExtractReceiptAsync(this Image image, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.ExtractReceipt(cancellationToken), cancellationToken);

    /// <summary>Reads an image with the given engine and reads the page as a receipt.</summary>
    /// <param name="image">The page image.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <returns>The receipt.</returns>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static Receipt ExtractReceipt(this Image image, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return (extractor ?? OneOcr.DefaultExtractor).ExtractReceipt(engine.Ocr(image, null, cancellationToken), 0, cancellationToken);
    }

    /// <summary>Like <see cref="ExtractReceipt(Image, OcrEngine, DocumentExtractor?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The receipt.</returns>
    public static Task<Receipt> ExtractReceiptAsync(this Image image, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.ExtractReceipt(engine, extractor, cancellationToken), cancellationToken);
    /// <summary>Reads a page with <see cref="OneOcr.DefaultExtractor"/> (a custom model) and returns its fields as <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">The caller's type.</typeparam>
    /// <param name="page">The page.</param>
    /// <param name="typeInfo">Source-generated type information for <typeparamref name="T"/>.</param>
    /// <returns>The value with per-field words, boxes and confidences; see <see cref="DocumentExtractor.Extract{T}"/>.</returns>
    /// <exception cref="OcrModelException">No default extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static ExtractionResult<T> Extract<T>(this OcrPage page, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, CancellationToken cancellationToken = default) =>
        OneOcr.DefaultExtractor.Extract(page, typeInfo, 0, cancellationToken);

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
        return extractor.Extract(page, typeInfo, 0, cancellationToken);
    }

    /// <summary>Reads an image with <see cref="OneOcr.Default"/>, then reads the page with <see cref="OneOcr.DefaultExtractor"/> (a custom model) into <typeparamref name="T"/>.</summary>
    /// <typeparam name="T">The caller's type.</typeparam>
    /// <param name="image">The page image.</param>
    /// <param name="typeInfo">Source-generated type information for <typeparamref name="T"/>.</param>
    /// <returns>The value with per-field words, boxes and confidences.</returns>
    /// <exception cref="OcrModelException">No default engine or extractor is configured.</exception>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static ExtractionResult<T> Extract<T>(this Image image, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, CancellationToken cancellationToken = default) =>
        OneOcr.DefaultExtractor.Extract(OneOcr.Default.Ocr(image, null, cancellationToken), typeInfo, 0, cancellationToken);

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
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <returns>The value with per-field words, boxes and confidences.</returns>
    /// <param name="cancellationToken">Cancels the work; the async twin then ends as cancelled.</param>
    public static ExtractionResult<T> Extract<T>(this Image image, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return (extractor ?? OneOcr.DefaultExtractor).Extract(engine.Ocr(image, null, cancellationToken), typeInfo, 0, cancellationToken);
    }

    /// <summary>Like <see cref="Extract{T}(Image, System.Text.Json.Serialization.Metadata.JsonTypeInfo{T}, OcrEngine, DocumentExtractor?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <typeparam name="T">The caller's type.</typeparam>
    /// <param name="image">The page image; keep it alive until the task completes.</param>
    /// <param name="typeInfo">Source-generated type information for <typeparamref name="T"/>.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The value with per-field words, boxes and confidences.</returns>
    public static Task<ExtractionResult<T>> ExtractAsync<T>(this Image image, System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, OcrEngine engine, DocumentExtractor? extractor = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => image.Extract(typeInfo, engine, extractor, cancellationToken), cancellationToken);
}
