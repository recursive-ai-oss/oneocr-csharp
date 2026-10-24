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
    public static Invoice ExtractInvoice(this OcrPage page) => OneOcr.DefaultExtractor.ExtractInvoice(page);

    /// <summary>Reads a page as an invoice.</summary>
    /// <param name="page">The page.</param>
    /// <param name="extractor">The extractor to use.</param>
    /// <returns>The invoice; fields not found are null.</returns>
    public static Invoice ExtractInvoice(this OcrPage page, DocumentExtractor extractor)
    {
        ArgumentNullException.ThrowIfNull(extractor);
        return extractor.ExtractInvoice(page);
    }

    /// <summary>Reads a page as a receipt with <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>The receipt; fields not found are null.</returns>
    /// <exception cref="OcrModelException">No default extractor is configured.</exception>
    public static Receipt ExtractReceipt(this OcrPage page) => OneOcr.DefaultExtractor.ExtractReceipt(page);

    /// <summary>Reads a page as a receipt.</summary>
    /// <param name="page">The page.</param>
    /// <param name="extractor">The extractor to use.</param>
    /// <returns>The receipt; fields not found are null.</returns>
    public static Receipt ExtractReceipt(this OcrPage page, DocumentExtractor extractor)
    {
        ArgumentNullException.ThrowIfNull(extractor);
        return extractor.ExtractReceipt(page);
    }

    /// <summary>Reads a page with the schema the model predicts for it, using <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="page">The page.</param>
    /// <returns>Fields, line items, tax rows and key–value pairs.</returns>
    /// <exception cref="OcrModelException">No default extractor is configured.</exception>
    public static DocumentExtraction Extract(this OcrPage page) => OneOcr.DefaultExtractor.Extract(page);

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
    public static Invoice ExtractInvoice(this Image image) => OneOcr.DefaultExtractor.ExtractInvoice(OneOcr.Default.Ocr(image));

    /// <summary>Reads an image with the given engine and reads the page as an invoice.</summary>
    /// <param name="image">The page image.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <returns>The invoice.</returns>
    public static Invoice ExtractInvoice(this Image image, OcrEngine engine, DocumentExtractor? extractor = null)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return (extractor ?? OneOcr.DefaultExtractor).ExtractInvoice(engine.Ocr(image));
    }

    /// <summary>Reads an image with <see cref="OneOcr.Default"/> and reads the page as a receipt with <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="image">The page image.</param>
    /// <returns>The receipt.</returns>
    /// <exception cref="OcrModelException">No default engine or extractor is configured.</exception>
    public static Receipt ExtractReceipt(this Image image) => OneOcr.DefaultExtractor.ExtractReceipt(OneOcr.Default.Ocr(image));

    /// <summary>Reads an image with the given engine and reads the page as a receipt.</summary>
    /// <param name="image">The page image.</param>
    /// <param name="engine">The engine that reads the image.</param>
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <returns>The receipt.</returns>
    public static Receipt ExtractReceipt(this Image image, OcrEngine engine, DocumentExtractor? extractor = null)
    {
        ArgumentNullException.ThrowIfNull(engine);
        return (extractor ?? OneOcr.DefaultExtractor).ExtractReceipt(engine.Ocr(image));
    }
}
