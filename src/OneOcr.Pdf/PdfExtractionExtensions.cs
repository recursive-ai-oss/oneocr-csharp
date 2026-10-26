using OneOcr.Documents;
using OnePdf;

namespace OneOcr.Pdf;

/// <summary>
/// Invoice and receipt extraction for PDF pages: the page is read into the page model first (text layer where there is one, local OCR
/// for scanned regions; see <see cref="PdfOcrExtensions"/>), then the extraction model labels its words. Everything runs locally.
/// </summary>
public static class PdfExtractionExtensions
{
    /// <summary>Reads a PDF page as an invoice with <see cref="OneOcr.Default"/> and <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="page">The page.</param>
    /// <param name="options">Per-call PDF options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The invoice; fields not found are null.</returns>
    /// <exception cref="OcrModelException">No default engine or extractor is configured.</exception>
    public static Invoice ExtractInvoice(this PdfPage page, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        OneOcr.DefaultExtractor.ExtractInvoice(page.Ocr(options, cancellationToken), 0, cancellationToken);

    /// <summary>Reads a PDF page as an invoice with the given engine and extractor.</summary>
    /// <param name="page">The page.</param>
    /// <param name="engine">The engine that reads scanned regions.</param>
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <param name="options">Per-call PDF options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The invoice; fields not found are null.</returns>
    public static Invoice ExtractInvoice(
        this PdfPage page, OcrEngine engine, DocumentExtractor? extractor = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        (extractor ?? OneOcr.DefaultExtractor).ExtractInvoice(page.Ocr(engine, options, cancellationToken), 0, cancellationToken);

    /// <summary>Reads a PDF page as a receipt with <see cref="OneOcr.Default"/> and <see cref="OneOcr.DefaultExtractor"/>.</summary>
    /// <param name="page">The page.</param>
    /// <param name="options">Per-call PDF options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The receipt; fields not found are null.</returns>
    /// <exception cref="OcrModelException">No default engine or extractor is configured.</exception>
    public static Receipt ExtractReceipt(this PdfPage page, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        OneOcr.DefaultExtractor.ExtractReceipt(page.Ocr(options, cancellationToken), 0, cancellationToken);

    /// <summary>Reads a PDF page as a receipt with the given engine and extractor.</summary>
    /// <param name="page">The page.</param>
    /// <param name="engine">The engine that reads scanned regions.</param>
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <param name="options">Per-call PDF options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The receipt; fields not found are null.</returns>
    public static Receipt ExtractReceipt(
        this PdfPage page, OcrEngine engine, DocumentExtractor? extractor = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        (extractor ?? OneOcr.DefaultExtractor).ExtractReceipt(page.Ocr(engine, options, cancellationToken), 0, cancellationToken);

    /// <summary>Like <see cref="ExtractInvoice(PdfPage, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="page">The page.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The invoice; fields not found are null.</returns>
    public static Task<Invoice> ExtractInvoiceAsync(this PdfPage page, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ExtractInvoice(page, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ExtractInvoice(PdfPage, OcrEngine, DocumentExtractor?, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="page">The page.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The invoice; fields not found are null.</returns>
    public static Task<Invoice> ExtractInvoiceAsync(
        this PdfPage page, OcrEngine engine, DocumentExtractor? extractor = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ExtractInvoice(page, engine, extractor, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ExtractReceipt(PdfPage, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="page">The page.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The receipt; fields not found are null.</returns>
    public static Task<Receipt> ExtractReceiptAsync(this PdfPage page, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ExtractReceipt(page, options, cancellationToken), cancellationToken);

    /// <summary>Like <see cref="ExtractReceipt(PdfPage, OcrEngine, DocumentExtractor?, PdfOcrOptions?, CancellationToken)"/>, on a thread-pool thread, with the same result.</summary>
    /// <param name="page">The page.</param>
    /// <param name="engine">The recognizer.</param>
    /// <param name="extractor">The extractor; <see cref="OneOcr.DefaultExtractor"/> when null.</param>
    /// <param name="options">Per-call options; null for the defaults.</param>
    /// <param name="cancellationToken">Cancels between pages, regions and lines; the task then ends as cancelled.</param>
    /// <returns>The receipt; fields not found are null.</returns>
    public static Task<Receipt> ExtractReceiptAsync(
        this PdfPage page, OcrEngine engine, DocumentExtractor? extractor = null, PdfOcrOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.Run(() => ExtractReceipt(page, engine, extractor, options, cancellationToken), cancellationToken);
}
