using OneImage;
using OneOcr.Kernels;
using OneOcr.Model;

namespace OneOcr.Documents;

/// <summary>
/// Reads fields and key–value pairs off a page with a local extraction model (<c>docextract-v0</c>). Nothing leaves the machine and no
/// network is involved. An extractor is immutable and thread-safe: share one for the life of the process.
/// </summary>
/// <remarks>
/// <para>The model labels the words of a page; deterministic code joins the labelled words into values. A value is therefore always the
/// text of words that are on the page, with their boxes, and the model cannot invent one.</para>
/// <para>Results are bit-identical on every CPU path and at every degree of parallelism.</para>
/// <para>A user-trained model with its own classes loads the same way; its fields appear in <see cref="DocumentExtraction.Fields"/> and
/// <see cref="DocumentExtraction.LineItems"/> under the class names it defines.</para>
/// </remarks>
public sealed class DocumentExtractor
{
    private readonly DocExtractModel _model;
    private readonly OcrLimits _limits;
    private readonly KernelPath _path;

    private DocumentExtractor(DocExtractModel model, ModelContainer container, OcrLimits limits, KernelPath path)
    {
        _model = model;
        _limits = limits;
        _path = path;
        Version = container.Version;
        Name = container.Name;
    }

    /// <summary>The model's own version string.</summary>
    public string Version { get; }

    /// <summary>The model's name.</summary>
    public string Name { get; }

    /// <summary>The field classes the model labels, in its order.</summary>
    public IReadOnlyList<string> Classes => _model.Schema.Classes;

    /// <summary>The limits the extractor enforces.</summary>
    public OcrLimits Limits => _limits;

    /// <summary>Loads a <c>docextract-v0</c> model file.</summary>
    /// <param name="path">The model container.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The extractor.</returns>
    /// <exception cref="OcrModelException">The file is malformed or is not a document-extraction model.</exception>
    /// <exception cref="OcrLimitException">The file exceeds <paramref name="limits"/>.</exception>
    public static DocumentExtractor Load(string path, OcrLimits? limits = null)
    {
        limits ??= OcrLimits.Default;
        return FromContainer(ModelContainer.Load(path, limits), limits);
    }

    /// <summary>Loads a <c>docextract-v0</c> model from a stream.</summary>
    /// <param name="model">The stream, positioned at the start of the container.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The extractor.</returns>
    /// <exception cref="OcrModelException">The data is malformed or is not a document-extraction model.</exception>
    /// <exception cref="OcrLimitException">The data exceeds <paramref name="limits"/>.</exception>
    public static DocumentExtractor Load(Stream model, OcrLimits? limits = null)
    {
        limits ??= OcrLimits.Default;
        return FromContainer(ModelContainer.Load(model, limits), limits);
    }

    /// <summary>Loads a <c>docextract-v0</c> model held in memory.</summary>
    /// <param name="model">The whole container; copied.</param>
    /// <param name="limits">Bounds to enforce; <see cref="OcrLimits.Default"/> when null.</param>
    /// <returns>The extractor.</returns>
    /// <exception cref="OcrModelException">The data is malformed or is not a document-extraction model.</exception>
    /// <exception cref="OcrLimitException">The data exceeds <paramref name="limits"/>.</exception>
    public static DocumentExtractor Load(ReadOnlySpan<byte> model, OcrLimits? limits = null)
    {
        limits ??= OcrLimits.Default;
        return FromContainer(ModelContainer.Parse(model, limits), limits);
    }

    private static DocumentExtractor FromContainer(ModelContainer container, OcrLimits limits) =>
        new(DocExtractModel.FromContainer(container, limits), container, limits, Dispatch.Best);

    /// <summary>The same extractor on a given kernel path; for tests that compare paths.</summary>
    internal DocumentExtractor WithPath(KernelPath path) => new(_model, _limits, path, Version, Name);

    private DocumentExtractor(DocExtractModel model, OcrLimits limits, KernelPath path, string version, string name)
    {
        _model = model;
        _limits = limits;
        _path = Dispatch.Require(path);
        Version = version;
        Name = name;
    }

    internal DocExtractModel Model => _model;

    /// <summary>Reads a page with the schema the model predicts for it.</summary>
    /// <param name="page">The page; its words, in reading order, are what the model reads.</param>
    /// <param name="maxDegreeOfParallelism">Threads to use; 0 for min(4, cores).</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>Fields, line items, tax rows and key–value pairs. A page without words gives an empty result of type <see cref="DocumentType.Other"/>.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="page"/> is null.</exception>
    /// <exception cref="OcrLimitException">The page has more words than <see cref="OcrLimits.MaxWordsPerPage"/>.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public DocumentExtraction Extract(OcrPage page, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default) =>
        Run(page, null, maxDegreeOfParallelism, cancellationToken);

    /// <summary>Reads a page as an invoice: only invoice fields are read, whatever type the model predicts (it is reported in <see cref="DocumentExtraction.DocumentType"/>).</summary>
    /// <param name="page">The page.</param>
    /// <param name="maxDegreeOfParallelism">Threads to use; 0 for min(4, cores).</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The invoice; fields not found are null.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="page"/> is null.</exception>
    /// <exception cref="OcrLimitException">The page has more words than <see cref="OcrLimits.MaxWordsPerPage"/>.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public Invoice ExtractInvoice(OcrPage page, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default) =>
        new(Run(page, "invoice", maxDegreeOfParallelism, cancellationToken));

    /// <summary>Reads a page as a receipt: only receipt fields are read, whatever type the model predicts.</summary>
    /// <param name="page">The page.</param>
    /// <param name="maxDegreeOfParallelism">Threads to use; 0 for min(4, cores).</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The receipt; fields not found are null.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="page"/> is null.</exception>
    /// <exception cref="OcrLimitException">The page has more words than <see cref="OcrLimits.MaxWordsPerPage"/>.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public Receipt ExtractReceipt(OcrPage page, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default) =>
        new(Run(page, "receipt", maxDegreeOfParallelism, cancellationToken));

    /// <summary>Finds generic key–value pairs on any page.</summary>
    /// <param name="page">The page.</param>
    /// <param name="maxDegreeOfParallelism">Threads to use; 0 for min(4, cores).</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The pairs, in the order of their values on the page.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="page"/> is null.</exception>
    /// <exception cref="OcrLimitException">The page has more words than <see cref="OcrLimits.MaxWordsPerPage"/>.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public IReadOnlyList<OcrKeyValue> FindKeyValues(OcrPage page, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default) =>
        Run(page, null, maxDegreeOfParallelism, cancellationToken).KeyValues;

    /// <summary>A copy of the page whose <see cref="OcrPage.KeyValues"/> are the pairs found by <see cref="FindKeyValues"/>; pairs already on the page are kept first.</summary>
    /// <param name="page">The page.</param>
    /// <param name="maxDegreeOfParallelism">Threads to use; 0 for min(4, cores).</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The new page; the original is unchanged.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="page"/> is null.</exception>
    /// <exception cref="OcrLimitException">The page has more words than <see cref="OcrLimits.MaxWordsPerPage"/>.</exception>
    /// <exception cref="OperationCanceledException">Cancelled.</exception>
    public OcrPage WithKeyValues(OcrPage page, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default)
    {
        IReadOnlyList<OcrKeyValue> found = FindKeyValues(page, maxDegreeOfParallelism, cancellationToken);
        return new OcrPage(
            page.Index, page.Width, page.Height, page.UnitsPerInch, page.Blocks, page.Regions, page.Tables, [.. page.KeyValues, .. found],
            page.Diagnostics, page.ToPdf, page.Orientation, page.SkewDegrees);
    }

    /// <summary>Asynchronous <see cref="ExtractInvoice"/>: runs the same local work on the thread pool.</summary>
    /// <param name="page">The page.</param>
    /// <param name="maxDegreeOfParallelism">Threads to use; 0 for min(4, cores).</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The invoice.</returns>
    public Task<Invoice> ExtractInvoiceAsync(OcrPage page, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default) =>
        Task.Run(() => ExtractInvoice(page, maxDegreeOfParallelism, cancellationToken), cancellationToken);

    /// <summary>Asynchronous <see cref="ExtractReceipt"/>: runs the same local work on the thread pool.</summary>
    /// <param name="page">The page.</param>
    /// <param name="maxDegreeOfParallelism">Threads to use; 0 for min(4, cores).</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>The receipt.</returns>
    public Task<Receipt> ExtractReceiptAsync(OcrPage page, int maxDegreeOfParallelism = 0, CancellationToken cancellationToken = default) =>
        Task.Run(() => ExtractReceipt(page, maxDegreeOfParallelism, cancellationToken), cancellationToken);

    internal DocumentExtraction Run(OcrPage page, string? forcedType, int maxDegreeOfParallelism, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(page);
        cancellationToken.ThrowIfCancellationRequested();
        IReadOnlyList<OcrWord> words = page.Words;
        if (words.Count > _limits.MaxWordsPerPage)
        {
            throw new OcrLimitException(nameof(OcrLimits.MaxWordsPerPage), _limits.MaxWordsPerPage, words.Count);
        }

        var texts = new List<string>(words.Count);
        var boxes = new List<OcrBox>(words.Count);
        var index = new List<int>(words.Count);
        var sources = new List<OcrSource>(words.Count);
        for (int i = 0; i < words.Count; i++)
        {
            if (words[i].Text.Length == 0)
            {
                continue;
            }

            texts.Add(words[i].Text);
            boxes.Add(words[i].Box);
            index.Add(i);
            sources.Add(words[i].Source);
        }

        if (texts.Count == 0)
        {
            return new DocumentExtraction(DocumentType.Other, 0f, [], [], [], [], ["The page has no words."]);
        }

        int degree = maxDegreeOfParallelism > 0 ? maxDegreeOfParallelism : Math.Min(4, Environment.ProcessorCount);
        DocFeatures features = DocFeatures.Build(_model, texts, boxes, page.Width, page.Height);
        DocOutputs outputs = DocForward.Run(_model, features, _path, degree, cancellationToken);
        NumberStyle style = ValueNormalize.DetectStyle(texts);
        if (style == NumberStyle.Auto)
        {
            style = NumberStyle.PointDecimal;
        }

        DateOrder order = ValueNormalize.DetectOrder(texts);
        return DocAssembly.Assemble(_model, outputs, texts, boxes, index, sources, forcedType, style, order);
    }
}
