using OneImage;

namespace OneOcr.Llm;

/// <summary>
/// How a page image is shrunk and encoded before it is uploaded. Every provider has its own defaults, taken from that
/// provider's public image guidance; set any property to override them. Images are never enlarged.
/// </summary>
public sealed record LlmImageOptions
{
    /// <summary>Longest side in pixels after shrinking. Default depends on the provider (see each provider class).</summary>
    public int? MaxLongEdge { get; init; }

    /// <summary>Longest allowed length of the shorter side in pixels, or null for no bound.</summary>
    public int? MaxShortEdge { get; init; }

    /// <summary>Most pixels after shrinking, or null for no bound.</summary>
    public long? MaxPixels { get; init; }

    /// <summary>Largest encoded image in bytes. A lossless PNG over this size is re-encoded as JPEG at falling quality; if even that does not fit, the call throws <see cref="OcrLimitException"/>.</summary>
    public long? MaxBytes { get; init; }

    /// <summary>JPEG quality (1 to 100) for the first lossy attempt. Default 90.</summary>
    public int JpegQuality { get; init; } = 90;
}

/// <summary>
/// Chooses regions of a page that must not be uploaded. Called once per page before anything is built; return the boxes
/// (page units, top-left origin) to mask. The pixels under each box are painted solid black in the uploaded image, and every
/// recognized word whose box touches one is left out of the word list the model sees.
/// </summary>
/// <param name="page">The page, so the callback can inspect words, lines and regions.</param>
/// <param name="pageNumber">Zero-based position of the page in this call.</param>
/// <returns>Boxes to mask; null or empty for none.</returns>
public delegate IReadOnlyList<OcrBox>? LlmRedactor(OcrPage page, int pageNumber);

/// <summary>One page of a call: its image and the page model read from it.</summary>
/// <param name="Image">The page image. It is shrunk and encoded for upload; the caller's image is not changed.</param>
/// <param name="Page">The page model. Its coordinates are mapped onto the image by the ratio of their sizes.</param>
public readonly record struct LlmPageInput(Image Image, OcrPage Page);

/// <summary>A word that was listed for the model, with the id it was listed under.</summary>
/// <param name="Id">The id the model sees and cites, <c>w0</c>, <c>w1</c>, … counted across all pages of the call.</param>
/// <param name="PageIndex">Zero-based index of the page within the call.</param>
/// <param name="WordIndex">Index of the word in that page's <see cref="OcrPage.Words"/>.</param>
/// <param name="Text">The word's text.</param>
/// <param name="Box">The word's box in page units.</param>
public sealed record LlmWordRef(string Id, int PageIndex, int WordIndex, string Text, OcrBox Box);

/// <summary>Token counts a provider reported.</summary>
/// <param name="InputTokens">Tokens read. When the provider gave no count this is the local estimate.</param>
/// <param name="OutputTokens">Tokens written, including any reasoning tokens the provider bills as output.</param>
/// <param name="Reported">True when the numbers came from the provider, false when they are estimates.</param>
public readonly record struct LlmUsage(long InputTokens, long OutputTokens, bool Reported);

/// <summary>What a call would cost at most, worked out before anything is sent.</summary>
/// <param name="InputTokens">Estimated input tokens: text at three bytes per token, images by the provider's documented formula.</param>
/// <param name="MaxOutputTokens">The output-token ceiling the call will request.</param>
/// <param name="WorstCaseCostUsd">Input estimate plus the whole output ceiling at the table price; null when the model has no price.</param>
/// <param name="Warnings">For example a price table older than 180 days.</param>
public sealed record LlmEstimate(long InputTokens, int MaxOutputTokens, decimal? WorstCaseCostUsd, IReadOnlyList<string> Warnings);

/// <summary>The model's answer as JSON, with what it cost and what was shown to it.</summary>
public sealed class LlmExtraction
{
    internal LlmExtraction(
        string json, LlmUsage usage, decimal? costUsd, bool fromCache, string provider, string model,
        IReadOnlyList<LlmWordRef> words, int maskedWords, IReadOnlyList<string> warnings)
    {
        Json = json;
        Usage = usage;
        CostUsd = costUsd;
        FromCache = fromCache;
        Provider = provider;
        Model = model;
        Words = words;
        MaskedWords = maskedWords;
        Warnings = warnings;
    }

    /// <summary>The model's answer: one JSON object that is valid JSON and nested no deeper than <see cref="OcrLimits.MaxLlmJsonDepth"/>. It was asked to follow the schema; it is not validated against it.</summary>
    public string Json { get; }

    /// <summary>Token counts for this call; zero tokens when the answer came from the cache.</summary>
    public LlmUsage Usage { get; }

    /// <summary>What the call cost in USD at the table price; 0 for a cached answer; null when the model has no price.</summary>
    public decimal? CostUsd { get; }

    /// <summary>True when the answer was served from the client's <see cref="LlmResultCache"/> and nothing was sent.</summary>
    public bool FromCache { get; }

    /// <summary>Provider name, for example <c>anthropic</c>.</summary>
    public string Provider { get; }

    /// <summary>The model string the caller named.</summary>
    public string Model { get; }

    /// <summary>The words listed for the model, in the order and under the ids it saw. Words removed by redaction are not here.</summary>
    public IReadOnlyList<LlmWordRef> Words { get; }

    /// <summary>How many words were left out of the list because a redaction box touched them.</summary>
    public int MaskedWords { get; }

    /// <summary>Notes about the call, such as a stale price table.</summary>
    public IReadOnlyList<string> Warnings { get; }
}
