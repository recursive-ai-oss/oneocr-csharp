using System.Text.Json;

namespace OneOcr.Llm;

/// <summary>The outputs a read can produce; combine with <c>|</c>. One model call produces every requested output.</summary>
[Flags]
public enum LlmOutputs
{
    /// <summary>No output; not allowed.</summary>
    None = 0,

    /// <summary>Exact text: the model's transcription merged word by word with the local reader's (<see cref="LlmReadResult.Text"/>).</summary>
    Text = 1,

    /// <summary>Markdown from the model, with the spans where it disagrees with the local reader flagged (<see cref="LlmReadResult.Markdown"/>).</summary>
    Markdown = 2,

    /// <summary>Structured values for a JSON Schema, each grounded against the page (<see cref="LlmReadResult.Json"/>, <see cref="LlmReadResult.Fields"/>).</summary>
    Json = 4,

    /// <summary>The merged page model: the local pages with the model's corrections applied, every word with its <see cref="OcrWord.Source"/> (<see cref="LlmReadResult.Pages"/>).</summary>
    PageModel = 8,
}

/// <summary>Where the work happens.</summary>
public enum LlmRouting
{
    /// <summary>The model is called for every requested output. The default.</summary>
    LlmOnly = 0,

    /// <summary>Local results are kept where they are confident; the model is called only for what is not. See <see cref="LlmReadOptions.ConfidenceThreshold"/>.</summary>
    LocalFirst = 1,

    /// <summary>The model is never called. Only outputs the local side can produce are returned.</summary>
    LocalOnly = 2,
}

/// <summary>What happens to a value that stays ungrounded after the one re-ask.</summary>
public enum UngroundedPolicy
{
    /// <summary>Keep the value in <see cref="LlmReadResult.Json"/> and mark it <see cref="GroundingState.Ungrounded"/> in <see cref="LlmReadResult.Fields"/>. The default.</summary>
    Flag = 0,

    /// <summary>Replace the value by <c>null</c> in <see cref="LlmReadResult.Json"/>; the field is still listed as ungrounded.</summary>
    Null = 1,

    /// <summary>Throw <see cref="OcrProviderException"/> with <see cref="LlmFailure.Ungrounded"/>.</summary>
    Throw = 2,
}

/// <summary>How well a value the model returned is backed by the page.</summary>
public enum GroundingState
{
    /// <summary>The value is the text of words on the page, up to spacing. The field has boxes.</summary>
    Grounded = 0,

    /// <summary>The value equals words on the page after normalization (case, date or amount format, currency symbol, punctuation at the ends). The field has boxes and <see cref="GroundedField.NormalizedValue"/>.</summary>
    GroundedNormalized = 1,

    /// <summary>The model says the value is computed or inferred rather than printed. It never has boxes; <see cref="GroundedField.Confidence"/> is high only when a sum, difference or product of the cited amounts reproduces it.</summary>
    Derived = 2,

    /// <summary>Nothing on the page supports the value: a possible hallucination.</summary>
    Ungrounded = 3,
}

/// <summary>A box on one page of a call.</summary>
/// <param name="PageIndex">Zero-based page index within the call.</param>
/// <param name="Box">The box in page units.</param>
public readonly record struct GroundedBox(int PageIndex, OcrBox Box);

/// <summary>One value the model returned, with how the page backs it.</summary>
/// <param name="Path">JSON Pointer to the value in <see cref="LlmReadResult.Json"/> (<c>/items/0/amount</c>).</param>
/// <param name="Value">The value as text: the string itself, or the JSON text of a number or boolean.</param>
/// <param name="State">The grounding state.</param>
/// <param name="Confidence">0 to 1: the state's base weight, lowered for a corrected citation, several candidate places, an inferred year or an ambiguous day/month, times the mean calibrated confidence of the words that back it. 0 when ungrounded.</param>
/// <param name="NormalizedValue">The canonical form for <see cref="GroundingState.GroundedNormalized"/>: <c>2026-10-03</c> for a date, <c>12.50</c> for an amount, the ISO code for a currency; otherwise null.</param>
/// <param name="Words">The page words that back the value; empty when derived or ungrounded.</param>
/// <param name="Boxes">One box per line of backing words; empty when derived or ungrounded.</param>
/// <param name="CitationCorrected">True when the model's cited words did not hold the value and the page was searched instead (or no words were cited).</param>
/// <param name="Reasked">True when this value comes from the re-ask.</param>
/// <param name="Note">A short explanation for anything but a clean grounding; null otherwise.</param>
public sealed record GroundedField(
    string Path, string Value, GroundingState State, float Confidence, string? NormalizedValue, IReadOnlyList<LlmWordRef> Words,
    IReadOnlyList<GroundedBox> Boxes, bool CitationCorrected = false, bool Reasked = false, string? Note = null);

/// <summary>How a word of the model's text and a word of the local reading differ.</summary>
public enum DisagreementKind
{
    /// <summary>Both have a word at this place and the texts differ.</summary>
    Substitution = 0,

    /// <summary>Only the model has this word; the local reader found nothing.</summary>
    LlmOnly = 1,

    /// <summary>Only the local reader has this word; the model's text lacks it.</summary>
    OcrOnly = 2,
}

/// <summary>One place where the model's text and the local reading differ.</summary>
/// <param name="PageIndex">Zero-based page index in the caller's list of pages.</param>
/// <param name="Kind">What differs.</param>
/// <param name="OcrText">The local word, or null for <see cref="DisagreementKind.LlmOnly"/>.</param>
/// <param name="LlmText">The model's word, or null for <see cref="DisagreementKind.OcrOnly"/>.</param>
/// <param name="OcrConfidence">The local word's calibrated confidence; 0 for <see cref="DisagreementKind.LlmOnly"/>.</param>
/// <param name="Chosen">For <see cref="LlmReadResult.Disagreements"/>: whose word is in the merged text, or null when the local word was dropped. For markdown flags: null.</param>
/// <param name="Box">The local word's box; null for <see cref="DisagreementKind.LlmOnly"/>.</param>
/// <param name="Start">Offset in <see cref="LlmReadResult.Text"/> (or <see cref="LlmReadResult.Markdown"/> for a markdown flag) where the span starts; the position of the gap for <see cref="DisagreementKind.OcrOnly"/> dropped words.</param>
/// <param name="Length">Length of the span in that string; 0 for a dropped word.</param>
public sealed record TextDisagreement(
    int PageIndex, DisagreementKind Kind, string? OcrText, string? LlmText, float OcrConfidence, OcrSource? Chosen, OcrBox? Box, int Start, int Length);

/// <summary>Fields found by a local model; the seam for local-first extraction.</summary>
/// <param name="Json">The values as one JSON object that fits the schema.</param>
/// <param name="Fields">Each value with its grounding.</param>
/// <param name="MinConfidence">The lowest confidence among the fields; the model is skipped under <see cref="LlmRouting.LocalFirst"/> when this reaches <see cref="LlmReadOptions.ConfidenceThreshold"/>.</param>
public sealed record LocalFieldResult(string Json, IReadOnlyList<GroundedField> Fields, float MinConfidence);

/// <summary>
/// A local extraction model. None ships yet; the interface is where one plugs in, so that <see cref="LlmRouting.LocalFirst"/> and
/// <see cref="LlmRouting.LocalOnly"/> can answer <see cref="LlmOutputs.Json"/> reads without the network.
/// </summary>
public interface ILocalFieldExtractor
{
    /// <summary>Extracts the schema's values from locally read pages.</summary>
    /// <param name="pages">The pages.</param>
    /// <param name="jsonSchema">The JSON Schema asked for.</param>
    /// <returns>The fields, or null when the extractor has nothing for this schema.</returns>
    LocalFieldResult? TryExtract(IReadOnlyList<OcrPage> pages, string jsonSchema);
}

/// <summary>What to read and how, for <see cref="LlmClient.ReadAsync(IReadOnlyList{LlmPageInput}, LlmReadOptions, CancellationToken)"/>. Immutable; build with object initializers.</summary>
public sealed class LlmReadOptions
{
    /// <summary>The outputs wanted. Default <see cref="LlmOutputs.Text"/>.</summary>
    public LlmOutputs Outputs { get; init; } = LlmOutputs.Text;

    /// <summary>The JSON Schema of the values, an object schema; required when <see cref="Outputs"/> has <see cref="LlmOutputs.Json"/>. Supported: objects with <c>properties</c>, arrays with <c>items</c>, scalar leaves, <c>$ref</c> to <c>$defs</c> (inlined, at most 8 levels), a nullable <c>anyOf</c> of one type and <c>null</c>. A leaf may be a string, number, integer or boolean.</summary>
    public string? JsonSchema { get; init; }

    /// <summary>Name for the schema sent to the provider. Default <c>extraction</c>.</summary>
    public string SchemaName { get; init; } = "extraction";

    /// <summary>Trusted extra instruction, for example what the document is. Default null.</summary>
    public string? Instruction { get; init; }

    /// <summary>Where the work happens. Default <see cref="LlmRouting.LlmOnly"/>.</summary>
    public LlmRouting Routing { get; init; } = LlmRouting.LlmOnly;

    /// <summary>
    /// Calibrated confidence from which a local result is kept. For text, a local word at or above it wins a disagreement and under
    /// <see cref="LlmRouting.LocalFirst"/> a page whose words are all at or above it is not sent. For fields, it is compared with
    /// <see cref="LocalFieldResult.MinConfidence"/>. Default 0.9.
    /// </summary>
    public float ConfidenceThreshold { get; init; } = 0.9f;

    /// <summary>A local extraction model for <see cref="LlmOutputs.Json"/>; default null (there is none yet).</summary>
    public ILocalFieldExtractor? LocalExtractor { get; init; }

    /// <summary>What happens to values that stay ungrounded after the re-ask. Default <see cref="UngroundedPolicy.Flag"/>.</summary>
    public UngroundedPolicy UngroundedPolicy { get; init; } = UngroundedPolicy.Flag;

    /// <summary>Re-ask once, in a second call, about ungrounded values. Default true. The second call counts against the cost bounds of the client like any call.</summary>
    public bool Reask { get; init; } = true;

    /// <summary>How numbers with one separator are read. Default <see cref="LlmNumberStyle.Auto"/>.</summary>
    public LlmNumberStyle NumberStyle { get; init; } = LlmNumberStyle.Auto;

    /// <summary>How numeric dates with day and month both 12 or less are read. Default <see cref="LlmDateOrder.Auto"/>.</summary>
    public LlmDateOrder DateOrder { get; init; } = LlmDateOrder.Auto;

    /// <summary>Whether words only the model has (the local reader found nothing) are kept in the merged text and page model. Default true; such words have a zero-width box at their neighbour's edge.</summary>
    public bool KeepLlmOnlyWords { get; init; } = true;
}

/// <summary>The outputs of one read, what it cost and what was shown.</summary>
public sealed class LlmReadResult
{
    internal LlmReadResult() { }

    /// <summary>The outputs that were requested.</summary>
    public LlmOutputs Outputs { get; internal init; }

    /// <summary>The merged text of all pages, pages set apart by a blank line; null unless <see cref="LlmOutputs.Text"/> was requested. When the model was not called it is the local text.</summary>
    public string? Text { get; internal init; }

    /// <summary>The model's own text, pages set apart by a blank line; null when the model was not asked for text.</summary>
    public string? LlmText { get; internal init; }

    /// <summary>The model's Markdown, pages set apart by a blank line; null unless <see cref="LlmOutputs.Markdown"/> was requested and the model was called.</summary>
    public string? Markdown { get; internal init; }

    /// <summary>The spans of <see cref="Markdown"/> where the model's words differ from the local reading, with offsets into it.</summary>
    public IReadOnlyList<TextDisagreement> MarkdownFlags { get; internal init; } = [];

    /// <summary>The values as one JSON object in the shape of the schema, with <see cref="UngroundedPolicy"/> applied; null unless <see cref="LlmOutputs.Json"/> was requested.</summary>
    public string? Json { get; internal init; }

    /// <summary>Every non-null value with its grounding.</summary>
    public IReadOnlyList<GroundedField> Fields { get; internal init; } = [];

    /// <summary>The pages: merged with the model's text when <see cref="LlmOutputs.PageModel"/> was requested and the model was called, otherwise the pages as given.</summary>
    public IReadOnlyList<OcrPage> Pages { get; internal init; } = [];

    /// <summary>Where the merged text and the local text differ, with the choice made for each.</summary>
    public IReadOnlyList<TextDisagreement> Disagreements { get; internal init; } = [];

    /// <summary>How many model calls were made: 0 when routing kept everything local, 1, or 2 with a re-ask.</summary>
    public int Calls { get; internal init; }

    /// <summary>Tokens used by all calls of this read.</summary>
    public LlmUsage Usage { get; internal init; }

    /// <summary>What all calls cost in USD; 0 when none was made; null when the model has no price.</summary>
    public decimal? CostUsd { get; internal init; }

    /// <summary>Provider name.</summary>
    public string Provider { get; internal init; } = "";

    /// <summary>Model string.</summary>
    public string Model { get; internal init; } = "";

    /// <summary>Notes: pages kept local, a skipped re-ask and why, answers that did not fit the requested shape.</summary>
    public IReadOnlyList<string> Diagnostics { get; internal init; } = [];

    /// <summary>Reads <see cref="Json"/> into a caller's type with source-generated type information.</summary>
    /// <typeparam name="T">The type.</typeparam>
    /// <param name="typeInfo">Type information from a <c>JsonSerializerContext</c>.</param>
    /// <returns>The value, or null when <see cref="Json"/> is null or the JSON is null.</returns>
    /// <exception cref="JsonException">The JSON does not fit the type.</exception>
    public T? GetValue<T>(System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo)
    {
        ArgumentNullException.ThrowIfNull(typeInfo);
        return Json is null ? default : JsonSerializer.Deserialize(Json, typeInfo);
    }

    /// <summary>The Markdown with each flagged span wrapped in <paramref name="open"/> and <paramref name="close"/>.</summary>
    /// <param name="open">Text put before each flagged span. Default <c>⟦</c>.</param>
    /// <param name="close">Text put after each flagged span. Default <c>⟧</c>.</param>
    /// <returns>The flagged Markdown, or null when there is no Markdown.</returns>
    public string? MarkdownWithFlags(string open = "\u27E6", string close = "\u27E7")
    {
        if (Markdown is null)
        {
            return null;
        }

        var sb = new System.Text.StringBuilder(Markdown.Length + (MarkdownFlags.Count * (open.Length + close.Length)));
        int pos = 0;
        foreach (TextDisagreement f in MarkdownFlags.Where(static f => f.Length > 0).OrderBy(static f => f.Start))
        {
            if (f.Start < pos || f.Start + f.Length > Markdown.Length)
            {
                continue;
            }

            sb.Append(Markdown, pos, f.Start - pos).Append(open).Append(Markdown, f.Start, f.Length).Append(close);
            pos = f.Start + f.Length;
        }

        sb.Append(Markdown, pos, Markdown.Length - pos);
        return sb.ToString();
    }
}
