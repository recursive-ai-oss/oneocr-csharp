namespace OneOcr.Llm;

/// <summary>
/// Policy for one <see cref="LlmClient"/>: privacy switches, cost bounds, retries and limits. Immutable; build with object
/// initializers. Nothing here is read from the environment.
/// </summary>
public sealed class LlmOptions
{
    /// <summary>When true every call throws <see cref="OcrProviderException"/> with <see cref="LlmFailure.Offline"/> before anything is built or sent, even if the answer is cached. Default false.</summary>
    public bool Offline { get; init; }

    /// <summary>Bounds for provider responses, schemas and requests. Default <see cref="OcrLimits.Default"/>.</summary>
    public OcrLimits Limits { get; init; } = OcrLimits.Default;

    /// <summary>Refuse a call whose worst-case cost (estimated input plus the whole output ceiling) is above this many USD. Default null: no bound. A bound with a model that has no price throws <see cref="OcrBudgetException"/>.</summary>
    public decimal? MaxCostPerCallUsd { get; init; }

    /// <summary>Refuse a call when the client's spend so far, the worst case of calls in flight and this call's worst case would together pass this many USD. Default null: no bound.</summary>
    public decimal? SpendCapUsd { get; init; }

    /// <summary>Prices for estimates and caps. Default <see cref="LlmPriceTable.Default"/>.</summary>
    public LlmPriceTable PriceTable { get; init; } = LlmPriceTable.Default;

    /// <summary>The clock used for the stale-price warning and for waiting between retries. Default <see cref="TimeProvider.System"/>.</summary>
    public TimeProvider TimeProvider { get; init; } = TimeProvider.System;

    /// <summary>Most calls in flight at once through one client. Default 4.</summary>
    public int MaxConcurrency { get; init; } = 4;

    /// <summary>Most extra attempts after a transient failure (connection error, time-out, 408, 429, 500, 502, 503, 504, 529). Default 2.</summary>
    public int MaxRetries { get; init; } = 2;

    /// <summary>Wait before the first retry; it doubles each time, with up to 25% added at random, and is replaced by the provider's <c>Retry-After</c> when that is longer. Default 500 ms.</summary>
    public TimeSpan RetryBaseDelay { get; init; } = TimeSpan.FromMilliseconds(500);

    /// <summary>Longest wait before a retry, whatever the provider asks for. Default 30 s.</summary>
    public TimeSpan MaxRetryDelay { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>Time allowed for one attempt, headers and body. Default 120 s.</summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(120);

    /// <summary>Output-token ceiling requested from the model, and the output side of the worst-case cost. Default 4096.</summary>
    public int MaxOutputTokens { get; init; } = 4096;

    /// <summary>Answers kept for identical requests (same images, prompt, model, provider and schema). Default null: no cache.</summary>
    public LlmResultCache? Cache { get; init; }

    /// <summary>Masks regions before upload. Default null: nothing is masked.</summary>
    public LlmRedactor? Redactor { get; init; }
}
