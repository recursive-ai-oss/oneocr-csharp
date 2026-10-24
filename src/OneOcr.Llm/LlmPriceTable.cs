namespace OneOcr.Llm;

/// <summary>Price of one model, in USD per million tokens.</summary>
/// <param name="Provider">Provider name as in <see cref="LlmProvider.PriceKey"/>: <c>anthropic</c>, <c>openai</c>, <c>gemini</c>, <c>azure-openai</c> or <c>openai-compatible</c>.</param>
/// <param name="ModelPrefix">The model string, or the start of it: the longest matching prefix wins, compared ignoring case.</param>
/// <param name="InputPerMillion">USD per million input tokens (images count as input).</param>
/// <param name="OutputPerMillion">USD per million output tokens.</param>
public sealed record LlmModelPrice(string Provider, string ModelPrefix, decimal InputPerMillion, decimal OutputPerMillion)
{
    /// <summary>The cost of a call with these token counts.</summary>
    /// <param name="inputTokens">Input tokens.</param>
    /// <param name="outputTokens">Output tokens.</param>
    /// <returns>USD.</returns>
    public decimal Cost(long inputTokens, long outputTokens) =>
        ((inputTokens * InputPerMillion) + (outputTokens * OutputPerMillion)) / 1_000_000m;
}

/// <summary>
/// Prices used for estimates and spend caps. The shipped <see cref="Default"/> is a snapshot of the providers' public list
/// prices on <see cref="AsOf"/>; prices change, so a table older than <see cref="StaleAfterDays"/> days makes every call carry a
/// warning. Replace or extend it with <see cref="With"/> or the constructor. Cached-token discounts, batch discounts and
/// long-prompt surcharges are not modelled: the estimate is the plain list price.
/// </summary>
public sealed class LlmPriceTable
{
    /// <summary>Days after <see cref="AsOf"/> when the table counts as stale: 180.</summary>
    public const int StaleAfterDays = 180;

    private readonly LlmModelPrice[] _prices;

    /// <summary>Creates a table.</summary>
    /// <param name="asOf">The date the prices were read from the providers' pages.</param>
    /// <param name="prices">The prices; earlier entries win over later ones with an equally long prefix.</param>
    public LlmPriceTable(DateOnly asOf, IEnumerable<LlmModelPrice> prices)
    {
        ArgumentNullException.ThrowIfNull(prices);
        AsOf = asOf;
        _prices = [.. prices];
    }

    /// <summary>
    /// The shipped snapshot, read on 2026-10-23 from the public list prices at
    /// https://docs.anthropic.com/en/docs/about-claude/pricing (Anthropic, standard tier),
    /// https://platform.openai.com/docs/pricing (OpenAI, standard tier) and
    /// https://ai.google.dev/gemini-api/docs/pricing (Gemini API, paid standard tier; Gemini 3.x prices are the
    /// introductory ones listed through 2026-12-31, and the 3.1 Pro entry is the price for prompts up to 200k tokens).
    /// Azure OpenAI is billed per deployment and region through Azure, so it has no shipped prices, and neither do
    /// OpenAI-compatible servers: add entries (a local server costs 0) before using a spend cap with them.
    /// </summary>
    public static LlmPriceTable Default { get; } = new(
        new DateOnly(2026, 10, 23),
        [
            new("anthropic", "claude-opus-5-5", 4m, 20m),
            new("anthropic", "claude-opus-5", 5m, 25m),
            new("anthropic", "claude-opus-4", 5m, 25m),
            new("anthropic", "claude-sonnet-5-5", 2m, 10m),
            new("anthropic", "claude-sonnet-5", 2m, 10m),
            new("anthropic", "claude-sonnet-4", 3m, 15m),
            new("anthropic", "claude-haiku-5-5", 0.10m, 0.50m),
            new("anthropic", "claude-haiku-4-5", 1m, 5m),
            new("openai", "gpt-6-astra", 10m, 50m),
            new("openai", "gpt-6.1-sol", 2m, 10m),
            new("openai", "gpt-6-luna", 0.10m, 0.50m),
            new("gemini", "gemini-3.8-flash", 0.75m, 3.75m),
            new("gemini", "gemini-3.5-flash-lite", 0.30m, 2.50m),
            new("gemini", "gemini-3.1-pro-preview", 2m, 12m),
            new("gemini", "gemini-2.5-pro", 1.25m, 10m),
            new("gemini", "gemini-2.5-flash-lite", 0.10m, 0.40m),
            new("gemini", "gemini-2.5-flash", 0.30m, 2.50m),
        ]);

    /// <summary>The date the prices were read.</summary>
    public DateOnly AsOf { get; }

    /// <summary>All entries, in lookup priority order.</summary>
    public IReadOnlyList<LlmModelPrice> Prices => _prices;

    /// <summary>Whether the table is older than <see cref="StaleAfterDays"/> days on the given day.</summary>
    /// <param name="today">The day to compare with.</param>
    /// <returns>True when stale.</returns>
    public bool IsStale(DateOnly today) => today.DayNumber - AsOf.DayNumber > StaleAfterDays;

    /// <summary>A copy of the table with extra entries that win over existing ones. <see cref="AsOf"/> is unchanged.</summary>
    /// <param name="overrides">The entries to add, for example <c>new LlmModelPrice("openai-compatible", "my-local-model", 0, 0)</c>.</param>
    /// <returns>The new table.</returns>
    public LlmPriceTable With(params LlmModelPrice[] overrides)
    {
        ArgumentNullException.ThrowIfNull(overrides);
        return new LlmPriceTable(AsOf, overrides.Concat(_prices));
    }

    /// <summary>Finds the price for a model: the longest <see cref="LlmModelPrice.ModelPrefix"/> that starts the model string, for the provider.</summary>
    /// <param name="provider">Provider key.</param>
    /// <param name="model">Model string as the caller named it.</param>
    /// <param name="price">The entry found.</param>
    /// <returns>True when there is one.</returns>
    public bool TryFind(string provider, string model, out LlmModelPrice price)
    {
        LlmModelPrice? best = null;
        foreach (LlmModelPrice p in _prices)
        {
            if (string.Equals(p.Provider, provider, StringComparison.OrdinalIgnoreCase)
                && model.StartsWith(p.ModelPrefix, StringComparison.OrdinalIgnoreCase)
                && (best is null || p.ModelPrefix.Length > best.ModelPrefix.Length))
            {
                best = p;
            }
        }

        price = best!;
        return best is not null;
    }
}
