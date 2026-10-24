namespace OneOcr;

/// <summary>Why a language-model call failed.</summary>
public enum LlmFailure
{
    /// <summary>The client is switched to offline; nothing was sent.</summary>
    Offline = 0,

    /// <summary>The request never got an answer: connection error or time-out, after the configured retries.</summary>
    Transport = 1,

    /// <summary>The provider answered with an error status, after the configured retries for transient ones.</summary>
    Status = 2,

    /// <summary>The answer was not what the provider's documented format promises, or was not valid JSON for the requested schema.</summary>
    Malformed = 3,

    /// <summary>The model declined to answer.</summary>
    Refusal = 4,

    /// <summary>The answer was cut off at the output-token limit.</summary>
    Truncated = 5,

    /// <summary>The provider's content filter blocked the request or the answer.</summary>
    Blocked = 6,

    /// <summary>Values stayed ungrounded after the re-ask and the read asked for <see cref="Llm.UngroundedPolicy.Throw"/>.</summary>
    Ungrounded = 7,
}

/// <summary>A language-model provider failed, was unreachable, or answered with something unusable. Never thrown by local methods.</summary>
public sealed class OcrProviderException : OcrException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="provider">Name of the provider, for example <c>anthropic</c>.</param>
    /// <param name="failure">What went wrong.</param>
    /// <param name="message">A description. Never contains the API key.</param>
    /// <param name="statusCode">The HTTP status, when there was one.</param>
    public OcrProviderException(string provider, LlmFailure failure, string message, int? statusCode = null)
        : base($"{provider}: {message}")
    {
        Provider = provider;
        Failure = failure;
        StatusCode = statusCode;
    }

    /// <summary>Creates the exception with the failure that caused it.</summary>
    /// <param name="provider">Name of the provider.</param>
    /// <param name="failure">What went wrong.</param>
    /// <param name="message">A description. Never contains the API key.</param>
    /// <param name="innerException">The underlying failure.</param>
    public OcrProviderException(string provider, LlmFailure failure, string message, Exception innerException)
        : base($"{provider}: {message}", innerException)
    {
        Provider = provider;
        Failure = failure;
    }

    /// <summary>Name of the provider.</summary>
    public string Provider { get; }

    /// <summary>What went wrong.</summary>
    public LlmFailure Failure { get; }

    /// <summary>The HTTP status code, or null when the failure was not an HTTP answer.</summary>
    public int? StatusCode { get; }
}

/// <summary>Which spending bound a call would have crossed.</summary>
public enum LlmBudgetKind
{
    /// <summary>The estimated cost of one call is above <c>MaxCostPerCallUsd</c>.</summary>
    PerCall = 0,

    /// <summary>The spend so far plus the estimate is above the client's <c>SpendCapUsd</c>.</summary>
    SpendCap = 1,

    /// <summary>A bound is set but the price table has no entry for the model, so the cost cannot be bounded.</summary>
    UnknownPrice = 2,
}

/// <summary>A call was refused before it was sent because it could cost more than the configured limits allow.</summary>
public sealed class OcrBudgetException : OcrException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="kind">Which bound.</param>
    /// <param name="message">A description.</param>
    /// <param name="limitUsd">The configured bound in USD, when there is one.</param>
    /// <param name="requiredUsd">The worst-case cost of the call in USD, when known.</param>
    /// <param name="spentUsd">What the client has spent so far in USD.</param>
    public OcrBudgetException(LlmBudgetKind kind, string message, decimal? limitUsd, decimal? requiredUsd, decimal spentUsd)
        : base(message)
    {
        Kind = kind;
        LimitUsd = limitUsd;
        RequiredUsd = requiredUsd;
        SpentUsd = spentUsd;
    }

    /// <summary>Which bound.</summary>
    public LlmBudgetKind Kind { get; }

    /// <summary>The configured bound in USD, or null for an unknown price.</summary>
    public decimal? LimitUsd { get; }

    /// <summary>The worst-case cost of the refused call in USD, or null when the price is unknown.</summary>
    public decimal? RequiredUsd { get; }

    /// <summary>What the client had spent when the call was refused.</summary>
    public decimal SpentUsd { get; }
}
