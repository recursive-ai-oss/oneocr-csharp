using System.Net;
using System.Text.Json;

namespace OneOcr.Llm;

/// <summary>One encoded page image ready to upload.</summary>
internal sealed record PreparedImage(string Label, byte[] Data, string MediaType, int Width, int Height);

/// <summary>Everything a provider needs to build one request.</summary>
internal sealed record LlmCall(
    string System, string UserText, IReadOnlyList<PreparedImage> Images, string SchemaName, string SchemaJson, int MaxOutputTokens);

/// <summary>The useful part of a provider's answer. A token count of -1 means the provider did not report one.</summary>
internal readonly record struct ProviderReply(string Json, long InputTokens, long OutputTokens);

/// <summary>
/// A language-model API. One subclass per API: <see cref="AnthropicProvider"/>, <see cref="OpenAiProvider"/>,
/// <see cref="OpenAiCompatibleProvider"/>, <see cref="GeminiProvider"/> and <see cref="AzureOpenAiProvider"/>. A provider
/// holds its model string, its key and its <see cref="HttpClient"/>; policy (budgets, retries, privacy, caching) lives in
/// <see cref="LlmClient"/>. The hierarchy is closed on purpose: the constructor is not accessible outside this assembly, so these five are the only providers, and the wire format of each is internal. To test code that uses <see cref="LlmClient"/>, give a provider an <see cref="HttpClient"/> with a fake <see cref="HttpMessageHandler"/>.
/// </summary>
[System.Diagnostics.DebuggerDisplay("{Name} ({Model})")]
public abstract class LlmProvider
{
    private readonly Secret? _secret;

    private static readonly Lazy<HttpClient> SharedHttp = new(static () => new HttpClient(new SocketsHttpHandler
    {
        // A redirect would carry the API key header to another host.
        AllowAutoRedirect = false,
        PooledConnectionLifetime = TimeSpan.FromMinutes(5),
    })
    {
        Timeout = System.Threading.Timeout.InfiniteTimeSpan,
    });

    private protected LlmProvider(string name, string model, HttpClient? http, LlmImageOptions? images, LlmImageOptions defaults, string? apiKey)
    {
        _secret = string.IsNullOrWhiteSpace(apiKey) ? null : new Secret(apiKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        Name = name;
        Model = model;
        Http = http ?? SharedHttp.Value;
        ImageOptions = (images ?? new LlmImageOptions()) with
        {
            MaxLongEdge = images?.MaxLongEdge ?? defaults.MaxLongEdge,
            MaxShortEdge = images?.MaxShortEdge ?? defaults.MaxShortEdge,
            MaxPixels = images?.MaxPixels ?? defaults.MaxPixels,
            MaxBytes = images?.MaxBytes ?? defaults.MaxBytes,
        };
    }

    /// <summary>The provider's short name: <c>anthropic</c>, <c>openai</c>, <c>openai-compatible</c>, <c>gemini</c> or <c>azure-openai</c>.</summary>
    public string Name { get; }

    /// <summary>The key into <see cref="LlmPriceTable"/>; the same as <see cref="Name"/>.</summary>
    public string PriceKey => Name;

    /// <summary>The model string the caller named. There is no default model.</summary>
    public string Model { get; }

    /// <summary>The image shrinking and size rules in effect: the provider's documented defaults with the caller's overrides applied.</summary>
    public LlmImageOptions ImageOptions { get; }

    /// <summary>
    /// Whether the provider is asked not to keep the request. On by default where the API has a per-request switch (OpenAI's
    /// <c>store: false</c>); a no-op for APIs without one (Anthropic, Gemini, Azure OpenAI, and servers that are not OpenAI).
    /// </summary>
    public bool NoRetention { get; init; } = true;

    /// <summary>Shows the provider and model, never the key.</summary>
    /// <returns>For example <c>anthropic (model-name)</c>.</returns>
    public override string ToString() => $"{Name} ({Model})";

    /// <summary>True when calls cost nothing unless the price table says otherwise (a server on this machine).</summary>
    internal virtual bool IsFreeByDefault => false;

    /// <summary>The key, for building request headers only.</summary>
    private protected string? ApiKey => _secret?.Value;

    /// <summary>Removes the key and any 8-character piece of it from text a server sent, and cuts the text to a few hundred characters.</summary>
    internal string Scrub(string text) => _secret is null ? (text.Length <= 400 ? text : text[..400]) : _secret.Scrub(text);

    /// <summary>Like <see cref="Scrub"/> but without cutting the text: for model answers, which are legitimately long.</summary>
    internal string ScrubKey(string text) => _secret is null ? text : _secret.ScrubAll(text);

    internal HttpClient Http { get; }

    internal abstract Uri RequestUri { get; }

    internal abstract void AddHeaders(HttpRequestMessage request);

    internal abstract byte[] BuildBody(LlmCall call, int maxJsonDepth);

    internal abstract ProviderReply ParseReply(ReadOnlySpan<byte> body, OcrLimits limits);

    internal abstract long EstimateImageTokens(int width, int height);

    internal HttpRequestMessage CreateRequest(byte[] body)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, RequestUri) { Content = new ByteArrayContent(body) };
        request.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");
        request.Headers.UserAgent.ParseAdd("RecursiveAI.OneOcr.Llm");
        AddHeaders(request);
        return request;
    }

    /// <summary>The key-bearing providers refuse to send a key over plain HTTP, except to the local machine.</summary>
    private protected static Uri CheckBase(Uri baseAddress, bool allowPlainHttp)
    {
        ArgumentNullException.ThrowIfNull(baseAddress);
        if (!baseAddress.IsAbsoluteUri || baseAddress.Scheme is not ("https" or "http"))
        {
            throw new ArgumentException("The base address must be an absolute http or https URL.", nameof(baseAddress));
        }

        if (baseAddress.Scheme == "http" && !allowPlainHttp && !baseAddress.IsLoopback)
        {
            throw new ArgumentException("An API key is never sent over plain HTTP; use https (plain HTTP is accepted only for localhost).", nameof(baseAddress));
        }

        string s = baseAddress.AbsoluteUri;
        return s.EndsWith('/') ? baseAddress : new Uri(s + "/");
    }

    private protected static string RequireKey(string apiKey)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(apiKey);
        return apiKey;
    }

    private protected static OcrProviderException Fail(string provider, LlmFailure failure, string message) => new(provider, failure, message);

    /// <summary>Whether an HTTP status is worth retrying.</summary>
    internal static bool IsTransient(HttpStatusCode status) =>
        status is HttpStatusCode.RequestTimeout or HttpStatusCode.TooManyRequests or HttpStatusCode.InternalServerError
            or HttpStatusCode.BadGateway or HttpStatusCode.ServiceUnavailable or HttpStatusCode.GatewayTimeout || (int)status == 529;

    /// <summary>Ceiling of tokens for plain text: three UTF-8 bytes per token, which errs high for English and about right for dense scripts.</summary>
    internal static long EstimateTextTokens(long utf8Bytes) => (utf8Bytes + 2) / 3;

    /// <summary>Writes <c>{ ... }</c> text of a content part list shared by every API: label, image, then the final instruction text.</summary>
    private protected static byte[] Write(Action<Utf8JsonWriter> body)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>(4096);
        using (var w = new Utf8JsonWriter(buffer))
        {
            body(w);
        }

        return buffer.WrittenSpan.ToArray();
    }
}
