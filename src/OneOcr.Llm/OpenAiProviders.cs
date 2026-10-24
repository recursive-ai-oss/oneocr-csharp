using System.Text.Json;
using System.Text.Json.Nodes;

namespace OneOcr.Llm;

/// <summary>
/// The Chat Completions wire format that OpenAI, Azure OpenAI and OpenAI-compatible servers share
/// (https://platform.openai.com/docs/api-reference/chat/create, structured output:
/// https://platform.openai.com/docs/guides/structured-outputs).
/// </summary>
internal static class OpenAiChatWire
{
    internal sealed record Style(string? Model, bool Strict, bool SendStoreFalse, string MaxTokensName, bool Temperature0, string? ImageDetail);

    public static byte[] Body(LlmCall call, Style style)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>(4096);
        using (var w = new Utf8JsonWriter(buffer))
        {
            w.WriteStartObject();
            if (style.Model is not null)
            {
                w.WriteString("model", style.Model);
            }

            w.WriteStartArray("messages");
            w.WriteStartObject();
            w.WriteString("role", "system");
            w.WriteString("content", call.System);
            w.WriteEndObject();
            w.WriteStartObject();
            w.WriteString("role", "user");
            w.WriteStartArray("content");
            foreach (PreparedImage image in call.Images)
            {
                w.WriteStartObject();
                w.WriteString("type", "text");
                w.WriteString("text", image.Label);
                w.WriteEndObject();
                w.WriteStartObject();
                w.WriteString("type", "image_url");
                w.WriteStartObject("image_url");
                w.WriteString("url", "data:" + image.MediaType + ";base64," + Convert.ToBase64String(image.Data));
                if (style.ImageDetail is not null)
                {
                    w.WriteString("detail", style.ImageDetail);
                }

                w.WriteEndObject();
                w.WriteEndObject();
            }

            w.WriteStartObject();
            w.WriteString("type", "text");
            w.WriteString("text", call.UserText);
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteStartObject("response_format");
            w.WriteString("type", "json_schema");
            w.WriteStartObject("json_schema");
            w.WriteString("name", call.SchemaName);
            if (style.Strict)
            {
                w.WriteBoolean("strict", true);
            }

            w.WritePropertyName("schema");
            w.WriteRawValue(call.SchemaJson, skipInputValidation: true);
            w.WriteEndObject();
            w.WriteEndObject();
            w.WriteNumber(style.MaxTokensName, call.MaxOutputTokens);
            if (style.Temperature0)
            {
                w.WriteNumber("temperature", 0);
            }

            if (style.SendStoreFalse)
            {
                w.WriteBoolean("store", false);
            }

            w.WriteEndObject();
        }

        return buffer.WrittenSpan.ToArray();
    }

    /// <summary>The schema text for the body: strict mode needs every property required and no extras.</summary>
    public static string Schema(string schemaJson) =>
        SchemaTools.ForStrict((JsonObject)JsonNode.Parse(schemaJson)!).ToJsonString();

    public static ProviderReply Parse(string provider, ReadOnlySpan<byte> body, OcrLimits limits)
    {
        using JsonDocument doc = ProviderJson.Parse(provider, body, limits);
        JsonElement root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw new OcrProviderException(provider, LlmFailure.Malformed, "the response is not a JSON object.");
        }

        if (root.TryGetProperty("error", out JsonElement err) && err.ValueKind is JsonValueKind.Object or JsonValueKind.String)
        {
            throw new OcrProviderException(provider, LlmFailure.Status, "the API reported an error: " + ProviderJson.ErrorMessage(body));
        }

        if (!root.TryGetProperty("choices", out JsonElement choices) || choices.ValueKind != JsonValueKind.Array || choices.GetArrayLength() == 0)
        {
            throw new OcrProviderException(provider, LlmFailure.Malformed, "the response has no choices.");
        }

        JsonElement choice = choices[0];
        string? finish = ProviderJson.Str(choice, "finish_reason");
        if (choice.ValueKind != JsonValueKind.Object || !choice.TryGetProperty("message", out JsonElement message) || message.ValueKind != JsonValueKind.Object)
        {
            throw new OcrProviderException(provider, LlmFailure.Malformed, "the first choice has no message.");
        }

        if (!string.IsNullOrEmpty(ProviderJson.Str(message, "refusal")))
        {
            throw new OcrProviderException(provider, LlmFailure.Refusal, "the model refused to answer.");
        }

        if (finish == "content_filter")
        {
            throw new OcrProviderException(provider, LlmFailure.Blocked, "the content filter stopped the answer.");
        }

        if (finish == "length")
        {
            throw new OcrProviderException(provider, LlmFailure.Truncated, "the answer was cut off at the output-token limit.");
        }

        string? content = ProviderJson.Str(message, "content");
        if (string.IsNullOrWhiteSpace(content))
        {
            throw new OcrProviderException(provider, LlmFailure.Malformed, "the message has no text content.");
        }

        string json = ProviderJson.ValidateAnswer(provider, content, limits);
        root.TryGetProperty("usage", out JsonElement usage);
        return new ProviderReply(json, ProviderJson.Count(usage, "prompt_tokens"), ProviderJson.Count(usage, "completion_tokens"));
    }
}

/// <summary>
/// OpenAI's Chat Completions API with strict <c>json_schema</c> structured output. Requests carry <c>store: false</c> unless
/// <see cref="LlmProvider.NoRetention"/> is turned off. Images are sent with <c>detail: "high"</c>; per the vision guide
/// (https://platform.openai.com/docs/guides/images-vision) the service fits the image in 2048 × 2048, then scales its shorter
/// side to 768 pixels, and bills 85 tokens plus 170 per 512-pixel tile, so that is what the default shrinks to. Schemas are
/// rewritten for strict mode: every property becomes required and extra properties are forbidden, so make optional values nullable.
/// </summary>
public sealed class OpenAiProvider : LlmProvider
{
    private readonly string _apiKey;
    private readonly Uri _uri;
    private readonly string? _organization;

    /// <summary>Creates the provider.</summary>
    /// <param name="apiKey">The API key, sent as a bearer token. Held only in memory, never logged or put in an exception message.</param>
    /// <param name="model">The model string. No default exists.</param>
    /// <param name="baseAddress">API root including the version path; default <c>https://api.openai.com/v1/</c>. Must be https (plain http only for localhost).</param>
    /// <param name="http">The client to send with; default a shared one that does not follow redirects.</param>
    /// <param name="images">Image size overrides; default 2048 long side, 768 short side, 20 MB.</param>
    /// <param name="organization">Optional organization id, sent as <c>OpenAI-Organization</c>.</param>
    public OpenAiProvider(string apiKey, string model, Uri? baseAddress = null, HttpClient? http = null, LlmImageOptions? images = null, string? organization = null)
        : base("openai", model, http, images, new LlmImageOptions { MaxLongEdge = 2048, MaxShortEdge = 768, MaxBytes = 20L * 1024 * 1024 })
    {
        _apiKey = RequireKey(apiKey);
        _organization = organization;
        _uri = new Uri(CheckBase(baseAddress ?? new Uri("https://api.openai.com/v1/"), allowPlainHttp: false), "chat/completions");
    }

    /// <summary>Creates the provider with a key read from an environment variable. This is the only place the library reads the environment.</summary>
    /// <param name="model">The model string.</param>
    /// <param name="variable">The variable to read; default <c>OPENAI_API_KEY</c>.</param>
    /// <param name="http">The client to send with.</param>
    /// <returns>The provider.</returns>
    /// <exception cref="InvalidOperationException">The variable is not set or is empty.</exception>
    public static OpenAiProvider FromEnvironment(string model, string variable = "OPENAI_API_KEY", HttpClient? http = null)
    {
        string? key = Environment.GetEnvironmentVariable(variable);
        return string.IsNullOrWhiteSpace(key)
            ? throw new InvalidOperationException($"The environment variable {variable} is not set.")
            : new OpenAiProvider(key, model, http: http);
    }

    internal override Uri RequestUri => _uri;

    internal override void AddHeaders(HttpRequestMessage request)
    {
        request.Headers.TryAddWithoutValidation("Authorization", "Bearer " + _apiKey);
        if (_organization is not null)
        {
            request.Headers.TryAddWithoutValidation("OpenAI-Organization", _organization);
        }
    }

    // OpenAI "high" detail: 85 + 170 per 512-pixel tile of the image after the 2048 / 768 fit.
    internal override long EstimateImageTokens(int width, int height) => 85 + (170L * ((width + 511) / 512) * ((height + 511) / 512));

    internal override byte[] BuildBody(LlmCall call, int maxJsonDepth) =>
        OpenAiChatWire.Body(
            call with { SchemaJson = OpenAiChatWire.Schema(call.SchemaJson) },
            new OpenAiChatWire.Style(Model, Strict: true, SendStoreFalse: NoRetention, "max_completion_tokens", Temperature0: false, ImageDetail: "high"));

    internal override ProviderReply ParseReply(ReadOnlySpan<byte> body, OcrLimits limits) => OpenAiChatWire.Parse(Name, body, limits);
}

/// <summary>
/// Any server that speaks the OpenAI Chat Completions format: Ollama, llama.cpp's <c>llama-server</c>, vLLM, LM Studio and
/// hosted look-alikes. Give it the base URL (for example <c>http://localhost:8080/v1/</c>) and the model name the server expects.
/// Structured output is sent as <c>response_format: json_schema</c>, which these servers turn into constrained decoding.
/// No <c>store</c> flag is sent because servers that are not OpenAI may reject unknown fields. The key is optional.
/// The default image bound (1536 pixels on the long side) is a guess that suits common vision encoders; pass
/// image options to the constructor for the model in use.
/// </summary>
public sealed class OpenAiCompatibleProvider : LlmProvider
{
    private readonly string? _apiKey;
    private readonly Uri _uri;

    /// <summary>Creates the provider.</summary>
    /// <param name="baseAddress">The server's API root including the version path, usually ending <c>/v1</c>. Plain http is accepted for any host here because local servers rarely have TLS; an API key then travels unencrypted, so only pass one over https.</param>
    /// <param name="model">The model name the server expects.</param>
    /// <param name="apiKey">Optional bearer token. Not accepted over plain http to a non-local host.</param>
    /// <param name="http">The client to send with; default a shared one that does not follow redirects.</param>
    /// <param name="images">Image size overrides.</param>
    public OpenAiCompatibleProvider(Uri baseAddress, string model, string? apiKey = null, HttpClient? http = null, LlmImageOptions? images = null)
        : base("openai-compatible", model, http, images, new LlmImageOptions { MaxLongEdge = 1536, MaxBytes = 20L * 1024 * 1024 })
    {
        Uri root = CheckBase(baseAddress, allowPlainHttp: true);
        if (!string.IsNullOrEmpty(apiKey) && root.Scheme == "http" && !root.IsLoopback)
        {
            throw new ArgumentException("An API key is never sent over plain HTTP to a non-local host.", nameof(apiKey));
        }

        _apiKey = string.IsNullOrEmpty(apiKey) ? null : apiKey;
        _uri = new Uri(root, "chat/completions");
    }

    /// <summary>Creates the provider with an optional key read from an environment variable. This is the only place the library reads the environment.</summary>
    /// <param name="baseAddress">The server's API root.</param>
    /// <param name="model">The model name the server expects.</param>
    /// <param name="variable">The variable to read; an unset or empty variable means no key.</param>
    /// <param name="http">The client to send with.</param>
    /// <returns>The provider.</returns>
    public static OpenAiCompatibleProvider FromEnvironment(Uri baseAddress, string model, string variable, HttpClient? http = null) =>
        new(baseAddress, model, Environment.GetEnvironmentVariable(variable), http);

    internal override Uri RequestUri => _uri;

    internal override void AddHeaders(HttpRequestMessage request)
    {
        if (_apiKey is not null)
        {
            request.Headers.TryAddWithoutValidation("Authorization", "Bearer " + _apiKey);
        }
    }

    // No documented formula for these servers: width * height / 750, the figure Anthropic documents and a fair midpoint.
    internal override long EstimateImageTokens(int width, int height) => ((long)width * height / 750) + 1;

    internal override byte[] BuildBody(LlmCall call, int maxJsonDepth) =>
        OpenAiChatWire.Body(
            call with { SchemaJson = OpenAiChatWire.Schema(call.SchemaJson) },
            new OpenAiChatWire.Style(Model, Strict: false, SendStoreFalse: false, "max_tokens", Temperature0: true, ImageDetail: null));

    internal override ProviderReply ParseReply(ReadOnlySpan<byte> body, OcrLimits limits) => OpenAiChatWire.Parse(Name, body, limits);
}

/// <summary>
/// Azure OpenAI Service chat completions on a deployment
/// (https://learn.microsoft.com/azure/ai-services/openai/reference): the URL names the deployment, the key goes in an
/// <c>api-key</c> header and an <c>api-version</c> query parameter is required. The "model" is the deployment name.
/// Azure does not store completions unless asked, so no <c>store</c> flag is sent and <see cref="LlmProvider.NoRetention"/>
/// changes nothing. Image rules and strict structured output are as for <see cref="OpenAiProvider"/>.
/// </summary>
public sealed class AzureOpenAiProvider : LlmProvider
{
    /// <summary>The API version used when none is given: the generally available 2024-10-21 data-plane release, which supports strict <c>json_schema</c>.</summary>
    public const string DefaultApiVersion = "2024-10-21";

    private readonly string _apiKey;
    private readonly Uri _uri;

    /// <summary>Creates the provider.</summary>
    /// <param name="endpoint">The resource endpoint, for example <c>https://my-resource.openai.azure.com/</c>. Must be https (plain http only for localhost).</param>
    /// <param name="deployment">The deployment name; used as the model string.</param>
    /// <param name="apiKey">The resource key, sent as <c>api-key</c>. Held only in memory.</param>
    /// <param name="apiVersion">The <c>api-version</c> query value; default <see cref="DefaultApiVersion"/>.</param>
    /// <param name="http">The client to send with; default a shared one that does not follow redirects.</param>
    /// <param name="images">Image size overrides; default as OpenAI.</param>
    public AzureOpenAiProvider(Uri endpoint, string deployment, string apiKey, string apiVersion = DefaultApiVersion, HttpClient? http = null, LlmImageOptions? images = null)
        : base("azure-openai", deployment, http, images, new LlmImageOptions { MaxLongEdge = 2048, MaxShortEdge = 768, MaxBytes = 20L * 1024 * 1024 })
    {
        _apiKey = RequireKey(apiKey);
        ArgumentException.ThrowIfNullOrWhiteSpace(apiVersion);
        Uri root = CheckBase(endpoint, allowPlainHttp: false);
        _uri = new Uri(root, $"openai/deployments/{Uri.EscapeDataString(deployment)}/chat/completions?api-version={Uri.EscapeDataString(apiVersion)}");
    }

    /// <summary>Creates the provider with a key read from an environment variable. This is the only place the library reads the environment.</summary>
    /// <param name="endpoint">The resource endpoint.</param>
    /// <param name="deployment">The deployment name.</param>
    /// <param name="variable">The variable to read; default <c>AZURE_OPENAI_API_KEY</c>.</param>
    /// <param name="apiVersion">The <c>api-version</c> value.</param>
    /// <param name="http">The client to send with.</param>
    /// <returns>The provider.</returns>
    /// <exception cref="InvalidOperationException">The variable is not set or is empty.</exception>
    public static AzureOpenAiProvider FromEnvironment(Uri endpoint, string deployment, string variable = "AZURE_OPENAI_API_KEY", string apiVersion = DefaultApiVersion, HttpClient? http = null)
    {
        string? key = Environment.GetEnvironmentVariable(variable);
        return string.IsNullOrWhiteSpace(key)
            ? throw new InvalidOperationException($"The environment variable {variable} is not set.")
            : new AzureOpenAiProvider(endpoint, deployment, key, apiVersion, http);
    }

    internal override Uri RequestUri => _uri;

    internal override void AddHeaders(HttpRequestMessage request) => request.Headers.TryAddWithoutValidation("api-key", _apiKey);

    internal override long EstimateImageTokens(int width, int height) => 85 + (170L * ((width + 511) / 512) * ((height + 511) / 512));

    internal override byte[] BuildBody(LlmCall call, int maxJsonDepth) =>
        OpenAiChatWire.Body(
            call with { SchemaJson = OpenAiChatWire.Schema(call.SchemaJson) },
            new OpenAiChatWire.Style(Model: null, Strict: true, SendStoreFalse: false, "max_completion_tokens", Temperature0: false, ImageDetail: "high"));

    internal override ProviderReply ParseReply(ReadOnlySpan<byte> body, OcrLimits limits) => OpenAiChatWire.Parse(Name, body, limits);
}
