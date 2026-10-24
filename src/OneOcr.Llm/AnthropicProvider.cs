using System.Text.Json;

namespace OneOcr.Llm;

/// <summary>
/// Anthropic's Messages API (<c>POST /v1/messages</c>). Structured output uses tool use: the schema is declared as the only
/// tool and the model is forced to call it, so the arguments of that call are the answer. The tool is an output schema only;
/// nothing is ever executed and no other tool is offered. Request and response shapes follow the public API reference
/// (https://docs.anthropic.com/en/api/messages) and the tool-use guide
/// (https://docs.anthropic.com/en/docs/build-with-claude/tool-use). Image guidance
/// (https://docs.anthropic.com/en/docs/build-with-claude/vision): images are resized to fit 1568 pixels on the long side and
/// about 1.15 megapixels, so larger uploads only cost time; the limit is 5 MB per image; tokens are about width × height / 750.
/// The API has no per-request retention switch, so <see cref="LlmProvider.NoRetention"/> changes nothing here; retention is an
/// organisation-level agreement with Anthropic.
/// </summary>
[System.Diagnostics.DebuggerDisplay("{Name} ({Model})")]
public sealed class AnthropicProvider : LlmProvider
{
    private const string Version = "2023-06-01";
    private readonly Uri _uri;

    /// <summary>Creates the provider.</summary>
    /// <param name="apiKey">The API key, sent as <c>x-api-key</c>. Held only in memory, never logged or put in an exception message.</param>
    /// <param name="model">The model string, for example one listed in the provider's model overview. No default exists.</param>
    /// <param name="baseAddress">API root; default <c>https://api.anthropic.com/</c>. Must be https (plain http only for localhost, for a test proxy).</param>
    /// <param name="http">The client to send with; default a shared one that does not follow redirects. Supply your own for proxies or handlers.</param>
    /// <param name="images">Image size overrides; default 1568 pixels long side, 1.15 megapixels, 5 MB.</param>
    public AnthropicProvider(string apiKey, string model, Uri? baseAddress = null, HttpClient? http = null, LlmImageOptions? images = null)
        : base("anthropic", model, http, images, new LlmImageOptions { MaxLongEdge = 1568, MaxPixels = 1_150_000, MaxBytes = 5L * 1024 * 1024 }, apiKey)
    {
        RequireKey(apiKey);
        _uri = new Uri(CheckBase(baseAddress ?? new Uri("https://api.anthropic.com/"), allowPlainHttp: false), "v1/messages");
    }

    /// <summary>Creates the provider with a key read from an environment variable. This is the only place the library reads the environment.</summary>
    /// <param name="model">The model string.</param>
    /// <param name="variable">The variable to read; default <c>ANTHROPIC_API_KEY</c>.</param>
    /// <param name="http">The client to send with.</param>
    /// <returns>The provider.</returns>
    /// <exception cref="InvalidOperationException">The variable is not set or is empty.</exception>
    public static AnthropicProvider FromEnvironment(string model, string variable = "ANTHROPIC_API_KEY", HttpClient? http = null)
    {
        string? key = Environment.GetEnvironmentVariable(variable);
        return string.IsNullOrWhiteSpace(key)
            ? throw new InvalidOperationException($"The environment variable {variable} is not set.")
            : new AnthropicProvider(key, model, http: http);
    }

    internal override Uri RequestUri => _uri;

    internal override void AddHeaders(HttpRequestMessage request)
    {
        request.Headers.TryAddWithoutValidation("x-api-key", ApiKey);
        request.Headers.TryAddWithoutValidation("anthropic-version", Version);
    }

    // Anthropic: width * height / 750 tokens per image.
    internal override long EstimateImageTokens(int width, int height) => ((long)width * height / 750) + 1;

    internal override byte[] BuildBody(LlmCall call, int maxJsonDepth) => Write(w =>
    {
        w.WriteStartObject();
        w.WriteString("model", Model);
        w.WriteNumber("max_tokens", call.MaxOutputTokens);
        w.WriteString("system", call.System);
        w.WriteStartArray("messages");
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
            w.WriteString("type", "image");
            w.WriteStartObject("source");
            w.WriteString("type", "base64");
            w.WriteString("media_type", image.MediaType);
            w.WriteBase64String("data", image.Data);
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
        w.WriteStartArray("tools");
        w.WriteStartObject();
        w.WriteString("name", call.SchemaName);
        w.WriteString("description", "Report the extracted data. This only records your answer.");
        w.WritePropertyName("input_schema");
        w.WriteRawValue(call.SchemaJson, skipInputValidation: true);
        w.WriteEndObject();
        w.WriteEndArray();
        w.WriteStartObject("tool_choice");
        w.WriteString("type", "tool");
        w.WriteString("name", call.SchemaName);
        w.WriteEndObject();
        w.WriteEndObject();
    });

    internal override ProviderReply ParseReply(ReadOnlySpan<byte> body, OcrLimits limits)
    {
        using JsonDocument doc = ProviderJson.Parse(Name, body, limits);
        JsonElement root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw new OcrProviderException(Name, LlmFailure.Malformed, "the response is not a JSON object.");
        }

        if (ProviderJson.Str(root, "type") == "error")
        {
            throw new OcrProviderException(Name, LlmFailure.Status, "the API reported an error: " + ProviderJson.ErrorMessage(body));
        }

        string? stop = ProviderJson.Str(root, "stop_reason");
        if (stop == "refusal")
        {
            throw new OcrProviderException(Name, LlmFailure.Refusal, "the model refused to answer.");
        }

        if (stop == "max_tokens")
        {
            throw new OcrProviderException(Name, LlmFailure.Truncated, "the answer was cut off at the output-token limit.");
        }

        string? json = null;
        if (root.TryGetProperty("content", out JsonElement content) && content.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement block in content.EnumerateArray())
            {
                if (ProviderJson.Str(block, "type") == "tool_use" && block.TryGetProperty("input", out JsonElement input) && input.ValueKind == JsonValueKind.Object)
                {
                    json = ProviderJson.ValidateAnswer(Name, input.GetRawText(), limits);
                    break;
                }
            }
        }

        if (json is null)
        {
            throw new OcrProviderException(Name, LlmFailure.Malformed, "the response has no tool_use block with an object input.");
        }

        root.TryGetProperty("usage", out JsonElement usage);
        return new ProviderReply(json, ProviderJson.Count(usage, "input_tokens"), ProviderJson.Count(usage, "output_tokens"));
    }
}
