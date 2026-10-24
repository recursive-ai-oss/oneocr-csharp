using System.Text.Json;
using System.Text.Json.Nodes;

namespace OneOcr.Llm;

/// <summary>
/// Google's Gemini API (<c>generateContent</c>, https://ai.google.dev/api/generate-content) with structured output through
/// <c>generationConfig.responseMimeType = "application/json"</c> and <c>responseSchema</c>
/// (https://ai.google.dev/gemini-api/docs/structured-output). Schemas are converted to Gemini's OpenAPI subset; a schema that
/// needs <c>$ref</c> cannot be sent. Images go inline as base64. Per the image guide
/// (https://ai.google.dev/gemini-api/docs/image-understanding) an image of 384 pixels or less on both sides costs 258 tokens and a
/// larger one is cut into 768-pixel tiles of 258 tokens each; the default long side of 2304 pixels is three tiles. The inline
/// request limit is 20 MB, so the per-image default is 15 MB. The API has no per-request retention switch, so
/// <see cref="LlmProvider.NoRetention"/> changes nothing; use a paid-tier key, whose data is not used to improve Google's products.
/// </summary>
[System.Diagnostics.DebuggerDisplay("{Name} ({Model})")]
public sealed class GeminiProvider : LlmProvider
{
    private readonly Uri _uri;

    /// <summary>Creates the provider.</summary>
    /// <param name="apiKey">The API key, sent as <c>x-goog-api-key</c>. Held only in memory, never logged or put in an exception message.</param>
    /// <param name="model">The model string, for example <c>gemini-2.5-flash</c>. No default exists.</param>
    /// <param name="baseAddress">API root; default <c>https://generativelanguage.googleapis.com/</c>. Must be https (plain http only for localhost).</param>
    /// <param name="http">The client to send with; default a shared one that does not follow redirects.</param>
    /// <param name="images">Image size overrides.</param>
    public GeminiProvider(string apiKey, string model, Uri? baseAddress = null, HttpClient? http = null, LlmImageOptions? images = null)
        : base("gemini", model, http, images, new LlmImageOptions { MaxLongEdge = 2304, MaxBytes = 15L * 1024 * 1024 }, apiKey)
    {
        RequireKey(apiKey);
        string id = model.StartsWith("models/", StringComparison.Ordinal) ? model["models/".Length..] : model;
        _uri = new Uri(CheckBase(baseAddress ?? new Uri("https://generativelanguage.googleapis.com/"), allowPlainHttp: false), $"v1beta/models/{Uri.EscapeDataString(id)}:generateContent");
    }

    /// <summary>Creates the provider with a key read from an environment variable. This is the only place the library reads the environment.</summary>
    /// <param name="model">The model string.</param>
    /// <param name="variable">The variable to read; default <c>GEMINI_API_KEY</c>.</param>
    /// <param name="http">The client to send with.</param>
    /// <returns>The provider.</returns>
    /// <exception cref="InvalidOperationException">The variable is not set or is empty.</exception>
    public static GeminiProvider FromEnvironment(string model, string variable = "GEMINI_API_KEY", HttpClient? http = null)
    {
        string? key = Environment.GetEnvironmentVariable(variable);
        return string.IsNullOrWhiteSpace(key)
            ? throw new InvalidOperationException($"The environment variable {variable} is not set.")
            : new GeminiProvider(key, model, http: http);
    }

    internal override Uri RequestUri => _uri;

    internal override void AddHeaders(HttpRequestMessage request) => request.Headers.TryAddWithoutValidation("x-goog-api-key", ApiKey);

    // Gemini: 258 tokens up to 384 x 384, otherwise 258 per 768 x 768 tile.
    internal override long EstimateImageTokens(int width, int height) =>
        width <= 384 && height <= 384 ? 258 : 258L * ((width + 767) / 768) * ((height + 767) / 768);

    internal override byte[] BuildBody(LlmCall call, int maxJsonDepth)
    {
        string schema = SchemaTools.ForGemini((JsonObject)JsonNode.Parse(call.SchemaJson)!).ToJsonString();
        return Write(w =>
        {
            w.WriteStartObject();
            w.WriteStartObject("systemInstruction");
            w.WriteStartArray("parts");
            w.WriteStartObject();
            w.WriteString("text", call.System);
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteStartArray("contents");
            w.WriteStartObject();
            w.WriteString("role", "user");
            w.WriteStartArray("parts");
            foreach (PreparedImage image in call.Images)
            {
                w.WriteStartObject();
                w.WriteString("text", image.Label);
                w.WriteEndObject();
                w.WriteStartObject();
                w.WriteStartObject("inline_data");
                w.WriteString("mime_type", image.MediaType);
                w.WriteBase64String("data", image.Data);
                w.WriteEndObject();
                w.WriteEndObject();
            }

            w.WriteStartObject();
            w.WriteString("text", call.UserText);
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteStartObject("generationConfig");
            w.WriteString("responseMimeType", "application/json");
            w.WritePropertyName("responseSchema");
            w.WriteRawValue(schema, skipInputValidation: true);
            w.WriteNumber("maxOutputTokens", call.MaxOutputTokens);
            w.WriteEndObject();
            w.WriteEndObject();
        });
    }

    internal override ProviderReply ParseReply(ReadOnlySpan<byte> body, OcrLimits limits)
    {
        using JsonDocument doc = ProviderJson.Parse(Name, body, limits);
        JsonElement root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw new OcrProviderException(Name, LlmFailure.Malformed, "the response is not a JSON object.");
        }

        if (root.TryGetProperty("error", out JsonElement err) && err.ValueKind == JsonValueKind.Object)
        {
            throw new OcrProviderException(Name, LlmFailure.Status, "the API reported an error: " + ProviderJson.ErrorMessage(body));
        }

        if (root.TryGetProperty("promptFeedback", out JsonElement feedback) && ProviderJson.Str(feedback, "blockReason") is { } reason)
        {
            throw new OcrProviderException(Name, LlmFailure.Blocked, "the prompt was blocked: " + reason + ".");
        }

        if (!root.TryGetProperty("candidates", out JsonElement candidates) || candidates.ValueKind != JsonValueKind.Array || candidates.GetArrayLength() == 0)
        {
            throw new OcrProviderException(Name, LlmFailure.Malformed, "the response has no candidates.");
        }

        JsonElement candidate = candidates[0];
        switch (ProviderJson.Str(candidate, "finishReason"))
        {
            case "MAX_TOKENS":
                throw new OcrProviderException(Name, LlmFailure.Truncated, "the answer was cut off at the output-token limit.");
            case "SAFETY" or "BLOCKLIST" or "PROHIBITED_CONTENT" or "SPII" or "RECITATION":
                throw new OcrProviderException(Name, LlmFailure.Blocked, "the answer was stopped by the provider's filters.");
        }

        var text = new System.Text.StringBuilder();
        if (candidate.ValueKind == JsonValueKind.Object && candidate.TryGetProperty("content", out JsonElement content)
            && content.ValueKind == JsonValueKind.Object && content.TryGetProperty("parts", out JsonElement parts) && parts.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement part in parts.EnumerateArray())
            {
                // Parts flagged as thoughts are the model's reasoning summary, not the answer.
                bool thought = part.ValueKind == JsonValueKind.Object && part.TryGetProperty("thought", out JsonElement t) && t.ValueKind == JsonValueKind.True;
                if (!thought && ProviderJson.Str(part, "text") is { } s)
                {
                    text.Append(s);
                }
            }
        }

        if (text.Length == 0)
        {
            throw new OcrProviderException(Name, LlmFailure.Malformed, "the candidate has no text.");
        }

        string json = ProviderJson.ValidateAnswer(Name, text.ToString(), limits);
        root.TryGetProperty("usageMetadata", out JsonElement usage);
        long output = ProviderJson.Count(usage, "candidatesTokenCount");
        long thoughts = ProviderJson.Count(usage, "thoughtsTokenCount");
        if (output >= 0 && thoughts > 0)
        {
            output += thoughts;
        }

        return new ProviderReply(json, ProviderJson.Count(usage, "promptTokenCount"), output);
    }
}
