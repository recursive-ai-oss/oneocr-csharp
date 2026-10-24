using System.Text;
using System.Text.Json;

namespace OneOcr.Llm;

/// <summary>JSON reading shared by the providers: depth-checked parsing and tolerant error-message extraction.</summary>
internal static class ProviderJson
{
    /// <summary>Walks the document once, throwing <see cref="OcrLimitException"/> when nesting passes <paramref name="limit"/> and <see cref="JsonException"/> when it is not JSON.</summary>
    public static void CheckDepth(ReadOnlySpan<byte> utf8, int limit)
    {
        var reader = new Utf8JsonReader(utf8, new JsonReaderOptions { MaxDepth = limit + 1 });
        while (reader.Read())
        {
            if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray && reader.CurrentDepth + 1 > limit)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxLlmJsonDepth), limit, reader.CurrentDepth + 1);
            }
        }
    }

    /// <summary>Parses a provider response body under the limits; malformed JSON becomes <see cref="OcrProviderException"/>.</summary>
    public static JsonDocument Parse(string provider, ReadOnlySpan<byte> body, OcrLimits limits)
    {
        // Reading a string value out of invalid UTF-8 throws InvalidOperationException, so refuse such bodies here.
        if (!System.Text.Unicode.Utf8.IsValid(body))
        {
            throw new OcrProviderException(provider, LlmFailure.Malformed, "the response is not valid UTF-8.");
        }

        try
        {
            CheckDepth(body, limits.MaxLlmJsonDepth);
            return JsonDocument.Parse(body.ToArray(), new JsonDocumentOptions { MaxDepth = limits.MaxLlmJsonDepth });
        }
        catch (JsonException e)
        {
            throw new OcrProviderException(provider, LlmFailure.Malformed, "the response is not valid JSON.", e);
        }
    }

    /// <summary>Checks that the model's answer text is one JSON object within the depth limit, after removing a Markdown code fence some servers add.</summary>
    public static string ValidateAnswer(string provider, string text, OcrLimits limits)
    {
        string answer = StripFence(text);
        byte[] bytes = Encoding.UTF8.GetBytes(answer);
        try
        {
            CheckDepth(bytes, limits.MaxLlmJsonDepth);
            var reader = new Utf8JsonReader(bytes);
            if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            {
                throw new OcrProviderException(provider, LlmFailure.Malformed, "the answer is not a JSON object.");
            }
        }
        catch (JsonException e)
        {
            throw new OcrProviderException(provider, LlmFailure.Malformed, "the answer is not valid JSON.", e);
        }

        return answer;
    }

    private static string StripFence(string text)
    {
        string t = text.Trim();
        if (t.StartsWith("```", StringComparison.Ordinal))
        {
            int nl = t.IndexOf('\n', StringComparison.Ordinal);
            int end = t.LastIndexOf("```", StringComparison.Ordinal);
            if (nl > 0 && end > nl)
            {
                return t[(nl + 1)..end].Trim();
            }
        }

        return t;
    }

    /// <summary>A token count from a usage object, or -1 when absent or not a non-negative integer.</summary>
    public static long Count(JsonElement parent, string name) =>
        parent.ValueKind == JsonValueKind.Object && parent.TryGetProperty(name, out JsonElement e)
        && e.ValueKind == JsonValueKind.Number && e.TryGetInt64(out long v) && v >= 0 ? v : -1;

    /// <summary>A string property, or null.</summary>
    public static string? Str(JsonElement parent, string name) =>
        parent.ValueKind == JsonValueKind.Object && parent.TryGetProperty(name, out JsonElement e) && e.ValueKind == JsonValueKind.String
            ? e.GetString() : null;

    /// <summary>Extracts a human-readable message from an error body, never throwing. Control characters are removed and the text is cut at 300 characters.</summary>
    public static string ErrorMessage(ReadOnlySpan<byte> body)
    {
        string? message = null;
        if (body.Length > 0 && body.Length <= 64 * 1024)
        {
            try
            {
                CheckDepth(body, 16);
                using JsonDocument doc = JsonDocument.Parse(body.ToArray(), new JsonDocumentOptions { MaxDepth = 16 });
                JsonElement root = doc.RootElement;
                if (root.ValueKind == JsonValueKind.Object)
                {
                    if (root.TryGetProperty("error", out JsonElement err))
                    {
                        message = err.ValueKind == JsonValueKind.String ? err.GetString() : Str(err, "message");
                    }

                    message ??= Str(root, "message");
                }
            }
            catch (Exception e) when (e is JsonException or OcrLimitException or InvalidOperationException)
            {
                message = null;
            }
        }

        message ??= Encoding.UTF8.GetString(body[..Math.Min(body.Length, 300)]);
        var sb = new StringBuilder(Math.Min(message.Length, 300));
        foreach (char c in message)
        {
            if (sb.Length >= 300)
            {
                break;
            }

            sb.Append(char.IsControl(c) ? ' ' : c);
        }

        return sb.ToString().Trim();
    }
}
