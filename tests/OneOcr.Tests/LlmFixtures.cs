namespace OneOcr.Tests;

/// <summary>
/// Hand-written provider responses in the shapes the public API references document, shared by the tests and the fuzz seed writer.
/// None comes from a real account: ids are made up and there are no keys.
/// </summary>
public static class LlmFixtures
{
    public const string Answer = "{\"vendor\":\"Acme\",\"total\":\"12.50\"}";

    public const string AnthropicOk =
        "{\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"test-model\",\"content\":[{\"type\":\"text\",\"text\":\"Recording.\"},"
        + "{\"type\":\"tool_use\",\"id\":\"toolu_01\",\"name\":\"extraction\",\"input\":{\"vendor\":\"Acme\",\"total\":\"12.50\"}}],"
        + "\"stop_reason\":\"tool_use\",\"usage\":{\"input_tokens\":1500,\"output_tokens\":40}}";

    public const string AnthropicRefusal = "{\"id\":\"msg_02\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"stop_reason\":\"refusal\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}";

    public const string AnthropicTruncated = "{\"id\":\"msg_03\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"t\",\"name\":\"extraction\",\"input\":{}}],\"stop_reason\":\"max_tokens\",\"usage\":{\"input_tokens\":10,\"output_tokens\":4096}}";

    public const string AnthropicError = "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}";

    public const string OpenAiOk =
        "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"model\":\"test-model\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
        + "\"content\":\"{\\\"vendor\\\":\\\"Acme\\\",\\\"total\\\":\\\"12.50\\\"}\",\"refusal\":null},\"finish_reason\":\"stop\"}],"
        + "\"usage\":{\"prompt_tokens\":1200,\"completion_tokens\":30,\"total_tokens\":1230}}";

    public const string OpenAiRefusal =
        "{\"id\":\"chatcmpl-2\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":null,\"refusal\":\"I can't help with that.\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}";

    public const string OpenAiLength =
        "{\"id\":\"chatcmpl-3\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"{\\\"vendor\\\":\\\"Ac\"},\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":4096}}";

    public const string OpenAiFenced =
        "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"```json\\n{\\\"vendor\\\":\\\"Acme\\\",\\\"total\\\":\\\"12.50\\\"}\\n```\"},\"finish_reason\":\"stop\"}]}";

    public const string OpenAiError = "{\"error\":{\"message\":\"Rate limit reached for requests\",\"type\":\"requests\",\"param\":null,\"code\":\"rate_limit_exceeded\"}}";

    public const string GeminiOk =
        "{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"{\\\"vendor\\\":\\\"Acme\\\",\\\"total\\\":\\\"12.50\\\"}\"}]},\"finishReason\":\"STOP\",\"index\":0}],"
        + "\"usageMetadata\":{\"promptTokenCount\":900,\"candidatesTokenCount\":25,\"totalTokenCount\":925}}";

    public const string GeminiBlocked = "{\"promptFeedback\":{\"blockReason\":\"SAFETY\"},\"usageMetadata\":{\"promptTokenCount\":900,\"totalTokenCount\":900}}";

    public const string GeminiMaxTokens =
        "{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"{\\\"vend\"}]},\"finishReason\":\"MAX_TOKENS\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":900,\"candidatesTokenCount\":4096}}";

    public const string GeminiError = "{\"error\":{\"code\":429,\"message\":\"Resource has been exhausted\",\"status\":\"RESOURCE_EXHAUSTED\"}}";

    public const string ObjectSchema =
        "{\"type\":\"object\",\"properties\":{\"vendor\":{\"type\":[\"string\",\"null\"]},\"total\":{\"type\":[\"string\",\"null\"],\"format\":\"uuid\"}},\"required\":[\"vendor\"],\"additionalProperties\":false}";

    /// <summary>Seeds for the response-parser fuzz targets, by provider.</summary>
    public static IReadOnlyList<(string Name, string Body)> Seeds(string provider) => provider switch
    {
        "anthropic" => [("ok", AnthropicOk), ("refusal", AnthropicRefusal), ("truncated", AnthropicTruncated), ("error", AnthropicError)],
        "openai" => [("ok", OpenAiOk), ("refusal", OpenAiRefusal), ("length", OpenAiLength), ("fenced", OpenAiFenced), ("error", OpenAiError)],
        "gemini" => [("ok", GeminiOk), ("blocked", GeminiBlocked), ("max-tokens", GeminiMaxTokens), ("error", GeminiError)],
        "schema" => [("object", ObjectSchema), ("nested", "{\"type\":\"object\",\"properties\":{\"items\":{\"type\":\"array\",\"items\":{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\"}}}}},\"$defs\":{\"a\":{\"type\":\"string\"}}}")],
        _ => throw new ArgumentException("unknown provider", nameof(provider)),
    };
}
