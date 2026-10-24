using System.Text;
using System.Text.Json;
using OneOcr;
using OneOcr.Llm;

namespace OneOcr.Fuzz;

/// <summary>
/// Provider response bytes through each provider's parser, and schema text through the schema checks. Only the typed exceptions
/// may come out; whatever parses must hand back a JSON object within the depth limit, and error-message extraction never throws.
/// </summary>
internal static class LlmTarget
{
    private static readonly OcrLimits Limits = new() { MaxLlmJsonDepth = 8, MaxLlmSchemaBytes = 4096 };
    private static readonly AnthropicProvider Anthropic = new("key", "model");
    private static readonly OpenAiProvider OpenAi = new("key", "model");
    private static readonly GeminiProvider Gemini = new("key", "model");

    public static long Parsed;

    public static void RunAnthropic(ReadOnlySpan<byte> data) => Reply(Anthropic, data);

    public static void RunOpenAi(ReadOnlySpan<byte> data) => Reply(OpenAi, data);

    public static void RunGemini(ReadOnlySpan<byte> data) => Reply(Gemini, data);

    public static void RunSchema(ReadOnlySpan<byte> data)
    {
        try
        {
            var schema = SchemaTools.Parse(Encoding.UTF8.GetString(data), Limits);
            SchemaTools.ForStrict(schema);
            try
            {
                SchemaTools.ForGemini(schema);
            }
            catch (ArgumentException)
            {
            }

            Parsed++;
        }
        catch (ArgumentException)
        {
        }
        catch (OcrLimitException)
        {
        }
    }

    private static void Reply(LlmProvider provider, ReadOnlySpan<byte> data)
    {
        if (ProviderJson.ErrorMessage(data).Length > 300)
        {
            throw new InvalidOperationException("error message not bounded");
        }

        ProviderReply reply;
        try
        {
            reply = provider.ParseReply(data, Limits);
        }
        catch (OcrProviderException)
        {
            return;
        }
        catch (OcrLimitException)
        {
            return;
        }

        Parsed++;
        using JsonDocument doc = JsonDocument.Parse(reply.Json, new JsonDocumentOptions { MaxDepth = Limits.MaxLlmJsonDepth });
        if (doc.RootElement.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidOperationException("answer is not an object");
        }
    }
}
