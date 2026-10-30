using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using OneImage;
using OneOcr.Llm;
using OneOcr.Tests;

namespace OneOcr.Fuzz;

/// <summary>
/// What a model says, through the whole read path: the first byte picks the requested outputs and the rest is the model's answer
/// text, delivered in an OpenAI-shaped reply. The answer is parsed, aligned with a small page, grounded against it and re-asked once
/// (the fake server gives the same answer again). Only the typed exceptions may come out, and the page models that come back must
/// write to JSON that reads back.
/// </summary>
internal static class LlmAnswerTarget
{
    private const string Schema =
        "{\"type\":\"object\",\"properties\":{\"vendor\":{\"type\":\"string\"},\"total\":{\"type\":\"number\"},\"paid\":{\"type\":\"boolean\"},"
        + "\"lines\":{\"type\":\"array\",\"items\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"qty\":{\"type\":\"integer\"}}}}}}";

    private static readonly OcrLimits Limits = new() { MaxLlmJsonDepth = 12, MaxLlmResponseBytes = 1 << 16 };

    private static readonly OcrPage Page = GroundingSupport.Layout(
    [
        new TestLine("Acme Corp invoice"), new TestLine("Total due 12.50 paid"), new TestLine("Widget 2", Conf: 0.4f), new TestLine("Gadget 1"),
    ]);

    private static readonly Image Blank = GroundingSupport.Blank(Page);

    /// <summary>Inputs that were read to the end (progress statistic for local runs).</summary>
    public static long Parsed;

    public static void Run(ReadOnlySpan<byte> data)
    {
        if (data.Length == 0)
        {
            return;
        }

        var outputs = (LlmOutputs)((data[0] & 7) == 0 ? 7 : data[0] & 7);
        bool json = (outputs & LlmOutputs.Json) != 0;
        string answer = Encoding.UTF8.GetString(data[1..]);
        string reply = "{\"id\":\"c1\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":"
            + JsonSerializer.Serialize(answer) + "},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":100}}";

        using var http = new HttpClient(new Repeat(reply));
        var options = new LlmOptions
        {
            Limits = Limits,
            PriceTable = LlmPriceTable.Default.With(new LlmModelPrice("openai", "model", 1m, 1m)),
            RetryBaseDelay = TimeSpan.FromMilliseconds(1),
            MaxRetryDelay = TimeSpan.FromMilliseconds(1),
        };
        using var client = new LlmClient(new OpenAiProvider("key", "model", http: http), options);
        var read = new LlmReadOptions { Outputs = outputs, JsonSchema = json ? Schema : null, UngroundedPolicy = (data[0] & 8) != 0 ? UngroundedPolicy.Null : UngroundedPolicy.Flag };
        LlmReadResult result;
        try
        {
            result = client.ReadAsync(Blank, Page, read).GetAwaiter().GetResult();
        }
        catch (OcrProviderException)
        {
            return;
        }
        catch (OcrLimitException)
        {
            return;
        }
        catch (OcrBudgetException)
        {
            return;
        }

        Parsed++;
        foreach (OcrPage p in result.Pages)
        {
            string text = OcrPageJson.ToJson([p]);
            _ = OcrPageJson.Parse(text);
        }
    }

    public static void WriteSeeds(string dir)
    {
        Directory.CreateDirectory(dir);
        void W(string name, int flags, string answer) => File.WriteAllBytes(Path.Combine(dir, name + ".seed"), [(byte)flags, .. Encoding.UTF8.GetBytes(answer)]);
        W("text", 1, "{\"pages\":[\"Acme Corp invoice\\nTotal due 12.50 paid\\nWidget 2\\nGadget 1\"]}");
        W("markdown", 2, "{\"markdown\":[\"# Acme Corp invoice\\n\\n| Item | Qty |\\n| --- | --- |\\n| Widget | 2 |\"]}");
        W("json", 4, "{\"vendor\":{\"value\":\"Acme Corp\",\"words\":[\"w0\",\"w1\"],\"derived\":false},\"total\":{\"value\":12.5,\"words\":[\"w5\"],\"derived\":false},"
            + "\"paid\":{\"value\":true,\"words\":[\"w6\"],\"derived\":false},\"lines\":[{\"name\":{\"value\":\"Widget\",\"words\":[\"w7\"],\"derived\":false},\"qty\":{\"value\":2,\"words\":[\"w8\"],\"derived\":false}}]}");
        W("ungrounded", 4 | 8, "{\"vendor\":{\"value\":\"Globex\",\"words\":[\"w99\"],\"derived\":false},\"total\":{\"value\":13.5,\"words\":[],\"derived\":true}}");
        W("all", 7, "{\"pages\":[\"Acme\"],\"markdown\":[\"# Acme\"],\"vendor\":{\"value\":\"Acme\",\"words\":[\"w0\"],\"derived\":false}}");
    }

    private sealed class Repeat(string body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(body, Encoding.UTF8, "application/json") });
    }
}
