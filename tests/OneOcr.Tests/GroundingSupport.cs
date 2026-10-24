using System.Threading.Tasks;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using OneImage;
using OneOcr.Llm;

namespace OneOcr.Tests;

/// <summary>A generated page: lines of text laid out into a page model (and optionally rendered by the caller).</summary>
public sealed record TestLine(string Text, float Size = 20f, float Conf = 0.97f, bool Hidden = false);

public static class GroundingSupport
{
    public const string Key = "sk-test-0123456789abcdef";

    /// <summary>Lays out lines top to bottom into a page model; <paramref name="measure"/> gives the width of text at a size.</summary>
    public static OcrPage Layout(IReadOnlyList<TestLine> lines, Func<string, float, float>? measure = null, double width = 800, bool includeHidden = false)
    {
        measure ??= static (s, size) => s.Length * size * 0.55f;
        var blockLines = new List<OcrLine>();
        double y = 20;
        foreach (TestLine tl in lines)
        {
            double x = 20;
            var words = new List<OcrWord>();
            string prefix = "";
            foreach (string w in tl.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                double left = x + measure(prefix, tl.Size);
                double right = left + measure(w, tl.Size);
                words.Add(new OcrWord(w, new OcrBox(left, y, right, y + (tl.Size * 1.2)), tl.Conf, tl.Conf, OcrSource.Ocr, []));
                prefix += w + " ";
            }

            if (words.Count > 0 && (includeHidden || !tl.Hidden))
            {
                var box = new OcrBox(words[0].Box.Left, y, words[^1].Box.Right, y + (tl.Size * 1.2));
                blockLines.Add(new OcrLine(string.Join(' ', words.Select(static w => w.Text)), words, [0], box, tl.Conf, OcrSource.Ocr));
            }

            y += tl.Size * 1.6;
        }

        var block = new OcrBlock(OcrBlockRole.Body, false, new OcrBox(0, 0, width, y), blockLines, 0.95f);
        return new OcrPage(0, width, y + 20, 96, [block]);
    }

    public static Image Blank(OcrPage page)
    {
        int w = (int)page.Width, h = (int)page.Height;
        var px = new Gray8[w * h];
        Array.Fill(px, new Gray8(255));
        return Image.FromPixels<Gray8>(px, w, h);
    }

    public static string Id(OcrPage page, string word, int occurrence = 0)
    {
        int n = 0;
        for (int i = 0; i < page.Words.Count; i++)
        {
            if (page.Words[i].Text == word && n++ == occurrence)
            {
                return "w" + i;
            }
        }

        throw new InvalidOperationException("no word " + word);
    }

    /// <summary>{"value":..,"words":[..],"derived":..} for a value given as JSON text.</summary>
    public static string V(string valueJson, string[] ids, bool derived = false) =>
        "{\"value\":" + valueJson + ",\"words\":[" + string.Join(",", ids.Select(static i => "\"" + i + "\"")) + "],\"derived\":" + (derived ? "true" : "false") + "}";

    public static string S(string s) => JsonSerializer.Serialize(s);

    /// <summary>The same answer in each provider's documented reply shape.</summary>
    public static string Shape(string provider, string answerJson) => provider switch
    {
        "anthropic" => "{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"extraction\",\"input\":" + answerJson
            + "}],\"stop_reason\":\"tool_use\",\"usage\":{\"input_tokens\":1000,\"output_tokens\":100}}",
        "gemini" => "{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":" + S(answerJson) + "}]},\"finishReason\":\"STOP\",\"index\":0}],"
            + "\"usageMetadata\":{\"promptTokenCount\":1000,\"candidatesTokenCount\":100}}",
        _ => "{\"id\":\"c1\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" + S(answerJson)
            + "},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":100}}",
    };

    public static readonly string[] Providers = ["anthropic", "openai", "openai-compatible", "gemini", "azure-openai"];

    public static LlmProvider Provider(string name, HttpClient http) => name switch
    {
        "anthropic" => new AnthropicProvider(Key, "test", http: http),
        "openai" => new OpenAiProvider(Key, "test", http: http),
        "openai-compatible" => new OpenAiCompatibleProvider(new Uri("http://localhost:8080/v1/"), "test", http: http),
        "gemini" => new GeminiProvider(Key, "test", http: http),
        "azure-openai" => new AzureOpenAiProvider(new Uri("https://example.openai.azure.com/"), "test", Key, http: http),
        _ => throw new ArgumentException(name),
    };

    public static LlmOptions Options(decimal? cap = null, LlmResultCache? cache = null) => new()
    {
        SpendCapUsd = cap,
        PriceTable = LlmPriceTable.Default.With(
            new LlmModelPrice("anthropic", "test", 3m, 15m), new LlmModelPrice("openai", "test", 2m, 8m), new LlmModelPrice("gemini", "test", 1m, 4m),
            new LlmModelPrice("azure-openai", "test", 2m, 8m), new LlmModelPrice("openai-compatible", "test", 0m, 0m)),
        Cache = cache,
        RetryBaseDelay = TimeSpan.FromMilliseconds(1),
        MaxRetryDelay = TimeSpan.FromMilliseconds(5),
        TimeProvider = new FixedClock(new DateTimeOffset(2026, 10, 24, 0, 0, 0, TimeSpan.Zero)),
    };

    private sealed class FixedClock(DateTimeOffset now) : TimeProvider
    {
        public override DateTimeOffset GetUtcNow() => now;
    }
}

/// <summary>Replays queued replies in order, remembers every request body, and can fail on any request.</summary>
public sealed class ScriptedHandler : HttpMessageHandler
{
    private readonly Queue<(HttpStatusCode Status, string Body)> _replies = new();
    private readonly bool _failOnAny;

    public ScriptedHandler(bool failOnAny = false) => _failOnAny = failOnAny;

    public List<string> Requests { get; } = [];

    public ScriptedHandler Then(string body, HttpStatusCode status = HttpStatusCode.OK)
    {
        _replies.Enqueue((status, body));
        return this;
    }

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        if (_failOnAny)
        {
            throw new InvalidOperationException("a request was sent: " + request.RequestUri);
        }

        string body = request.Content is null ? "" : await request.Content.ReadAsStringAsync(cancellationToken);
        lock (Requests)
        {
            Requests.Add(body);
        }

        (HttpStatusCode status, string reply) = _replies.Count > 1 ? _replies.Dequeue() : _replies.Peek();
        return new HttpResponseMessage(status) { Content = new StringContent(reply, Encoding.UTF8, "application/json") };
    }
}
