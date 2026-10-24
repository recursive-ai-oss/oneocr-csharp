using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using OneImage;
using OneOcr.Llm;

namespace OneOcr.Tests;

/// <summary>An API key must never reach an exception, a result, a cache entry or a printed provider, whatever a server echoes.</summary>
public class LlmSecretTests
{
    private const string Key = "sk-test-AbCdEfGh12345678-XYZ";

    private static bool Leaks(string? text)
    {
        if (text is null)
        {
            return false;
        }

        for (int i = 0; i + 8 <= Key.Length; i++)
        {
            if (text.Contains(Key.AsSpan(i, 8).ToString(), StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    private static void AssertClean(Exception e)
    {
        for (Exception? x = e; x is not null; x = x.InnerException)
        {
            Assert.False(Leaks(x.Message), "message leaks: " + x.Message);
            Assert.False(Leaks(x.ToString()), "ToString leaks");
        }
    }

    private sealed class Echo(Func<int, HttpResponseMessage> respond) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(respond(1));
    }

    private sealed class Throwing : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            throw new HttpRequestException("connection to proxy failed for " + Key, new InvalidOperationException("auth " + Key));
    }

    public static IEnumerable<object[]> Providers() => new[] { "anthropic", "openai", "compatible", "azure", "gemini" }.Select(p => new object[] { p });

    private static LlmProvider Make(string kind, HttpMessageHandler h)
    {
        var http = new HttpClient(h);
        return kind switch
        {
            "anthropic" => new AnthropicProvider(Key, "test-model", http: http),
            "openai" => new OpenAiProvider(Key, "test-model", http: http),
            "compatible" => new OpenAiCompatibleProvider(new Uri("http://localhost:1/v1"), "test-model", Key, http),
            "azure" => new AzureOpenAiProvider(new Uri("https://x.openai.azure.com"), "dep", Key, http: http),
            _ => new GeminiProvider(Key, "test-model", http: http),
        };
    }

    private static LlmClient Client(LlmProvider p, LlmResultCache? cache = null) => new(p, new LlmOptions { MaxRetries = 0, Cache = cache });

    private static (Image, OcrPage) Page()
    {
        var px = new Gray8[100 * 60];
        Array.Fill(px, new Gray8(255));
        var w = new OcrWord("hi", new OcrBox(5, 5, 20, 15), 0.9f, 0.9f, OcrSource.Ocr, []);
        var line = new OcrLine("hi", [w], [0], w.Box, 0.9f, OcrSource.Ocr);
        return (Image.FromPixels<Gray8>(px, 100, 60), new OcrPage(0, 100, 60, 72, [new OcrBlock(OcrBlockRole.Body, false, w.Box, [line], 0.9f)]));
    }

    private static HttpResponseMessage WithEchoHeaders(HttpStatusCode status, string body)
    {
        var r = new HttpResponseMessage(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        r.Headers.TryAddWithoutValidation("x-echo", Key);
        r.Headers.TryAddWithoutValidation("www-authenticate", "Bearer " + Key);
        r.ReasonPhrase = "bad key " + Key;
        return r;
    }

    [Theory]
    [MemberData(nameof(Providers))]
    public async Task Echoed_keys_in_error_bodies_headers_and_json_fields_never_reach_the_exception(string kind)
    {
        string[] bodies =
        [
            "{\"error\":{\"message\":\"Invalid key " + Key + "\",\"key\":\"  " + Key + " \"}}",
            "Bad request for key EfGh12345678 and also " + Key + "\n" + new string('z', 600),
            "{\"type\":\"error\",\"error\":{\"message\":\"Incorrect API key provided: " + Key + "\"}}",
            "{\"message\":\"" + Key + "\"}",
        ];
        foreach (string body in bodies)
        {
            foreach (HttpStatusCode status in new[] { HttpStatusCode.BadRequest, HttpStatusCode.OK })
            {
                using var client = Client(Make(kind, new Echo(_ => WithEchoHeaders(status, body))));
                var (img, page) = Page();
                var e = await Assert.ThrowsAsync<OcrProviderException>(() => client.ExtractAsync(img, page, "x", LlmFixtures.ObjectSchema));
                AssertClean(e);
                Assert.True(e.Message.Length < 600, "echoed text is capped");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Providers))]
    public async Task Transport_errors_that_mention_the_key_are_scrubbed_and_drop_the_inner_exception(string kind)
    {
        using var client = Client(Make(kind, new Throwing()));
        var (img, page) = Page();
        var e = await Assert.ThrowsAsync<OcrProviderException>(() => client.ExtractAsync(img, page, "x", LlmFixtures.ObjectSchema));
        Assert.Equal(LlmFailure.Transport, e.Failure);
        AssertClean(e);
        Assert.Contains("[redacted]", e.Message);
    }

    [Theory]
    [InlineData("anthropic")]
    [InlineData("openai")]
    [InlineData("gemini")]
    public async Task A_model_that_repeats_the_key_in_its_answer_does_not_get_it_into_the_result_or_cache(string kind)
    {
        string ok = kind switch
        {
            "anthropic" => LlmFixtures.AnthropicOk,
            "openai" => LlmFixtures.OpenAiOk,
            _ => LlmFixtures.GeminiOk,
        };
        ok = ok.Replace("Acme", Key, StringComparison.Ordinal);
        var cache = new LlmResultCache();
        using var client = Client(Make(kind, new Echo(_ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(ok, Encoding.UTF8, "application/json") })), cache);
        var (img, page) = Page();
        LlmExtraction r = await client.ExtractAsync(img, page, "x", LlmFixtures.ObjectSchema);
        Assert.False(Leaks(r.Json));
        Assert.Contains("[redacted]", r.Json);
        LlmExtraction cached = await client.ExtractAsync(img, page, "x", LlmFixtures.ObjectSchema);
        Assert.True(cached.FromCache);
        Assert.False(Leaks(cached.Json));
        Assert.All(r.Warnings, w => Assert.False(Leaks(w)));
    }

    [Theory]
    [MemberData(nameof(Providers))]
    public void Printing_a_provider_or_the_client_shows_no_key(string kind)
    {
        LlmProvider p = Make(kind, new Throwing());
        Assert.False(Leaks(p.ToString()));
        Assert.False(Leaks($"{p}"));
        using var client = new LlmClient(p, new LlmOptions());
        Assert.False(Leaks(client.ToString()));
        Assert.False(Leaks(client.Provider.ToString()));
        Assert.False(Leaks(new LlmOptions().ToString()));
        // Nothing public exposes the key either.
        foreach (var prop in typeof(LlmProvider).GetProperties())
        {
            if (prop.GetIndexParameters().Length == 0 && prop.GetValue(p) is { } v)
            {
                Assert.False(Leaks(v.ToString()), prop.Name);
            }
        }
    }

    [Fact]
    public void Scrub_removes_the_key_padded_keys_and_eight_character_pieces()
    {
        var secret = new Secret("  " + Key + "\t");
        Assert.Equal("a [redacted] b", secret.Scrub("a " + Key + " b"));
        Assert.Equal("see [redacted] now", secret.Scrub("see EfGh12345678 now"));
        Assert.Equal("short EfGh123 stays", secret.Scrub("short EfGh123 stays"));
        Assert.Equal("[redacted]", secret.ToString());
        Assert.True(secret.Scrub(new string('q', 5000)).Length <= 400);
        Assert.Equal(5000, secret.ScrubAll(new string('q', 5000)).Length);
        Assert.Equal("nothing here", new Secret("   ").Scrub("nothing here"));
    }
}
