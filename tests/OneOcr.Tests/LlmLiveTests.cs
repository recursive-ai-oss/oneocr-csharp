using System.Net.Http;
using System.Numerics;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using OneImage;
using OneImage.Text;
using OneOcr.Llm;

namespace OneOcr.Tests;

public sealed record LiveAnswer(string Text, string Amount);

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(LiveAnswer))]
internal sealed partial class LiveContext : JsonSerializerContext;

/// <summary>
/// One end-to-end call against a real OpenAI-compatible server (for example llama.cpp's llama-server with a small vision
/// model). Runs only when ONEOCR_LLM_LIVE_URL names the server's API root; ONEOCR_LLM_LIVE_MODEL names the model and
/// ONEOCR_LLM_LIVE_FONT a font file for drawing the test page. ONEOCR_LLM_LIVE_OUT, when set, receives a short report.
/// Nothing here uses a commercial service or a key.
/// </summary>
public class LlmLiveTests
{
    [Fact]
    public async Task A_local_openai_compatible_server_answers_a_structured_question_about_a_rendered_page()
    {
        string? url = Environment.GetEnvironmentVariable("ONEOCR_LLM_LIVE_URL");
        string? fontPath = Environment.GetEnvironmentVariable("ONEOCR_LLM_LIVE_FONT");
        Assert.SkipWhen(url is null || fontPath is null, "ONEOCR_LLM_LIVE_URL and ONEOCR_LLM_LIVE_FONT are not set.");
        string model = Environment.GetEnvironmentVariable("ONEOCR_LLM_LIVE_MODEL") ?? "local";

        const string Text = "TOTAL 427.50";
        Font font = new FontCollection().Add(fontPath!).CreateFont(72);
        var options = new TextOptions(font) { Origin = new Vector2(30, 30) };
        float advance = TextMeasurer.Measure(Text, options).Advance.Width;
        const int W = 640, H = 160;
        var px = new Rgba32[W * H];
        Array.Fill(px, new Rgba32(255, 255, 255, 255));
        Image image = Image.FromPixels<Rgba32>(px, W, H).DrawText(Text, options, new Rgba32(0, 0, 0, 255));

        // The page model a local reader would give for this image: two words with boxes.
        double split = 30 + (advance * 0.47);
        OcrWord Word(string t, double l, double r) => new(t, new OcrBox(l, 30, r, 110), 0.95f, 0.95f, OcrSource.Ocr, []);
        var words = new List<OcrWord> { Word("TOTAL", 30, split), Word("427.50", split + 20, 30 + advance) };
        var line = new OcrLine(Text, words, [0], new OcrBox(30, 30, 30 + advance, 110), 0.95f, OcrSource.Ocr);
        var page = new OcrPage(0, W, H, 96, [new OcrBlock(OcrBlockRole.Body, false, line.Box, [line], 0.95f)]);

        var provider = new OpenAiCompatibleProvider(new Uri(url!), model, images: new LlmImageOptions { MaxLongEdge = 1024 });
        using var client = new LlmClient(provider, new LlmOptions
        {
            Timeout = TimeSpan.FromMinutes(5),
            MaxConcurrency = 1,
            MaxOutputTokens = 128,
            PriceTable = LlmPriceTable.Default.With(new LlmModelPrice("openai-compatible", model, 0m, 0m)),
            SpendCapUsd = 0m,
        });

        LlmReadResult<LiveAnswer> r = await client.ExtractAsync(
            image, page, "Copy the text written on the page into 'text' and the number into 'amount'.", LiveContext.Default.LiveAnswer);

        Assert.False(string.IsNullOrWhiteSpace(r.Value!.Text));
        Assert.NotNull(r.Value!.Amount);
        Assert.Equal(0m, r.Read.CostUsd);
        Assert.Equal(2, r.Read.Pages.Count);
        string report = $"model={model} json={r.Read.Json} usage={r.Read.Usage} text_match={r.Value!.Text.Contains("427", StringComparison.Ordinal)}";
        if (Environment.GetEnvironmentVariable("ONEOCR_LLM_LIVE_OUT") is { } outPath)
        {
            await File.WriteAllTextAsync(outPath, report + "\n");
        }
    }
}
