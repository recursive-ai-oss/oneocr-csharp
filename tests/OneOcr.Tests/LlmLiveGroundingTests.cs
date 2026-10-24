using System.Globalization;
using System.Net.Http;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using OneImage;
using OneImage.Text;
using OneOcr.Llm;

namespace OneOcr.Tests;

/// <summary>Draws generated pages with a real font so a vision model can read them; word boxes come from the same font metrics.</summary>
internal sealed class LivePages
{
    private readonly Func<float, Font> _make;

    public LivePages(string fontPath)
    {
        var family = new FontCollection().Add(fontPath);
        _make = size => family.CreateFont(size);
    }

    private Font Font(float size) => _make(size);

    public float Measure(string text, float size) =>
        text.Length == 0 ? 0f : TextMeasurer.Measure(text, new TextOptions(Font(size))).Advance.Width;

    /// <summary>The page model (hidden lines left out) and the image (hidden lines drawn).</summary>
    public (Image Image, OcrPage Page) Render(IReadOnlyList<TestLine> lines)
    {
        OcrPage page = GroundingSupport.Layout(lines, Measure);
        OcrPage full = GroundingSupport.Layout(lines, Measure, includeHidden: true);
        int w = (int)full.Width, h = (int)full.Height;
        var px = new Rgba32[w * h];
        Array.Fill(px, new Rgba32(255, 255, 255, 255));
        Image image = Image.FromPixels<Rgba32>(px, w, h);
        double y = 20;
        foreach (TestLine tl in lines)
        {
            image = image.DrawText(tl.Text, new TextOptions(Font(tl.Size)) { Origin = new Vector2(20, (float)y) }, new Rgba32(0, 0, 0, 255));
            y += tl.Size * 1.6;
        }

        return (image, page);
    }
}

/// <summary>A tiny generated form with exact truth, for scoring a live model's answers against grounding.</summary>
internal sealed record LiveForm(string Vendor, string Number, string PrintedDate, string IsoDate, string PrintedTotal, decimal Total, TestLine[] Lines)
{
    public const string Schema =
        "{\"type\":\"object\",\"properties\":{\"vendor\":{\"type\":[\"string\",\"null\"]},\"invoice_number\":{\"type\":[\"string\",\"null\"]},"
        + "\"date\":{\"type\":[\"string\",\"null\"],\"format\":\"date\"},\"total\":{\"type\":[\"number\",\"null\"]}}}";

    public static List<LiveForm> Generate(int n, int seed)
    {
        var forms = new List<LiveForm>();
        string[] vendors = ["Northwind Trading", "Contoso GmbH", "Globex Supplies", "Initech Ltd", "Hooli Inc", "Wonka BV"];
        for (int i = 0; i < n; i++)
        {
            var rng = new Random(seed + i);
            string vendor = vendors[rng.Next(vendors.Length)];
            string number = "INV-" + rng.Next(1000, 9999).ToString(CultureInfo.InvariantCulture);
            DateTime d = new DateTime(2025, 1, 1).AddDays(rng.Next(0, 600));
            string printed = rng.Next(3) switch { 0 => d.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture), 1 => d.ToString("d MMMM yyyy", CultureInfo.InvariantCulture), _ => d.ToString("dd.MM.yyyy", CultureInfo.InvariantCulture) };
            decimal total = Math.Round((decimal)(rng.NextDouble() * 4000) + 5, 2);
            string tp = total.ToString("#,##0.00", CultureInfo.InvariantCulture);
            forms.Add(new LiveForm(vendor, number, printed, d.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture), tp, total,
                [new TestLine(vendor, 28), new TestLine("Invoice No: " + number, 24), new TestLine("Date: " + printed, 24), new TestLine("Total EUR " + tp, 28)]));
        }

        return forms;
    }

    /// <summary>Whether the value the model wrote is the true one, by a check independent of grounding.</summary>
    public bool IsRight(string field, GroundedField f) => field switch
    {
        "vendor" => string.Equals(f.Value.Replace(" ", ""), Vendor.Replace(" ", ""), StringComparison.OrdinalIgnoreCase),
        "invoice_number" => string.Equals(f.Value.Trim(), Number, StringComparison.OrdinalIgnoreCase),
        "date" => f.Value.Trim() == IsoDate,
        _ => decimal.TryParse(f.Value, NumberStyles.Float, CultureInfo.InvariantCulture, out decimal v) && v == Total,
    };
}

internal static class LiveRun
{
    /// <summary>Runs the injection set and the forms against a client; returns a Markdown report. Never throws for a provider failure: the case is reported as NoValue.</summary>
    public static async Task<string> RunAsync(LlmClient client, LivePages pages, string title, int forms, Func<LlmReadOptions, LlmReadOptions>? tweak = null)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"### {title}");
        sb.AppendLine("| attack | outcome | field value | state | calls |");
        sb.AppendLine("|---|---|---|---|---|");
        foreach (InjectionCase c in InjectionSet.Cases)
        {
            (Image image, OcrPage page) = pages.Render(c.Lines);
            var opts = new LlmReadOptions { Outputs = LlmOutputs.Json, JsonSchema = InjectionSet.Schema, Instruction = InjectionSet.Instruction };
            try
            {
                LlmReadResult r = await client.ReadAsync(image, page, tweak?.Invoke(opts) ?? opts);
                GroundedField? f = r.Fields.FirstOrDefault(x => x.Path == "/" + c.Field);
                sb.AppendLine($"| {c.Name} | {InjectionSet.Classify(r, c)} | {Short(f?.Value)} | {f?.State.ToString() ?? "-"} | {r.Calls} |");
            }
            catch (Exception e) when (e is OcrProviderException or OcrBudgetException or OcrLimitException)
            {
                sb.AppendLine($"| {c.Name} | NoValue | - | {e.GetType().Name} | - |");
            }
        }

        var failureKinds = new List<string>();
        int rightGrounded = 0, wrongGrounded = 0, rightUngrounded = 0, wrongUngrounded = 0, derived = 0, nullValue = 0, failures = 0, total = 0;
        foreach (LiveForm form in LiveForm.Generate(forms, 77))
        {
            (Image image, OcrPage page) = pages.Render(form.Lines);
            var opts = new LlmReadOptions { Outputs = LlmOutputs.Json, JsonSchema = LiveForm.Schema, Instruction = "Extract the vendor, invoice number, date and total of this invoice." };
            LlmReadResult r;
            try
            {
                r = await client.ReadAsync(image, page, tweak?.Invoke(opts) ?? opts);
            }
            catch (Exception e) when (e is OcrProviderException or OcrBudgetException or OcrLimitException)
            {
                failures++;
                failureKinds.Add(e is OcrProviderException pe ? "provider:" + pe.Failure : e.GetType().Name);
                continue;
            }

            foreach (string field in new[] { "vendor", "invoice_number", "date", "total" })
            {
                total++;
                GroundedField? f = r.Fields.FirstOrDefault(x => x.Path == "/" + field);
                if (f is null)
                {
                    nullValue++;
                    continue;
                }

                bool right = form.IsRight(field, f);
                switch (f.State)
                {
                    case GroundingState.Grounded or GroundingState.GroundedNormalized:
                        if (right)
                        {
                            rightGrounded++;
                        }
                        else
                        {
                            wrongGrounded++;
                        }

                        break;
                    case GroundingState.Ungrounded:
                        if (right)
                        {
                            rightUngrounded++;
                        }
                        else
                        {
                            wrongUngrounded++;
                        }

                        break;
                    default:
                        derived++;
                        break;
                }
            }
        }

        sb.AppendLine();
        sb.AppendLine($"forms: {forms} pages, {failures} failed calls ({string.Join(", ", failureKinds)}), {total} fields returned a slot, {nullValue} null");
        sb.AppendLine("| | value right | value wrong |");
        sb.AppendLine("|---|---|---|");
        sb.AppendLine($"| grounded (either kind) | {rightGrounded} | {wrongGrounded} |");
        sb.AppendLine($"| ungrounded | {rightUngrounded} | {wrongUngrounded} |");
        sb.AppendLine($"| derived | {derived} (not scored) | |");
        int g = rightGrounded + wrongGrounded, wrong = wrongGrounded + wrongUngrounded;
        sb.AppendLine($"grounded-is-right precision {(g == 0 ? "n/a" : ((double)rightGrounded / g).ToString("P1", CultureInfo.InvariantCulture))}; wrong-value-is-ungrounded recall {(wrong == 0 ? "n/a" : ((double)wrongUngrounded / wrong).ToString("P1", CultureInfo.InvariantCulture))}");
        return sb.ToString();
    }

    private static string Short(string? s) => s is null ? "-" : s.Length > 40 ? s[..40] + "…" : s;
}

/// <summary>
/// The injection set and a few generated forms against a real OpenAI-compatible server (llama.cpp's llama-server with a small vision
/// model). Runs only when ONEOCR_LLM_LIVE_URL, ONEOCR_LLM_LIVE_FONT are set; ONEOCR_LLM_LIVE_MODEL names the model and
/// ONEOCR_LLM_GROUNDING_OUT receives the report. No commercial service and no key.
/// </summary>
public class LlmLiveGroundingTests
{
    [Fact]
    public async Task A_local_model_is_scored_on_the_injection_set_and_generated_forms()
    {
        string? url = Environment.GetEnvironmentVariable("ONEOCR_LLM_LIVE_URL");
        string? font = Environment.GetEnvironmentVariable("ONEOCR_LLM_LIVE_FONT");
        Assert.SkipWhen(url is null || font is null, "ONEOCR_LLM_LIVE_URL and ONEOCR_LLM_LIVE_FONT are not set.");
        string model = Environment.GetEnvironmentVariable("ONEOCR_LLM_LIVE_MODEL") ?? "local";
        int forms = int.Parse(Environment.GetEnvironmentVariable("ONEOCR_LLM_LIVE_FORMS") ?? "8", CultureInfo.InvariantCulture);
        var provider = new OpenAiCompatibleProvider(new Uri(url!), model, images: new LlmImageOptions { MaxLongEdge = 1024 });
        using var client = new LlmClient(provider, new LlmOptions
        {
            Timeout = TimeSpan.FromMinutes(5),
            MaxConcurrency = 1,
            MaxOutputTokens = 512,
            PriceTable = LlmPriceTable.Default.With(new LlmModelPrice("openai-compatible", model, 0m, 0m)),
            SpendCapUsd = 0m,
        });
        string report = await LiveRun.RunAsync(client, new LivePages(font!), $"live local model {model} through the OpenAI-compatible provider", forms);
        Environment.GetEnvironmentVariable("ONEOCR_LLM_GROUNDING_OUT").AppendReport(report);
        Assert.Contains("| attack |", report, StringComparison.Ordinal);
    }
}

/// <summary>
/// A hand-run against real providers, for after review: set ONEOCR_HANDRUN=1, ONEOCR_LLM_LIVE_FONT, and for each provider you want its
/// key variable (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, AZURE_OPENAI_API_KEY plus ONEOCR_HANDRUN_AZURE_ENDPOINT)
/// and its model (ONEOCR_HANDRUN_ANTHROPIC_MODEL, _OPENAI_MODEL, _GEMINI_MODEL; _AZURE_MODEL holds the deployment name). Keys are read at
/// run time through each provider's FromEnvironment; a provider without a key or model is reported as skipped. Spend is capped at
/// 2 USD per provider by the client's own cost controls (SpendCapUsd, plus 0.25 USD per call), counting the re-ask. The report
/// (ONEOCR_HANDRUN_OUT) holds provider names, models, outcomes and spend, never keys or request text.
/// </summary>
public class LlmProviderHandRun
{
    private const decimal CapUsd = 2m;

    [Fact]
    public async Task Providers_with_keys_run_the_injection_set_and_forms_under_a_two_dollar_cap()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("ONEOCR_HANDRUN") == "1", "ONEOCR_HANDRUN=1 is not set.");
        string? font = Environment.GetEnvironmentVariable("ONEOCR_LLM_LIVE_FONT");
        Assert.SkipWhen(font is null, "ONEOCR_LLM_LIVE_FONT is not set.");
        var pages = new LivePages(font!);
        var report = new StringBuilder();
        int ran = 0;
        foreach ((string name, Func<string, LlmProvider?> make, string keyVar) in Providers())
        {
            string? model = Environment.GetEnvironmentVariable($"ONEOCR_HANDRUN_{name.ToUpperInvariant()}_MODEL");
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(keyVar)) || string.IsNullOrEmpty(model))
            {
                report.AppendLine($"### {name}: skipped ({(string.IsNullOrEmpty(Environment.GetEnvironmentVariable(keyVar)) ? "no key" : "no model")})");
                continue;
            }

            LlmProvider? provider = make(model);
            if (provider is null)
            {
                report.AppendLine($"### {name}: skipped (incomplete configuration)");
                continue;
            }

            using var client = new LlmClient(provider, new LlmOptions { SpendCapUsd = CapUsd, MaxCostPerCallUsd = 0.25m, MaxConcurrency = 1, MaxOutputTokens = 1024 });
            string section;
            try
            {
                section = await LiveRun.RunAsync(client, pages, $"{name} {model}", forms: 6);
            }
            catch (OcrBudgetException e)
            {
                section = $"### {name} {model}: stopped by the cost controls ({e.Kind})";
            }

            report.AppendLine(section);
            report.AppendLine($"spent {client.SpentUsd.ToString("0.0000", CultureInfo.InvariantCulture)} USD of the {CapUsd} USD cap");
            ran++;
        }

        Environment.GetEnvironmentVariable("ONEOCR_HANDRUN_OUT").AppendReport(report.ToString());
        Assert.SkipWhen(ran == 0, "No provider had a key and a model.");
    }

    private static IEnumerable<(string Name, Func<string, LlmProvider?> Make, string KeyVar)> Providers()
    {
        yield return ("anthropic", static m => AnthropicProvider.FromEnvironment(m), "ANTHROPIC_API_KEY");
        yield return ("openai", static m => OpenAiProvider.FromEnvironment(m), "OPENAI_API_KEY");
        yield return ("gemini", static m => GeminiProvider.FromEnvironment(m), "GEMINI_API_KEY");
        yield return ("azure", static m =>
        {
            string? endpoint = Environment.GetEnvironmentVariable("ONEOCR_HANDRUN_AZURE_ENDPOINT");
            return endpoint is null ? null : AzureOpenAiProvider.FromEnvironment(new Uri(endpoint), m);
        }, "AZURE_OPENAI_API_KEY");
    }
}
