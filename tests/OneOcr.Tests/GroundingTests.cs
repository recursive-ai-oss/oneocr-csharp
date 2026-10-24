using System.Threading.Tasks;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using OneImage;
using OneOcr.Llm;

namespace OneOcr.Tests;

public class GroundingTests
{
    private const string Schema =
        "{\"type\":\"object\",\"properties\":{\"number\":{\"type\":[\"string\",\"null\"]},\"date\":{\"type\":[\"string\",\"null\"],\"format\":\"date\"},"
        + "\"total\":{\"type\":[\"number\",\"null\"]},\"currency\":{\"type\":[\"string\",\"null\"]},\"paid\":{\"type\":[\"boolean\",\"null\"]},"
        + "\"items\":{\"type\":\"array\",\"items\":{\"type\":\"object\",\"properties\":{\"desc\":{\"type\":\"string\"},\"amount\":{\"type\":\"number\"}}}}}}";

    private static readonly OcrPage Page = GroundingSupport.Layout(
    [
        new TestLine("Invoice No: INV-1001"),
        new TestLine("Date: 03.10.2026"),
        new TestLine("Total EUR 1.234,50"),
        new TestLine("Widget 1.000,00 Gadget 234,50"),
        new TestLine("Tax 19% 197,06"),
    ]);

    private static string Id(string w, int n = 0) => GroundingSupport.Id(Page, w, n);

    private static string Data(params (string Name, string V)[] props) =>
        "{\"data\":{" + string.Join(",", props.Select(static p => "\"" + p.Name + "\":" + p.V)) + ",\"items\":[]}}";

    private static string Wrap(string provider, string a) => GroundingSupport.Shape(provider, a);

    private static async Task<(LlmReadResult R, ScriptedHandler H)> Run(
        string first, string? second = null, LlmReadOptions? o = null, decimal? cap = null, string provider = "openai")
    {
        var h = new ScriptedHandler().Then(Wrap(provider, first));
        if (second is not null)
        {
            h.Then(Wrap(provider, second));
        }

        using var http = new HttpClient(h);
        using var client = new LlmClient(GroundingSupport.Provider(provider, http), GroundingSupport.Options(cap));
        LlmReadOptions opts = o ?? new LlmReadOptions { Outputs = LlmOutputs.Json, JsonSchema = Schema };
        LlmReadResult r = await client.ReadAsync(GroundingSupport.Blank(Page), Page, opts);
        return (r, h);
    }

    [Theory]
    [InlineData("1.234,50", "1234.50", true)]
    [InlineData("€ 1.234,5", "1234.50", true)]
    [InlineData("1,234.50", "1234.50", true)]
    [InlineData("1 234,50", "1234.50", true)]
    [InlineData("(12.50)", "-12.50", true)]
    [InlineData("12.50-", "-12.50", true)]
    [InlineData("USD 3", "3", true)]
    [InlineData("1,23,4", "", false)]
    [InlineData("12a", "", false)]
    [InlineData("1.234.567", "1234567", true)]
    [InlineData("0,125", "0.125", true)]
    public void Amounts_parse_and_canonicalize(string text, string canonical, bool ok)
    {
        Assert.Equal(ok, GroundingNormalize.TryParseAmount(text, LlmNumberStyle.Auto, out ParsedAmount a));
        if (ok)
        {
            Assert.Equal(canonical, GroundingNormalize.CanonicalAmount(a.Value));
        }
    }

    [Fact]
    public void A_single_separator_with_three_digits_follows_the_page_style()
    {
        Assert.True(GroundingNormalize.TryParseAmount("1,234", LlmNumberStyle.PointDecimal, out ParsedAmount a));
        Assert.Equal(1234m, a.Value);
        Assert.True(GroundingNormalize.TryParseAmount("1,234", LlmNumberStyle.CommaDecimal, out a));
        Assert.Equal(1.234m, a.Value);
        Assert.Equal(LlmNumberStyle.CommaDecimal, GroundingNormalize.DetectStyle(["12,50", "3,99", "1.000,00"]));
        Assert.Equal(LlmNumberStyle.PointDecimal, GroundingNormalize.DetectStyle(["12.50", "1,000.00"]));
    }

    [Theory]
    [InlineData("2026-10-03", "2026-10-03")]
    [InlineData("13.10.2026", "2026-10-13")]
    [InlineData("3 October 2026", "2026-10-03")]
    [InlineData("Oct 3, 2026", "2026-10-03")]
    [InlineData("3rd of October 2026", "2026-10-03")]
    [InlineData("3. Oktober 2026", "2026-10-03")]
    [InlineData("13/10/26", "2026-10-13")]
    [InlineData("10/13/2026", "2026-10-13")]
    [InlineData("20261003", "2026-10-03")]
    [InlineData("Oct 3", "--10-03")]
    public void Dates_are_read_into_iso(string text, string iso)
    {
        List<ParsedDate> d = GroundingNormalize.ParseDate(text, LlmDateOrder.Auto);
        Assert.Single(d);
        Assert.Equal(iso, GroundingNormalize.CanonicalDate(d[0]));
    }

    [Fact]
    public void An_ambiguous_numeric_date_has_both_readings_unless_the_order_is_known()
    {
        Assert.Equal(2, GroundingNormalize.ParseDate("03/10/2026", LlmDateOrder.Auto).Count);
        Assert.Equal("2026-10-03", GroundingNormalize.CanonicalDate(GroundingNormalize.ParseDate("03/10/2026", LlmDateOrder.DayFirst).Single()));
        Assert.Equal("2026-03-10", GroundingNormalize.CanonicalDate(GroundingNormalize.ParseDate("03/10/2026", LlmDateOrder.MonthFirst).Single()));
        Assert.Empty(GroundingNormalize.ParseDate("31.02.2026", LlmDateOrder.Auto));
        Assert.Empty(GroundingNormalize.ParseDate("Invoice date", LlmDateOrder.Auto));
        Assert.Equal(LlmDateOrder.DayFirst, GroundingNormalize.DetectOrder(["13.10.2026", "03/10/2026"]));
    }

    [Fact]
    public async Task Printed_values_are_grounded_normalized_values_say_so_and_everything_has_boxes()
    {
        string answer = Data(
            ("number", GroundingSupport.V("\"INV-1001\"", [Id("INV-1001")])),
            ("date", GroundingSupport.V("\"2026-10-03\"", [Id("Date:"), Id("03.10.2026")])),
            ("total", GroundingSupport.V("1234.5", [Id("Total"), Id("EUR"), Id("1.234,50")])),
            ("currency", GroundingSupport.V("\"EUR\"", [Id("EUR")])),
            ("paid", GroundingSupport.V("null", [])));
        (LlmReadResult r, ScriptedHandler h) = await Run(answer);
        Assert.Equal(1, r.Calls);
        Assert.Single(h.Requests);
        GroundedField F(string p) => r.Fields.Single(f => f.Path == p);
        Assert.Equal(GroundingState.Grounded, F("/number").State);
        Assert.Equal(GroundingState.GroundedNormalized, F("/date").State);
        Assert.Equal("2026-10-03", F("/date").NormalizedValue);
        Assert.Equal(GroundingState.GroundedNormalized, F("/total").State);
        Assert.Equal("1234.50", F("/total").NormalizedValue);
        Assert.Equal(GroundingState.Grounded, F("/currency").State);
        Assert.DoesNotContain(r.Fields, f => f.Path == "/paid");
        Assert.All(r.Fields, f => Assert.NotEmpty(f.Boxes));
        Assert.All(r.Fields.Where(f => f.Path != "/date"), f => Assert.True(f.Confidence > 0.85f));
        Assert.Contains("ambiguous", F("/date").Note, StringComparison.Ordinal); // 03.10 could be 3 Oct or 10 Mar, and the page has no 13.xx to tell
        using JsonDocument doc = JsonDocument.Parse(r.Json!);
        Assert.Equal("INV-1001", doc.RootElement.GetProperty("number").GetString());
        Assert.Equal(1234.5m, doc.RootElement.GetProperty("total").GetDecimal());
        Assert.Equal(JsonValueKind.Null, doc.RootElement.GetProperty("paid").ValueKind);
        Assert.Equal(0, doc.RootElement.GetProperty("items").GetArrayLength());
    }

    [Fact]
    public async Task A_wrong_or_missing_citation_is_corrected_by_searching_the_page_at_lower_confidence()
    {
        string answer = Data(
            ("number", GroundingSupport.V("\"INV-1001\"", [Id("Widget")])),
            ("currency", GroundingSupport.V("\"EUR\"", [])));
        (LlmReadResult r, _) = await Run(answer);
        GroundedField n = r.Fields.Single(f => f.Path == "/number");
        Assert.Equal(GroundingState.Grounded, n.State);
        Assert.True(n.CitationCorrected);
        Assert.Equal("INV-1001", n.Words.Single().Text);
        Assert.True(n.Confidence < 0.9f);
        Assert.True(r.Fields.Single(f => f.Path == "/currency").CitationCorrected);
    }

    [Fact]
    public async Task Folded_case_currency_symbols_and_trailing_punctuation_ground_as_normalized()
    {
        string answer = Data(
            ("number", GroundingSupport.V("\"inv-1001\"", [Id("INV-1001")])),
            ("total", GroundingSupport.V("1234.50", [Id("1.234,50")])),
            ("currency", GroundingSupport.V("\"Date\"", [Id("Date:")])));
        (LlmReadResult r, _) = await Run(answer);
        Assert.All(r.Fields, f => Assert.Equal(GroundingState.GroundedNormalized, f.State));
    }

    [Fact]
    public async Task A_hallucinated_value_is_re_asked_once_and_flagged_when_it_stays_wrong()
    {
        string bad = Data(("total", GroundingSupport.V("9999.99", [Id("Total")])));
        (LlmReadResult r, ScriptedHandler h) = await Run(bad, bad);
        Assert.Equal(2, r.Calls);
        Assert.Equal(2, h.Requests.Count);
        GroundedField f = r.Fields.Single();
        Assert.Equal(GroundingState.Ungrounded, f.State);
        Assert.Equal(0f, f.Confidence);
        Assert.True(f.Reasked);
        Assert.Contains("9999.99", r.Json);
        Assert.Contains("CHECK", h.Requests[1]);
        Assert.DoesNotContain("CHECK", h.Requests[0]);
        Assert.Equal(0.0056m, r.CostUsd); // two calls of 1000 in / 100 out at 2 / 8 USD per million tokens
    }

    [Fact]
    public async Task The_re_ask_can_fix_a_value_or_make_the_model_withdraw_it()
    {
        string bad = Data(("total", GroundingSupport.V("9999.99", [Id("Total")])), ("number", GroundingSupport.V("\"INV-9999\"", [Id("Invoice")])));
        string better = Data(("total", GroundingSupport.V("\"1.234,50\"", [Id("1.234,50")])), ("number", GroundingSupport.V("null", [])));
        (LlmReadResult r, _) = await Run(bad, better);
        GroundedField t = r.Fields.Single(f => f.Path == "/total");
        Assert.Equal(GroundingState.Grounded, t.State);
        Assert.True(t.Reasked);
        Assert.DoesNotContain(r.Fields, f => f.Path == "/number");
        Assert.Contains(r.Diagnostics, d => d.Contains("/number", StringComparison.Ordinal) && d.Contains("withdrew", StringComparison.Ordinal));
        using JsonDocument doc = JsonDocument.Parse(r.Json!);
        Assert.Equal(JsonValueKind.Null, doc.RootElement.GetProperty("number").ValueKind);
    }

    [Fact]
    public async Task A_correct_answer_makes_no_second_call_and_reask_can_be_turned_off()
    {
        string good = Data(("number", GroundingSupport.V("\"INV-1001\"", [Id("INV-1001")])));
        Assert.Equal(1, (await Run(good)).R.Calls);
        string bad = Data(("number", GroundingSupport.V("\"X\"", [])));
        (LlmReadResult r, ScriptedHandler h) = await Run(bad, o: new LlmReadOptions { Outputs = LlmOutputs.Json, JsonSchema = Schema, Reask = false });
        Assert.Equal(1, r.Calls);
        Assert.Single(h.Requests);
    }

    [Fact]
    public async Task The_re_ask_counts_against_the_spend_cap_and_is_skipped_when_it_would_pass_it()
    {
        string bad = Data(("total", GroundingSupport.V("9999.99", [])));
        // Output priced at 1 USD per token, so a call's worst case is 4096 USD (the output ceiling) and the real cost of the first is 100 USD:
        // a cap of 4100 admits the first call and refuses the second (100 spent + 4096 worst case).
        var h = new ScriptedHandler().Then(Wrap("openai", bad));
        using var http = new HttpClient(h);
        LlmOptions lo = new() { SpendCapUsd = 4100m, PriceTable = LlmPriceTable.Default.With(new LlmModelPrice("openai", "test", 0m, 1_000_000m)), RetryBaseDelay = TimeSpan.FromMilliseconds(1) };
        using var client = new LlmClient(GroundingSupport.Provider("openai", http), lo);
        LlmReadResult r = await client.ReadAsync(GroundingSupport.Blank(Page), Page, new LlmReadOptions { Outputs = LlmOutputs.Json, JsonSchema = Schema });
        Assert.Equal(1, r.Calls);
        Assert.Single(h.Requests);
        Assert.Contains(r.Diagnostics, d => d.Contains("re-ask was skipped", StringComparison.Ordinal));
        Assert.Equal(GroundingState.Ungrounded, r.Fields.Single().State);
        Assert.True(client.SpentUsd > 0m);
    }

    [Fact]
    public async Task Ungrounded_policy_flag_null_and_throw()
    {
        string bad = Data(("total", GroundingSupport.V("9999.99", [])), ("number", GroundingSupport.V("\"INV-1001\"", [Id("INV-1001")])));
        LlmReadOptions O(UngroundedPolicy p) => new() { Outputs = LlmOutputs.Json, JsonSchema = Schema, UngroundedPolicy = p, Reask = false };
        (LlmReadResult flag, _) = await Run(bad, o: O(UngroundedPolicy.Flag));
        Assert.Equal(9999.99m, JsonDocument.Parse(flag.Json!).RootElement.GetProperty("total").GetDecimal());
        (LlmReadResult nul, _) = await Run(bad, o: O(UngroundedPolicy.Null));
        Assert.Equal(JsonValueKind.Null, JsonDocument.Parse(nul.Json!).RootElement.GetProperty("total").ValueKind);
        Assert.Equal(GroundingState.Ungrounded, nul.Fields.Single(f => f.Path == "/total").State);
        OcrProviderException e = await Assert.ThrowsAsync<OcrProviderException>(async () => await Run(bad, o: O(UngroundedPolicy.Throw)));
        Assert.Equal(LlmFailure.Ungrounded, e.Failure);
        Assert.Contains("/total", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Derived_values_are_never_boxed_and_only_verified_sums_get_high_confidence()
    {
        // 1.000,00 + 234,50 = 1.234,50 is printed; use a sum that is not printed: 1234.50 + 197.06 = 1431.56
        string answer = Data(
            ("total", GroundingSupport.V("1431.56", [Id("1.234,50"), Id("197,06")], derived: true)),
            ("number", GroundingSupport.V("\"INV-1001-A\"", [Id("INV-1001")], derived: true)),
            ("paid", GroundingSupport.V("true", [], derived: true)));
        (LlmReadResult r, _) = await Run(answer);
        GroundedField sum = r.Fields.Single(f => f.Path == "/total");
        Assert.Equal(GroundingState.Derived, sum.State);
        Assert.Empty(sum.Boxes);
        Assert.Equal(0.9f, sum.Confidence);
        GroundedField guess = r.Fields.Single(f => f.Path == "/number");
        Assert.Equal(GroundingState.Derived, guess.State);
        Assert.Equal(0.3f, guess.Confidence);
        Assert.Equal(GroundingState.Derived, r.Fields.Single(f => f.Path == "/paid").State);
        Assert.Equal(1, r.Calls);
    }

    [Fact]
    public async Task Array_items_get_indexed_paths_and_a_value_appearing_twice_is_flagged_ambiguous()
    {
        string items = "{\"data\":{\"items\":[{\"desc\":" + GroundingSupport.V("\"Widget\"", [Id("Widget")]) + ",\"amount\":" + GroundingSupport.V("1000", [Id("1.000,00")])
            + "},{\"desc\":" + GroundingSupport.V("\"Gadget\"", [Id("Gadget")]) + ",\"amount\":" + GroundingSupport.V("234.5", [Id("Gadget")]) + "}]}}";
        (LlmReadResult r, _) = await Run(items);
        Assert.Contains(r.Fields, f => f.Path == "/items/0/amount" && f.State == GroundingState.GroundedNormalized);
        GroundedField bad = r.Fields.Single(f => f.Path == "/items/1/amount");
        Assert.True(bad.CitationCorrected);
        Assert.Equal("234,50", bad.Words.Single().Text);
    }

    [Fact]
    public async Task Unknown_cited_ids_are_noted_and_the_page_is_searched()
    {
        string answer = Data(("number", GroundingSupport.V("\"INV-1001\"", ["w9999", "x"])));
        (LlmReadResult r, _) = await Run(answer);
        GroundedField f = r.Fields.Single();
        Assert.Equal(GroundingState.Grounded, f.State);
        Assert.Contains("not in the word list", f.Note, StringComparison.Ordinal);
    }

    [Fact]
    public async Task A_schema_with_refs_and_nullable_any_of_is_wrapped_and_read_back()
    {
        const string S = "{\"type\":\"object\",\"properties\":{\"seller\":{\"$ref\":\"#/$defs/Party\"},\"note\":{\"anyOf\":[{\"type\":\"string\"},{\"type\":\"null\"}]}},"
            + "\"$defs\":{\"Party\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}}}}}";
        string answer = "{\"data\":{\"seller\":{\"name\":" + GroundingSupport.V("\"Widget\"", [Id("Widget")]) + "},\"note\":" + GroundingSupport.V("null", []) + "}}";
        (LlmReadResult r, ScriptedHandler h) = await Run(answer, o: new LlmReadOptions { Outputs = LlmOutputs.Json, JsonSchema = S });
        Assert.Equal("/seller/name", r.Fields.Single().Path);
        Assert.Contains("\"derived\"", h.Requests[0], StringComparison.Ordinal);
        Assert.DoesNotContain("$ref", h.Requests[0], StringComparison.Ordinal);
    }

    [Fact]
    public async Task Schemas_that_cannot_be_grounded_are_argument_errors()
    {
        await Assert.ThrowsAsync<ArgumentException>(async () => await Run("{}", o: new LlmReadOptions
        {
            Outputs = LlmOutputs.Json,
            JsonSchema = "{\"type\":\"object\",\"properties\":{\"a\":{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"integer\"}]}}}",
        }));
        await Assert.ThrowsAsync<ArgumentException>(async () => await Run("{}", o: new LlmReadOptions { Outputs = LlmOutputs.Json }));
    }

    [Fact]
    public async Task A_key_echoed_in_the_re_ask_error_does_not_reach_the_diagnostics()
    {
        string bad = Data(("total", GroundingSupport.V("9999.99", [])));
        var h = new ScriptedHandler().Then(Wrap("openai", bad)).Then("{\"error\":{\"message\":\"bad key " + GroundingSupport.Key + "\"}}", HttpStatusCode.BadRequest);
        using var http = new HttpClient(h);
        using var client = new LlmClient(GroundingSupport.Provider("openai", http), GroundingSupport.Options());
        LlmReadResult r = await client.ReadAsync(GroundingSupport.Blank(Page), Page, new LlmReadOptions { Outputs = LlmOutputs.Json, JsonSchema = Schema });
        Assert.Contains(r.Diagnostics, d => d.Contains("re-ask was skipped", StringComparison.Ordinal));
        Assert.DoesNotContain(r.Diagnostics, d => d.Contains(GroundingSupport.Key, StringComparison.Ordinal) || d.Contains("0123456789", StringComparison.Ordinal));
    }
}
