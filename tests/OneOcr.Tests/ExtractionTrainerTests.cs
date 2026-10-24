using System.Net.Http;
using System.Threading.Tasks;
using System.Globalization;
using System.Text.Json.Serialization;
using OneOcr.Documents;
using OneOcr.Llm;
using OneOcr.Model;
using OneOcr.Training;

namespace OneOcr.Tests;

public sealed class CustomForm
{
    public string? OrderNo { get; set; }

    public decimal? Amount { get; set; }

    public string? Date { get; set; }

    public string[]? Lines { get; set; }
}

public sealed class StrictForm
{
    public required string OrderNo { get; set; }
}

public sealed class OtherForm
{
    public string? Unknown { get; set; }

    public decimal? Amount { get; set; }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(CustomForm))]
[JsonSerializable(typeof(StrictForm))]
[JsonSerializable(typeof(OtherForm))]
internal sealed partial class CustomFormContext : JsonSerializerContext
{
}

/// <summary>
/// Fine-tuning for a custom document type: the container writer, the schema, labelling from a scripted grounded LLM read (no network),
/// training on a random-weight backbone (a stand-in: the point is that the new head learns the schema from grounded labels and beats the
/// untuned head on held-out pages), resume to the same bits, the budget cap, and <c>Extract&lt;T&gt;</c> on the tuned model.
/// </summary>
public class ExtractionTrainerTests : IDisposable
{
    private static readonly TestDocModels.Shape Shape = TestDocModels.Small;
    private static readonly OcrLimits Limits = OcrLimits.Default;
    private readonly string _root = Path.Combine(Path.GetTempPath(), "oneocr-xt-" + Guid.NewGuid().ToString("N"));

    public ExtractionTrainerTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try
        {
            Directory.Delete(_root, true);
        }
        catch (IOException)
        {
        }
    }

    private static byte[] BaseModel(int seed = 5) => TestDocModels.RandomContainer(Shape, seed);

    // ---- A synthetic document type with exact truth ----

    private sealed record Form(OcrPage Page, string? OrderNo, string? Amount, string? Date, string AnswerJson, bool Invented);

    private static readonly string[][] Fillers =
    [
        ["Thank", "you"], ["Page", "1", "of", "1"], ["Keep", "this", "slip"], ["Open", "daily"], ["Customer", "copy"],
    ];

    private static Form MakeForm(int seed, bool invent = false)
    {
        var rng = new Random(seed);
        string? order = rng.NextDouble() < 0.15 ? null : "PO-" + rng.Next(1000, 9999).ToString(CultureInfo.InvariantCulture);
        string? amount = rng.NextDouble() < 0.15 ? null : rng.Next(1, 999).ToString(CultureInfo.InvariantCulture) + "." + rng.Next(10, 99).ToString(CultureInfo.InvariantCulture);
        string? date = rng.NextDouble() < 0.15 ? null : $"{rng.Next(10, 28)}/{rng.Next(10, 12)}/2026";
        var lines = new List<string>();
        if (order is not null) { lines.Add("Order " + order); }
        if (amount is not null) { lines.Add("Total " + amount); }
        if (date is not null) { lines.Add("Date " + date); }
        for (int i = rng.Next(1, 3); i > 0; i--) { lines.Add(string.Join(' ', Fillers[rng.Next(Fillers.Length)])); }
        for (int i = lines.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (lines[i], lines[j]) = (lines[j], lines[i]);
        }

        OcrPage page = GroundingSupport.Layout([.. lines.Select(static l => new TestLine(l, Conf: 0.99f))]);
        string Field(string? value, string word, bool number = false) => value is null
            ? GroundingSupport.V("null", [])
            : GroundingSupport.V(number ? value : GroundingSupport.S(value), [GroundingSupport.Id(page, word)]);
        string answer = "{\"data\":{\"orderNo\":" + (invent ? GroundingSupport.V("\"PO-0000\"", []) : Field(order, order ?? "")) + ",\"amount\":" + Field(amount, amount ?? "", true) + ",\"date\":" + Field(date, date ?? "") + "}}";
        return new Form(page, order, amount, date, answer, invent);
    }

    private static ExtractionSchema Schema() => ExtractionSchema.From(CustomFormContext.Default.CustomForm);

    private ExtractionTrainerOptions Options(string work, string model, int steps = 120, double backbone = 0, int seed = 3, int stopAfter = 0, string? output = null, double lr = 3e-3, double tagWeight = 3) => new()
    {
        ModelPath = model,
        Schema = Schema(),
        WorkDirectory = Path.Combine(_root, work),
        OutputModelPath = output is null ? null : Path.Combine(_root, output),
        Steps = steps,
        BatchSize = 8,
        Seed = seed,
        BackboneScale = backbone,
        LearningRate = lr,
        FieldTagWeight = tagWeight,
        StopAfterSteps = stopAfter,
        MaxDegreeOfParallelism = 2,
        CheckpointEvery = 10,
    };

    private async Task<ExtractionLabelSummary> LabelAsync(string work, string model, IReadOnlyList<Form> forms, decimal? cap = null)
    {
        var handler = new ScriptedHandler();
        var docs = new List<ExtractionDocument>();
        for (int i = 0; i < forms.Count; i++)
        {
            string reply = GroundingSupport.Shape("openai", forms[i].AnswerJson);
            handler.Then(reply);
            if (forms[i].Invented)
            {
                handler.Then(reply); // the re-ask of the ungrounded value gets the same answer
            }

            docs.Add(new ExtractionDocument("doc" + i.ToString(CultureInfo.InvariantCulture), "form" + i.ToString(CultureInfo.InvariantCulture), GroundingSupport.Blank(forms[i].Page), forms[i].Page));
        }

        using var http = new HttpClient(handler);
        using var llm = new LlmClient(GroundingSupport.Provider("openai", http), GroundingSupport.Options(cap));
        string path = Path.Combine(_root, "base.onm");
        File.WriteAllBytes(path, BaseModel());
        return await new ExtractionTrainer(Options(work, path), llm).LabelAsync(docs);
    }

    private static List<Form> Forms(int count, int invented = 0, int seed0 = 100) =>
        [.. Enumerable.Range(0, count).Select(i => MakeForm(seed0 + i, i < invented))];

    private string CopyLabels(string from, string to)
    {
        string dst = Path.Combine(_root, to, "extract-labels");
        Directory.CreateDirectory(dst);
        foreach (string f in Directory.GetFiles(Path.Combine(_root, from, "extract-labels")))
        {
            File.Copy(f, Path.Combine(dst, Path.GetFileName(f)), true);
        }

        return to;
    }

    [Fact]
    public void The_container_writer_keeps_the_format_and_round_trips_weights()
    {
        ModelContainer baseContainer = ModelContainer.Parse(BaseModel(), Limits);
        byte[] head = DocContainerWriter.WithNewHead(baseContainer, ["orderNo", "amount", "date"], 7, Limits);
        DocumentExtractor ex = DocumentExtractor.Load(head);
        Assert.Equal(["orderNo", "amount", "date"], ex.Classes);
        ModelContainer h = ModelContainer.Parse(head, Limits);
        Assert.Equal(baseContainer.Tensors.Select(static t => t.Name), h.Tensors.Select(static t => t.Name));
        Assert.Equal(baseContainer.ReadTensor("blocks.0.qkv.weight"), h.ReadTensor("blocks.0.qkv.weight"));
        Assert.Equal(7, h.FindTensor("tag_head.weight")!.Shape[0]);
        Assert.Equal(head, DocContainerWriter.WithNewHead(baseContainer, ["orderNo", "amount", "date"], 7, Limits));
        Assert.NotEqual(head, DocContainerWriter.WithNewHead(baseContainer, ["orderNo", "amount", "date"], 8, Limits));

        // Writing the weights back that were just read reproduces every tensor exactly (fp16 is exact through fp32).
        DocExtractModel m = DocExtractModel.FromContainer(h, Limits);
        byte[] again = DocContainerWriter.WithTrained(h, m, "round trip");
        ModelContainer a = ModelContainer.Parse(again, Limits);
        foreach (TensorInfo t in h.Tensors)
        {
            Assert.Equal(h.ReadTensor(t.Name), a.ReadTensor(t.Name));
        }

        foreach (string[] bad in new[] { new[] { "BT-9" }, ["receipt.total"], ["a/b"], ["a~b"], ["x", "x"], [""], [] })
        {
            Assert.Throws<ArgumentException>(() => DocContainerWriter.WithNewHead(baseContainer, bad, 1, Limits));
        }
    }

    [Fact]
    public void The_schema_reads_scalar_properties_and_notes_the_rest()
    {
        ExtractionSchema s = Schema();
        Assert.Equal(["orderNo", "amount", "date"], s.Fields.Select(static f => f.Name));
        Assert.Equal("number", s.Fields[1].Type);
        Assert.Contains(s.Notes, static n => n.Contains("lines", StringComparison.Ordinal));
        Assert.Contains("\"null\"", s.ToJsonSchema(), StringComparison.Ordinal);
        ExtractionSchema raw = ExtractionSchema.FromJsonSchema("{\"type\":\"object\",\"properties\":{\"po\":{\"type\":[\"string\",\"null\"],\"description\":\"PO number\"},\"n\":{\"type\":\"integer\"}}}");
        Assert.Equal("PO number", raw.Fields[0].Description);
        Assert.Throws<ArgumentException>(() => ExtractionSchema.FromJsonSchema("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"object\"}}}"));
        Assert.Throws<ArgumentException>(() => ExtractionSchema.FromJsonSchema("{\"type\":\"object\",\"properties\":{\"BT-1\":{\"type\":\"string\"}}}"));
    }

    [Fact]
    public async Task Only_grounded_values_become_labels_and_a_page_with_an_invented_value_is_dropped()
    {
        List<Form> forms = Forms(10, invented: 2);
        ExtractionLabelSummary s = await LabelAsync("lab", Path.Combine(_root, "base.onm"), forms);
        Assert.Equal(10, s.Documents);
        Assert.Equal(8, s.Usable);
        Assert.Equal(2, s.DroppedUngrounded);
        Assert.False(s.BudgetReached);
        // doc2 is a normal page: its label is the exact words of the truth.
        string json = File.ReadAllText(Path.Combine(_root, "lab", "extract-labels", "doc2.json"));
        Form f = forms[2];
        if (f.OrderNo is not null)
        {
            Assert.Contains("\"value\":\"" + f.OrderNo + "\"", json, StringComparison.Ordinal);
        }

        // Labelling again costs nothing: the pages already have labels.
        ExtractionLabelSummary again = await LabelAsync("lab", Path.Combine(_root, "base.onm"), forms);
        Assert.Equal(10, again.AlreadyLabelled);
        Assert.Equal(0, again.Documents);
    }

    [Fact]
    public async Task The_spend_cap_stops_labelling_and_keeps_what_was_done()
    {
        ExtractionLabelSummary s = await LabelAsync("cap", Path.Combine(_root, "base.onm"), Forms(3), cap: 0.0000001m);
        Assert.True(s.BudgetReached);
        Assert.Equal(0, s.Documents);
    }

    private (ExtractionReport Report, string Out) TrainRun(string work, int steps = 400, double backbone = 0.3, int stopAfter = 0, string? output = null)
    {
        string model = Path.Combine(_root, "base.onm");
        File.WriteAllBytes(model, BaseModel());
        ExtractionTrainerOptions o = Options(work, model, steps, backbone, 3, stopAfter, output ?? work + ".onm");
        return (new ExtractionTrainer(o).Train(), o.OutputModelPath!);
    }

    [Fact]
    public async Task A_tuned_head_beats_the_untuned_head_on_held_out_pages_and_resumes_to_the_same_model()
    {
        await LabelAsync("a", Path.Combine(_root, "base.onm"), Forms(60, invented: 3));
        CopyLabels("a", "b");
        (ExtractionReport r, string outPath) = TrainRun("a");
        TestNotes.Add($"extraction e2e: train {r.TrainDocuments}, held out {r.HeldOutDocuments}, loss {r.FirstLoss:F3} -> {r.LastLoss:F3}, accuracy {r.AccuracyBefore:F3} -> {r.AccuracyAfter:F3}, agreement {r.AgreementBefore:F3} -> {r.AgreementAfter:F3}, threshold {r.SuggestedLocalFirstThreshold}");
        Assert.Equal(57, r.UsableDocuments);
        Assert.Equal(3, r.DroppedUngrounded);
        Assert.True(r.HeldOutDocuments >= 10);
        Assert.True(r.LastLoss < r.FirstLoss);
        Assert.True(r.AccuracyAfter > r.AccuracyBefore + 0.1, $"{r.AccuracyBefore} -> {r.AccuracyAfter}");
        Assert.True(r.AgreementAfter > r.AgreementBefore);
        Assert.Equal(3, r.Fields.Count);
        Assert.True(File.Exists(Path.Combine(_root, "a", "extract-report.json")));

        // Stop part-way, then continue from the checkpoint: the same model bytes as the uninterrupted run.
        (ExtractionReport part, _) = TrainRun("b", stopAfter: 47);
        Assert.Equal(string.Empty, part.ModelPath);
        Assert.False(File.Exists(Path.Combine(_root, "b.onm")));
        TrainRun("b");
        Assert.Equal(File.ReadAllBytes(outPath), File.ReadAllBytes(Path.Combine(_root, "b.onm")));

        // The tuned model is an ordinary model file and fills a type through Extract<T>.
        DocumentExtractor tuned = DocumentExtractor.Load(outPath);
        int right = 0, asked = 0;
        for (int i = 0; i < 20; i++)
        {
            Form f = MakeForm(5000 + i);
            ExtractionResult<CustomForm> x = tuned.Extract(f.Page, CustomFormContext.Default.CustomForm, 1);
            asked += 3;
            right += (x.Value.OrderNo == f.OrderNo ? 1 : 0)
                + (x.Value.Amount == (f.Amount is null ? null : decimal.Parse(f.Amount, CultureInfo.InvariantCulture)) ? 1 : 0)
                + (x.Value.Date == f.Date ? 1 : 0);
        }

        TestNotes.Add($"extraction e2e Extract<T>: {right}/{asked} fields exact on 20 fresh pages");
        Assert.True(right >= asked * 0.7, $"{right}/{asked}");
    }

    [Fact]
    public void Extract_of_T_notes_a_property_the_model_has_no_class_for_and_rejects_a_missing_required_member()
    {
        byte[] head = DocContainerWriter.WithNewHead(ModelContainer.Parse(BaseModel(), Limits), ["amount"], 2, Limits);
        DocumentExtractor ex = DocumentExtractor.Load(head);
        OcrPage page = MakeForm(1).Page;
        ExtractionResult<OtherForm> r = ex.Extract(page, CustomFormContext.Default.OtherForm, 1);
        Assert.Null(r.Value.Unknown);
        Assert.Contains(r.Diagnostics, static d => d.Contains("no class 'unknown'", StringComparison.Ordinal));
        Assert.Throws<ArgumentException>(() => ex.Extract(page, CustomFormContext.Default.StrictForm, 1));
        Assert.Throws<ArgumentNullException>(() => ex.Extract<OtherForm>(page, null!));
    }
}
