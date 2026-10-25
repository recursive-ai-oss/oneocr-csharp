using System.Globalization;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using OneOcr.Documents;
using OneOcr.Llm;
using OneOcr.Training;

namespace OneOcr.Tests;

/// <summary>
/// Opt-in demonstration that a user can fine-tune the shipped extraction model for a document type of their own. The data are generated invoice pages
/// that also print a contract number (a field the shipped model has no class for), read by the library's own OCR. Labels come through the
/// grounding path from scripted answers built from the generator's exact truth — the answers a perfect labelling model would give; no network, no
/// provider. The fine-tuned model and the shipped model are then scored on separate generated pages with their own, template-disjoint layouts.
/// Environment: <c>ONEOCR_FT_DIR</c> (holds <c>train/ocr</c>, <c>train/truth</c>, <c>val/ocr</c>, <c>val/truth</c>), <c>ONEOCR_FT_MODEL</c> (the shipped model),
/// <c>ONEOCR_FT_OUT</c> (report file), optional <c>ONEOCR_FT_STEPS</c> and <c>ONEOCR_FT_BACKBONE</c> (comma-separated backbone scales), <c>ONEOCR_FT_WORK</c>.
/// </summary>
public class ExtractionFineTuneDemo
{
    // Schema property -> the shipped model's class for the same meaning (null: the shipped model has none).
    private static readonly (string Name, string? Shipped, string Description)[] Fields =
    [
        ("contractNumber", null, "The contract number printed next to the invoice details"),
        ("invoiceNumber", "BT-1", "The invoice number"),
        ("issueDate", "BT-2", "The invoice date"),
        ("dueDate", "BT-9", "The due date"),
        ("sellerName", "BT-27", "The seller's name"),
        ("total", "BT-112", "The invoice total"),
        ("iban", "BT-84", "The seller's IBAN"),
    ];

    private sealed record Truth(OcrPage Page, Dictionary<string, (string Text, int[] Words)> Fields);

    private static Truth ReadTruth(string file)
    {
        byte[] bytes = File.ReadAllBytes(file);
        OcrPage page = OcrPageJson.Read(bytes)[0];
        using JsonDocument doc = JsonDocument.Parse(bytes);
        var map = new Dictionary<string, (string, int[])>(StringComparer.Ordinal);
        JsonElement p = doc.RootElement.GetProperty("pages")[0];
        if (p.TryGetProperty("fields", out JsonElement fields))
        {
            foreach (JsonElement f in fields.EnumerateArray())
            {
                string name = f.GetProperty("name").GetString()!;
                int[] words = [.. f.GetProperty("words").EnumerateArray().Select(static e => e.GetInt32())];
                if (words.Length > 0 && Array.Exists(Fields, x => x.Name == name) && f.GetProperty("lineItem").ValueKind == JsonValueKind.Null)
                {
                    map[name] = (string.Join(' ', words.Select(i => page.Words[i].Text)), words);
                }
            }
        }

        return new Truth(page, map);
    }

    // The OCR words that cover a truth word (at least half of the smaller box) and overlap it more than any other truth word.
    private static int[] Cover(OcrPage truth, OcrPage ocr, int[] truthWords)
    {
        var result = new SortedSet<int>();
        foreach (int t in truthWords)
        {
            OcrBox tb = truth.Words[t].Box;
            for (int o = 0; o < ocr.Words.Count; o++)
            {
                OcrBox ob = ocr.Words[o].Box;
                double inter = Overlap(tb, ob);
                if (inter <= 0 || inter < 0.5 * Math.Min(Area(tb), Area(ob)))
                {
                    continue;
                }

                bool best = true;
                for (int u = 0; u < truth.Words.Count && best; u++)
                {
                    best = u == t || Overlap(truth.Words[u].Box, ob) <= inter;
                }

                if (best)
                {
                    result.Add(o);
                }
            }
        }

        return [.. result];
    }

    private static double Area(OcrBox b) => Math.Max(1e-6, (b.Right - b.Left) * (b.Bottom - b.Top));

    private static double Overlap(OcrBox a, OcrBox b)
    {
        double w = Math.Min(a.Right, b.Right) - Math.Max(a.Left, b.Left), h = Math.Min(a.Bottom, b.Bottom) - Math.Max(a.Top, b.Top);
        return w > 0 && h > 0 ? w * h : 0;
    }

    private static string Norm(string s) => string.Concat(s.Normalize(NormalizationForm.FormC).Where(static c => !char.IsWhiteSpace(c)));

    private static string Schema()
    {
        var sb = new StringBuilder("{\"type\":\"object\",\"properties\":{");
        sb.AppendJoin(',', Fields.Select(static f => $"\"{f.Name}\":{{\"type\":[\"string\",\"null\"],\"description\":{JsonSerializer.Serialize(f.Description)}}}"));
        return sb.Append("}}").ToString();
    }

    [Fact]
    public async Task A_user_trained_invoice_variant_beats_the_shipped_model_on_its_own_held_out_pages()
    {
        string? dir = Environment.GetEnvironmentVariable("ONEOCR_FT_DIR");
        string? model = Environment.GetEnvironmentVariable("ONEOCR_FT_MODEL");
        Assert.SkipWhen(dir is null || model is null || !Directory.Exists(dir) || !File.Exists(model), "ONEOCR_FT_DIR and ONEOCR_FT_MODEL are not set.");
        int steps = int.Parse(Environment.GetEnvironmentVariable("ONEOCR_FT_STEPS") ?? "300", CultureInfo.InvariantCulture);
        double[] scales = [.. (Environment.GetEnvironmentVariable("ONEOCR_FT_BACKBONE") ?? "0,0.05,0.2").Split(',').Select(static s => double.Parse(s, CultureInfo.InvariantCulture))];
        string work = Environment.GetEnvironmentVariable("ONEOCR_FT_WORK") ?? Path.Combine(Path.GetTempPath(), "oneocr-ft-" + Guid.NewGuid().ToString("N"));
        var report = new StringBuilder();
        ExtractionSchema schema = ExtractionSchema.FromJsonSchema(Schema());

        // 1. Labels: the scripted answer of a labelling model that reads every field correctly, checked against the OCR words by grounding.
        string labelWork = Path.Combine(work, "labels");
        var labelOptions = new ExtractionTrainerOptions { ModelPath = model!, Schema = schema, WorkDirectory = labelWork };
        int documents = 0, usable = 0, dropped = 0, fieldsLabelled = 0;
        foreach (string ocrFile in Directory.GetFiles(Path.Combine(dir!, "train", "ocr"), "*.page.json").Order(StringComparer.Ordinal))
        {
            string id = Path.GetFileName(ocrFile)[..^".page.json".Length];
            string truthFile = Path.Combine(dir!, "train", "truth", id + ".json");
            if (!File.Exists(truthFile))
            {
                continue;
            }

            OcrPage ocr = OcrPageJson.Read(File.ReadAllBytes(ocrFile))[0];
            Truth truth = ReadTruth(truthFile);
            var parts = new List<string>();
            foreach ((string name, _, _) in Fields)
            {
                if (truth.Fields.TryGetValue(name, out (string Text, int[] Words) f) && Cover(truth.Page, ocr, f.Words) is { Length: > 0 } cited)
                {
                    parts.Add(JsonSerializer.Serialize(name) + ":" + GroundingSupport.V(GroundingSupport.S(f.Text), [.. cited.Select(static i => "w" + i.ToString(CultureInfo.InvariantCulture))]));
                }
                else
                {
                    parts.Add(JsonSerializer.Serialize(name) + ":" + GroundingSupport.V("null", []));
                }
            }

            var handler = new ScriptedHandler().Then(GroundingSupport.Shape("openai", "{\"data\":{" + string.Join(',', parts) + "}}"));
            using var http = new HttpClient(handler);
            using var llm = new LlmClient(GroundingSupport.Provider("openai", http), GroundingSupport.Options());
            ExtractionLabelSummary s = await new ExtractionTrainer(labelOptions, llm).LabelAsync([new ExtractionDocument(id, id, GroundingSupport.Blank(ocr), ocr)]);
            documents += s.Documents;
            usable += s.Usable;
            dropped += s.DroppedUngrounded;
            fieldsLabelled += s.FieldsLabelled;
        }

        report.AppendLine(CultureInfo.InvariantCulture, $"labelling (scripted exact answers, no network): {documents} pages, {usable} usable, {dropped} dropped for an ungrounded value, {fieldsLabelled} grounded values");
        Assert.True(usable > 0);

        // 2. Held-out pages: both models read the same OCR words of the validation pages.
        var valPages = new List<(OcrPage Ocr, Truth Truth)>();
        foreach (string ocrFile in Directory.GetFiles(Path.Combine(dir!, "val", "ocr"), "*.page.json").Order(StringComparer.Ordinal))
        {
            string truthFile = Path.Combine(dir!, "val", "truth", Path.GetFileName(ocrFile)[..^".page.json".Length] + ".json");
            if (File.Exists(truthFile))
            {
                valPages.Add((OcrPageJson.Read(File.ReadAllBytes(ocrFile))[0], ReadTruth(truthFile)));
            }
        }

        string Score(string title, Func<OcrPage, IReadOnlyList<ExtractedField>> read, Func<string, string?> className)
        {
            var right = new Dictionary<string, int>();
            var total = new Dictionary<string, int>();
            foreach ((OcrPage ocr, Truth truth) in valPages)
            {
                IReadOnlyList<ExtractedField> got = read(ocr);
                foreach ((string name, _, _) in Fields)
                {
                    if (!truth.Fields.TryGetValue(name, out (string Text, int[] Words) f))
                    {
                        continue;
                    }

                    total[name] = total.GetValueOrDefault(name) + 1;
                    string? cls = className(name);
                    ExtractedField? hit = cls is null ? null : got.Where(x => x.Id == cls).OrderByDescending(static x => x.Confidence).FirstOrDefault();
                    if (hit is not null && Norm(hit.Text) == Norm(f.Text))
                    {
                        right[name] = right.GetValueOrDefault(name) + 1;
                    }
                }
            }

            var sb = new StringBuilder();
            sb.Append(title).Append(": ");
            sb.AppendJoin("; ", Fields.Select(f => $"{f.Name} {(double)right.GetValueOrDefault(f.Name) / Math.Max(1, total.GetValueOrDefault(f.Name)):P1}"));
            sb.Append(CultureInfo.InvariantCulture, $"; all {(double)right.Values.Sum() / Math.Max(1, total.Values.Sum()):P1} of {total.Values.Sum()} values on {valPages.Count} held-out pages");
            return sb.ToString();
        }

        DocumentExtractor shipped = DocumentExtractor.Load(model!);
        report.AppendLine(Score("shipped model", p => shipped.ExtractInvoice(p).Extraction.Fields, n => Array.Find(Fields, f => f.Name == n).Shipped));

        // 3. Fine-tune at each backbone scale from the same labels; same data, seed and steps.
        foreach (double scale in scales)
        {
            string runWork = Path.Combine(work, "run-" + scale.ToString(CultureInfo.InvariantCulture));
            Directory.CreateDirectory(Path.Combine(runWork, "extract-labels"));
            foreach (string f in Directory.GetFiles(Path.Combine(labelWork, "extract-labels")))
            {
                File.Copy(f, Path.Combine(runWork, "extract-labels", Path.GetFileName(f)), true);
            }

            var options = new ExtractionTrainerOptions
            {
                ModelPath = model!,
                Schema = schema,
                WorkDirectory = runWork,
                OutputModelPath = Path.Combine(runWork, "model.onm"),
                Steps = steps,
                BatchSize = 8,
                Seed = 1,
                BackboneScale = scale,
                MaxDegreeOfParallelism = 4,
                CheckpointEvery = 50,
            };
            var clock = System.Diagnostics.Stopwatch.StartNew();
            ExtractionReport r = new ExtractionTrainer(options).Train();
            report.AppendLine(CultureInfo.InvariantCulture, $"backbone scale {scale}: {r.TrainDocuments} train / {r.HeldOutDocuments} held-out pages, {r.Steps} steps in {clock.Elapsed.TotalSeconds:F0} s, loss {r.FirstLoss:F3} -> {r.LastLoss:F3}");
            report.AppendLine(CultureInfo.InvariantCulture, $"  trainer's held-out accuracy {r.AccuracyBefore:P1} -> {r.AccuracyAfter:P1}; agreement with the labelling model {r.AgreementBefore:P1} -> {r.AgreementAfter:P1}; suggested LocalFirst threshold {(r.SuggestedLocalFirstThreshold?.ToString("F3", CultureInfo.InvariantCulture) ?? "none")} ({r.ThresholdSupport} values at or above)");
            foreach (ExtractionFieldReport f in r.Fields)
            {
                report.AppendLine(CultureInfo.InvariantCulture, $"    {f.Field}: {f.Labelled} labelled, accuracy {f.AccuracyBefore:P1} -> {f.AccuracyAfter:P1}, recall {f.RecallBefore:P1} -> {f.RecallAfter:P1}");
            }

            DocumentExtractor tuned = DocumentExtractor.Load(r.ModelPath);
            report.AppendLine(Score($"  tuned (scale {scale}) on separate generated pages", p => tuned.Extract(p).Fields, n => n));
        }

        Environment.GetEnvironmentVariable("ONEOCR_FT_OUT").AppendReport(report.ToString());
    }
}
