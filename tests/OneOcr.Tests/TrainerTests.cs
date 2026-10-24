using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using OneImage;
using OneOcr.Kernels;
using OneOcr.Llm;
using OneOcr.Model;
using OneOcr.Recognition;
using OneOcr.Training;

namespace OneOcr.Tests;

public class TrainerTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "oneocr-train-" + Guid.NewGuid().ToString("N"));

    public TrainerTests() => Directory.CreateDirectory(_root);

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

    private string Dir(string name)
    {
        string p = Path.Combine(_root, name);
        Directory.CreateDirectory(p);
        return p;
    }

    private string WriteModel(byte[] bytes)
    {
        string p = Path.Combine(_root, "start.onm");
        File.WriteAllBytes(p, bytes);
        return p;
    }

    /// <summary>Writes pages in style B (named so that they sort in order) and returns each page's true lines.</summary>
    private string[][] WriteDocs(string folder, int pages, int seed, bool styleB = true)
    {
        var all = new string[pages][];
        for (int p = 0; p < pages; p++)
        {
            all[p] = GlyphPages.Lines(seed + p, 4);
            GlyphPages.Page(all[p], styleB).Save(Path.Combine(folder, $"page{p:D2}.png"));
        }

        return all;
    }

    private static async Task<(LlmClient Client, ScriptedHandler Handler)> Labeller(IEnumerable<string> pageTexts)
    {
        var handler = new ScriptedHandler();
        foreach (string text in pageTexts)
        {
            handler.Then(GroundingSupport.Shape("openai-compatible", JsonSerializer.Serialize(new { pages = new[] { text } })));
        }

        await Task.CompletedTask;
        var http = new HttpClient(handler);
        return (new LlmClient(GroundingSupport.Provider("openai-compatible", http), GroundingSupport.Options()), handler);
    }

    private OcrTrainerOptions Options(string work, string model, Action<OcrTrainerOptionsBuilder>? tweak = null)
    {
        var b = new OcrTrainerOptionsBuilder { Work = work, Model = model };
        tweak?.Invoke(b);
        return new OcrTrainerOptions
        {
            ModelPath = b.Model,
            WorkDirectory = b.Work,
            Seed = b.Seed,
            Steps = b.Steps,
            BatchSize = b.Batch,
            LearningRate = b.Lr,
            HeldOutFraction = 0.25,
            MaxAgreementCer = b.MaxCer,
            MaxDegreeOfParallelism = b.Degree,
            StopAfterSteps = b.StopAfter,
            CheckpointEvery = 7,
            KernelPath = b.Path,
            OutputModelPath = b.Out,
            Restart = b.Restart,
        };
    }

    private sealed class OcrTrainerOptionsBuilder
    {
        public required string Work { get; init; }
        public required string Model { get; init; }
        public int Seed { get; set; } = 1;
        public int Steps { get; set; } = 120;
        public int Batch { get; set; } = 8;
        public double Lr { get; set; } = 1e-3;
        public double MaxCer { get; set; } = 0.5;
        public int Degree { get; set; } = 2;
        public int StopAfter { get; set; }
        public KernelPath? Path { get; set; }
        public string? Out { get; set; }
        public bool Restart { get; set; }
    }

    [Fact]
    public async Task Fine_tuning_on_a_shifted_style_lowers_held_out_error_and_the_model_loads_through_the_engine()
    {
        string model = WriteModel(PretrainedA.Model.Value);
        string docs = Dir("docs");
        string[][] truth = WriteDocs(docs, 16, 200);
        (LlmClient client, ScriptedHandler handler) = await Labeller(truth.Select(GlyphPages.Text));
        using (client)
        {
            var trainer = new OcrTrainer(Options(Dir("work"), model), client);
            TrainingReport report = await trainer.RunAsync(docs);

            Assert.Equal(16, report.Documents);
            Assert.Equal(16, handler.Requests.Count); // one call per page
            Assert.True(report.TrainLines >= 8, $"train lines {report.TrainLines}");
            Assert.NotNull(report.Evaluation);
            BeforeAfter e = report.Evaluation!;
            TestNotes.Add($"agreement {report.AgreementRate:F2}, train {report.TrainLines}, held-out {report.HeldOutLines}: " +
                $"before cer {e.HeldOut.Cer:F3} wer {e.HeldOut.Wer:F3}, after cer {e.Tuned.Cer:F3} wer {e.Tuned.Wer:F3}, loss {report.FirstLoss:F2} -> {report.LastLoss:F2}");
            Assert.True(e.HeldOut.Cer > 0.05, "the start should be weak on the shifted style");
            Assert.True(e.Tuned.Cer < e.HeldOut.Cer, $"{e.Tuned.Cer} !< {e.HeldOut.Cer}");
            Assert.True(report.LastLoss < report.FirstLoss);

            // The written file is a normal model: the engine loads it and reads a page of the shifted style better than the start did.
            OcrEngine engine = OcrEngine.Load(report.ModelPath);
            Assert.NotEmpty(engine.ToText(GlyphPages.Page(truth[0], true)));
            Assert.True(PretrainedA.Cer(File.ReadAllBytes(report.ModelPath), true, 7000) < PretrainedA.Cer(PretrainedA.Model.Value, true, 7000));
            Assert.Equal(ModelContainer.Parse(PretrainedA.Model.Value).Charset, ModelContainer.Load(report.ModelPath).Charset);
            Assert.True(File.Exists(Path.Combine(Dir("work"), "report.json")));
        }
    }

    [Fact]
    public async Task Lines_the_language_model_reads_differently_never_train_and_land_in_the_review_export()
    {
        string model = WriteModel(PretrainedA.Model.Value);
        string docs = Dir("docs");
        string[][] truth = WriteDocs(docs, 4, 300, styleB: false); // style A: the start reads it, so agreement is exact
        string[][] told = [.. truth.Select(static p => (string[])p.Clone())];
        told[1][2] = "hhhh gggg"; // the scripted model says something else for this one line
        (LlmClient client, _) = await Labeller(told.Select(GlyphPages.Text));
        string work = Dir("work");
        using (client)
        {
            var trainer = new OcrTrainer(Options(work, model, b => b.MaxCer = 0), client);
            LabelSummary s = await trainer.LabelAsync(docs);
            Assert.Equal(4, s.Documents);
            Assert.True(s.Disagreed + s.Unaligned >= 1);
            Assert.True(s.Agreed >= 10, $"agreed {s.Agreed}");

            string export = Dir("export");
            ReviewExport r = LineReview.Export(work, export);
            Assert.True(r.Items >= 1);
            using JsonDocument json = JsonDocument.Parse(File.ReadAllBytes(r.ReviewFile));
            JsonElement[] items = [.. json.RootElement.GetProperty("items").EnumerateArray()];
            Assert.Equal(r.Items, items.Length);
            JsonElement bad = items.First(static i => i.GetProperty("state").GetString() is "disagreed" or "unaligned");
            Assert.True(File.Exists(Path.Combine(export, bad.GetProperty("crop").GetString()!)));
            string badId = bad.GetProperty("id").GetString()!;

            // No exported line is in the training set, and the set holds only lines that agreed (label = the model's text).
            List<(string Id, string Label)> training = trainer.TrainingSetForTests();
            Assert.DoesNotContain(training, t => items.Any(i => i.GetProperty("id").GetString() == t.Id));
            Assert.NotEmpty(training);
            Assert.All(training, t => Assert.Contains(t.Label, told.SelectMany(static p => p).ToList()));
            Assert.DoesNotContain(training, t => t.Label == "hhhh gggg");
            Assert.NotNull(badId);
        }
    }

    [Fact]
    public async Task A_corrected_line_trains_the_next_run_and_a_review_file_round_trips()
    {
        string model = WriteModel(PretrainedA.Model.Value);
        string docs = Dir("docs");
        string[][] truth = WriteDocs(docs, 4, 400, styleB: false);
        string[][] told = [.. truth.Select(static p => (string[])p.Clone())];
        told[0][1] = "hhhh gggg";
        (LlmClient client, _) = await Labeller(told.Select(GlyphPages.Text));
        string work = Dir("work");
        using (client)
        {
            var trainer = new OcrTrainer(Options(work, model, b => b.MaxCer = 0), client);
            await trainer.LabelAsync(docs);
        }

        ReviewExport r = LineReview.Export(work, Dir("export"));
        byte[] file = File.ReadAllBytes(r.ReviewFile);
        using JsonDocument doc = JsonDocument.Parse(file);
        JsonElement item = doc.RootElement.GetProperty("items")[0];
        string id = item.GetProperty("id").GetString()!;
        string edited = Encoding.UTF8.GetString(file).Replace("\"text\": null", "\"text\": \"  abab   cdcd \"", StringComparison.Ordinal);
        string editedPath = Path.Combine(_root, "edited.json");
        File.WriteAllText(editedPath, edited);

        ReviewImportResult imported = LineReview.Import(work, editedPath);
        Assert.True(imported.Accepted >= 1);
        Assert.Equal(0, imported.UnknownIds);
        // Whitespace is normalized and the corrected line is now a training line with that text (unless held out).
        var trainer2 = new OcrTrainer(Options(work, model, b => b.MaxCer = 0));
        List<(string Id, string Label)> training = trainer2.TrainingSetForTests();
        Assert.All(training.Where(t => t.Label.StartsWith("abab", StringComparison.Ordinal)), t => Assert.Equal("abab cdcd", t.Label));
        // Items already corrected leave the next export.
        Assert.Equal(0, LineReview.Export(work, Dir("export2")).Items);
        Assert.Equal(r.Items, LineReview.Export(work, Dir("export3"), includeReviewed: true).Items);
        Assert.NotNull(id);

        ReviewImportResult again = LineReview.Import(work, edited.Select(static c => (byte)c).ToArray().AsSpan(), null);
        Assert.Equal(imported.Accepted, again.Replaced);
    }

    [Fact]
    public async Task Stopping_and_resuming_gives_the_same_weights_as_one_run_on_every_path_and_degree()
    {
        string model = WriteModel(PretrainedA.Model.Value);
        string docs = Dir("docs");
        string[][] truth = WriteDocs(docs, 10, 500);
        (LlmClient client, _) = await Labeller(truth.Select(GlyphPages.Text));
        string labelled = Dir("labelled");
        using (client)
        {
            await new OcrTrainer(Options(labelled, model), client).LabelAsync(docs);
        }

        byte[] Run(string name, KernelPath? path, int degree, int stopFirst)
        {
            string work = Dir(name);
            CopyDirectory(Path.Combine(labelled, "labels"), Path.Combine(work, "labels"));
            string output = Path.Combine(_root, name + ".onm");
            if (stopFirst > 0)
            {
                TrainingReport partial = new OcrTrainer(Options(work, model, b => { b.Path = path; b.Degree = degree; b.StopAfter = stopFirst; b.Steps = 30; b.Out = output; })).Train();
                Assert.Equal(stopFirst, partial.Steps);
                Assert.False(File.Exists(output)); // a stopped run writes no model
            }

            new OcrTrainer(Options(work, model, b => { b.Path = path; b.Degree = degree; b.Steps = 30; b.Out = output; })).Train();
            return File.ReadAllBytes(output);
        }

        byte[] straight = Run("a", Dispatch.Best, 2, 0);
        byte[] resumed = Run("b", Dispatch.Best, 2, 11);   // stopped between checkpoints (every 7) and at one
        byte[] scalarSerial = Run("c", KernelPath.Scalar, 1, 14);
        Assert.Equal(Convert.ToHexString(SHA256.HashData(straight)), Convert.ToHexString(SHA256.HashData(resumed)));
        Assert.Equal(Convert.ToHexString(SHA256.HashData(straight)), Convert.ToHexString(SHA256.HashData(scalarSerial)));
        Assert.NotEqual(Convert.ToHexString(SHA256.HashData(straight)), Convert.ToHexString(SHA256.HashData(PretrainedA.Model.Value)));
    }

    [Fact]
    public async Task A_checkpoint_of_another_run_is_refused_and_a_cancelled_run_resumes()
    {
        string model = WriteModel(PretrainedA.Model.Value);
        string docs = Dir("docs");
        string[][] truth = WriteDocs(docs, 8, 600);
        (LlmClient client, _) = await Labeller(truth.Select(GlyphPages.Text));
        string work = Dir("work");
        using (client)
        {
            await new OcrTrainer(Options(work, model), client).LabelAsync(docs);
        }

        new OcrTrainer(Options(work, model, b => { b.StopAfter = 5; b.Steps = 20; })).Train();
        var other = new OcrTrainer(Options(work, model, b => { b.Steps = 20; b.Lr = 5e-4; }));
        Assert.Throws<InvalidOperationException>(() => other.Train());
        // Restart discards the checkpoint instead.
        Assert.Equal(20, new OcrTrainer(Options(work, model, b => { b.Steps = 20; b.Lr = 5e-4; b.Restart = true; })).Train().Steps);
        File.Delete(Path.Combine(work, "checkpoint.bin"));
        new OcrTrainer(Options(work, model, b => { b.StopAfter = 5; b.Steps = 20; })).Train();

        // Cancelling before the first step saves a checkpoint at the last completed step and rethrows.
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var trainer = new OcrTrainer(Options(work, model, b => b.Steps = 20));
        Assert.ThrowsAny<OperationCanceledException>(() => trainer.Train(cts.Token));
        TrainingReport done = trainer.Train();
        Assert.Equal(20, done.Steps);
    }

    [Fact]
    public async Task Evaluate_review_import_and_training_from_stored_labels_make_no_network_request()
    {
        string model = WriteModel(PretrainedA.Model.Value);
        string docs = Dir("docs");
        string[][] truth = WriteDocs(docs, 8, 700);
        (LlmClient labeller, _) = await Labeller(truth.Select(GlyphPages.Text));
        string work = Dir("work");
        using (labeller)
        {
            await new OcrTrainer(Options(work, model), labeller).LabelAsync(docs);
        }

        // A client whose every request would throw: nothing below may use it.
        var failing = new ScriptedHandler(failOnAny: true);
        using var http = new HttpClient(failing);
        using var client = new LlmClient(GroundingSupport.Provider("openai", http), GroundingSupport.Options());
        var trainer = new OcrTrainer(Options(work, model, b => b.Steps = 10), client);
        TrainingReport report = trainer.Train();
        EvaluationReport eval = OcrTrainer.Evaluate(report.ModelPath, Options(work, model));
        ReviewExport export = LineReview.Export(work, Dir("export"));
        LineReview.Import(work, export.ReviewFile);
        Assert.Empty(failing.Requests);
        Assert.Equal(report.Evaluation!.Tuned.Cer, eval.Rates.Cer);

        // And without a client, labelling is refused rather than attempted.
        var none = new OcrTrainer(Options(work, model));
        await Assert.ThrowsAsync<InvalidOperationException>(() => none.LabelAsync(docs));
    }

    [Fact]
    public async Task A_provider_budget_stops_labelling_and_a_later_call_continues()
    {
        string model = WriteModel(PretrainedA.Model.Value);
        string docs = Dir("docs");
        string[][] truth = WriteDocs(docs, 4, 800);
        var handler = new ScriptedHandler();
        foreach (string[] page in truth)
        {
            handler.Then(GroundingSupport.Shape("openai", JsonSerializer.Serialize(new { pages = new[] { GlyphPages.Text(page) } })));
        }

        using var http = new HttpClient(handler);
        // The "openai" test price is $2/$8 per million tokens, so a cap of one tenth of a cent allows no call.
        using var client = new LlmClient(GroundingSupport.Provider("openai", http), GroundingSupport.Options(cap: 0.000001m));
        var trainer = new OcrTrainer(Options(Dir("work"), model), client);
        LabelSummary s = await trainer.LabelAsync(docs);
        Assert.True(s.BudgetReached);
        Assert.Equal(0, s.Documents);
        Assert.Empty(handler.Requests);
    }

    [Fact]
    public void A_written_container_reads_back_the_same_weights()
    {
        ModelContainer c = ModelContainer.Parse(PretrainedA.Model.Value);
        CrnnModel m = CrnnModel.FromContainer(c, OcrLimits.Default);
        ModelContainer back = ModelContainer.Parse(ContainerWriter.Write(c, m, "round trip"));
        CrnnModel m2 = CrnnModel.FromContainer(back, OcrLimits.Default);
        Assert.Equal(c.Tensors.Select(static t => t.Name), back.Tensors.Select(static t => t.Name));
        for (int i = 0; i < m.Conv.Length; i++)
        {
            Assert.Equal(m.Conv[i].Weights, m2.Conv[i].Weights);
            Assert.Equal(m.Conv[i].Bias, m2.Conv[i].Bias);
        }

        for (int l = 0; l < m.Lstm.Length; l++)
        {
            Assert.Equal(m.Lstm[l].Forward.Bias, m2.Lstm[l].Forward.Bias);
            Assert.Equal(m.Lstm[l].Reverse.WeightHh, m2.Lstm[l].Reverse.WeightHh);
        }

        Assert.Equal(m.FcWeight, m2.FcWeight);
        Assert.Equal(c.FormatVersion, back.FormatVersion);
    }

    [Fact]
    public void The_written_container_keeps_the_calibration_of_a_version_1_start()
    {
        const string Cal = "{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[0,0.5,1],\"y\":[0,0.4,0.95]}}}";
        byte[] start = TestModels.RandomContainerV1(TestModels.Small, 4, Cal);
        ModelContainer c = ModelContainer.Parse(start);
        ModelContainer back = ModelContainer.Parse(ContainerWriter.Write(c, CrnnModel.FromContainer(c, OcrLimits.Default), "x"));
        Assert.Equal(1u, back.FormatVersion);
        Assert.NotNull(back.Calibration);
        Assert.Equal(c.Calibration!.Apply(OcrScript.Latin, 0.7f), back.Calibration!.Apply(OcrScript.Latin, 0.7f));
    }

    [Fact]
    public void Alignment_finds_each_line_in_the_model_text_and_marks_the_rest()
    {
        string llm = "Total due\nAcme Corp\nsomething extra Invoice 42";
        LineLabelling.Match[] m = LineLabelling.Align(["Total due", "Acrne Corp", "zzzzzz qqqqq", "Invoice 42"], llm, 0, 0.5);
        Assert.Equal(LineAgreement.Agreed, m[0].Agreement);
        Assert.Equal(LineAgreement.Disagreed, m[1].Agreement);
        Assert.Equal("Acme Corp", m[1].LlmText);
        Assert.Equal(LineAgreement.Unaligned, m[2].Agreement);
        Assert.Equal(LineAgreement.Agreed, m[3].Agreement); // found past the words the line did not need
    }

    private static void CopyDirectory(string from, string to)
    {
        Directory.CreateDirectory(to);
        foreach (string f in Directory.GetFiles(from))
        {
            File.Copy(f, Path.Combine(to, Path.GetFileName(f)), true);
        }
    }
}
