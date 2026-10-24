using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using OneImage;
using OneOcr.Documents;
using OneOcr.Kernels;
using OneOcr.Llm;
using OneOcr.Model;

namespace OneOcr.Training;

/// <summary>Settings of an <see cref="ExtractionTrainer"/>.</summary>
public sealed class ExtractionTrainerOptions
{
    /// <summary>The document-extraction model (<c>docextract-v0</c> container) to start from. Required.</summary>
    public required string ModelPath { get; init; }

    /// <summary>The fields of the custom document type. Required.</summary>
    public required ExtractionSchema Schema { get; init; }

    /// <summary>The folder for labels, the checkpoint and the report. Required; created when missing.</summary>
    public required string WorkDirectory { get; init; }

    /// <summary>Where the tuned model is written; default <c>model.onm</c> in the work directory.</summary>
    public string? OutputModelPath { get; init; }

    /// <summary>Seeds the new tag head, the split into training and held-out documents, and the order of training pages. Default 1.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>Optimizer steps. Default 200.</summary>
    public int Steps { get; init; } = 200;

    /// <summary>Pages per step. Default 8.</summary>
    public int BatchSize { get; init; } = 8;

    /// <summary>Peak learning rate of the tag head (linear warm-up, then linear decay to 10%). Default 3e-3.</summary>
    public double LearningRate { get; init; } = 3e-3;

    /// <summary>
    /// Learning-rate multiplier of everything except the tag head. Default 0.05: the backbone is tuned gently with the new head. 0 keeps the
    /// shipped backbone fixed and trains only the head (fast, and nothing drifts, but the head can only use features the backbone already has).
    /// </summary>
    public double BackboneScale { get; init; } = 0.05;

    /// <summary>Weight of every B-/I- tag against O in the loss (O counts 1). Default 3: most words are O, and a head that says O everywhere must not look good.</summary>
    public double FieldTagWeight { get; init; } = 3;

    /// <summary>Warm-up steps; default min(20, Steps/10).</summary>
    public int? WarmupSteps { get; init; }

    /// <summary>AdamW weight decay on weight matrices. Default 0.01.</summary>
    public double WeightDecay { get; init; } = 0.01;

    /// <summary>Global gradient-norm clip; 0 turns it off. Default 5.</summary>
    public double MaxGradNorm { get; init; } = 5;

    /// <summary>Share of documents (by a hash of their content and the seed) kept out of training for the report. Default 0.2; with fewer than two usable documents nothing is held out.</summary>
    public double HeldOutFraction { get; init; } = 0.2;

    /// <summary>
    /// Pages with more words than this are not used (labelling skips them before any model call). A training page is one attention window, and
    /// its memory grows with the square of its words; the default 384 keeps a batch of 4 pages of the shipped model near 1 GB. The model's own
    /// window is the other bound.
    /// </summary>
    public int MaxWordsPerPage { get; init; } = 384;

    /// <summary>Extra instruction for the labelling model; null for none.</summary>
    public string? Instruction { get; init; }

    /// <summary>Threads; 0 for min(4, cores).</summary>
    public int MaxDegreeOfParallelism { get; init; }

    /// <summary>Steps between checkpoints. Default 25.</summary>
    public int CheckpointEvery { get; init; } = 25;

    /// <summary>Stop after this many steps in total (0: run all); a later call continues from the checkpoint. No model is written until the last step.</summary>
    public int StopAfterSteps { get; init; }

    /// <summary>Delete an existing checkpoint and start over.</summary>
    public bool Restart { get; init; }

    /// <summary>Share of fields at or above the suggested <c>LocalFirst</c> threshold that must be exactly right. Default 0.99.</summary>
    public double LocalFirstPrecision { get; init; } = 0.99;

    /// <summary>Bounds; <see cref="OcrLimits.Default"/> when null.</summary>
    public OcrLimits? Limits { get; init; }

    /// <summary>Receives short progress lines.</summary>
    public IProgress<string>? Progress { get; init; }

    internal KernelPath? KernelPath { get; init; }

    internal void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(WorkDirectory);
        ArgumentNullException.ThrowIfNull(Schema);
        if (Steps < 1 || BatchSize < 1 || CheckpointEvery < 1 || StopAfterSteps < 0 || (WarmupSteps ?? 0) < 0 || MaxDegreeOfParallelism < 0 || MaxWordsPerPage < 1)
        {
            throw new ArgumentException("Steps, BatchSize, CheckpointEvery and MaxWordsPerPage must be positive; the other counts cannot be negative.");
        }

        if (!(LearningRate > 0) || !(WeightDecay >= 0) || !(MaxGradNorm >= 0) || !(BackboneScale >= 0) || !(FieldTagWeight > 0)
            || !(HeldOutFraction >= 0 && HeldOutFraction < 1) || !(LocalFirstPrecision > 0 && LocalFirstPrecision <= 1))
        {
            throw new ArgumentException("A rate, weight or fraction is out of range.");
        }
    }
}

/// <summary>One page for labelling: the image the labelling model sees and the page model the local reader made from it.</summary>
/// <param name="Id">A stable id (a hash of the file, for example); the label file is named after it.</param>
/// <param name="Name">A display name, such as the file name.</param>
/// <param name="Image">The page image.</param>
/// <param name="Page">The page model; its words are what the model learns to tag.</param>
public sealed record ExtractionDocument(string Id, string Name, Image Image, OcrPage Page);

/// <summary>What <see cref="ExtractionTrainer.LabelAsync(IEnumerable{ExtractionDocument}, CancellationToken)"/> did.</summary>
/// <param name="Documents">Pages sent to the labelling model in this call.</param>
/// <param name="AlreadyLabelled">Pages whose labels were already in the work directory.</param>
/// <param name="Unreadable">Pages skipped: not an image, too many words, or the provider failed.</param>
/// <param name="Usable">Pages (of those labelled now) every cited value of which is grounded.</param>
/// <param name="DroppedUngrounded">Pages kept out because the labelling model gave a value the page does not show.</param>
/// <param name="FieldsLabelled">Grounded values turned into word labels.</param>
/// <param name="CostUsd">Spent on this call.</param>
/// <param name="BudgetReached">The client's spend cap stopped the run; the pages labelled so far are kept.</param>
/// <param name="Diagnostics">Why pages were skipped.</param>
public sealed record ExtractionLabelSummary(
    int Documents, int AlreadyLabelled, int Unreadable, int Usable, int DroppedUngrounded, int FieldsLabelled, decimal CostUsd, bool BudgetReached, IReadOnlyList<string> Diagnostics);

/// <summary>Accuracy of one field on the held-out documents.</summary>
/// <param name="Field">The field name.</param>
/// <param name="Labelled">Held-out documents the labelling model gave a grounded value for.</param>
/// <param name="AccuracyBefore">Share of held-out documents where the untuned head's words equal the label's (no value and no label also count as equal).</param>
/// <param name="AccuracyAfter">The same for the tuned model.</param>
/// <param name="RecallBefore">Share of labelled documents where the untuned head found exactly the label's words.</param>
/// <param name="RecallAfter">The same for the tuned model.</param>
public sealed record ExtractionFieldReport(string Field, int Labelled, double AccuracyBefore, double AccuracyAfter, double RecallBefore, double RecallAfter);

/// <summary>The outcome of a fine-tuning run for a custom schema.</summary>
/// <param name="Documents">Pages with labels in the work directory.</param>
/// <param name="UsableDocuments">Pages usable for training and evaluation.</param>
/// <param name="DroppedUngrounded">Pages dropped for an ungrounded value.</param>
/// <param name="TrainDocuments">Pages trained on.</param>
/// <param name="HeldOutDocuments">Pages kept out for the numbers below.</param>
/// <param name="Steps">Steps run.</param>
/// <param name="FirstLoss">Loss of the first step.</param>
/// <param name="LastLoss">Loss of the last step.</param>
/// <param name="Fields">Per field, on the held-out documents.</param>
/// <param name="AccuracyBefore">Over all held-out (document, field) pairs, the untuned head.</param>
/// <param name="AccuracyAfter">The same for the tuned model.</param>
/// <param name="AgreementBefore">Among pairs where the labelling model or the local model has a value, the share where they have the same words: the untuned head.</param>
/// <param name="AgreementAfter">The same for the tuned model.</param>
/// <param name="SuggestedLocalFirstThreshold">The lowest field confidence from which at least <see cref="ExtractionTrainerOptions.LocalFirstPrecision"/> of the tuned model's values were exactly right on the held-out pages (with at least 5 values); null when none qualifies.</param>
/// <param name="ThresholdSupport">Values at or above that threshold.</param>
/// <param name="ModelPath">The tuned model file; empty when the run stopped early.</param>
/// <param name="Diagnostics">Notes.</param>
public sealed record ExtractionReport(
    int Documents, int UsableDocuments, int DroppedUngrounded, int TrainDocuments, int HeldOutDocuments, int Steps, double FirstLoss, double LastLoss,
    IReadOnlyList<ExtractionFieldReport> Fields, double AccuracyBefore, double AccuracyAfter, double AgreementBefore, double AgreementAfter,
    float? SuggestedLocalFirstThreshold, int ThresholdSupport, string ModelPath, IReadOnlyList<string> Diagnostics);

internal sealed class ExtractLabelsDto
{
    public string Format { get; set; } = ExtractionTrainer.LabelsFormat;
    public int Version { get; set; } = 1;
    public string Id { get; set; } = "";
    public string Document { get; set; } = "";
    public double Width { get; set; }
    public double Height { get; set; }
    public decimal CostUsd { get; set; }
    public int Calls { get; set; }

    /// <summary>Whether the page may train and evaluate (no ungrounded value).</summary>
    public bool Usable { get; set; }

    public List<string> Words { get; set; } = [];

    /// <summary>Four numbers per word: left, top, right, bottom.</summary>
    public List<double> Boxes { get; set; } = [];

    public List<ExtractLabelDto> Fields { get; set; } = [];
    public List<string> Notes { get; set; } = [];
}

internal sealed class ExtractLabelDto
{
    public string Name { get; set; } = "";
    public string Value { get; set; } = "";

    /// <summary>Indices into <see cref="ExtractLabelsDto.Words"/> (words with text), ascending.</summary>
    public List<int> Words { get; set; } = [];

    /// <summary>False when the words are not one run; such a field is neither trained on nor scored.</summary>
    public bool Contiguous { get; set; }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = false)]
[JsonSerializable(typeof(ExtractLabelsDto))]
internal sealed partial class ExtractJsonContext : JsonSerializerContext
{
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = true)]
[JsonSerializable(typeof(ExtractionReport))]
[JsonSerializable(typeof(ExtractionLabelSummary))]
internal sealed partial class ExtractReportJsonContext : JsonSerializerContext
{
}

/// <summary>
/// Fine-tunes the document-extraction model for a custom document type from the user's own pages. An LLM reads each page for the
/// schema's fields; only values the grounding step finds on the page (<c>Grounded</c> or <c>GroundedNormalized</c>) become labels, so
/// the local model learns from words that are really there. The labelled words become BIO tags for a new tag head on the shipped
/// backbone (docs/spec/docextract-backward.md); a page with a value the page does not show is left out entirely. A share of the pages
/// (chosen by a hash of their content) is held out and scored before and after, with the agreement between the local model and the
/// LLM and a suggested <c>LocalFirst</c> threshold. Training is deterministic for a seed and resumes from a checkpoint to the same
/// result; labelling stops when the client's spend cap is reached.
/// </summary>
/// <remarks>
/// Labelling sends page images and words to the provider the client is configured with, and nothing else in this class touches the network.
/// Using a provider's output to train a model is governed by your own agreement with that provider; read it first, and consider a local
/// open-weight model behind the OpenAI-compatible provider for labelling.
/// </remarks>
public sealed class ExtractionTrainer
{
    internal const string LabelsFormat = "oneocr.extraction-labels";
    private const long MaxDocumentBytes = 256L * 1024 * 1024;
    private static readonly string[] Extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"];

    private readonly ExtractionTrainerOptions _options;
    private readonly LlmClient? _llm;
    private readonly OcrLimits _limits;
    private readonly KernelPath _path;
    private readonly int _degree;
    private readonly string _labelDir;

    /// <summary>Creates a trainer.</summary>
    /// <param name="options">The settings.</param>
    /// <param name="labeller">The client that reads pages for the schema; null when the labels are already in the work directory, in which case labelling throws.</param>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentException">A setting is out of range.</exception>
    public ExtractionTrainer(ExtractionTrainerOptions options, LlmClient? labeller = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        _options = options;
        _llm = labeller;
        _limits = options.Limits ?? OcrLimits.Default;
        _path = Dispatch.Require(options.KernelPath ?? Dispatch.Best);
        _degree = options.MaxDegreeOfParallelism > 0 ? options.MaxDegreeOfParallelism : Math.Min(4, Environment.ProcessorCount);
        _labelDir = Path.Combine(options.WorkDirectory, "extract-labels");
        Directory.CreateDirectory(_labelDir);
    }

    private void Say(string message) => _options.Progress?.Report(message);

    private string LabelPath(string id) => Path.Combine(_labelDir, id + ".json");

    private static bool ValidId(string id) => id.Length is > 0 and <= 64 && id.All(static c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_');

    // ---- Labelling ----

    /// <summary>
    /// Reads every image in a folder with the local engine and labels the pages (see <see cref="LabelAsync(IEnumerable{ExtractionDocument}, CancellationToken)"/>).
    /// A page is identified by a hash of its file, so a changed file is labelled again and an unchanged one never costs twice.
    /// </summary>
    /// <param name="documentsFolder">A folder of page images.</param>
    /// <param name="engine">The engine that reads the pages; null loads the bundled models.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>What was labelled.</returns>
    /// <exception cref="InvalidOperationException">The trainer has no labelling client.</exception>
    public async Task<ExtractionLabelSummary> LabelFolderAsync(string documentsFolder, OcrEngine? engine = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(documentsFolder);
        _ = _llm ?? throw new InvalidOperationException("Labelling needs a labelling client; create the trainer with one.");
        int unreadable = 0;
        var notes = new List<string>();
        OcrEngine? ocr = engine;
        IEnumerable<ExtractionDocument> Pages()
        {
            foreach (string file in Directory.EnumerateFiles(documentsFolder).Where(static f => Extensions.Contains(Path.GetExtension(f), StringComparer.OrdinalIgnoreCase)).Order(StringComparer.Ordinal))
            {
                cancellationToken.ThrowIfCancellationRequested();
                string name = Path.GetFileName(file);
                if (new FileInfo(file).Length > MaxDocumentBytes)
                {
                    unreadable++;
                    notes.Add($"{name}: skipped, over the {MaxDocumentBytes}-byte file limit.");
                    continue;
                }

                byte[] bytes = File.ReadAllBytes(file);
                string id = Convert.ToHexStringLower(SHA256.HashData(bytes))[..16];
                if (File.Exists(LabelPath(id)))
                {
                    yield return new ExtractionDocument(id, name, null!, null!);
                    continue;
                }

                Image image;
                OcrPage page;
                try
                {
                    image = Image.Load(bytes, cancellationToken: cancellationToken);
                    ocr ??= OcrEngine.LoadBundled(_limits);
                    page = ocr.Ocr(image, new OcrOptions { MaxDegreeOfParallelism = _degree, IncludeChars = false }, cancellationToken);
                }
                catch (ImageException ex)
                {
                    unreadable++;
                    notes.Add($"{name}: not readable as an image ({ex.Message}).");
                    continue;
                }
                catch (OcrLimitException ex)
                {
                    unreadable++;
                    notes.Add($"{name}: skipped, over a limit ({ex.Message}).");
                    continue;
                }

                yield return new ExtractionDocument(id, name, image, page);
            }
        }

        ExtractionLabelSummary s = await LabelAsync(Pages(), cancellationToken).ConfigureAwait(false);
        return s with { Unreadable = s.Unreadable + unreadable, Diagnostics = [.. notes, .. s.Diagnostics] };
    }

    /// <summary>
    /// Asks the labelling model for the schema's fields on each page and stores the grounded ones as word labels. A page whose id already has
    /// labels is not sent again. A value the grounding step cannot find on the page marks the whole page unusable (the model is then not
    /// trained on a page where it might have been taught the wrong words as "not a field"); a value computed rather than printed is not labelled.
    /// </summary>
    /// <param name="documents">The pages.</param>
    /// <param name="cancellationToken">Cancels the work.</param>
    /// <returns>What was labelled. When the client's spend cap is reached the call returns what it has, with <see cref="ExtractionLabelSummary.BudgetReached"/> set.</returns>
    /// <exception cref="InvalidOperationException">The trainer has no labelling client.</exception>
    public async Task<ExtractionLabelSummary> LabelAsync(IEnumerable<ExtractionDocument> documents, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(documents);
        LlmClient llm = _llm ?? throw new InvalidOperationException("Labelling needs a labelling client; create the trainer with one.");
        string schemaJson = _options.Schema.ToJsonSchema();
        int sent = 0, already = 0, unreadable = 0, usable = 0, dropped = 0, fields = 0;
        decimal cost = 0;
        bool budget = false;
        var notes = new List<string>();
        int maxWords = Math.Min(_options.MaxWordsPerPage, DocLimit());
        foreach (ExtractionDocument d in documents)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!ValidId(d.Id))
            {
                throw new ArgumentException($"The document id '{d.Id}' must be 1 to 64 letters, digits, '-' or '_'.", nameof(documents));
            }

            if (File.Exists(LabelPath(d.Id)))
            {
                already++;
                continue;
            }

            var words = new List<OcrWord>();
            var map = new Dictionary<int, int>();
            for (int i = 0; i < d.Page.Words.Count; i++)
            {
                if (d.Page.Words[i].Text.Length > 0)
                {
                    map[i] = words.Count;
                    words.Add(d.Page.Words[i]);
                }
            }

            if (words.Count == 0 || words.Count > maxWords)
            {
                unreadable++;
                notes.Add($"{d.Name}: skipped, {words.Count} words (needs 1 to {maxWords}).");
                continue;
            }

            decimal before = llm.SpentUsd;
            LlmReadResult result;
            try
            {
                result = await llm.ReadAsync(
                    d.Image, d.Page, new LlmReadOptions { Outputs = LlmOutputs.Json, JsonSchema = schemaJson, SchemaName = "extraction", Instruction = _options.Instruction },
                    cancellationToken).ConfigureAwait(false);
            }
            catch (OcrBudgetException ex)
            {
                budget = true;
                notes.Add($"{d.Name}: stopped, {ex.Message}");
                break;
            }
            catch (OcrProviderException ex)
            {
                unreadable++;
                notes.Add($"{d.Name}: the provider failed ({ex.Failure}): {ex.Message}");
                continue;
            }

            decimal spent = llm.SpentUsd - before;
            cost += spent;
            sent++;
            var dto = new ExtractLabelsDto
            {
                Id = d.Id, Document = d.Name, Width = d.Page.Width, Height = d.Page.Height, CostUsd = spent, Calls = result.Calls, Usable = true,
                Words = [.. words.Select(static w => w.Text)],
            };
            foreach (OcrWord w in words)
            {
                dto.Boxes.Add(w.Box.Left);
                dto.Boxes.Add(w.Box.Top);
                dto.Boxes.Add(w.Box.Right);
                dto.Boxes.Add(w.Box.Bottom);
            }

            foreach (GroundedField g in result.Fields)
            {
                string name = g.Path.StartsWith('/') ? g.Path[1..] : g.Path;
                if (!_options.Schema.Fields.Any(f => f.Name == name))
                {
                    continue;
                }

                if (g.State == GroundingState.Ungrounded)
                {
                    dto.Usable = false;
                    dto.Notes.Add($"'{name}' = '{g.Value}' is not on the page.");
                    continue;
                }

                if (g.State == GroundingState.Derived)
                {
                    dto.Notes.Add($"'{name}' is computed, not printed; not labelled.");
                    continue;
                }

                var idx = new SortedSet<int>();
                bool mapped = true;
                foreach (LlmWordRef r in g.Words)
                {
                    if (r.PageIndex == 0 && map.TryGetValue(r.WordIndex, out int ne))
                    {
                        idx.Add(ne);
                    }
                    else
                    {
                        mapped = false;
                    }
                }

                if (idx.Count == 0 || !mapped)
                {
                    dto.Notes.Add($"'{name}': grounded, but its words could not be placed; not labelled.");
                    continue;
                }

                bool run = idx.Max - idx.Min + 1 == idx.Count;
                dto.Fields.Add(new ExtractLabelDto { Name = name, Value = g.Value, Words = [.. idx], Contiguous = run });
                fields++;
            }

            if (dto.Usable)
            {
                usable++;
            }
            else
            {
                dropped++;
            }

            WorkStore.WriteAtomic(LabelPath(d.Id), JsonSerializer.SerializeToUtf8Bytes(dto, ExtractJsonContext.Default.ExtractLabelsDto));
            Say($"labelled {d.Name}: {dto.Fields.Count} fields{(dto.Usable ? "" : ", dropped (ungrounded value)")}");
        }

        return new ExtractionLabelSummary(sent, already, unreadable, usable, dropped, fields, cost, budget, notes);
    }

    private int DocLimit()
    {
        ModelContainer c = ModelContainer.Load(_options.ModelPath, _limits);
        return (c.WordInput ?? throw new OcrModelException("Not a docextract model.")).MaxWords;
    }

    // ---- Training ----

    /// <summary>Labels the folder with the local engine and trains (see <see cref="LabelFolderAsync"/> and <see cref="Train"/>).</summary>
    /// <param name="documentsFolder">A folder of page images.</param>
    /// <param name="engine">The engine that reads the pages; null loads the bundled models.</param>
    /// <param name="cancellationToken">Cancels the work; a cancelled run leaves a checkpoint and can be run again.</param>
    /// <returns>The report.</returns>
    public async Task<ExtractionReport> RunAsync(string documentsFolder, OcrEngine? engine = null, CancellationToken cancellationToken = default)
    {
        await LabelFolderAsync(documentsFolder, engine, cancellationToken).ConfigureAwait(false);
        return Train(cancellationToken);
    }

    private sealed record Labelled(ExtractLabelsDto Doc, string Hash, bool HeldOut);

    private sealed record Truth(string Field, int[] Words);

    /// <summary>
    /// Trains the tag head (and, with a non-zero <see cref="ExtractionTrainerOptions.BackboneScale"/>, the rest) on the usable labelled pages outside
    /// the held-out split, writes the model, and scores the held-out pages before and after. Needs labels in the work directory (from
    /// <see cref="LabelAsync(IEnumerable{ExtractionDocument}, CancellationToken)"/> or an earlier run). Never touches the network.
    /// Resumes from the work directory's checkpoint when the model, schema, settings and labels are the same, to the bit-identical result.
    /// </summary>
    /// <param name="cancellationToken">Cancels training; the checkpoint is kept.</param>
    /// <returns>The report; with <see cref="ExtractionTrainerOptions.StopAfterSteps"/> short of the run, a report without a model.</returns>
    /// <exception cref="InvalidOperationException">There are no usable labelled pages.</exception>
    /// <exception cref="OcrFormatException">A label file is not valid.</exception>
    /// <exception cref="OcrModelException">The model is not a <c>docextract-v0</c> container.</exception>
    public ExtractionReport Train(CancellationToken cancellationToken = default)
    {
        var diagnostics = new List<string>(_options.Schema.Notes);
        byte[] baseBytes = File.ReadAllBytes(_options.ModelPath);
        ModelContainer baseContainer = ModelContainer.Parse(baseBytes, _limits);
        string[] classes = [.. _options.Schema.Fields.Select(static f => f.Name)];
        byte[] headBytes = DocContainerWriter.WithNewHead(baseContainer, classes, _options.Seed, _limits);
        ModelContainer head = ModelContainer.Parse(headBytes, _limits);
        DocExtractModel model = DocExtractModel.FromContainer(head, _limits);

        List<Labelled> all = LoadLabels(out int total, out int unusable);
        if (all.Count == 0)
        {
            throw new InvalidOperationException("There are no usable labelled pages. Label documents first.");
        }

        List<Labelled> train = [.. all.Where(static l => !l.HeldOut)];
        List<Labelled> held = [.. all.Where(static l => l.HeldOut)];
        if (held.Count == 0)
        {
            diagnostics.Add("Nothing was held out (too few usable pages or HeldOutFraction 0), so there are no before/after numbers.");
        }

        var examples = new List<DocExample>(train.Count);
        int labelledFields = 0, skippedFields = 0;
        foreach (Labelled l in train)
        {
            examples.Add(Example(model, l.Doc, classes, ref labelledFields, ref skippedFields));
        }

        if (skippedFields > 0)
        {
            diagnostics.Add($"{skippedFields} grounded values were not used for training because their words are not one run; those words are ignored by the loss.");
        }

        if (labelledFields == 0)
        {
            diagnostics.Add("No training page has a labelled field: the model can only learn to tag nothing.");
        }

        int steps = _options.Steps;
        int warmup = _options.WarmupSteps ?? Math.Min(20, steps / 10);
        float[] tagWeights = new float[model.Tags];
        Array.Fill(tagWeights, (float)_options.FieldTagWeight);
        tagWeights[0] = 1f;
        var trainer = new DocTrainer(
            model, _path,
            new DocTrainSettings
            {
                Adam = new AdamWOptions { LearningRate = _options.LearningRate, WeightDecay = _options.WeightDecay },
                BackboneScale = _options.BackboneScale,
                Weights = new DocLossWeights { Tag = 1f },
                TagClassWeights = tagWeights,
                MaxGradNorm = _options.MaxGradNorm,
            });

        byte[] fingerprint = Checkpoint.Fingerprint(
        [
            Convert.ToHexStringLower(SHA256.HashData(baseBytes)),
            string.Join('|', classes),
            string.Join('|', _options.Seed, steps, _options.BatchSize, F(_options.LearningRate), warmup, F(_options.WeightDecay), F(_options.MaxGradNorm), F(_options.BackboneScale),
                F(_options.FieldTagWeight), _options.MaxWordsPerPage),
            .. train.Select(static l => l.Doc.Id + ":" + l.Hash),
        ]);

        int[] lengths = [.. trainer.Parameters.Select(static p => p.Length)];
        string checkpointPath = Path.Combine(_options.WorkDirectory, "extract-checkpoint.bin");
        long step = 0;
        double firstLoss = double.NaN, lastLoss = double.NaN;
        if (_options.Restart && File.Exists(checkpointPath))
        {
            File.Delete(checkpointPath);
        }

        Checkpoint.State? saved = Checkpoint.TryLoad(checkpointPath, fingerprint, lengths);
        if (saved is not null)
        {
            for (int i = 0; i < lengths.Length; i++)
            {
                saved.Weights[i].CopyTo(trainer.Parameters[i], 0);
            }

            trainer.Optimizer.Restore(saved.Step, saved.M, saved.V);
            (step, firstLoss, lastLoss) = (saved.Step, saved.FirstLoss, saved.LastLoss);
            Say($"resuming at step {step} of {steps}");
        }

        void Save()
        {
            (float[][] m, float[][] v) = trainer.Optimizer.Moments();
            Checkpoint.Save(checkpointPath, fingerprint, new Checkpoint.State(step, firstLoss, lastLoss, [.. trainer.Parameters], m, v));
        }

        int stopAt = _options.StopAfterSteps > 0 ? Math.Min(_options.StopAfterSteps, steps) : steps;
        int[] order = [];
        long orderEpoch = -1;
        try
        {
            while (step < stopAt)
            {
                var batch = new List<DocExample>(_options.BatchSize);
                for (int j = 0; j < _options.BatchSize; j++)
                {
                    long position = (step * _options.BatchSize) + j;
                    long epoch = position / examples.Count;
                    if (epoch != orderEpoch)
                    {
                        order = SplitMix.Permutation(((ulong)(uint)_options.Seed << 32) ^ (ulong)epoch, examples.Count);
                        orderEpoch = epoch;
                    }

                    batch.Add(examples[order[position % examples.Count]]);
                }

                double loss = trainer.Step(batch, _degree, LearningRateAt((int)step, steps, warmup), cancellationToken);
                if (step == 0)
                {
                    firstLoss = loss;
                }

                lastLoss = loss;
                step++;
                if (step % _options.CheckpointEvery == 0 || step == stopAt)
                {
                    Save();
                    Say($"step {step}/{steps} loss {loss:F4}");
                }
            }
        }
        catch (OperationCanceledException)
        {
            // The interrupted step changed nothing: weights are updated only after all gradients are in.
            Save();
            throw;
        }

        if (step < steps)
        {
            diagnostics.Add($"Stopped after {step} of {steps} steps; call Train again to continue. No model was written.");
            return new ExtractionReport(
                total, all.Count, unusable, train.Count, held.Count, (int)step, firstLoss, lastLoss, [], double.NaN, double.NaN, double.NaN, double.NaN, null, 0, string.Empty, diagnostics);
        }

        byte[] tuned = DocContainerWriter.WithTrained(head, model, $"{train.Count} pages, {steps} steps, seed {_options.Seed}");
        string outputPath = _options.OutputModelPath ?? Path.Combine(_options.WorkDirectory, "model.onm");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
        WorkStore.WriteAtomic(outputPath, tuned);

        List<ExtractionFieldReport> perField = [];
        double accBefore = double.NaN, accAfter = double.NaN, agreeBefore = double.NaN, agreeAfter = double.NaN;
        float? threshold = null;
        int support = 0;
        if (held.Count > 0)
        {
            Score before = Evaluate(headBytes, held, classes, cancellationToken);
            Score after = Evaluate(tuned, held, classes, cancellationToken);
            for (int c = 0; c < classes.Length; c++)
            {
                perField.Add(new ExtractionFieldReport(
                    classes[c], after.Labelled[c], Ratio(before.Correct[c], held.Count), Ratio(after.Correct[c], held.Count), Ratio(before.Found[c], before.Labelled[c]),
                    Ratio(after.Found[c], after.Labelled[c])));
            }

            long pairs = (long)held.Count * classes.Length;
            accBefore = Ratio(before.Correct.Sum(), pairs);
            accAfter = Ratio(after.Correct.Sum(), pairs);
            agreeBefore = Ratio(before.AgreeCorrect, before.AgreePairs);
            agreeAfter = Ratio(after.AgreeCorrect, after.AgreePairs);
            (threshold, support) = Threshold(after.Predictions, _options.LocalFirstPrecision);
        }

        var report = new ExtractionReport(
            total, all.Count, unusable, train.Count, held.Count, (int)step, firstLoss, lastLoss, perField, accBefore, accAfter, agreeBefore, agreeAfter, threshold, support,
            outputPath, diagnostics);
        WorkStore.WriteAtomic(Path.Combine(_options.WorkDirectory, "extract-report.json"), JsonSerializer.SerializeToUtf8Bytes(report, ExtractReportJsonContext.Default.ExtractionReport));
        return report;

        static string F(double v) => v.ToString("R", CultureInfo.InvariantCulture);
    }

    private static double Ratio(long n, long d) => d == 0 ? double.NaN : n / (double)d;

    private double LearningRateAt(int step, int steps, int warmup)
    {
        double peak = _options.LearningRate;
        if (warmup > 0 && step < warmup)
        {
            return peak * (step + 1) / warmup;
        }

        int span = steps - warmup - 1;
        double progress = span <= 0 ? 1.0 : (double)(step - warmup) / span;
        return peak * (1.0 - (0.9 * progress));
    }

    // Usable pages with validated labels, split by a hash of (seed, id); the ids sort the same on every run and machine.
    private List<Labelled> LoadLabels(out int total, out int unusable)
    {
        var docs = new List<Labelled>();
        total = 0;
        unusable = 0;
        int maxWords = Math.Min(_options.MaxWordsPerPage, DocLimit());
        foreach (string file in Directory.EnumerateFiles(_labelDir, "*.json").Order(StringComparer.Ordinal))
        {
            var fi = new FileInfo(file);
            if (fi.Length > _limits.MaxModelFileBytes)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxModelFileBytes), _limits.MaxModelFileBytes, fi.Length);
            }

            byte[] bytes = File.ReadAllBytes(file);
            ExtractLabelsDto doc;
            try
            {
                var reader = new Utf8JsonReader(bytes, new JsonReaderOptions { MaxDepth = _limits.MaxJsonDepth });
                doc = JsonSerializer.Deserialize(ref reader, ExtractJsonContext.Default.ExtractLabelsDto) ?? throw new OcrFormatException($"'{file}' is empty.");
            }
            catch (JsonException ex)
            {
                throw new OcrFormatException($"'{file}' is not valid: {ex.Message}", ex);
            }

            if (doc.Format != LabelsFormat || doc.Version != 1 || !ValidId(doc.Id) || doc.Id != Path.GetFileNameWithoutExtension(file))
            {
                throw new OcrFormatException($"'{file}' is not an extraction label file of this version.");
            }

            if (doc.Words.Count > _limits.MaxWordsPerPage)
            {
                throw new OcrLimitException(nameof(OcrLimits.MaxWordsPerPage), _limits.MaxWordsPerPage, doc.Words.Count);
            }

            if (doc.Boxes.Count != 4L * doc.Words.Count || !(doc.Width > 0) || !(doc.Height > 0) || !double.IsFinite(doc.Width) || !double.IsFinite(doc.Height)
                || doc.Boxes.Any(static b => !double.IsFinite(b)) || doc.Fields.Any(f => f.Words.Any(w => w < 0 || w >= doc.Words.Count)))
            {
                throw new OcrFormatException($"'{file}' has boxes or word indices that do not match its words.");
            }

            total++;
            if (!doc.Usable || doc.Words.Count == 0 || doc.Words.Count > maxWords)
            {
                unusable++;
                continue;
            }

            docs.Add(new Labelled(doc, Convert.ToHexStringLower(SHA256.HashData(bytes)), false));
        }

        // Split on the usable pages: the held-out ones are those with the smallest keys.
        int n = docs.Count;
        int heldCount = _options.HeldOutFraction <= 0 || n < 2 ? 0 : Math.Min(n - 1, Math.Max(1, (int)Math.Round(n * _options.HeldOutFraction, MidpointRounding.ToEven)));
        HashSet<string> heldIds = [.. docs.OrderBy(d => Key(_options.Seed, d.Doc.Id)).ThenBy(static d => d.Doc.Id, StringComparer.Ordinal).Take(heldCount).Select(static d => d.Doc.Id)];
        return [.. docs.Select(d => d with { HeldOut = heldIds.Contains(d.Doc.Id) })];
    }

    private static ulong Key(int seed, string id) =>
        BitConverter.ToUInt64(SHA256.HashData(Encoding.UTF8.GetBytes(seed.ToString(CultureInfo.InvariantCulture) + ":" + id)), 0);

    private static OcrBox[] Boxes(ExtractLabelsDto d) =>
        [.. Enumerable.Range(0, d.Words.Count).Select(i => new OcrBox(d.Boxes[4 * i], d.Boxes[(4 * i) + 1], d.Boxes[(4 * i) + 2], d.Boxes[(4 * i) + 3]))];

    // BIO tags per word: O, then B and I per class. Words of a value that is not one run, and words claimed by two fields, are ignored by the loss.
    private static DocExample Example(DocExtractModel m, ExtractLabelsDto d, string[] classes, ref int labelled, ref int skipped)
    {
        DocFeatures f = DocFeatures.Build(m, d.Words, Boxes(d), d.Width, d.Height);
        int[] tags = new int[d.Words.Count];
        var claimed = new int[d.Words.Count];
        foreach (ExtractLabelDto field in d.Fields)
        {
            foreach (int w in field.Words)
            {
                claimed[w]++;
            }
        }

        foreach (ExtractLabelDto field in d.Fields)
        {
            int c = Array.IndexOf(classes, field.Name);
            if (c < 0)
            {
                continue;
            }

            if (!field.Contiguous)
            {
                skipped++;
                foreach (int w in field.Words)
                {
                    tags[w] = -1;
                }

                continue;
            }

            labelled++;
            for (int k = 0; k < field.Words.Count; k++)
            {
                int w = field.Words[k];
                tags[w] = claimed[w] > 1 ? -1 : (k == 0 ? 1 + (2 * c) : 2 + (2 * c));
            }
        }

        return new DocExample(f, new DocTargets(tags));
    }

    // ---- Scoring ----

    private sealed record Prediction(float Confidence, bool Exact);

    private sealed class Score
    {
        public required int[] Correct { get; init; }
        public required int[] Found { get; init; }
        public required int[] Labelled { get; init; }
        public long AgreePairs { get; set; }
        public long AgreeCorrect { get; set; }
        public List<Prediction> Predictions { get; } = [];
    }

    private Score Evaluate(byte[] modelBytes, List<Labelled> docs, string[] classes, CancellationToken ct)
    {
        DocumentExtractor extractor = DocumentExtractor.Load(modelBytes, _limits).WithPath(_path);
        var results = new DocumentExtraction[docs.Count];
        ParallelRun.For(docs.Count, _degree, i =>
        {
            ExtractLabelsDto d = docs[i].Doc;
            results[i] = extractor.RunWords(
                d.Words, Boxes(d), [.. Enumerable.Range(0, d.Words.Count)], [.. Enumerable.Repeat(OcrSource.Ocr, d.Words.Count)], d.Width, d.Height, null, 1, ct);
        }, ct);

        var s = new Score { Correct = new int[classes.Length], Found = new int[classes.Length], Labelled = new int[classes.Length] };
        for (int i = 0; i < docs.Count; i++)
        {
            ExtractLabelsDto d = docs[i].Doc;
            for (int c = 0; c < classes.Length; c++)
            {
                ExtractLabelDto? label = d.Fields.FirstOrDefault(f => f.Name == classes[c]);
                if (label is { Contiguous: false })
                {
                    // Not scored: there is no single answer to compare with. Counted as correct so it lowers no accuracy.
                    s.Correct[c]++;
                    continue;
                }

                ExtractedField? got = results[i].Fields.FirstOrDefault(f => f.Id == classes[c]);
                bool exact = label is null ? got is null : got is not null && got.WordIndices.OrderBy(static x => x).SequenceEqual(label.Words);
                if (exact)
                {
                    s.Correct[c]++;
                }

                if (label is not null)
                {
                    s.Labelled[c]++;
                    if (exact)
                    {
                        s.Found[c]++;
                    }
                }

                if (label is not null || got is not null)
                {
                    s.AgreePairs++;
                    if (exact)
                    {
                        s.AgreeCorrect++;
                    }
                }

                if (got is not null)
                {
                    s.Predictions.Add(new Prediction(got.Confidence, exact));
                }
            }
        }

        return s;
    }

    // The lowest confidence from which at least `precision` of the values were exactly right, with a little support behind it.
    private static (float? Threshold, int Support) Threshold(List<Prediction> predictions, double precision)
    {
        const int MinSupport = 5;
        List<Prediction> sorted = [.. predictions.OrderBy(static p => p.Confidence)];
        var exactFrom = new int[sorted.Count + 1];
        for (int i = sorted.Count - 1; i >= 0; i--)
        {
            exactFrom[i] = exactFrom[i + 1] + (sorted[i].Exact ? 1 : 0);
        }

        for (int i = 0; i < sorted.Count; i++)
        {
            int n = sorted.Count - i;
            bool firstOfValue = i == 0 || sorted[i].Confidence != sorted[i - 1].Confidence;
            if (firstOfValue && n >= MinSupport && exactFrom[i] >= precision * n)
            {
                return (sorted[i].Confidence, n);
            }
        }

        return (null, 0);
    }

    /// <summary>The report as indented JSON (camelCase members).</summary>
    /// <param name="report">The report.</param>
    /// <returns>JSON text.</returns>
    public static string ToJson(ExtractionReport report) =>
        JsonSerializer.Serialize(report ?? throw new ArgumentNullException(nameof(report)), ExtractReportJsonContext.Default.ExtractionReport);

    /// <summary>The labelling summary as indented JSON (camelCase members).</summary>
    /// <param name="summary">The summary.</param>
    /// <returns>JSON text.</returns>
    public static string ToJson(ExtractionLabelSummary summary) =>
        JsonSerializer.Serialize(summary ?? throw new ArgumentNullException(nameof(summary)), ExtractReportJsonContext.Default.ExtractionLabelSummary);
}
