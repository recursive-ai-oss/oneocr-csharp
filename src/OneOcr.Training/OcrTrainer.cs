using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using OneImage;
using OneOcr.Kernels;
using OneOcr.Llm;
using OneOcr.Model;
using OneOcr.Recognition;

namespace OneOcr.Training;

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals)]
[JsonSerializable(typeof(TrainingReport))]
[JsonSerializable(typeof(EvaluationReport))]
[JsonSerializable(typeof(LabelSummary))]
internal sealed partial class ReportJsonContext : JsonSerializerContext
{
}

/// <summary>
/// Fine-tunes a recognition model on a user's own documents. The language model you give it transcribes each page; the local
/// reader reads the same page line by line; a line becomes training data only when the two agree (see
/// <see cref="OcrTrainerOptions.MaxAgreementCer"/>). Lines they read differently go to a review queue
/// (<see cref="LineReview"/>), where a person can correct them and add them to the next run.
/// <para>
/// <b>Whose terms apply.</b> The labels come from the provider you configure, under your own agreement with that provider. Some
/// providers' terms limit using their output to train models; OpenAI's, for instance, exempts models used only to classify or
/// organize data and not distributed to others, which may cover an internal recognizer of your own, while other providers' terms
/// differ. Read the terms of the service you use. Labelling with a local open-weight model served through
/// <see cref="OpenAiCompatibleProvider"/> avoids the question. This library does not decide it for you.
/// </para>
/// <para>
/// Training itself runs on the CPU with fp32 master weights, and its results (weights, losses, report) are bit-identical on every
/// supported CPU path and at every degree of parallelism. Only <see cref="LabelAsync"/> and <see cref="RunAsync"/> reach the
/// network, and only through the <see cref="LlmClient"/> you pass in.
/// </para>
/// </summary>
public sealed class OcrTrainer
{
    // Scratch the trainer keeps for reuse. Only memory that was in use at once can be kept, so this does not raise the peak.
    private const long ScratchBytes = 1L << 30;
    private const long MaxDocumentBytes = 256L * 1024 * 1024;
    private static readonly string[] Extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"];

    private readonly OcrTrainerOptions _options;
    private readonly LlmClient? _llm;
    private readonly OcrLimits _limits;
    private readonly WorkStore _store;
    private readonly KernelPath _path;
    private readonly int _degree;

    /// <summary>Creates a trainer.</summary>
    /// <param name="options">The settings.</param>
    /// <param name="labeller">The client that transcribes pages; null when the labels are already in the work directory (from an earlier run or a review import), in which case <see cref="LabelAsync"/> throws.</param>
    /// <exception cref="ArgumentNullException"><paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentException">A setting is out of range.</exception>
    public OcrTrainer(OcrTrainerOptions options, LlmClient? labeller = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        _options = options;
        _llm = labeller;
        _limits = options.Limits ?? OcrLimits.Default;
        _store = new WorkStore(options.WorkDirectory, _limits);
        _path = Dispatch.Require(options.KernelPath ?? Dispatch.Best);
        _degree = options.MaxDegreeOfParallelism > 0 ? options.MaxDegreeOfParallelism : Math.Min(4, Environment.ProcessorCount);
    }

    private string OutputPath => _options.OutputModelPath ?? Path.Combine(_options.WorkDirectory, "model.onm");

    private void Say(string message) => _options.Progress?.Report(message);

    // ---- Labelling ----

    /// <summary>
    /// Reads each image in <paramref name="documentsFolder"/> (PNG, JPEG, BMP, GIF, TIFF and WebP files, this folder only, in name
    /// order) with the starting model, asks the labelling client for its transcription of the page, and stores the lines with
    /// whether the two agree. A document that an earlier call labelled (same file content) is skipped, so a stopped run can be
    /// repeated. Pages go to the provider; nothing else leaves the machine.
    /// </summary>
    /// <param name="documentsFolder">A folder of page images.</param>
    /// <param name="cancellationToken">Cancels between documents and during a provider call.</param>
    /// <returns>What was labelled. When the client's spend cap is reached the call returns what it has, with <see cref="LabelSummary.BudgetReached"/> set.</returns>
    /// <exception cref="InvalidOperationException">The trainer was created without a labelling client.</exception>
    /// <exception cref="DirectoryNotFoundException">The folder does not exist.</exception>
    /// <exception cref="OcrModelException">The starting model is not a usable container.</exception>
    /// <exception cref="OcrBudgetException">The very first page would pass a cost bound before any spend (the client's own per-call bound).</exception>
    public async Task<LabelSummary> LabelAsync(string documentsFolder, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(documentsFolder);
        LlmClient llm = _llm ?? throw new InvalidOperationException("Labelling needs a labelling client; create the trainer with one.");
        ModelContainer container = ModelContainer.Load(_options.ModelPath, _limits);
        OcrEngine engine = OcrEngine.Create(container, _limits, _path);
        int height = container.Input.Height;

        int documents = 0, already = 0, unreadable = 0, lines = 0, agreed = 0, disagreed = 0, unaligned = 0;
        bool budget = false;
        decimal cost = 0;
        var notes = new List<string>();
        foreach (string file in Directory.EnumerateFiles(documentsFolder).Where(static f => Extensions.Contains(Path.GetExtension(f), StringComparer.OrdinalIgnoreCase)).Order(StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            string name = Path.GetFileName(file);
            long size = new FileInfo(file).Length;
            if (size > MaxDocumentBytes)
            {
                unreadable++;
                notes.Add($"{name}: skipped, {size} bytes is over the {MaxDocumentBytes}-byte file limit.");
                continue;
            }

            byte[] bytes = await File.ReadAllBytesAsync(file, cancellationToken).ConfigureAwait(false);
            string id = Convert.ToHexStringLower(SHA256.HashData(bytes))[..16];
            if (_store.HasDocument(id))
            {
                already++;
                continue;
            }

            List<OcrEngine.LineSample> samples;
            OcrPage page;
            Image image;
            try
            {
                image = Image.Load(bytes, cancellationToken: cancellationToken);
            }
            catch (ImageException ex)
            {
                unreadable++;
                notes.Add($"{name}: not readable as an image ({ex.Message}).");
                continue;
            }

            {
                try
                {
                    samples = engine.ReadLineSamples(image, _degree, cancellationToken);
                    page = engine.Ocr(image, new OcrOptions { MaxDegreeOfParallelism = _degree, IncludeChars = false }, cancellationToken);
                }
                catch (OcrLimitException ex)
                {
                    unreadable++;
                    notes.Add($"{name}: skipped, over a limit ({ex.Message}).");
                    continue;
                }

                samples = [.. samples.Where(static s => TextMetrics.Normalize(s.Text).Length > 0)];
                var doc = new DocLabelsDto { Id = id, Document = name, Height = height };
                if (samples.Count > 0)
                {
                    decimal before = llm.SpentUsd;
                    LlmReadResult result;
                    try
                    {
                        result = await llm.ReadAsync(image, page, new LlmReadOptions { Outputs = LlmOutputs.Text }, cancellationToken).ConfigureAwait(false);
                    }
                    catch (OcrBudgetException ex)
                    {
                        budget = true;
                        notes.Add($"{name}: stopped, {ex.Message}");
                        break;
                    }
                    catch (OcrProviderException ex)
                    {
                        unreadable++;
                        notes.Add($"{name}: the provider failed ({ex.Failure}): {ex.Message}");
                        continue;
                    }

                    decimal spent = llm.SpentUsd - before;
                    cost += spent;
                    doc.CostUsd = spent;
                    doc.Calls = result.Calls;
                    LineLabelling.Match[] matches = LineLabelling.Align(
                        [.. samples.Select(static s => s.Text)], result.LlmText ?? result.Text ?? string.Empty, _options.MaxAgreementCer, _options.AlignMaxCer);
                    for (int i = 0; i < samples.Count; i++)
                    {
                        OcrEngine.LineSample s = samples[i];
                        LineLabelling.Match m = matches[i];
                        doc.Lines.Add(new LineLabelDto
                        {
                            Index = i, X = s.X, Y = s.Y, Width = s.Width, Height = s.Height, Local = s.Text, Confidence = s.Confidence,
                            Llm = m.LlmText, State = m.Agreement, Cer = m.Cer, CropWidth = s.CropWidth, Crop = s.Gray,
                        });
                        switch (m.Agreement)
                        {
                            case LineAgreement.Agreed: agreed++; break;
                            case LineAgreement.Disagreed: disagreed++; break;
                            default: unaligned++; break;
                        }
                    }

                    lines += samples.Count;
                }

                _store.Save(doc);
                documents++;
                Say($"labelled {name}: {doc.Lines.Count} lines");
            }
        }

        return new LabelSummary(documents, already, unreadable, lines, agreed, disagreed, unaligned, cost, budget, notes);
    }

    // ---- Training ----

    /// <summary>Labels the folder (see <see cref="LabelAsync"/>) and then trains (see <see cref="Train"/>).</summary>
    /// <param name="documentsFolder">A folder of page images.</param>
    /// <param name="cancellationToken">Cancels labelling and training; a cancelled run leaves a checkpoint and can be run again.</param>
    /// <returns>The report of the training run.</returns>
    public async Task<TrainingReport> RunAsync(string documentsFolder, CancellationToken cancellationToken = default)
    {
        LabelSummary summary = await LabelAsync(documentsFolder, cancellationToken).ConfigureAwait(false);
        Say($"labelling done: {summary.Documents} new documents, {summary.Agreed} agreed / {summary.Disagreed} disagreed / {summary.Unaligned} unaligned lines");
        return Train(cancellationToken);
    }

    private sealed record Candidate(string Id, DocLabelsDto Doc, LineLabelDto Line, string? Reference, bool Reviewed, bool HeldOut);

    /// <summary>
    /// Fine-tunes the starting model on the labelled lines in the work directory and writes the result. Lines that train: those
    /// the local reader and the language model agreed on, plus lines a reviewer corrected, outside the held-out split, whose text
    /// the model's charset can write. The held-out lines (each with the reviewer's text or else the language model's text as the
    /// reference) are read by the starting model and the tuned one, through the normal recognizer path, for the report. The tuned
    /// model keeps the starting model's calibration (the line and word confidences it reports are those of the starting model's
    /// fit; refitting needs a dedicated split of much more data than a fine-tune has). This never touches the network.
    /// <para>
    /// A checkpoint is written every <see cref="OcrTrainerOptions.CheckpointEvery"/> steps and on cancellation; calling again with
    /// the same settings resumes and ends with weights identical, bit for bit, to an uninterrupted run.
    /// </para>
    /// </summary>
    /// <param name="cancellationToken">Observed between lines of a step.</param>
    /// <returns>The report. With <see cref="OcrTrainerOptions.StopAfterSteps"/> set below <see cref="OcrTrainerOptions.Steps"/> the run stops early, writes no model, and says so in the diagnostics.</returns>
    /// <exception cref="InvalidOperationException">There is nothing to train on, or the checkpoint belongs to another run.</exception>
    /// <exception cref="OcrFormatException">A work file is damaged, or was made for a model with another input height.</exception>
    /// <exception cref="OcrModelException">The starting model is not a usable container.</exception>
    /// <exception cref="OperationCanceledException">Cancelled; a checkpoint was saved.</exception>
    public TrainingReport Train(CancellationToken cancellationToken = default)
    {
        byte[] modelBytes = File.ReadAllBytes(_options.ModelPath);
        ModelContainer container = ModelContainer.Parse(modelBytes, _limits);
        CrnnModel model = CrnnModel.FromContainer(container, _limits);
        List<Candidate> all = LoadCandidates(container, out int documents, out Dictionary<string, string> reviewed);
        var diagnostics = new List<string>();

        int agreed = all.Count(static c => c.Line.State == LineAgreement.Agreed);
        int disagreed = all.Count(static c => c.Line.State == LineAgreement.Disagreed);
        int unaligned = all.Count(static c => c.Line.State == LineAgreement.Unaligned);
        int total = agreed + disagreed + unaligned;

        var charset = new HashSet<string>(model.Charset.Skip(1), StringComparer.Ordinal);
        var trainer = new CrnnTrainer(
            model, _path, new AdamWOptions { LearningRate = _options.LearningRate, WeightDecay = _options.WeightDecay }, _limits, _options.MaxGradNorm, poolBytes: ScratchBytes);
        List<Candidate> selected = SelectTrainingLines(all, charset, out int outside);
        var lines = new List<TrainLine>(selected.Count);
        var labels = new List<string>(selected.Count);
        foreach (Candidate c in selected)
        {
            lines.Add(new TrainLine(c.Line.Crop, c.Line.CropWidth, trainer.Encode(c.Reference!)));
            labels.Add(c.Id + "\n" + c.Reference);
        }

        if (lines.Count == 0)
        {
            throw new InvalidOperationException("There are no lines to train on: none were agreed or corrected outside the held-out split. Label more documents, relax MaxAgreementCer, or import review corrections.");
        }

        if (outside > 0)
        {
            diagnostics.Add($"{outside} candidate lines were left out because their text has characters outside the model's charset.");
        }

        int steps = _options.Steps;
        int warmup = _options.WarmupSteps ?? Math.Min(20, steps / 10);
        byte[] fingerprint = Checkpoint.Fingerprint(
        [
            Convert.ToHexStringLower(SHA256.HashData(modelBytes)),
            string.Join('|', _options.Seed, steps, _options.BatchSize, _options.LearningRate.ToString("R", CultureInfo.InvariantCulture), warmup,
                _options.WeightDecay.ToString("R", CultureInfo.InvariantCulture), _options.MaxGradNorm.ToString("R", CultureInfo.InvariantCulture)),
            .. labels,
        ]);

        int[] lengths = [.. trainer.Parameters.Select(static p => p.Length)];
        long step = 0;
        double firstLoss = double.NaN, lastLoss = double.NaN;
        if (_options.Restart && File.Exists(_store.CheckpointPath))
        {
            File.Delete(_store.CheckpointPath);
        }

        Checkpoint.State? saved = Checkpoint.TryLoad(_store.CheckpointPath, fingerprint, lengths);
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
            Checkpoint.Save(_store.CheckpointPath, fingerprint, new Checkpoint.State(step, firstLoss, lastLoss, [.. trainer.Parameters], m, v));
        }

        int stopAt = _options.StopAfterSteps > 0 ? Math.Min(_options.StopAfterSteps, steps) : steps;
        int[] order = [];
        long orderEpoch = -1;
        try
        {
            while (step < stopAt)
            {
                var batch = new List<TrainLine>(_options.BatchSize);
                for (int j = 0; j < _options.BatchSize; j++)
                {
                    long position = (step * _options.BatchSize) + j;
                    long epoch = position / lines.Count;
                    if (epoch != orderEpoch)
                    {
                        order = SplitMix.Permutation(((ulong)(uint)_options.Seed << 32) ^ (ulong)epoch, lines.Count);
                        orderEpoch = epoch;
                    }

                    batch.Add(lines[order[position % lines.Count]]);
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
            // The step that was interrupted changed nothing (the weights are updated only after all gradients are in).
            Save();
            throw;
        }

        if (step < steps)
        {
            diagnostics.Add($"Stopped after {step} of {steps} steps; call Train again to continue. No model was written.");
            return new TrainingReport(
                documents, total, agreed, disagreed, unaligned, all.Count(static c => c.Reviewed), total == 0 ? 0 : agreed / (double)total, lines.Count, outside,
                all.Count(static c => c.HeldOut && c.Reference is not null), (int)step, firstLoss, lastLoss, null, null, string.Empty, diagnostics);
        }

        byte[] tuned = ContainerWriter.Write(container, model, $"{lines.Count} lines, {steps} steps, seed {_options.Seed}");
        string outputPath = OutputPath;
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
        WorkStore.WriteAtomic(outputPath, tuned);

        List<EvalLine> held = [.. all.Where(static c => c.HeldOut && !string.IsNullOrEmpty(c.Reference)).Select(static c => new EvalLine(c.Line.Crop, c.Line.CropWidth, c.Reference!))];
        BeforeAfter? evaluation = null;
        float? threshold = null;
        if (held.Count > 0)
        {
            EvaluationReport before = EvaluateLines(container, held, _limits, _path, _degree, _options.LocalFirstPrecision);
            EvaluationReport after = EvaluateLines(ModelContainer.Parse(tuned, _limits), held, _limits, _path, _degree, _options.LocalFirstPrecision);
            evaluation = new BeforeAfter(before.Rates, after.Rates);
            threshold = after.SuggestedLocalFirstThreshold;
        }
        else
        {
            diagnostics.Add("Nothing was held out with a reference text, so there are no before/after numbers.");
        }

        var report = new TrainingReport(
            documents, total, agreed, disagreed, unaligned, all.Count(static c => c.Reviewed), total == 0 ? 0 : agreed / (double)total, lines.Count, outside,
            held.Count, (int)step, firstLoss, lastLoss, evaluation, threshold, outputPath, diagnostics);
        WorkStore.WriteAtomic(_store.ReportPath, JsonSerializer.SerializeToUtf8Bytes(report, ReportJsonContext.Default.TrainingReport));
        return report;
    }

    /// <summary>
    /// The lines a run trains on: outside the held-out split, either corrected by a reviewer or agreed, with a non-empty reference
    /// whose characters the charset can write. A disagreed or unaligned line never qualifies unless a reviewer corrected it.
    /// </summary>
    private static List<Candidate> SelectTrainingLines(List<Candidate> all, HashSet<string> charset, out int outsideCharset)
    {
        var selected = new List<Candidate>();
        outsideCharset = 0;
        foreach (Candidate c in all.Where(static c => !c.HeldOut))
        {
            if (string.IsNullOrEmpty(c.Reference) || !(c.Reviewed || c.Line.State == LineAgreement.Agreed))
            {
                continue;
            }

            if (!InCharset(c.Reference, charset))
            {
                outsideCharset++;
                continue;
            }

            selected.Add(c);
        }

        return selected;
    }

    /// <summary>The ids and labels of the lines <see cref="Train"/> would use, for tests.</summary>
    internal List<(string Id, string Label)> TrainingSetForTests()
    {
        ModelContainer container = ModelContainer.Load(_options.ModelPath, _limits);
        var charset = new HashSet<string>(container.Charset.Skip(1), StringComparer.Ordinal);
        return [.. SelectTrainingLines(LoadCandidates(container, out _, out _), charset, out _).Select(static c => (c.Id, c.Reference!))];
    }

    // Linear warm-up, then a straight line down to a tenth of the peak at the last step. Plain arithmetic, so the rate is the same everywhere.
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

    private static bool InCharset(string text, HashSet<string> charset)
    {
        foreach (Rune r in text.EnumerateRunes())
        {
            if (!charset.Contains(r.ToString()))
            {
                return false;
            }
        }

        return true;
    }

    private List<Candidate> LoadCandidates(ModelContainer container, out int documents, out Dictionary<string, string> reviewed)
    {
        List<DocLabelsDto> docs = _store.LoadAll();
        documents = docs.Count;
        reviewed = _store.LoadReviewed();
        foreach (DocLabelsDto d in docs)
        {
            if (d.Height != container.Input.Height)
            {
                throw new OcrFormatException($"The labels of '{d.Document}' were made for input height {d.Height}, but the model's is {container.Input.Height}. Label again in a new work directory.");
            }
        }

        HashSet<string> held = HeldOutLineIds(docs, _options.Seed, _options.HeldOutFraction);
        var list = new List<Candidate>();
        foreach (DocLabelsDto d in docs)
        {
            foreach (LineLabelDto l in d.Lines)
            {
                string id = WorkStore.LineId(d.Id, l.Index);
                bool isReviewed = reviewed.TryGetValue(id, out string? text);
                string? reference = isReviewed ? text : (l.State == LineAgreement.Unaligned ? null : l.Llm);
                list.Add(new Candidate(id, d, l, reference is null ? null : TextMetrics.Normalize(reference), isReviewed, held.Contains(id)));
            }
        }

        return list;
    }

    /// <summary>
    /// The lines kept out of training: whole documents, chosen by a hash of the seed and the document id, at least one and at most
    /// all but one; with a single document, lines chosen the same way.
    /// </summary>
    internal static HashSet<string> HeldOutLineIds(List<DocLabelsDto> docs, int seed, double fraction)
    {
        var held = new HashSet<string>(StringComparer.Ordinal);
        if (fraction <= 0)
        {
            return held;
        }

        if (docs.Count >= 2)
        {
            int k = Math.Clamp((int)Math.Round(docs.Count * fraction), 1, docs.Count - 1);
            foreach (DocLabelsDto d in docs.OrderBy(d => Key(seed, d.Id)).ThenBy(static d => d.Id, StringComparer.Ordinal).Take(k))
            {
                foreach (LineLabelDto l in d.Lines)
                {
                    held.Add(WorkStore.LineId(d.Id, l.Index));
                }
            }
        }
        else if (docs.Count == 1 && docs[0].Lines.Count >= 2)
        {
            DocLabelsDto d = docs[0];
            int k = Math.Clamp((int)Math.Round(d.Lines.Count * fraction), 1, d.Lines.Count - 1);
            foreach (LineLabelDto l in d.Lines.OrderBy(l => Key(seed, WorkStore.LineId(d.Id, l.Index))).ThenBy(static l => l.Index).Take(k))
            {
                held.Add(WorkStore.LineId(d.Id, l.Index));
            }
        }

        return held;
    }

    private static ulong Key(int seed, string id) =>
        BitConverter.ToUInt64(SHA256.HashData(Encoding.UTF8.GetBytes(seed.ToString(CultureInfo.InvariantCulture) + ":" + id)), 0);

    // ---- Evaluation ----

    internal sealed record EvalLine(byte[] Gray, int Width, string Reference);

    /// <summary>
    /// Measures a model file against the held-out lines of a work directory: the same split, references and report numbers as a
    /// training run's "before" and "after", for any model. Never touches the network.
    /// </summary>
    /// <param name="modelPath">The model to measure.</param>
    /// <param name="options">Supplies the work directory, the seed and held-out fraction (use the run's), the limits and <see cref="OcrTrainerOptions.LocalFirstPrecision"/>.</param>
    /// <returns>Error rates on the held-out lines and the suggested <c>LocalFirst</c> threshold.</returns>
    /// <exception cref="InvalidOperationException">No held-out line has a reference text.</exception>
    public static EvaluationReport Evaluate(string modelPath, OcrTrainerOptions options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        OcrLimits limits = options.Limits ?? OcrLimits.Default;
        ModelContainer container = ModelContainer.Load(modelPath, limits);
        var trainer = new OcrTrainer(options);
        List<Candidate> all = trainer.LoadCandidates(container, out _, out _);
        List<EvalLine> held = [.. all.Where(static c => c.HeldOut && !string.IsNullOrEmpty(c.Reference)).Select(static c => new EvalLine(c.Line.Crop, c.Line.CropWidth, c.Reference!))];
        if (held.Count == 0)
        {
            throw new InvalidOperationException("No held-out line has a reference text. Label documents first (and keep HeldOutFraction above 0).");
        }

        return EvaluateLines(container, held, limits, trainer._path, trainer._degree, options.LocalFirstPrecision);
    }

    private static EvaluationReport EvaluateLines(ModelContainer container, List<EvalLine> lines, OcrLimits limits, KernelPath path, int degree, double precision)
    {
        var recognizer = new LineRecognizer(CrnnModel.FromContainer(container, limits), limits, path);
        var reads = new RecognizedLine[lines.Count];
        ParallelRun.For(lines.Count, degree, i => reads[i] = recognizer.Recognize(lines[i].Gray, lines[i].Width, null, 1), CancellationToken.None);

        var tally = new ErrorTally();
        var scored = new List<(float Confidence, bool Exact)>(lines.Count);
        for (int i = 0; i < lines.Count; i++)
        {
            tally.Add(lines[i].Reference, reads[i].Text);
            scored.Add((reads[i].Confidence, TextMetrics.Normalize(reads[i].Text) == TextMetrics.Normalize(lines[i].Reference)));
        }

        // The lowest confidence such that the lines at or above it were exact often enough, with a little support behind it.
        scored.Sort(static (a, b) => a.Confidence.CompareTo(b.Confidence));
        const int MinSupport = 5;
        var exactFrom = new int[scored.Count + 1];
        for (int i = scored.Count - 1; i >= 0; i--)
        {
            exactFrom[i] = exactFrom[i + 1] + (scored[i].Exact ? 1 : 0);
        }

        float? threshold = null;
        int support = 0;
        for (int i = 0; i < scored.Count && threshold is null; i++)
        {
            int n = scored.Count - i;
            bool firstOfValue = i == 0 || scored[i].Confidence != scored[i - 1].Confidence;
            if (firstOfValue && n >= MinSupport && exactFrom[i] >= precision * n)
            {
                threshold = scored[i].Confidence;
                support = n;
            }
        }

        return new EvaluationReport(tally.Rates(), threshold, support);
    }

    // ---- Reports as JSON ----

    /// <summary>The report as indented JSON (camelCase members).</summary>
    /// <param name="report">The report.</param>
    /// <returns>The JSON text.</returns>
    public static string ToJson(TrainingReport report) => JsonSerializer.Serialize(report ?? throw new ArgumentNullException(nameof(report)), ReportJsonContext.Default.TrainingReport);

    /// <summary>The report as indented JSON (camelCase members).</summary>
    /// <param name="report">The report.</param>
    /// <returns>The JSON text.</returns>
    public static string ToJson(EvaluationReport report) => JsonSerializer.Serialize(report ?? throw new ArgumentNullException(nameof(report)), ReportJsonContext.Default.EvaluationReport);

    /// <summary>The summary as indented JSON (camelCase members).</summary>
    /// <param name="summary">The summary.</param>
    /// <returns>The JSON text.</returns>
    public static string ToJson(LabelSummary summary) => JsonSerializer.Serialize(summary ?? throw new ArgumentNullException(nameof(summary)), ReportJsonContext.Default.LabelSummary);
}
