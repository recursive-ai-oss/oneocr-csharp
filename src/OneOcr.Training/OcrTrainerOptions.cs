using OneOcr.Kernels;

namespace OneOcr.Training;

/// <summary>Settings of an <see cref="OcrTrainer"/> run. Every value has a default except the two paths.</summary>
public sealed class OcrTrainerOptions
{
    /// <summary>The model to start from: a container file with architecture <c>crnn-v0</c>, normally the recognizer that ships with the library. Required.</summary>
    public required string ModelPath { get; init; }

    /// <summary>
    /// The run's directory: labelled documents, the reviewer's corrections, the checkpoint and the report. Created if missing.
    /// Running again with the same directory continues where the last run stopped. Required.
    /// </summary>
    public required string WorkDirectory { get; init; }

    /// <summary>Where the fine-tuned model is written. Default: <c>model.onm</c> inside <see cref="WorkDirectory"/>.</summary>
    public string? OutputModelPath { get; init; }

    /// <summary>Seed of the batch order and of the held-out split. Same seed, same labels and same options give the same weights on every machine. Default 1.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>Training steps in total (a resumed run counts the steps already taken). Default 200.</summary>
    public int Steps { get; init; } = 200;

    /// <summary>Lines per step. Default 16.</summary>
    public int BatchSize { get; init; } = 16;

    /// <summary>
    /// Peak learning rate. The rate rises linearly over <see cref="WarmupSteps"/> and then falls linearly to a tenth of this value at
    /// the last step. Fine-tuning wants a small rate; default 1e-4.
    /// </summary>
    public double LearningRate { get; init; } = 1e-4;

    /// <summary>Steps of linear warm-up. Default: a tenth of <see cref="Steps"/>, at most 20.</summary>
    public int? WarmupSteps { get; init; }

    /// <summary>Decoupled weight decay on weight matrices (not biases). Default 0.01.</summary>
    public double WeightDecay { get; init; } = 0.01;

    /// <summary>Clip the global gradient norm to this value; 0 turns clipping off. Default 5.</summary>
    public double MaxGradNorm { get; init; } = 5;

    /// <summary>
    /// Share of the documents kept out of training and used for the before/after report. With a single document the split is by
    /// line. Never less than one document (or line) when at least two exist. 0 keeps nothing out. Default 0.2.
    /// </summary>
    public double HeldOutFraction { get; init; } = 0.2;

    /// <summary>
    /// How far a local line may differ from the language model's text and still count as agreeing, as a character error rate. 0 (the
    /// default) means the two must read the same after whitespace is normalized. The agreed line trains with the language model's text.
    /// </summary>
    public double MaxAgreementCer { get; init; }

    /// <summary>How far a stretch of the language model's text may be from a local line, as a character error rate, and still be taken as the same line. Beyond it the line is unaligned. Default 0.5.</summary>
    public double AlignMaxCer { get; init; } = 0.5;

    /// <summary>Most threads to use; 0 means min(4, cores). Weights and report do not depend on it.</summary>
    public int MaxDegreeOfParallelism { get; init; }

    /// <summary>Steps between checkpoints. Default 25.</summary>
    public int CheckpointEvery { get; init; } = 25;

    /// <summary>Stops after this many steps in total and leaves a checkpoint, without writing the model; 0 (the default) runs to <see cref="Steps"/>.</summary>
    public int StopAfterSteps { get; init; }

    /// <summary>
    /// Discards any checkpoint in the work directory and starts from the starting model. Needed after labels change (a review
    /// import, more documents), because a checkpoint only continues the run whose model, labels and settings it was made with.
    /// Default false.
    /// </summary>
    public bool Restart { get; init; }

    /// <summary>The share of lines the suggested <c>LocalFirst</c> threshold must read exactly as the language model does. Default 0.99.</summary>
    public double LocalFirstPrecision { get; init; } = 0.99;

    /// <summary>Size limits; null means <see cref="OcrLimits.Default"/>.</summary>
    public OcrLimits? Limits { get; init; }

    /// <summary>Receives a line of text for each stage and each checkpoint.</summary>
    public IProgress<string>? Progress { get; init; }

    internal KernelPath? KernelPath { get; init; }

    internal void Validate()
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(WorkDirectory);
        if (Steps < 1 || BatchSize < 1 || CheckpointEvery < 1 || StopAfterSteps < 0 || (WarmupSteps ?? 0) < 0 || MaxDegreeOfParallelism < 0)
        {
            throw new ArgumentException("Steps, BatchSize and CheckpointEvery must be positive; the other counts cannot be negative.");
        }

        if (!(LearningRate > 0) || !(WeightDecay >= 0) || !(MaxGradNorm >= 0) || !(HeldOutFraction >= 0 && HeldOutFraction < 1)
            || !(MaxAgreementCer >= 0 && MaxAgreementCer <= 1) || !(AlignMaxCer >= MaxAgreementCer && AlignMaxCer <= 1)
            || !(LocalFirstPrecision > 0 && LocalFirstPrecision <= 1))
        {
            throw new ArgumentException("A rate, fraction or tolerance is out of range (AlignMaxCer must be at least MaxAgreementCer).");
        }
    }
}

/// <summary>What labelling a folder of documents produced.</summary>
/// <param name="Documents">Documents read this call.</param>
/// <param name="AlreadyLabelled">Documents skipped because an earlier call labelled them.</param>
/// <param name="Unreadable">Documents that could not be read as images, or that the provider failed on; the names are in <paramref name="Diagnostics"/>.</param>
/// <param name="Lines">Lines found in the documents read this call.</param>
/// <param name="Agreed">Lines the language model and the local reader agree on.</param>
/// <param name="Disagreed">Lines they read differently (sent to the review queue).</param>
/// <param name="Unaligned">Lines with no matching stretch in the language model's text (sent to the review queue).</param>
/// <param name="CostUsd">Estimated spend of this call, from the client's price table; null when the model has no price.</param>
/// <param name="BudgetReached">True when the client's spend cap stopped labelling before every document was done; a later call continues.</param>
/// <param name="Diagnostics">Notes, one per problem.</param>
public sealed record LabelSummary(
    int Documents, int AlreadyLabelled, int Unreadable, int Lines, int Agreed, int Disagreed, int Unaligned, decimal CostUsd, bool BudgetReached, IReadOnlyList<string> Diagnostics);

/// <summary>Before and after numbers of a fine-tuning run, on lines held out of training.</summary>
/// <param name="HeldOut">The starting model's reading of the held-out lines against the reference text.</param>
/// <param name="Tuned">The fine-tuned model's reading of the same lines.</param>
public sealed record BeforeAfter(TextErrorRates HeldOut, TextErrorRates Tuned);

/// <summary>The result of measuring one model against the held-out lines of a work directory.</summary>
/// <param name="Rates">Error rates against the reference text: a reviewer's correction where there is one, otherwise the language model's text.</param>
/// <param name="SuggestedLocalFirstThreshold">
/// The lowest line confidence at which lines were read exactly as the reference at least <see cref="OcrTrainerOptions.LocalFirstPrecision"/>
/// of the time (with at least five such lines); null when no threshold reached that. Use it as the confidence above which
/// local reading can be trusted without asking the language model.
/// </param>
/// <param name="ThresholdSupport">Held-out lines at or above the suggested threshold.</param>
public sealed record EvaluationReport(TextErrorRates Rates, float? SuggestedLocalFirstThreshold, int ThresholdSupport);

/// <summary>The outcome of a fine-tuning run.</summary>
/// <param name="Documents">Labelled documents in the work directory.</param>
/// <param name="Lines">Lines in them.</param>
/// <param name="Agreed">Lines the local reader and the language model agreed on.</param>
/// <param name="Disagreed">Lines they read differently.</param>
/// <param name="Unaligned">Lines with no matching stretch in the language model's text.</param>
/// <param name="Reviewed">Lines with a correction from the review queue.</param>
/// <param name="AgreementRate">Agreed lines over agreed, disagreed and unaligned lines (0 when there are none).</param>
/// <param name="TrainLines">Lines the model was trained on: agreed and reviewed lines outside the held-out split, with text inside the model's charset.</param>
/// <param name="SkippedOutsideCharset">Candidate lines left out because their text has a character the model cannot output.</param>
/// <param name="HeldOutLines">Lines in the held-out split that have a reference text.</param>
/// <param name="Steps">Steps taken in total.</param>
/// <param name="FirstLoss">Mean loss of the first step.</param>
/// <param name="LastLoss">Mean loss of the last step.</param>
/// <param name="Evaluation">Error rates before and after on the held-out lines; null when nothing was held out.</param>
/// <param name="SuggestedLocalFirstThreshold">See <see cref="EvaluationReport.SuggestedLocalFirstThreshold"/>, for the fine-tuned model.</param>
/// <param name="ModelPath">The fine-tuned model file.</param>
/// <param name="Diagnostics">Notes, one per problem.</param>
public sealed record TrainingReport(
    int Documents, int Lines, int Agreed, int Disagreed, int Unaligned, int Reviewed, double AgreementRate, int TrainLines, int SkippedOutsideCharset,
    int HeldOutLines, int Steps, double FirstLoss, double LastLoss, BeforeAfter? Evaluation, float? SuggestedLocalFirstThreshold, string ModelPath,
    IReadOnlyList<string> Diagnostics);
