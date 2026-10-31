using OneImage;
using System.Globalization;
using OneOcr.Llm;
using OneOcr.Training;

namespace OneOcr.Tool;

/// <summary>The <c>oneocr</c> command line: thin commands over <see cref="OcrTrainer"/> and <see cref="LineReview"/>.</summary>
internal static class Program
{
    private const string Usage = """
        oneocr - read text from images, and fine-tune the OneOcr line recognizer on your own documents

        usage:
          oneocr --version
          oneocr ocr    --in <image or folder> [--out <dir>] [--model <file> --detector <file> [--wordlm <file>] [--wordfreq <file>]] [--degree N] [--json] [--stats <file>]
          oneocr train  --model <file> --work <dir> [--docs <dir>] [labelling options] [training options]
          oneocr train  --extract <schema.json> --model <docextract file> --work <dir> [--docs <dir>] [labelling options] [extraction options]
          oneocr eval   --model <file> --work <dir> [--seed N] [--held-out F]
          oneocr review export --work <dir> --out <dir> [--include-reviewed]
          oneocr review import --work <dir> --file <review.json>

        ocr reads PNG and JPEG pages with the bundled models (or with --model and --detector files, and --wordlm for the word character model, --wordfreq for the word frequency table) and writes <name>.txt per
        page into --out (one page with no --out prints the text); --json adds <name>.page.json, the page model; --stats writes
        the model load time, time per page and peak memory as JSON. Local only, never touches the network.

        train labels the images in --docs with a language model (when --docs is given), then fine-tunes --model on the
        lines the local reader and the model agree on. Without --docs it trains on the labels already in --work (for
        example after a review import) and never touches the network. Run it again with the same --work to resume.

        labelling options (needed with --docs):
          --provider openai|anthropic|gemini|azure-openai|openai-compatible
          --llm-model <name>        model name (the deployment name for azure-openai)
          --base-url <url>          required for azure-openai (the endpoint) and openai-compatible
          --key <value>             the API key; or --key-from-env <VAR> to read it from that environment variable
                                    (nothing is read from the environment unless you name the variable)
          --budget-usd <amount>     spend cap for this run; required except for openai-compatible

        training options:
          --out <file>        tuned model (default <work>/model.onm)    --steps N (200)       --batch N (16)
          --lr X (1e-4)       --seed N (1)       --held-out F (0.2)     --max-cer F (0: lines must read the same)
          --degree N (min(4, cores))             --stop-after N
          --restart           throw away the checkpoint and start over (needed after a review import or new labels)

        train --extract fine-tunes the document-extraction model for your own document type: --extract names a JSON Schema
        whose top-level scalar properties are the fields; the labelling model reads each page for them, only values found
        on the page become labels, and a new tag head (and, gently, the backbone) is trained on them. --model is a
        docextract model file. The report gives per-field accuracy on held-out pages before and after, agreement with the
        labelling model and a suggested LocalFirst threshold. Pages are read locally with the bundled models.

        extraction options:
          --out <file> (<work>/model.onm)  --steps N (200)  --batch N (8)  --lr X (3e-3)  --seed N (1)  --held-out F (0.2)
          --backbone-scale X (0.05; 0 trains only the new head)  --field-weight X (3)  --max-words N (384)
          --instruction <text>  --degree N  --stop-after N  --restart

        Keys given with --key are visible to other users of the machine in the process list; prefer --key-from-env.
        """;

    private static async Task<int> Main(string[] args)
    {
        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };
        try
        {
            if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
            {
                Console.Out.WriteLine(Usage);
                return args.Length == 0 ? 2 : 0;
            }

            switch (args[0])
            {
                case "--version" or "-v" or "version":
                    Console.Out.WriteLine(VersionText());
                    return 0;
                case "ocr":
                    return OcrCommand.Run(new Flags(args.AsSpan(1)));
                case "train":
                    return await TrainAsync(new Flags(args.AsSpan(1)), cts.Token).ConfigureAwait(false);
                case "eval":
                    return Eval(new Flags(args.AsSpan(1)));
                case "review" when args.Length >= 2 && args[1] == "export":
                    return Export(new Flags(args.AsSpan(2)));
                case "review" when args.Length >= 2 && args[1] == "import":
                    return Import(new Flags(args.AsSpan(2)));
                default:
                    Console.Error.WriteLine("oneocr: unknown command '" + args[0] + "'. Run 'oneocr help'.");
                    return 2;
            }
        }
        catch (UsageException ex)
        {
            Console.Error.WriteLine("oneocr: " + ex.Message);
            return 2;
        }
        catch (OperationCanceledException)
        {
            Console.Error.WriteLine("oneocr: cancelled; a checkpoint was saved, run the same command again to continue.");
            return 130;
        }
        catch (Exception ex) when (ex is OcrException or ImageException or ArgumentException or InvalidOperationException or IOException or UnauthorizedAccessException)
        {
            Console.Error.WriteLine("oneocr: " + ex.Message);
            return 1;
        }
        catch (Exception ex)
        {
            // A bug, not bad input: say so in one line and exit non-zero instead of dumping a stack trace and a core file.
            Console.Error.WriteLine("oneocr: unexpected " + ex.GetType().Name + ": " + ex.Message);
            return 70;
        }
    }

    private static string VersionText()
    {
        string? v = typeof(Program).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
            is [System.Reflection.AssemblyInformationalVersionAttribute attr, ..] ? attr.InformationalVersion : null;
        int plus = v?.IndexOf('+', StringComparison.Ordinal) ?? -1;
        return "oneocr " + (v is null ? "(unknown version)" : plus >= 0 ? v[..plus] : v);
    }

    private static async Task<int> TrainAsync(Flags f, CancellationToken ct)
    {
        string docs = f.Optional("docs") ?? string.Empty;
        if (f.Optional("extract") is { } schemaFile)
        {
            return await TrainExtractionAsync(f, schemaFile, docs, ct).ConfigureAwait(false);
        }

        OcrTrainerOptions options = Options(f, new Progress<string>(static m => Console.Error.WriteLine(m)));
        LlmClient? client = null;
        try
        {
            if (docs.Length > 0)
            {
                client = Client(f);
            }

            f.Done();
            var trainer = new OcrTrainer(options, client);
            if (docs.Length > 0)
            {
                LabelSummary summary = await trainer.LabelAsync(docs, ct).ConfigureAwait(false);
                Console.Error.WriteLine(
                    $"labelled {summary.Documents} documents ({summary.AlreadyLabelled} already done, {summary.Unreadable} skipped): " +
                    $"{summary.Agreed} agreed, {summary.Disagreed} disagreed, {summary.Unaligned} unaligned lines; spent about ${summary.CostUsd.ToString("0.0000", CultureInfo.InvariantCulture)}");
                foreach (string note in summary.Diagnostics)
                {
                    Console.Error.WriteLine("  " + note);
                }

                if (summary.BudgetReached)
                {
                    Console.Error.WriteLine("The spend cap was reached; training on what is labelled. Run again with a higher --budget-usd to label the rest.");
                }
            }

            TrainingReport report = trainer.Train(ct);
            Console.Out.WriteLine(OcrTrainer.ToJson(report));
            return 0;
        }
        finally
        {
            client?.Dispose();
        }
    }

    private static async Task<int> TrainExtractionAsync(Flags f, string schemaFile, string docs, CancellationToken ct)
    {
        ExtractionSchema schema = ExtractionSchema.FromJsonSchema(File.ReadAllText(schemaFile));
        foreach (string note in schema.Notes)
        {
            Console.Error.WriteLine("schema: " + note);
        }

        var options = new ExtractionTrainerOptions
        {
            ModelPath = f.Required("model"),
            Schema = schema,
            WorkDirectory = f.Required("work"),
            OutputModelPath = f.Optional("out"),
            Steps = f.Int("steps", 200),
            BatchSize = f.Int("batch", 8),
            LearningRate = f.Double("lr", 3e-3),
            Seed = f.Int("seed", 1),
            HeldOutFraction = f.Double("held-out", 0.2),
            BackboneScale = f.Double("backbone-scale", 0.05),
            FieldTagWeight = f.Double("field-weight", 3),
            MaxWordsPerPage = f.Int("max-words", 384),
            Instruction = f.Optional("instruction"),
            MaxDegreeOfParallelism = f.Int("degree", 0),
            StopAfterSteps = f.Int("stop-after", 0),
            Restart = f.Switch("restart"),
            Progress = new Progress<string>(static m => Console.Error.WriteLine(m)),
        };
        LlmClient? client = docs.Length > 0 ? Client(f) : null;
        try
        {
            f.Done();
            var trainer = new ExtractionTrainer(options, client);
            if (docs.Length > 0)
            {
                ExtractionLabelSummary s = await trainer.LabelFolderAsync(docs, null, ct).ConfigureAwait(false);
                Console.Error.WriteLine(
                    $"labelled {s.Documents} documents ({s.AlreadyLabelled} already done, {s.Unreadable} skipped): {s.Usable} usable, {s.DroppedUngrounded} dropped for a value not on the page, " +
                    $"{s.FieldsLabelled} fields; spent about ${s.CostUsd.ToString("0.0000", CultureInfo.InvariantCulture)}");
                foreach (string note in s.Diagnostics)
                {
                    Console.Error.WriteLine("  " + note);
                }

                if (s.BudgetReached)
                {
                    Console.Error.WriteLine("The spend cap was reached; training on what is labelled. Run again with a higher --budget-usd to label the rest.");
                }
            }

            Console.Out.WriteLine(ExtractionTrainer.ToJson(trainer.Train(ct)));
            return 0;
        }
        finally
        {
            client?.Dispose();
        }
    }

    private static int Eval(Flags f)
    {
        string model = f.Required("model");
        OcrTrainerOptions options = Options(f, null, modelIsEvaluated: model);
        f.Done();
        Console.Out.WriteLine(OcrTrainer.ToJson(OcrTrainer.Evaluate(model, options)));
        return 0;
    }

    private static int Export(Flags f)
    {
        string work = f.Required("work"), output = f.Required("out");
        bool includeReviewed = f.Switch("include-reviewed");
        f.Done();
        ReviewExport result = LineReview.Export(work, output, includeReviewed);
        Console.Out.WriteLine($"{result.Items} lines to review: edit 'text' in {result.ReviewFile}, then run 'oneocr review import'.");
        return 0;
    }

    private static int Import(Flags f)
    {
        string work = f.Required("work"), file = f.Required("file");
        f.Done();
        ReviewImportResult r = LineReview.Import(work, file);
        Console.Out.WriteLine($"{r.Accepted} corrections stored ({r.Replaced} replaced earlier ones); {r.Unreviewed} items still unreviewed; {r.UnknownIds} unknown ids ignored.");
        foreach (string note in r.Diagnostics)
        {
            Console.Out.WriteLine("  " + note);
        }

        return 0;
    }

    private static OcrTrainerOptions Options(Flags f, IProgress<string>? progress, string? modelIsEvaluated = null)
    {
        string model = modelIsEvaluated ?? f.Required("model");
        return new OcrTrainerOptions
        {
            ModelPath = model,
            WorkDirectory = f.Required("work"),
            OutputModelPath = f.Optional("out"),
            Steps = f.Int("steps", 200),
            BatchSize = f.Int("batch", 16),
            LearningRate = f.Double("lr", 1e-4),
            Seed = f.Int("seed", 1),
            HeldOutFraction = f.Double("held-out", 0.2),
            MaxAgreementCer = f.Double("max-cer", 0),
            MaxDegreeOfParallelism = f.Int("degree", 0),
            StopAfterSteps = f.Int("stop-after", 0),
            Restart = f.Switch("restart"),
            Progress = progress,
        };
    }

    // Builds the labelling client from explicit flags only.
    private static LlmClient Client(Flags f)
    {
        string providerName = f.Required("provider");
        string model = f.Required("llm-model");
        string? baseUrl = f.Optional("base-url");
        string? key = f.Optional("key");
        string? keyVariable = f.Optional("key-from-env");
        if (key is not null && keyVariable is not null)
        {
            throw new UsageException("Give --key or --key-from-env, not both.");
        }

        if (keyVariable is not null)
        {
            key = Environment.GetEnvironmentVariable(keyVariable);
            if (string.IsNullOrEmpty(key))
            {
                throw new UsageException($"The environment variable {keyVariable} is not set.");
            }
        }

        string? budgetText = f.Optional("budget-usd");
        decimal? budget = budgetText is null ? null : ParseDecimal(budgetText);
        Uri? uri = baseUrl is null ? null : (Uri.TryCreate(baseUrl, UriKind.Absolute, out Uri? u) ? u : throw new UsageException("--base-url is not an absolute URL."));
        if (budget is null && providerName != "openai-compatible")
        {
            throw new UsageException("--budget-usd is required: it is the most this run may spend at the provider.");
        }

        LlmProvider provider = providerName switch
        {
            "openai" => new OpenAiProvider(NeedKey(key), model, uri),
            "anthropic" => new AnthropicProvider(NeedKey(key), model, uri),
            "gemini" => new GeminiProvider(NeedKey(key), model, uri),
            "azure-openai" => new AzureOpenAiProvider(uri ?? throw new UsageException("azure-openai needs --base-url (the endpoint)."), model, NeedKey(key)),
            "openai-compatible" => new OpenAiCompatibleProvider(uri ?? throw new UsageException("openai-compatible needs --base-url."), model, key),
            _ => throw new UsageException("Unknown --provider '" + providerName + "'."),
        };
        return new LlmClient(provider, new LlmOptions { SpendCapUsd = budget });
    }

    private static string NeedKey(string? key) => string.IsNullOrEmpty(key) ? throw new UsageException("This provider needs --key or --key-from-env.") : key;

    private static decimal ParseDecimal(string text) =>
        decimal.TryParse(text, NumberStyles.Number, CultureInfo.InvariantCulture, out decimal v) && v >= 0 ? v : throw new UsageException("--budget-usd must be a non-negative number.");
}

internal sealed class UsageException(string message) : Exception(message);

/// <summary>Flags of the form <c>--name value</c> (or <c>--name</c> alone for switches); anything left over is an error.</summary>
internal sealed class Flags
{
    private readonly Dictionary<string, string?> _values = new(StringComparer.Ordinal);
    private readonly HashSet<string> _used = new(StringComparer.Ordinal);

    public Flags(ReadOnlySpan<string> args)
    {
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (!a.StartsWith("--", StringComparison.Ordinal) || a.Length < 3)
            {
                throw new UsageException("Unexpected argument '" + a + "'.");
            }

            string name = a[2..];
            if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                _values[name] = args[++i];
            }
            else
            {
                _values[name] = null;
            }
        }
    }

    public string? Optional(string name)
    {
        _used.Add(name);
        if (!_values.TryGetValue(name, out string? v))
        {
            return null;
        }

        return v ?? throw new UsageException("--" + name + " needs a value.");
    }

    public string Required(string name) => Optional(name) ?? throw new UsageException("--" + name + " is required.");

    public bool Switch(string name)
    {
        _used.Add(name);
        return _values.TryGetValue(name, out string? v) && v is null ? true : (_values.ContainsKey(name) ? throw new UsageException("--" + name + " takes no value.") : false);
    }

    public int Int(string name, int fallback) =>
        Optional(name) is { } s ? (int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v : throw new UsageException("--" + name + " must be an integer.")) : fallback;

    public double Double(string name, double fallback) =>
        Optional(name) is { } s ? (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) ? v : throw new UsageException("--" + name + " must be a number.")) : fallback;

    public void Done()
    {
        foreach (string name in _values.Keys)
        {
            if (!_used.Contains(name))
            {
                throw new UsageException("Unknown option --" + name + ".");
            }
        }
    }
}
