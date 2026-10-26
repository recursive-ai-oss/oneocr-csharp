// Every `#region` below is a code sample of the README, word for word. They are compiled here so they cannot drift from the API;
// `check_readme.py` verifies that the README still shows exactly these texts. Nothing in this program is run.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using OneImage;
using OneOcr;
using OneOcr.Decoding;
using OneOcr.Documents;
using OneOcr.Llm;
using OneOcr.Models.Documents;
using OneOcr.Pdf;
using OneOcr.Training;

#region CustomType
public sealed record PurchaseOrder(string? Number, string? Supplier, decimal? Total);

[JsonSerializable(typeof(PurchaseOrder))]
internal sealed partial class OrderContext : JsonSerializerContext;
#endregion

internal static class Samples
{
    private static void Main()
    {
    }

    public static void QuickStart()
    {
        #region QuickStart
        // Once at startup: the recognizer, the page detector and the word model that ship in RecursiveAI.OneOcr.Models.
        OcrEngine engine = OcrEngine.LoadBundled();

        Image image = Image.Load("invoice.png");
        string text = image.ToText(engine);              // plain text in reading order
        OcrPage page = image.Ocr(engine);                // the page model: blocks, lines, words, characters, tables
        string markdown = image.ToMarkdown(engine);      // headings, lists and pipe tables
        #endregion
    }

    public static void DefaultEngine()
    {
        #region DefaultEngine
        OneOcr.OneOcr.Configure();                       // once; the bundled models become OneOcr.OneOcr.Default

        string text = Image.Load("page.png").ToText();   // the extension methods without an engine use the default
        OcrPage page = Image.Load("page.png").Ocr();
        #endregion
    }

    public static void Options(OcrEngine engine, Image image)
    {
        #region Options
        OcrPage page = image.Ocr(engine, new OcrOptions
        {
            Dpi = 300,                    // when the file does not say; otherwise estimated, with a diagnostic
            MaxDegreeOfParallelism = 2,   // 0 = min(4, cores); the result does not depend on it
            IncludeChars = false,         // skip per-character boxes
            DetectOrientation = true,     // 0/90/180/270 (the default)
            Deskew = true,                // tilts up to about 15 degrees (the default)
        });

        foreach (OcrBlock block in page.Blocks)
        {
            foreach (OcrLine line in block.Lines)
            {
                foreach (OcrWord word in line.Words)
                {
                    Console.WriteLine($"{word.Text}  {word.Confidence:F3}  ({word.Box.Left:F0}, {word.Box.Top:F0})");
                }
            }
        }
        #endregion
    }

    public static void PageJson(OcrPage page)
    {
        #region PageJson
        string json = OcrPageJson.ToJson([page], new OcrPageJsonOptions { Indented = true });
        IReadOnlyList<OcrPage> again = OcrPageJson.Parse(json);   // checked against OcrLimits; throws OcrFormatException on bad input
        #endregion
    }

    public static void Decoding(string modelPath, string lmPath)
    {
        #region Decoding
        OcrEngine engine = OcrEngine.Load(modelPath).WithDecoding(new DecodeOptions
        {
            Mode = CtcDecoding.Beam,
            BeamWidth = 8,
            LanguageModel = CharLanguageModel.Load(lmPath),
        });
        #endregion
    }

    public static void Documents(OcrEngine engine, Image image)
    {
        #region Documents
        DocumentModels.Configure();                      // the extraction model from RecursiveAI.OneOcr.Models.Documents, once

        Invoice invoice = image.ExtractInvoice(engine);
        Console.WriteLine($"{invoice.Number?.Value}  {invoice.IssueDate?.Value}  {invoice.Total?.Value} {invoice.Currency?.Value}");
        foreach (InvoiceLine line in invoice.Lines)
        {
            Console.WriteLine($"{line.Description?.Text}  {line.Quantity?.Value} x {line.UnitPrice?.Value} = {line.Amount?.Value}");
        }

        Receipt receipt = image.ExtractReceipt(engine);
        Console.WriteLine($"{receipt.Merchant?.Text}  {receipt.Total?.Value}");
        #endregion
    }

    public static void CustomExtraction(OcrEngine engine, Image image)
    {
        #region CustomExtraction
        DocumentExtractor extractor = DocumentExtractor.Load("purchase-orders.onm");   // a model you fine-tuned (see Training)
        ExtractionResult<PurchaseOrder> result = image.Extract(OrderContext.Default.PurchaseOrder, engine, extractor);

        Console.WriteLine($"{result.Value.Number}  {result.Value.Total}");
        IReadOnlyList<OcrBox> where = result.WordBoxes("Total");   // where on the page the value was read
        #endregion
    }

    public static void Pdf(OcrEngine engine)
    {
        #region Pdf
        IReadOnlyList<OcrPage> pages = PdfOcr.Read("contract.pdf", engine, new PdfOcrOptions { RenderDpi = 300 });
        foreach (OcrPage page in pages)
        {
            Console.WriteLine($"page {page.Index + 1}: {page.Words.Count} words");
        }

        string markdown = pages.ToMarkdown();
        #endregion
    }

    public static async Task Llm(OcrEngine engine, Image image, string schemaJson)
    {
        #region Llm
        OcrPage page = image.Ocr(engine);

        LlmProvider provider = AnthropicProvider.FromEnvironment("your-model-name");   // reads ANTHROPIC_API_KEY, because you called it
        using var llm = new LlmClient(provider, new LlmOptions { MaxCostPerCallUsd = 0.25m, SpendCapUsd = 5m });

        LlmReadResult result = await llm.ReadAsync(image, page, new LlmReadOptions
        {
            Outputs = LlmOutputs.Text | LlmOutputs.Json,   // one call produces every requested output
            JsonSchema = schemaJson,
            UngroundedPolicy = UngroundedPolicy.Flag,
        });

        foreach (GroundedField field in result.Fields)
        {
            Console.WriteLine($"{field.Path} = {field.Value}  [{field.State}, {field.Confidence:F2}]  {field.Boxes.Count} box(es)");
        }
        #endregion
    }

    public static void Estimate(LlmClient llm, Image image, OcrPage page, string schemaJson)
    {
        #region Estimate
        LlmEstimate estimate = llm.Estimate([new LlmPageInput(image, page)], "Read the invoice.", schemaJson);
        Console.WriteLine($"about {estimate.InputTokens} input tokens, at most ${estimate.WorstCaseCostUsd}");
        #endregion
    }

    public static async Task LocalFirst(LlmClient llm, DocumentExtractor extractor, Image image, OcrPage page, string schemaJson)
    {
        #region LocalFirst
        LlmReadResult result = await llm.ReadAsync(image, page, new LlmReadOptions
        {
            Outputs = LlmOutputs.Json,
            JsonSchema = schemaJson,
            Routing = LlmRouting.LocalFirst,                          // the model is called only for fields the local model is unsure of
            LocalExtractor = new DocumentFieldExtractor(extractor),
            ConfidenceThreshold = 0.9f,
        });
        Console.WriteLine($"{result.Calls} call(s), ${result.CostUsd}");
        #endregion
    }

    public static async Task Redaction(OcrEngine engine, Image image)
    {
        #region Redaction
        var options = new LlmOptions
        {
            Offline = false,                                          // true makes every call throw OcrProviderException
            Redactor = (page, pageNumber) =>                          // boxes to mask before upload; masked words are not grounded
                [.. page.Words.Where(w => w.Text.StartsWith("DE", StringComparison.Ordinal) && w.Text.Length > 15).Select(w => w.Box)],
        };
        using var llm = new LlmClient(OpenAiProvider.FromEnvironment("your-model-name"), options);
        LlmReadResult result = await llm.ReadAsync(image, image.Ocr(engine), new LlmReadOptions { Outputs = LlmOutputs.Markdown });
        #endregion
    }

    public static async Task TrainRecognizer(LlmClient llm)
    {
        #region TrainRecognizer
        var trainer = new OcrTrainer(new OcrTrainerOptions { ModelPath = "recognizer.onm", WorkDirectory = "run1" }, llm);
        TrainingReport report = await trainer.RunAsync("scans/");
        Console.WriteLine(OcrTrainer.ToJson(report));

        LineReview.Export("run1", "review/");                       // lines the two readers disagreed on, with crops
        // ...set "text" on the lines you checked in review/review.json, then:
        LineReview.Import("run1", "review/review.json");
        #endregion
    }

    public static async Task TrainExtraction(LlmClient llm)
    {
        #region TrainExtraction
        ExtractionSchema schema = ExtractionSchema.From(OrderContext.Default.PurchaseOrder);   // top-level scalar properties are the fields
        var trainer = new ExtractionTrainer(
            new ExtractionTrainerOptions { ModelPath = "docextract.onm", Schema = schema, WorkDirectory = "po-run" }, llm);
        ExtractionReport report = await trainer.RunAsync("purchase-orders/");

        DocumentExtractor tuned = DocumentExtractor.Load(report.ModelPath);
        #endregion
    }

    public static void TesseractText(OcrEngine engine)
    {
        #region TesseractText
        // tesseract page.png out -l eng            ->  out.txt
        string text = Image.Load("page.png").ToText(engine);
        #endregion
    }

    public static void WordTable(OcrEngine engine, Image image)
    {
        #region WordTable
        OcrPage page = image.Ocr(engine);
        for (int b = 0; b < page.Blocks.Count; b++)
        {
            for (int l = 0; l < page.Blocks[b].Lines.Count; l++)
            {
                foreach (OcrWord word in page.Blocks[b].Lines[l].Words)
                {
                    // block, line, left, top, width, height, confidence 0-100, text: the columns of Tesseract's TSV
                    Console.WriteLine(string.Join('\t', b, l,
                        (int)word.Box.Left, (int)word.Box.Top,
                        (int)(word.Box.Right - word.Box.Left), (int)(word.Box.Bottom - word.Box.Top),
                        (int)Math.Round(word.Confidence * 100), word.Text));
                }
            }
        }
        #endregion
    }

    public static void Batch(OcrEngine engine, string folder)
    {
        #region Batch
        Parallel.ForEach(
            Directory.EnumerateFiles(folder, "*.png"),
            new ParallelOptions { MaxDegreeOfParallelism = 4 },
            path => File.WriteAllText(Path.ChangeExtension(path, ".txt"), Image.Load(path).ToText(engine)));
        #endregion
    }

    public static void OneLine(OcrEngine engine, string cropPath)
    {
        #region OneLine
        // A crop that is one line of text: the page reader finds that line and reads it.
        OcrPage page = Image.Load(cropPath).Ocr(engine, new OcrOptions { DetectOrientation = false, Deskew = false });
        string line = page.Text;
        float confidence = page.Words.Count == 0 ? 0 : page.Words.Min(w => w.Confidence);
        #endregion
    }

    public static void LowConfidence(OcrEngine engine, Image image)
    {
        #region LowConfidence
        foreach (OcrWord word in image.Ocr(engine).Words.Where(w => w.Confidence < 0.8f))
        {
            Console.WriteLine($"check: {word.Text} ({word.Confidence:P0}) at {word.Box.Left:F0},{word.Box.Top:F0}");
        }
        #endregion
    }

    public static async Task AsyncOcr(OcrEngine engine, Image image)
    {
        #region AsyncOcr
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        OcrPage page = await image.OcrAsync(engine, new OcrOptions { Deskew = false }, cts.Token);
        string text = await image.ToTextAsync(engine, cts.Token);
        IReadOnlyList<OcrPage> frames = await engine.OcrAsync([image, image], cancellationToken: cts.Token);
        #endregion
    }

    public static void KernelPath(OcrEngine engine)
    {
        #region KernelPath
        Console.WriteLine(engine.KernelPath);                       // Avx512, Avx2, Vector128 or Scalar: the text is the same on all four
        if (engine.PerformanceNotice is { } notice)
        {
            Console.WriteLine(notice);                              // the scalar path is running on a CPU that could do better
        }
        #endregion
    }

    public static void Limits(string modelPath)
    {
        #region Limits
        var limits = new OcrLimits { MaxImagePixels = 50_000_000, MaxLinesPerPage = 1000 };
        OcrEngine engine = OcrEngine.Load(modelPath, limits);       // OcrLimitException before anything large is allocated
        #endregion
    }
}
