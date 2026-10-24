using System;
using System.Linq;
using OneOcr;
using OneOcr.Model;
using OneOcr.Recognition;
using OneOcr.Tests;

// Builds a small random-weight recognizer, runs it, and checks the failure paths, all under NativeAOT.
var shape = new TestModels.Shape(8, [(4, 2, 2), (4, 2, 2), (8, 2, 1)], 8, 2, 12);
byte[] file = TestModels.RandomContainer(shape, 1);
LineRecognizer recognizer = LineRecognizer.FromContainer(ModelContainer.Parse(file));
byte[] image = TestModels.RandomImage(8, 96, 2);
RecognizedLine a = recognizer.Recognize(image, 96, 1);
RecognizedLine b = recognizer.Recognize(image, 96, 4);
if (a != b)
{
    Console.Error.WriteLine("degrees disagree");
    return 1;
}

try
{
    ModelContainer.Parse(file.AsSpan(0, file.Length - 1));
    Console.Error.WriteLine("truncated file was accepted");
    return 1;
}
catch (OcrModelException)
{
}

// A page of four text-like lines through the whole pipeline: line finding, scaling, recognition, and the default engine.
var pageShape = new TestModels.Shape(32, [(8, 2, 2), (16, 2, 2), (16, 2, 1), (32, 2, 1), (32, 2, 1)], 24, 2, 41);
OcrEngine engine = OcrEngine.Load(new System.IO.MemoryStream(TestModels.RandomContainer(pageShape, 3)));
var pixels = new OneImage.Gray8[600 * 300];
Array.Fill(pixels, new OneImage.Gray8(255));
foreach (int top in new[] { 40, 110, 180 })
{
    for (int y = top; y < top + 30; y++)
    {
        for (int x = 60; x < 500; x++)
        {
            if ((x / 7) % 3 != 0)
            {
                pixels[(y * 600) + x] = new OneImage.Gray8(10);
            }
        }
    }
}

OneImage.Image page = OneImage.Image.FromPixels<OneImage.Gray8>(pixels, 600, 300);
string t1 = engine.ToText(page, 1), t4 = page.ToText(engine);
if (t1 != t4 || t1.Split('\n').Length != 3)
{
    Console.Error.WriteLine("ToText disagrees or found the wrong number of lines");
    return 1;
}

// Beam search with a language model and a lexicon: same text at every degree of parallelism, and a model that fits the charset.
string[] charset = TestModels.Charset(41);
string[] alphabet = [.. charset.Skip(2).Take(12)];
var beam = new OneOcr.Decoding.DecodeOptions
{
    Mode = OneOcr.Decoding.CtcDecoding.Beam,
    BeamWidth = 4,
    LanguageModel = OneOcr.Decoding.CharLanguageModel.Parse(TestLm.Build("smoke", alphabet, [string.Concat(alphabet.Take(4)), string.Concat(alphabet.Skip(2).Take(5))], 3)),
    Lexicon = OneOcr.Decoding.Lexicon.FromWords([string.Concat(alphabet.Take(3)), "x1"]),
};
string b1 = engine.ToText(page, beam, 1), b4 = engine.WithDecoding(beam).ToText(page, 4);
if (b1 != b4 || b1.Split('\n').Length != 3)
{
    Console.Error.WriteLine("beam decoding disagrees across degrees");
    return 1;
}

// A version 1 container: calibration is parsed (source-generated JSON), applied, and words come back with boxes.
OcrEngine calibrated = OcrEngine.Load(new System.IO.MemoryStream(TestModels.RandomContainerV1(pageShape, 3,
    "{\"raw\":\"word-mean-max-run\",\"scripts\":{\"Latin\":{\"x\":[0,1],\"y\":[0,0.5]}}}")));
OneOcr.OcrPage words = calibrated.Ocr(page, new OneOcr.OcrOptions { MaxDegreeOfParallelism = 1 });
if (words.ToText() != t1 || words.Blocks.Sum(bl => bl.Lines.Count) != 3 || calibrated.Ocr(page, new OneOcr.OcrOptions { MaxDegreeOfParallelism = 4 }).ToText() != t1)
{
    Console.Error.WriteLine("Ocr disagrees with ToText");
    return 1;
}

// The page model's JSON: write, read, write again (source-generated, no reflection), and a malformed document is a typed error.
string json = OneOcr.OcrPageJson.ToJson([words]);
System.Collections.Generic.IReadOnlyList<OneOcr.OcrPage> back = OneOcr.OcrPageJson.Parse(json);
if (back.Count != 1 || OneOcr.OcrPageJson.ToJson(back) != json || back[0].ToText() != t1)
{
    Console.Error.WriteLine("page JSON does not round trip");
    return 1;
}

try
{
    OneOcr.OcrPageJson.Parse(json.Replace("\"oneocr.page\"", "\"other\""));
    Console.Error.WriteLine("wrong format accepted");
    return 1;
}
catch (OneOcr.OcrFormatException)
{
}

OneOcr.OneOcr.Configure(engine);
if (page.ToText() != t1)
{
    Console.Error.WriteLine("default engine disagrees");
    return 1;
}

// The LLM layer: a typed extraction through a fake handler, with the schema exported from source-generated type information.
var ocrWord = new OneOcr.OcrWord("Total", new OneOcr.OcrBox(10, 10, 60, 30), 0.9f, 0.9f, OneOcr.OcrSource.Ocr, new System.Collections.Generic.List<OneOcr.OcrChar>());
var ocrLine = new OneOcr.OcrLine("Total", new[] { ocrWord }, new[] { 0 }, ocrWord.Box, 0.9f, OneOcr.OcrSource.Ocr);
var llmPage = new OneOcr.OcrPage(0, 600, 300, 72, new[] { new OneOcr.OcrBlock(OneOcr.OcrBlockRole.Body, false, ocrWord.Box, new[] { ocrLine }, 0.9f) });
var llmHandler = new SmokeHandler();
var anthropic = new OneOcr.Llm.AnthropicProvider("smoke-key", "smoke-model", http: new System.Net.Http.HttpClient(llmHandler));
using (var llm = new OneOcr.Llm.LlmClient(anthropic, new OneOcr.Llm.LlmOptions { Cache = new OneOcr.Llm.LlmResultCache() }))
{
    var typed = llm.ExtractAsync(page, llmPage, "Read the total.", SmokeContext.Default.SmokeAnswer).GetAwaiter().GetResult();
    var again = llm.ExtractAsync(page, llmPage, "Read the total.", SmokeContext.Default.SmokeAnswer).GetAwaiter().GetResult();
    if (typed.Value.Total != "12.50" || !again.Details.FromCache || llmHandler.Calls != 1)
    {
        Console.Error.WriteLine("LLM extraction failed");
        return 1;
    }
}

using (var offline = new OneOcr.Llm.LlmClient(anthropic, new OneOcr.Llm.LlmOptions { Offline = true }))
{
    try
    {
        offline.ExtractAsync(page, llmPage, "x", SmokeContext.Default.SmokeAnswer).GetAwaiter().GetResult();
        Console.Error.WriteLine("offline client sent a request");
        return 1;
    }
    catch (OneOcr.OcrProviderException e) when (e.Failure == OneOcr.LlmFailure.Offline)
    {
    }
}

foreach (var provider in new OneOcr.Llm.LlmProvider[]
{
    new OneOcr.Llm.OpenAiProvider("k", "m", http: new System.Net.Http.HttpClient(llmHandler)),
    new OneOcr.Llm.GeminiProvider("k", "m", http: new System.Net.Http.HttpClient(llmHandler)),
    new OneOcr.Llm.AzureOpenAiProvider(new Uri("https://x.openai.azure.com"), "d", "k", http: new System.Net.Http.HttpClient(llmHandler)),
    new OneOcr.Llm.OpenAiCompatibleProvider(new Uri("http://localhost:1/v1"), "m", http: new System.Net.Http.HttpClient(llmHandler)),
})
{
    using var c = new OneOcr.Llm.LlmClient(provider);
    var estimate = c.Estimate(new[] { new OneOcr.Llm.LlmPageInput(page, llmPage) }, "x", "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":[\"string\",\"null\"]}}}");
    if (estimate.InputTokens <= 0)
    {
        Console.Error.WriteLine("no estimate");
        return 1;
    }
}

// Grounded read: every requested output from one call, values checked against the page.
var readHandler = new ReadHandler();
using (var rc = new OneOcr.Llm.LlmClient(new OneOcr.Llm.AnthropicProvider("smoke-key", "smoke-model", http: new System.Net.Http.HttpClient(readHandler))))
{
    var read = rc.ReadAsync(page, llmPage, new OneOcr.Llm.LlmReadOptions
    {
        Outputs = OneOcr.Llm.LlmOutputs.Text | OneOcr.Llm.LlmOutputs.Markdown | OneOcr.Llm.LlmOutputs.Json | OneOcr.Llm.LlmOutputs.PageModel,
        JsonSchema = "{\"type\":\"object\",\"properties\":{\"word\":{\"type\":[\"string\",\"null\"]}}}",
    }).GetAwaiter().GetResult();
    if (read.Calls != 1 || readHandler.Calls != 1 || read.Text != "Total" || read.Markdown != "**Total**" || read.Json != "{\"word\":\"Total\"}"
        || read.Fields.Count != 1 || read.Fields[0].State != OneOcr.Llm.GroundingState.Grounded || read.Pages.Count != 1)
    {
        Console.Error.WriteLine("grounded read failed");
        return 1;
    }
}

// Fine-tuning: stored labels in, a tuned model and a report out, then the review queue both ways.
string trainRoot = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "oneocr-aot-" + Guid.NewGuid().ToString("N"));
System.IO.Directory.CreateDirectory(trainRoot);
try
{
    var trainShape = new TestModels.Shape(32, [(8, 2, 2), (16, 2, 2), (16, 2, 1), (32, 2, 1), (32, 2, 1)], 24, 2, 41);
    string startPath = System.IO.Path.Combine(trainRoot, "start.onm");
    System.IO.File.WriteAllBytes(startPath, TestModels.RandomContainer(trainShape, 5));
    string work = System.IO.Path.Combine(trainRoot, "work");
    string[] trainCharset = TestModels.Charset(41);
    var store = new OneOcr.Training.WorkStore(work, OcrLimits.Default);
    for (int d = 0; d < 3; d++)
    {
        var doc = new OneOcr.Training.DocLabelsDto { Id = "doc" + d, Document = "d" + d + ".png", Height = 32 };
        for (int i = 0; i < 3; i++)
        {
            doc.Lines.Add(new OneOcr.Training.LineLabelDto
            {
                Index = i, Local = "x", Llm = trainCharset[1 + i] + trainCharset[2 + i], State = i == 2 ? OneOcr.Training.LineAgreement.Disagreed : OneOcr.Training.LineAgreement.Agreed,
                CropWidth = 96, Width = 96, Height = 32, Crop = TestModels.RandomImage(32, 96, (d * 3) + i),
            });
        }

        store.Save(doc);
    }

    var trainOptions = new OneOcr.Training.OcrTrainerOptions { ModelPath = startPath, WorkDirectory = work, Steps = 3, BatchSize = 4, HeldOutFraction = 0.34 };
    OneOcr.Training.TrainingReport trained = new OneOcr.Training.OcrTrainer(trainOptions).Train();
    if (trained.TrainLines != 4 || trained.Evaluation is null || trained.Steps != 3 || !System.IO.File.Exists(trained.ModelPath)
        || !OneOcr.Training.OcrTrainer.ToJson(trained).Contains("\"agreementRate\"", StringComparison.Ordinal))
    {
        Console.Error.WriteLine("fine-tuning failed");
        return 1;
    }

    OcrEngine.Load(trained.ModelPath);
    OneOcr.Training.ReviewExport exported = OneOcr.Training.LineReview.Export(work, System.IO.Path.Combine(trainRoot, "export"));
    string edited = System.IO.File.ReadAllText(exported.ReviewFile).Replace("\"text\": null", "\"text\": \"ab\"", StringComparison.Ordinal);
    OneOcr.Training.ReviewImportResult imported = OneOcr.Training.LineReview.Import(work, System.Text.Encoding.UTF8.GetBytes(edited));
    if (exported.Items != 3 || imported.Accepted != 3 || imported.UnknownIds != 0)
    {
        Console.Error.WriteLine("review round trip failed");
        return 1;
    }
}
finally
{
    System.IO.Directory.Delete(trainRoot, true);
}

Console.WriteLine($"ok: {a.Text.Length} characters, confidence {a.Confidence:F3}");
return 0;

internal sealed record SmokeAnswer(string? Total);

[System.Text.Json.Serialization.JsonSourceGenerationOptions(PropertyNamingPolicy = System.Text.Json.Serialization.JsonKnownNamingPolicy.CamelCase)]
[System.Text.Json.Serialization.JsonSerializable(typeof(SmokeAnswer))]
internal sealed partial class SmokeContext : System.Text.Json.Serialization.JsonSerializerContext;

internal sealed class SmokeHandler : System.Net.Http.HttpMessageHandler
{
    public int Calls;

    protected override System.Threading.Tasks.Task<System.Net.Http.HttpResponseMessage> SendAsync(System.Net.Http.HttpRequestMessage request, System.Threading.CancellationToken cancellationToken)
    {
        Calls++;
        const string Body = "{\"type\":\"message\",\"content\":[{\"type\":\"tool_use\",\"name\":\"extraction\",\"input\":{\"total\":\"12.50\"}}],\"stop_reason\":\"tool_use\",\"usage\":{\"input_tokens\":100,\"output_tokens\":10}}";
        return System.Threading.Tasks.Task.FromResult(new System.Net.Http.HttpResponseMessage(System.Net.HttpStatusCode.OK)
        {
            Content = new System.Net.Http.StringContent(Body, System.Text.Encoding.UTF8, "application/json"),
        });
    }
}

internal sealed class ReadHandler : System.Net.Http.HttpMessageHandler
{
    public int Calls;

    protected override System.Threading.Tasks.Task<System.Net.Http.HttpResponseMessage> SendAsync(System.Net.Http.HttpRequestMessage request, System.Threading.CancellationToken cancellationToken)
    {
        Calls++;
        const string Body = "{\"type\":\"message\",\"content\":[{\"type\":\"tool_use\",\"name\":\"extraction\",\"input\":{\"pages\":[\"Total\"],\"markdown\":[\"**Total**\"],\"data\":{\"word\":{\"value\":\"Total\",\"words\":[\"w0\"],\"derived\":false}}}}],\"stop_reason\":\"tool_use\",\"usage\":{\"input_tokens\":100,\"output_tokens\":10}}";
        return System.Threading.Tasks.Task.FromResult(new System.Net.Http.HttpResponseMessage(System.Net.HttpStatusCode.OK)
        {
            Content = new System.Net.Http.StringContent(Body, System.Text.Encoding.UTF8, "application/json"),
        });
    }
}
