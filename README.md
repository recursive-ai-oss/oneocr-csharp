# OneOcr

**Local OCR and document understanding for .NET: a pure-managed, deterministic reader with
small models of our own, a page model with boxes and calibrated confidences, invoices and
receipts as fields, and optional LLM vision that is checked against the page.** Pure C# with no
dependency beyond .NET itself (and the OneImage / OnePDF libraries for pixels and PDFs). No native
binaries, no `tessdata`, no network unless you configure a provider.

[![NuGet](https://img.shields.io/nuget/v/RecursiveAI.OneOcr)](https://www.nuget.org/packages/RecursiveAI.OneOcr)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-512BD4)](#install)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

Results against Tesseract, PaddleOCR, docTR and Apple Vision, per category, are in
[Benchmarks](#benchmarks). Every number there comes from one run of the release gate on one
machine; the method is written next to the tables so you can repeat it.

## Why OneOcr

- **Free, fast, local.** The recognizer (5 MB), the page detector (1.4 MB) and the word model
  (1.4 MB) ship inside the `RecursiveAI.OneOcr.Models` package and run on the CPU through our own
  SIMD kernels. Scripts: Latin (including the EU languages, Turkish and Vietnamese), Greek and
  Cyrillic in one model.
- **A page, not a string.** The result is an `OcrPage`: blocks, lines, words and characters with
  boxes (and quads for rotated text), tables with row and column spans, key–value pairs, reading
  order, and a `Source` on every element. It serializes to a documented, versioned JSON format,
  `oneocr.page`, that shares its member names with
  [OnePDF's `onepdf.extraction`](https://www.nuget.org/packages/RecursiveAI.OnePdf)
  ([spec](docs/spec/oneocr-page-v1.md)).
- **Confidence you can use.** A word's `Confidence` is the calibrated probability that the word
  is exactly right, calibrated per script on held-out data. The release gate requires an expected
  calibration error of at most 0.05 in every category.
- **Same bytes everywhere.** Text, boxes and confidences are bit-identical on AVX-512, AVX2,
  128-bit vectors (SSE and NEON) and the scalar path, and at every degree of parallelism. The
  rules are written down in [docs/spec/numerics.md](docs/spec/numerics.md).
- **Documents as fields.** `ExtractInvoice()` and `ExtractReceipt()` return typed fields, each
  with the words and boxes it was read from; a small local model labels words on the page, so it
  cannot invent a value. Invoice fields carry their EN 16931 business-term ids.
- **LLM vision that is held to the page.** Opt-in providers (Anthropic, OpenAI, OpenAI-compatible
  servers, Gemini, Azure OpenAI) over plain `HttpClient`. Every value the model returns is
  checked against the words on the page and comes back `Grounded`, `GroundedNormalized`,
  `Derived` or `Ungrounded`, with boxes where it has them.
- **Train it on your documents.** Fine-tune the recognizer or a new extraction head from your
  own scans, on the CPU, deterministically and resumably, with a review queue for the lines a
  model and the reader disagree on.
- **Safe on files you did not write.** Model files, images, page JSON, PDFs, review files and LLM
  responses are read by bounds-checked managed code under limits checked before anything large is
  allocated, and each of those readers has a fuzz target. The only `unsafe` code is in the SIMD
  kernels, which run on buffers that safe callers validated. NativeAOT and trimming are
  supported.

## Install

```sh
dotnet add package RecursiveAI.OneOcr
```

`RecursiveAI.OneOcr` pulls in `RecursiveAI.OneOcr.Models` (the weights), so one install gives you
the one-liner. The rest is opt-in:

| Package | What it adds |
|---|---|
| `RecursiveAI.OneOcr` | Local OCR, the page model and `oneocr.page` JSON, local Markdown, invoice / receipt / custom extraction (inference). No networking code. |
| `RecursiveAI.OneOcr.Models` | The recognizer, page detector and word model, embedded and checked against their SHA-256 pins on every load. |
| `RecursiveAI.OneOcr.Models.Documents` | The invoice and receipt extraction model. |
| `RecursiveAI.OneOcr.Llm` | LLM vision: providers, modes, grounding, cost controls, routing. The only package that uses the network. |
| `RecursiveAI.OneOcr.Training` | `OcrTrainer`, `ExtractionTrainer` and the review queue. |
| `RecursiveAI.OneOcr.Pdf` | PDF pages: text layer plus OCR of image regions. |
| `RecursiveAI.OneOcr.Tool` | The `oneocr` command-line tool: `ocr`, `train`, `eval`, `review`. |

```sh
dotnet add package RecursiveAI.OneOcr.Llm
dotnet add package RecursiveAI.OneOcr.Pdf
dotnet add package RecursiveAI.OneOcr.Models.Documents
dotnet add package RecursiveAI.OneOcr.Training
dotnet tool install --global RecursiveAI.OneOcr.Tool
```

Target framework `net10.0`. Namespaces: `OneOcr` (core), `OneOcr.Documents`, `OneOcr.Decoding`,
`OneOcr.Models.Documents`, `OneOcr.Llm`, `OneOcr.Pdf`, `OneOcr.Training`. Pixels come from
[`RecursiveAI.OneImage`](https://www.nuget.org/packages/RecursiveAI.OneImage), so any `OneImage.Image`
(PNG, JPEG, WebP, TIFF and so on) is an input. Moving from Tesseract? See the
[migration guide](docs/migrating-from-tesseract.md).

## Quick start

<!-- sample: QuickStart -->
```csharp
// Once at startup: the recognizer, the page detector and the word model that ship in RecursiveAI.OneOcr.Models.
OcrEngine engine = OcrEngine.LoadBundled();

Image image = Image.Load("invoice.png");
string text = image.ToText(engine);              // plain text in reading order
OcrPage page = image.Ocr(engine);                // the page model: blocks, lines, words, characters, tables
string markdown = image.ToMarkdown(engine);      // headings, lists and pipe tables
```

An `OcrEngine` is immutable and thread-safe; load it once and share it. If you would rather not
pass it around, set a process-wide default once and use the extension methods without an engine.
(The static class is `OneOcr.OneOcr`, because the namespace is `OneOcr` too.)

<!-- sample: DefaultEngine -->
```csharp
OneOcr.OneOcr.Configure();                       // once; the bundled models become OneOcr.OneOcr.Default

string text = Image.Load("page.png").ToText();   // the extension methods without an engine use the default
OcrPage page = Image.Load("page.png").Ocr();
```

From the command line (`dotnet tool install --global RecursiveAI.OneOcr.Tool`):

```sh
oneocr ocr --in page.png                      # text to stdout
oneocr ocr --in scans/ --out text/ --json     # one .txt and one .page.json per image
```

`ocr` reads PNG and JPEG with the bundled models and never touches the network; `--degree N` caps
the threads (the text does not depend on it). The same tool trains models, see
[Training](#training).

## Reading a page: `OcrEngine` and options

`ToText` reads a clean, upright, single-column page fast and returns the text. `Ocr` does the
whole job: finds the text lines with the page detector, turns the page upright if it came in
rotated by 90, 180 or 270 degrees, straightens tilted lines, reads every line, decides word
breaks from the line's pixels as well as from the network, drops specks that were read as text,
works out reading order across columns and sidebars, and finds tables.

<!-- sample: Options -->
```csharp
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
```

Other `OcrOptions` switches are `DropNoise`, `FixWordSpacing`, `FixDigitLookalikes` (reads the
letter `O` as `0` inside identifiers and dates) and `FixSeparators` (reads a middle dot, bullet or
vertical bar between words from its pixels when the network gave it as a hyphen or another small
mark). These two are the only steps after recognition that change a character. All are on by
default and documented on the type.

- **Errors.** Public methods throw only the typed exceptions (all derive from `OcrException`),
  argument exceptions, I/O exceptions and `OperationCanceledException`: `OcrFormatException`
  (bytes that are not what they claim to be), `OcrLimitException`, `OcrModelException`,
  `OcrProviderException` and `OcrBudgetException` (in the LLM package). Where something is
  recoverable the page says so in `page.Diagnostics` instead of throwing.
- **Threading and async.** Every local call takes a `CancellationToken`, checked between pages and
  between lines, and throws `OperationCanceledException`. Every local call also has an `Async`
  twin (`OcrAsync`, `ToTextAsync`, `ToMarkdownAsync`, `ExtractInvoiceAsync`, `ExtractReceiptAsync`,
  `ExtractAsync<T>`, and the same on PDF pages and documents). A twin runs the very same code on a
  thread-pool thread, so its result is byte-identical to the sync call, and a cancelled token ends
  the task as cancelled. Keep the `Image` alive until the task completes. `engine.Ocr(images)` and
  `engine.OcrAsync(images)` read several images (for example the frames of a TIFF) in order, with
  consecutive `OcrPage.Index` values. The LLM package is async only.

<!-- sample: AsyncOcr -->
```csharp
using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
OcrPage page = await image.OcrAsync(engine, new OcrOptions { Deskew = false }, cts.Token);
string text = await image.ToTextAsync(engine, cts.Token);
IReadOnlyList<OcrPage> frames = await engine.OcrAsync([image, image], cancellationToken: cts.Token);
```
- **Decoding.** Page reading (`Ocr`, `ToMarkdown`) is always greedy CTC, because the word boxes come
  from the greedy alignment. `engine.WithDecoding(...)` sets the decoding the plain-text call
  `engine.ToText(image, engine.Decoding)` uses: beam search with a character language model or a
  lexicon.

<!-- sample: Decoding -->
```csharp
OcrEngine engine = OcrEngine.Load(modelPath).WithDecoding(new DecodeOptions
{
    Mode = CtcDecoding.Beam,
    BeamWidth = 8,
    LanguageModel = CharLanguageModel.Load(lmPath),
});
```

  Words that contain digits or code symbols are read exactly as greedy decoding reads them, so
  the language model never "corrects" an identifier. There is no word dictionary by default.
- **Your own model.** `OcrEngine.Load(path)` reads a model container (for example one you
  fine-tuned) and `.WithDetector(path)` adds a detector; without one the classical line finder
  runs.

## The page model and `oneocr.page` JSON

`OcrPage → OcrBlock → OcrLine → OcrWord → OcrChar`, plus `Regions` (roles such as title, list,
table or figure, where the detector reports them), `Tables` (cells with row and column spans that
point at words), `KeyValues` and `Diagnostics`.

| Member | Meaning |
|---|---|
| Coordinates | Top-left origin, y down, in page units: pixels for images, points for PDF pages. `Width`, `Height`, `UnitsPerInch`; PDF pages also carry `ToPdf`, the matrix back to PDF user space. |
| `Source` | `Ocr`, `TextLayer` or `Llm`, on every element. |
| `Confidence` | Calibrated probability that the word is exactly right. `RawConfidence` is the uncalibrated mean of the characters' best softmax probabilities. Lines and blocks average their words'. |
| Boxes and quads | Axis-aligned `OcrBox` everywhere; an `OcrQuad` where text was rotated or straightened. Character boxes from OCR are approximate (they come from the network's alignment) and say so. |
| `Orientation`, `SkewDegrees` | How far the page was turned and straightened before reading. All coordinates stay in the image as you gave it. |

<!-- sample: PageJson -->
```csharp
string json = OcrPageJson.ToJson([page], new OcrPageJsonOptions { Indented = true });
IReadOnlyList<OcrPage> again = OcrPageJson.Parse(json);   // checked against OcrLimits; throws OcrFormatException on bad input
```

The JSON is deterministic (the same pages give the same bytes), versioned (`"format":
"oneocr.page"`, `"version": 1`), and specified member by member in
[docs/spec/oneocr-page-v1.md](docs/spec/oneocr-page-v1.md), including how it differs from
`onepdf.extraction` (y runs down here, boxes are `[left, top, right, bottom]`).

`page.ToMarkdown()` turns a page into Markdown without any model call: headings from text size,
lists, pipe tables from detected tables, headers, footers and page numbers dropped (see
`MarkdownOptions`). `page.ToText()` joins blocks with a blank line, lines with a newline and
table cells with a tab.

## Invoices, receipts and your own document types

A small layout-aware model tags the words of a page; deterministic code assembles the fields from
the tagged words. Each `ExtractedField` has the exact text, a normalized value (an ISO date, a
plain decimal amount, an ISO currency code), the word indices and the box.

<!-- sample: Documents -->
```csharp
DocumentModels.Configure();                      // the extraction model from RecursiveAI.OneOcr.Models.Documents, once

Invoice invoice = image.ExtractInvoice(engine);
Console.WriteLine($"{invoice.Number?.Value}  {invoice.IssueDate?.Value}  {invoice.Total?.Value} {invoice.Currency?.Value}");
foreach (InvoiceLine line in invoice.Lines)
{
    Console.WriteLine($"{line.Description?.Text}  {line.Quantity?.Value} x {line.UnitPrice?.Value} = {line.Amount?.Value}");
}

Receipt receipt = image.ExtractReceipt(engine);
Console.WriteLine($"{receipt.Merchant?.Text}  {receipt.Total?.Value}");
```

- **Invoice** fields follow EN 16931 business terms (`BT-1` number, `BT-2` issue date, `BT-5`
  currency, `BT-27` seller name, `BT-44` buyer name, `BT-84` IBAN, `BT-106` subtotal,
  `BT-110` tax, `BT-112` total, `BT-115` amount due, and so on), plus line items (`BT-126` to
  `BT-153`) and the tax breakdown per rate. Look any field up with `invoice.Extraction.Get("BT-112")`.
- **Receipt** fields: merchant, date, time, subtotal, tax, tip, total, payment method and items.
- A field the model did not find is `null`, never a guess. `Confidence` on an extracted field is
  the mean probability of its words' tags; it is not a calibrated probability.
- `page.FindKeyValues()` pairs keys with values on any page and `page.WithKeyValues()` returns the
  page with the pairs filled in.

For your own type, give `Extract<T>` a source-generated `JsonTypeInfo<T>` (no reflection, so it
works under NativeAOT) and a model you trained for it (see [Training](#training)):

<!-- sample: CustomType -->
```csharp
public sealed record PurchaseOrder(string? Number, string? Supplier, decimal? Total);

[JsonSerializable(typeof(PurchaseOrder))]
internal sealed partial class OrderContext : JsonSerializerContext;
```

<!-- sample: CustomExtraction -->
```csharp
DocumentExtractor extractor = DocumentExtractor.Load("purchase-orders.onm");   // a model you fine-tuned (see Training)
ExtractionResult<PurchaseOrder> result = image.Extract(OrderContext.Default.PurchaseOrder, engine, extractor);

Console.WriteLine($"{result.Value.Number}  {result.Value.Total}");
IReadOnlyList<OcrBox> where = result.WordBoxes("Total");   // where on the page the value was read
```

The model format and its inference are specified in [docs/spec/docextract-v0.md](docs/spec/docextract-v0.md).

## PDF

`RecursiveAI.OneOcr.Pdf` reads each page as one `OcrPage`: text-layer words wherever the PDF has
them (from [OnePDF](https://www.nuget.org/packages/RecursiveAI.OnePdf)), and OCR of image regions
that the text does not cover, rendered at 300 DPI through
[`RecursiveAI.OneImage.Pdf`](https://www.nuget.org/packages/RecursiveAI.OneImage.Pdf). Every word
keeps its `Source`.

<!-- sample: Pdf -->
```csharp
IReadOnlyList<OcrPage> pages = PdfOcr.Read("contract.pdf", engine, new PdfOcrOptions { RenderDpi = 300 });
foreach (OcrPage page in pages)
{
    Console.WriteLine($"page {page.Index + 1}: {page.Words.Count} words");
}

string markdown = pages.ToMarkdown();
```

Extension methods on OnePDF's `PdfPage` and `PdfDocument` (`Ocr`, `ToText`, `ToMarkdown`,
`ExtractInvoice`, `ExtractReceipt`, each with an `Async` twin) do the same on a document you already hold.

- A text layer is not trusted blindly. An invisible layer laid over a scan (a previous OCR pass)
  is re-read with our engine unless you set `TrustExistingOcrLayer`. Lines that look garbled
  (U+FFFD, private-use code points, control characters) are re-read from a render, and a
  cross-check of a few sampled lines drops the whole layer when it disagrees with what the page
  looks like.
- Page coordinates are PDF points, y down; `ToPdf` maps them back to PDF user space.
- Not yet: text drawn as outlined glyphs, annotations.

## LLM vision, held to the page

`RecursiveAI.OneOcr.Llm` sends a page image and the local reader's word list (id, text, box) to a
model and asks for text, Markdown, a page model or JSON for your schema in one call. Everything it
returns is checked against the page.

| Provider class | API |
|---|---|
| `AnthropicProvider` | Anthropic Messages |
| `OpenAiProvider` | OpenAI |
| `OpenAiCompatibleProvider` | Ollama, llama.cpp, vLLM, LM Studio and other OpenAI-style servers (a local model needs no key) |
| `GeminiProvider` | Google Gemini |
| `AzureOpenAiProvider` | Azure OpenAI (endpoint and deployment) |

There is no default model: you name the model string. Providers take an injected `HttpClient` (a
shared one that never follows redirects is used otherwise, so a key header cannot be sent to
another host), write requests with `Utf8JsonWriter` and read responses with `System.Text.Json` under
size and depth limits; no vendor SDKs, no reflection. Structured output uses each provider's own
mechanism (OpenAI `json_schema`, a single forced tool call for Anthropic, `responseSchema` for
Gemini), and the model is never offered tools to run.

<!-- sample: Llm -->
```csharp
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
```

**Outputs** (`LlmOutputs`, combine with `|`): `Text` (the model's transcription merged with the
local reading word by word: the local word wins where it is confident, the model's where the
reader is unsure, and each disagreement is listed), `Markdown` (the model's Markdown with the spans
where it disagrees with the local reader marked), `Json` (values for your schema, grounded) and
`PageModel` (the merged page, every word with its `Source`). `ExtractAsync<T>` takes a
source-generated `JsonTypeInfo<T>` instead of a schema string.

**Grounding.** For every scalar the model returns it must cite the words it read. We check the
cited words against the value after normalizing dates, amounts, currency, whitespace and case; a
missing or wrong citation triggers a search of the page. Each value ends in one of five states:

| `GroundingState` | Meaning |
|---|---|
| `Grounded` | The value is the text of words on the page. Boxes provided. |
| `GroundedNormalized` | Equal to words on the page after normalization (for example "Oct 3" and `2026-10-03`). Boxes and `NormalizedValue` provided. |
| `Derived` | The model says it computed or inferred the value. Never boxed; confidence is high only when a sum, difference or product of the cited amounts reproduces it. |
| `Ungrounded` | Nothing on the page supports it: a possible hallucination. |
| `RoleConflict` | The value is on the page, but a local extraction model reads those words as a different field. Counts like `Ungrounded`. |

Ungrounded values are re-asked once in a second call that shows the mismatch (counted against
your spend), then `UngroundedPolicy` applies: `Flag` (default: keep and mark), `Null` or `Throw`.
The rules are in [docs/spec/grounding.md](docs/spec/grounding.md). One risk grounding cannot
remove, and the spec says so: a value that is printed on the page but sits in the wrong role can
ground. The role check catches this when a local extraction model is available.

**Cost controls.** An estimate before you call, per-call and per-client USD caps from a dated
price table you can replace (a warning appears when it is older than 180 days), a concurrency
limit, bounded retries with backoff, and an opt-in result cache keyed by image, prompt, model and
schema.

<!-- sample: Estimate -->
```csharp
LlmEstimate estimate = llm.Estimate([new LlmPageInput(image, page)], "Read the invoice.", schemaJson);
Console.WriteLine($"about {estimate.InputTokens} input tokens, at most ${estimate.WorstCaseCostUsd}");
```

**Routing.** `LlmRouting.LlmOnly` (default), `LocalOnly`, or `LocalFirst`: the local extraction
model answers the fields it is confident about and the model is called, with a reduced schema, only
for the rest. When every field is local, no call is made and nothing is sent.

<!-- sample: LocalFirst -->
```csharp
LlmReadResult result = await llm.ReadAsync(image, page, new LlmReadOptions
{
    Outputs = LlmOutputs.Json,
    JsonSchema = schemaJson,
    Routing = LlmRouting.LocalFirst,                          // the model is called only for fields the local model is unsure of
    LocalExtractor = new DocumentFieldExtractor(extractor),
    ConfidenceThreshold = 0.9f,
});
Console.WriteLine($"{result.Calls} call(s), ${result.CostUsd}");
```

**Privacy.**

- Nothing leaves the machine unless you configure a provider; local methods never touch the
  network, even with a provider configured. There is no telemetry.
- API keys are passed explicitly or through an explicit `FromEnvironment()` call; nothing reads
  environment variables behind your back. Keys are kept in a type that never prints them and are
  scrubbed from error messages.
- `LlmOptions.Offline = true` makes every call throw.
- A redaction hook masks regions before upload, and masked words are left out of grounding:

<!-- sample: Redaction -->
```csharp
var options = new LlmOptions
{
    Offline = false,                                          // true makes every call throw OcrProviderException
    Redactor = (page, pageNumber) =>                          // boxes to mask before upload; masked words are not grounded
        [.. page.Words.Where(w => w.Text.StartsWith("DE", StringComparison.Ordinal) && w.Text.Length > 15).Select(w => w.Box)],
};
using var llm = new LlmClient(OpenAiProvider.FromEnvironment("your-model-name"), options);
LlmReadResult result = await llm.ReadAsync(image, image.Ocr(engine), new LlmReadOptions { Outputs = LlmOutputs.Markdown });
```

- `LlmProvider.NoRetention` is on by default and sends `store: false` to OpenAI. Anthropic, Gemini
  and Azure OpenAI have no per-request switch, so retention there is set by your account
  agreement with them.
- Document text is sent as delimited, untrusted data and the model gets no tools. Grounding stops
  invented values; the remaining risk (the model picking a different printed number) is described
  in the grounding spec.

## Training

Training in C# is fine-tuning from the shipped weights, on the CPU, with our own backward passes
(convolution with folded batch norm, ReLU, max pooling, bidirectional LSTM, linear layers, CTC and
the extraction transformer), AdamW and checkpoints. Given the same labels, settings and seed, the
weights are the same on every machine and at every degree of parallelism, and a run resumed from a
checkpoint ends with exactly the weights an uninterrupted run would have. Base models are trained
in PyTorch by [`training/`](training/); every layer's forward and backward pass is tested against
PyTorch references.

**Recognition.** The local reader reads each page and the model you choose transcribes it. A line
becomes training data only when the two agree; disagreements go to a review queue you can export,
correct and import.

<!-- sample: TrainRecognizer -->
```csharp
var trainer = new OcrTrainer(new OcrTrainerOptions { ModelPath = "recognizer.onm", WorkDirectory = "run1" }, llm);
TrainingReport report = await trainer.RunAsync("scans/");
Console.WriteLine(OcrTrainer.ToJson(report));

LineReview.Export("run1", "review/");                       // lines the two readers disagreed on, with crops
// ...set "text" on the lines you checked in review/review.json, then:
LineReview.Import("run1", "review/review.json");
```

**Extraction.** Give a schema; the model reads your pages for its fields; only values that are
found on the page become labels (a page with a value that is not on the page is left out
entirely); a new tag head is trained on the shipped backbone, which is tuned gently along with it.

<!-- sample: TrainExtraction -->
```csharp
ExtractionSchema schema = ExtractionSchema.From(OrderContext.Default.PurchaseOrder);   // top-level scalar properties are the fields
var trainer = new ExtractionTrainer(
    new ExtractionTrainerOptions { ModelPath = "docextract.onm", Schema = schema, WorkDirectory = "po-run" }, llm);
ExtractionReport report = await trainer.RunAsync("purchase-orders/");

DocumentExtractor tuned = DocumentExtractor.Load(report.ModelPath);
```

Every run holds out a share of the documents (by content hash) and reports error rates or
per-field accuracy before and after, agreement with the labelling model, and a suggested
`LocalFirst` threshold. The held-out references are the labelling model's text or grounded values
(or your corrections), so the numbers measure agreement with them; read a few pages yourself before
you rely on them. The same runs are available from the command line:

```sh
oneocr train --model recognizer.onm --work run1 --docs scans/ \
    --provider openai --llm-model <model> --key-from-env OPENAI_API_KEY --budget-usd 2
oneocr review export --work run1 --out review/
oneocr review import --work run1 --file review/review.json
oneocr train --model recognizer.onm --work run1 --restart     # trains on the labels already in run1; no network
oneocr eval  --model run1/model.onm --work run1
oneocr train --extract schema.json --model docextract.onm --work po-run --docs purchase-orders/ \
    --provider openai-compatible --base-url http://localhost:8080/v1 --llm-model <local model>
```

`eval`, `review` and `train` without `--docs` never touch the network. `oneocr help` lists every
option.

**Provider terms.** Using a provider's output to train a model is governed by your own agreement
with that provider. As read on 2026-10-23: Anthropic's Commercial Terms of Service (section D.4)
bar using the services "to build a competing product or service, including to train competing AI
models"; OpenAI's Services Agreement bars using output "to develop artificial intelligence models
that compete with OpenAI's products and services", except a model "primarily intended to categorize,
classify, or organize data (e.g., embeddings or classifiers), if these models are not distributed
or made commercially available to third parties". A model you train for your own internal
extraction may fall under that exception; check your own agreement. A local open-weight model
behind the OpenAI-compatible provider avoids the question.

## Determinism

Text, boxes and confidences hash the same on every kernel path, every operating system and every
degree of parallelism. This is a design rule, not luck:

- Dot products and reductions use fused multiply-add with a fixed 16-lane accumulation order:
  element *k* always lands in partial sum *k mod 16*, whatever the vector width, and the 16
  partials reduce in one fixed tree. The scalar path uses `Math.FusedMultiplyAdd`; hardware without
  FMA takes the same scalar path (exact, slower).
- `exp`, `tanh`, `sigmoid`, `log` and softmax are our own polynomial implementations with a fixed
  operation order, never the platform's math library.
- Parallelism splits independent outputs; no sum crosses threads. Weights are stored as half
  precision and computed in single precision.
- Preprocessing, decoding and the page model are written down in [docs/spec/](docs/spec/) so another
  implementation can reproduce them.

Tests check every vector path against the scalar reference by hash, over randomized shapes
including odd sizes and tails, at degree 1 and at the default degree.

<!-- sample: KernelPath -->
```csharp
Console.WriteLine(engine.KernelPath);                       // Avx512, Avx2, Vector128 or Scalar: the text is the same on all four
if (engine.PerformanceNotice is { } notice)
{
    Console.WriteLine(notice);                              // the scalar path is running on a CPU that could do better
}
```

## Limits and security

> Nothing leaves the machine unless you configure a provider; local methods never touch the network.

- **Limits before allocation.** `OcrLimits` bounds image pixels (134 million by default), pages per
  call, lines, words and characters per page, model file and tensor sizes, page-JSON size, LLM
  response size, schema size, JSON depth and review-import size. Each bound is checked against the
  declared or running value before the work it guards, and a breach is an `OcrLimitException` that
  names the limit. Recursion never follows input depth.

<!-- sample: Limits -->
```csharp
var limits = new OcrLimits { MaxImagePixels = 50_000_000, MaxLinesPerPage = 1000 };
OcrEngine engine = OcrEngine.Load(modelPath, limits);       // OcrLimitException before anything large is allocated
```

- **Untrusted bytes.** Model containers, images, page JSON, PDFs, LLM responses, JSON Schemas and
  review files are parsed by safe, bounds-checked code with checked arithmetic on every size and
  offset taken from the input, and each has a fuzz target under [`fuzz/`](fuzz/README.md). No crash
  or hang on any input is a release requirement.
- **`unsafe` only in the kernels.** `unsafe`, `Unsafe.*` and `MemoryMarshal` appear only under
  `src/OneOcr/Kernels/`, over buffers that safe callers validated.
- **No networking in the core package.** `HttpClient` appears only in `RecursiveAI.OneOcr.Llm`.
  There is no telemetry.
- **No runtime dependencies** beyond the .NET base class library, OneImage (pixels) and, for the
  PDF package, OnePDF and OneImage.Pdf.

## NativeAOT

The libraries and the tool are trim- and AOT-clean (no reflection, no runtime code generation).
A NativeAOT binary is compiled for one instruction-set level, and .NET's default for x64
(`x86-64-v2`) has no AVX2 or FMA. At that level the kernels fall back to their exact scalar
path: the text is the same, but a page takes about a hundred times longer. The
`RecursiveAI.OneOcr` package therefore sets `IlcInstructionSet` to `x86-64-v3` (AVX2 and FMA, any
x86-64 CPU since about 2015) for x64 NativeAOT publishes whose project has not chosen a level; a
value in your project or on the command line always wins. Use `x86-64-v4` for the AVX-512 kernels.
A binary built for v3 refuses to start on an older CPU. Arm needs nothing.

To see what a process runs, read `OcrEngine.KernelPath`; `OcrEngine.PerformanceNotice` is non-null
when the scalar path is running on a CPU that could run AVX2 and says how to fix it (an app
published below v3, or `DOTNET_EnableAVX2=0` / `DOTNET_EnableHWIntrinsic=0` in the environment).
It never throws and is not part of the page, so page JSON is the same on every path.

The `oneocr` tool can be published the same way for fast starts in scripts:
`dotnet publish src/OneOcr.Tool -c Release -r linux-x64` (see [src/OneOcr.Tool](src/OneOcr.Tool/README.md)).

## Models and licences

Code and model weights are licensed `MIT OR Apache-2.0`. Weights are stored as half precision in
our own versioned container format ([docs/spec/model-container-v1.md](docs/spec/model-container-v1.md)),
which is the contract between the PyTorch exporter and the C# reader; user-trained models use the
same format.

| Model | Role | Architecture | Size | Package |
|---|---|---|---|---|
| `crnn-v0` 0.6.2 | Line recognizer: 622 classes (Latin, Greek, Cyrillic, punctuation, currency), per-script calibrated confidence | convolutions, 2-layer BiLSTM, CTC; 2.5 M parameters | 5.0 MB | `.Models` |
| `dbnet-lite-v1` 0.2.2 | Page detector: text lines (oriented), page orientation, layout classes | DBNet-style, 0.7 M parameters; [spec](docs/spec/detector-v1.md) | 1.4 MB | `.Models` |
| `charlm-v1` | Character n-gram model for word breaks and optional beam search; [spec](docs/spec/char-lm-v1.md) | interpolated Kneser–Ney, Latin + Greek + Cyrillic | 1.4 MB | `.Models` |
| `docextract-v0` | Invoice and receipt word tagger; [spec](docs/spec/docextract-v0.md) | character-CNN word embedding, 2-D position embedding, small transformer | about 4 MB | `.Models.Documents` |

Training-data attributions and font licences are in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## How the models were trained

Everything is reproducible from [`training/`](training/) (a Python project; pinned fetch scripts,
generator seeds and configs). Weights start from random initialization or from our own earlier
checkpoints; no third-party pretrained weights are used, and no weights, images or training data are
committed to the repository.

- **Recognizer.** Text lines rendered from public-domain books (Project Gutenberg) in 77 open font
  families (SIL OFL, Apache-2.0 and Ubuntu Font Licence), in Latin, Greek and Cyrillic, then degraded
  (blur, noise, JPEG, low resolution, faint ink, perspective, uneven light) with exact ground
  truth; plus word crops of the CORD v2 receipts training split with its labels and synthetic
  receipt and interface text. Held-out font families and held-out books form the evaluation sets.
- **Detector.** Page images and text-line boxes from DocLayNet and PubLayNet (only articles under
  CC0 and CC BY), CORD v2 word quads, and synthetic pages rendered in Chromium with exact word
  boxes.
- **Extraction.** Generated invoices and receipts with exact field truth, plus CORD v2 training
  labels.
- **Labels.** Only these may label data for shipped weights: exact-truth synthetic data,
  dataset-provided labels (CORD, DocLayNet, PubLayNet) and permissively licensed open-weight
  models. Commercial LLM APIs are never used to label training data for distributed weights; they
  may appear only as benchmark rows for the LLM layer. Nobody hand-labels images.
- **Clean room.** Other OCR systems were studied only through their papers, public documentation and
  the output of their released binaries. Their source code and model files were never read or used.

## Benchmarks

**Status: the tables below are placeholders (`TBD`) until the release gate has run.** Nothing here
is estimated; each number will come from one run of the gate on one machine, with the CPU, core
count and load, .NET SDK and runtime, and each engine's version and settings printed beside it.

*Accuracy, error rate %, CER / WER / reading order. Lower is better; the best engine per row is
marked in bold.*

| Category | OneOcr | Tesseract (best) | Tesseract (fast) | PaddleOCR | docTR | Apple Vision |
|---|---|---|---|---|---|---|
| Clean 300 DPI print | TBD | TBD | TBD | TBD | TBD | TBD |
| Degraded scans | TBD | TBD | TBD | TBD | TBD | TBD |
| Screenshots and UI | TBD | TBD | TBD | TBD | TBD | TBD |
| Complex layout | TBD | TBD | TBD | TBD | TBD | TBD |
| Phone photos | TBD | TBD | TBD | TBD | TBD | TBD |
| Receipts (recall-CER / recall-WER) | TBD | TBD | TBD | TBD | TBD | TBD |
| Real web screenshots | TBD | TBD | TBD | TBD | TBD | TBD |
| Real born-digital PDFs: clean | TBD | TBD | TBD | TBD | TBD | TBD |
| Real born-digital PDFs: layout | TBD | TBD | TBD | TBD | TBD | TBD |
| Real PDF pages, scan-degraded | TBD | TBD | TBD | TBD | TBD | TBD |

The last four rows use real pages (rendered web pages and PDFs with a verified text layer); they
are reported apart from the synthetic rows because their truth and their difficulty differ.

*Calibration and detection.*

| Category | ECE (target ≤ 0.05) | Word-box F1 at IoU 0.5 |
|---|---|---|
| each category above | TBD | TBD |

*Speed and memory, whole process per page (start-up and model load included), NativeAOT build.*

| Category | OneOcr, 1 thread | OneOcr, 4 threads | Tesseract fast | Tesseract best | OneOcr peak RSS (target ≤ 300 MB) |
|---|---|---|---|---|---|
| each category above | TBD | TBD | TBD | TBD | TBD |

**Method.**

- **Metrics.** CER and WER are order-free: each truth word is paired with its closest leftover
  output word. Reading-order error is the edit distance over the word sequence divided by the
  number of truth words (it includes recognition errors, so read it next to WER). Text is
  normalized by NFC, typographic quotes and dashes folded to ASCII, ligatures split, whitespace
  collapsed; case and punctuation are kept. Receipts have partial truth (CORD leaves store and
  payment blocks unannotated), so they are judged on recall-CER and recall-WER, the error over truth
  words only. Word-box F1 uses boxes tight to the printed ink. ECE uses 15 equal-width bins over
  word confidence, a word being correct when it matches its truth exactly.
- **Truth.** Never produced by an OCR engine: browser layout geometry, the geometry of our own
  transforms, a dataset's shipped annotation, or a PDF's text layer checked by two independent
  readers. Pages and truth are hashed in manifests; the generators and seeds are in
  [`bench/`](bench/README.md).
- **Engines.** Tesseract is the distribution's 5.3.4 binary with its own `tessdata_best` and
  `tessdata_fast` models, run as a black box; for each metric the best of page-segmentation modes
  3, 4, 6 and 11 is taken (the hardest baseline). PaddleOCR (PP-OCRv5 mobile and server, PP-OCRv6),
  docTR (default and mobile) and Apple Vision (accurate) run as released packages with their stock
  models. Their output is a comparison row, never training data.
- **Rule.** An engine is judged against the best value any measured engine reaches per metric. On
  clean print, an error within 0.2 percentage points of the best is reported as a tie.
- **Speed.** One process per run, the shipped `oneocr ocr` published NativeAOT for the host, three
  runs, mean; peak RSS from the operating system. Tesseract is timed the same way at one thread and
  at the same thread count as OneOcr.
- **Reproduce.** `python -m oneocr_bench gate` in [`bench/`](bench/README.md) prints the quality
  and calibration table for an evaluation directory; `bench/scripts/full_run.sh` runs the whole
  comparison on one machine.

## Not yet

Searchable-PDF output and int8 inference (planned after 1.0); CJK, Arabic and Indic scripts (the
page model already has right-to-left and vertical flags, but the local reader does not produce
them); handwriting; curved-page dewarping and scene text; text drawn as outlined glyphs in PDFs;
AWS Bedrock and Google Vertex providers; a built-in PII redactor (the redaction hook is yours to
fill); a word-level character whitelist.

## Where to go

- **Formats and rules** ([docs/spec/](docs/spec)): the [model container](docs/spec/model-container-v1.md),
  [page JSON](docs/spec/oneocr-page-v1.md), [numeric rules](docs/spec/numerics.md),
  [decoding](docs/spec/decoding.md), [text normalization](docs/spec/text-normalization-v1.md),
  [Markdown](docs/spec/markdown-v1.md), [tables](docs/spec/table-structure-v1.md),
  [orientation](docs/spec/page-orientation-v1.md), [grounding](docs/spec/grounding.md),
  [PDF bridge](docs/spec/pdf-bridge.md), [review queue](docs/spec/review-queue.md).
- **Migrating:** [from Tesseract](docs/migrating-from-tesseract.md).
- **Evidence:** [benchmark tooling](bench/README.md), [fuzzing](fuzz/README.md).
- **The API reference** is the XML documentation your IDE shows.

## License

Code and model weights are licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option.

Copyright (c) Recursive AI, LLC. Third-party data and fonts: [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
