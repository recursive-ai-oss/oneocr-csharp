# RecursiveAI.OneOcr

Local, deterministic OCR for .NET. Pure managed code with no dependency beyond the base class library and `RecursiveAI.OneImage`: no native binaries, no network. Text, boxes and confidences are bit-identical on every CPU path and at every degree of parallelism.

```sh
dotnet add package RecursiveAI.OneOcr
```

```csharp
using OneImage;
using OneOcr;

OcrEngine engine = OcrEngine.LoadBundled();     // once; the engine is immutable and thread-safe
Image image = Image.Load("invoice.png");

string text = image.ToText(engine);
OcrPage page = image.Ocr(engine);               // blocks, lines, words, tables, boxes and calibrated confidences
string markdown = image.ToMarkdown(engine);
```

`OcrEngine.ConfigureDefault()` sets a process-wide engine for the extension methods that take none. Every call takes an optional `OcrOptions` (an engine carries its own in `engine.Options`). `Ocr(Image)` reads one page; `OcrPages` reads every frame of a multi-page file, takes the resolution a PNG, JPEG or TIFF declares, and reports files it cannot read as `OcrFormatException` or `OcrLimitException`. Boxes are always in the image as you gave it, also for pages that came in rotated, and `page.Diagnostics` carry stable codes.

The package holds the reader, the page model and its `oneocr.page` JSON, local Markdown, and inference for the invoice, receipt and custom extraction models. The weights come with `RecursiveAI.OneOcr.Models`, which this package depends on. LLM vision, training, PDF pages and the command-line tool are separate opt-in packages.

Full documentation, benchmarks and the format specifications: https://github.com/recursive-ai-oss/oneocr-csharp

Licensed under MIT OR Apache-2.0. Third-party notices are in THIRD-PARTY-NOTICES.md.
