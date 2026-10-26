# Moving from Tesseract to OneOcr

This guide maps what you do with the `tesseract` command line (or a wrapper around it) to the OneOcr API. It is based on
Tesseract's public documentation (its manual pages and `--help` output) and on its observable behaviour, nothing else.
Where a setting has no counterpart, the table says so rather than inventing one.

The short version: OneOcr is a managed library, so there is no executable to install, no `tessdata` folder to point at, and no
language code to choose for Latin, Greek and Cyrillic text. You load an engine once, hand it an image and read the page
model.

<!-- sample: TesseractText -->
```csharp
// tesseract page.png out -l eng            ->  out.txt
string text = Image.Load("page.png").ToText(engine);
```

Every code block in this guide compiles; the samples live in `tests/OneOcr.Samples/`.

## Concepts side by side

| Tesseract | OneOcr |
|---|---|
| `tesseract in.png out` | `image.ToText(engine)` for text, `image.Ocr(engine)` for the page model |
| `tessdata`, `TESSDATA_PREFIX`, `-l eng` | None. `OcrEngine.LoadBundled()` loads the recognizer, page detector and word model from `RecursiveAI.OneOcr.Models`. `OcrEngine.Load(path)` reads a model container you trained or fine-tuned |
| `tessdata_fast` / `tessdata_best` | One bundled set of models. There is no speed/accuracy switch to flip |
| Languages: `-l eng+deu`, `-l ell`, `-l rus` | One recognizer reads Latin (English, German, French, Turkish, Vietnamese and the other European languages), Greek and Cyrillic together, so mixed text needs no language list. CJK, Arabic and Indic scripts are not supported yet |
| Page segmentation modes `--psm 0`–`13` | No modes. The page detector finds text lines and the layout code orders them into columns, tables and blocks. See "Page segmentation modes" below |
| `--oem` | Not applicable: there is one engine |
| `--dpi N` | `OcrOptions.Dpi`. It only sets the page's `UnitsPerInch`; recognition does not depend on it. When it is absent the DPI is estimated and a diagnostic says so |
| `OMP_THREAD_LIMIT` | `OcrOptions.MaxDegreeOfParallelism` (0 means min(4, cores)). The result is bit-identical at every setting |
| `--psm 0` (orientation and script detection) | `OcrOptions.DetectOrientation` (on by default; 0, 90, 180 and 270 degrees) and `OcrOptions.Deskew` (on by default). The result is in `OcrPage.Orientation` and `OcrPage.SkewDegrees` |
| `-c tessedit_char_whitelist=...` | No whitelist. Filter the words you get back |
| Dictionaries, `load_system_dawg`, user words | No dictionary by default, so identifiers and codes are never "corrected". Beam search with a character language model and a lexicon is opt-in (`DecodeOptions`) |
| `preserve_interword_spaces` | Not applicable. Word breaks are decided per line from the pixels, and a line's text is its words joined by one space; cells of a table row are joined by a tab |
| Output `txt` | `image.ToText()`, or `OcrPage.Text` with blank lines between blocks |
| Output `tsv` | Walk `page.Blocks[i].Lines[j].Words`, see the recipe below |
| Output `hocr` | `OcrPage` → `OcrBlock` → `OcrLine` → `OcrWord`, serialized as `oneocr.page` JSON |
| Output `pdf` (searchable PDF) | Not yet |
| Output `alto`, `lstmbox`, `makebox` | Not provided |

## Page segmentation modes

Tesseract asks you to say what the page looks like (a single block, a single line, a single word, sparse text), and the
best mode differs from page to page. OneOcr takes no such hint.

| Tesseract mode | In OneOcr |
|---|---|
| 3, fully automatic (the default), 1, 2, 4 | The normal path: `image.Ocr(engine)`. Columns, headers, footers and tables are found for you |
| 0, orientation only | `DetectOrientation` and `Deskew` are options of the normal path; there is no call that only detects |
| 5, 6, a single block | The normal path. Text boxes and screenshots read the same as pages |
| 7, 13, a single line | Pass the crop to `Ocr` with orientation and deskew off, see the recipe below |
| 8, 10, a single word or character | Same as a single line; the result has one word |
| 11, 12, sparse text | The normal path. The detector looks for text lines anywhere on the page, not only in blocks |

If you used a mode only because the default read your page badly, try the default first: it is the one OneOcr is built and
measured on.

## Confidence

| | Tesseract | OneOcr |
|---|---|---|
| Scale | Per word, 0 to 100 (TSV `conf`, hOCR `x_wconf`) | Per word, 0 to 1 (`OcrWord.Confidence`); lines, blocks and the page aggregate it |
| Meaning | A score from the recogniser's internals | The calibrated probability that the word is exactly right |
| Calibration | None documented; thresholds are tuned per project | Fitted per script on held-out data. Words at 0.9 are right about 90% of the time. The expected calibration error is published per category in the README |
| Raw value | – | `OcrWord.RawConfidence` is the uncalibrated mean over the word's characters |

Because the number is a probability you can set one threshold and expect it to mean the same thing on every document. To
move an existing threshold over, divide it by 100 as a starting point and then look at the words just below it:

<!-- sample: LowConfidence -->
```csharp
foreach (OcrWord word in image.Ocr(engine).Words.Where(w => w.Confidence < 0.8f))
{
    Console.WriteLine($"check: {word.Text} ({word.Confidence:P0}) at {word.Box.Left:F0},{word.Box.Top:F0}");
}
```

## Output formats

### Plain text

`image.ToText(engine)` returns the page in reading order: lines are separated by a newline and table cells by a tab; `page.Text` also puts a blank
line between blocks. Reading order is worked out from the layout, so two-column pages come out column by column.

### TSV

Tesseract's TSV has one row per page, block, paragraph, line and word, with `left top width height conf text`. The word rows
map directly onto `OcrWord`. Boxes are top-left origin, y down, in pixels for images (in points for PDF pages), as
Tesseract's are.

<!-- sample: WordTable -->
```csharp
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
```

### hOCR

hOCR describes a page as nested elements: `ocr_page`, `ocr_carea` (a block of the page), `ocr_par`, `ocr_line` and
`ocrx_word`, each with a `bbox` and words with `x_wconf`. The page model has the same nesting:

| hOCR | Page model |
|---|---|
| `ocr_page` (`bbox`) | `OcrPage` (`Width`, `Height`) |
| `ocr_carea` / `ocr_par` | `OcrBlock` (role body, header or footer; `IsTable` for grids) |
| `ocr_line` | `OcrLine` |
| `ocrx_word`, `x_wconf` | `OcrWord`, `Confidence` |
| `ocrx_cinfo` | `OcrChar`, only with `OcrOptions.IncludeChars`; character boxes are approximate and flagged as such |
| `bbox x0 y0 x1 y1` | `OcrBox` (`Left`, `Top`, `Right`, `Bottom`) |
| `textangle` | `OcrQuad` on rotated elements; `OcrPage.SkewDegrees` for the page |

Nothing in the library writes hOCR. If you have a consumer that needs it, the walk above is all it takes. For storage and
exchange, use `oneocr.page` JSON, which is documented in [docs/spec/oneocr-page-v1.md](spec/oneocr-page-v1.md) and reads
back with `OcrPageJson.Parse`.

### PDF

OneOcr does not write searchable PDFs yet. It does read PDFs: `PdfOcr.Read` uses a PDF's own text where it has any and reads
the scanned parts (see the README).

## Recipes

### Many files at once

The engine is immutable and thread-safe, so one engine serves every thread. Run files in parallel and leave each page at its
default degree, or lower the degree per page when you run many pages at once.

<!-- sample: Batch -->
```csharp
Parallel.ForEach(
    Directory.EnumerateFiles(folder, "*.png"),
    new ParallelOptions { MaxDegreeOfParallelism = 4 },
    path => File.WriteAllText(Path.ChangeExtension(path, ".txt"), Image.Load(path).ToText(engine)));
```

### One line, one word

Where you used `--psm 7` or `--psm 8`, pass the crop to `Ocr`. Turning orientation and deskew off stops the reader second
guessing a short strip.

<!-- sample: OneLine -->
```csharp
// A crop that is one line of text: the page reader finds that line and reads it.
OcrPage page = Image.Load(cropPath).Ocr(engine, new OcrOptions { DetectOrientation = false, Deskew = false });
string line = page.Text;
float confidence = page.Words.Count == 0 ? 0 : page.Words.Min(w => w.Confidence);
```

### JSON for another program

Where you ran `tesseract in out hocr` or `tsv` for a downstream step, write the page model instead:

<!-- sample: PageJson -->
```csharp
string json = OcrPageJson.ToJson([page], new OcrPageJsonOptions { Indented = true });
IReadOnlyList<OcrPage> again = OcrPageJson.Parse(json);   // checked against OcrLimits; throws OcrFormatException on bad input
```

### Tables, key–value pairs, Markdown

These replace the post-processing most Tesseract pipelines grow. `page.Tables` holds cells with row and column spans that
refer to words, `image.ToMarkdown(engine)` renders headings, lists and pipe tables, and `image.ExtractInvoice(...)`,
`ExtractReceipt` and `Extract<T>` return typed fields with their boxes. The README has the details.

## What is different

- **No native binaries.** The library is managed code over the .NET base class library, so it runs wherever .NET 10 does,
  including NativeAOT, with no `libtesseract`, no `leptonica` and no data files to deploy beside the application.
- **Deterministic.** Text, boxes and confidences are bit-identical on AVX-512, AVX2, 128-bit vector and scalar hardware, on
  every operating system, and at every degree of parallelism. `engine.KernelPath` says which path is running.
- **Calibrated confidence.** See above.
- **A page model, not a flat result.** Blocks, lines, words, tables, key–value pairs and diagnostics come back in one object,
  with a `Source` on each element, and the same shape is used for text read from a PDF and text read by an LLM.
- **Limits and typed errors.** Image size, lines per page and model sizes are bounded by `OcrLimits` before anything large
  is allocated, and the public methods throw only the typed exceptions in the README.
- **Optional LLM vision, never by default.** `RecursiveAI.OneOcr.Llm` adds grounded reading through a provider you
  configure. The core package has no networking code.

## What is not there yet

Searchable-PDF output, CJK, Arabic and Indic scripts, handwriting, curved-page dewarping, word whitelists and a word
dictionary. See "Not yet" in the README.
