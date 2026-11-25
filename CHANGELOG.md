# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Searchable PDFs** (`RecursiveAI.OneOcr.Pdf`): `image.ToSearchablePdf(...)`, `SearchablePdf.FromImages/FromFiles/FromData` (many images or a multi-frame TIFF into one PDF) and `SearchablePdf.MakeSearchable` (PDF in, PDF out) put the page model's words on the page as an invisible text layer, so selection, search and copy match what the page shows. A JPEG is placed as its own bytes, bilevel pages stay 1-bit, nothing is re-encoded lossily, and rendering is unchanged. `MakeSearchable` leaves pages that already have trusted text exactly as they were, and replaces an invisible layer from an earlier OCR step instead of stacking another on it. PDF/A-2b output is an option. Output is deterministic. `PdfTextLayerWriter` is the low-level writer for one `OcrPage`. Specified in `docs/spec/searchable-pdf.md`.
- **Running headers and footers on OCR'd pages.** Blocks near the top or bottom edge that repeat across the pages of one call (page numbers count as the same text) get the header or footer role, so local Markdown drops them as it does for text layers. `OcrOptions.MarkHeadersAndFooters` (default on) turns it off.
- **Paragraph splitting.** A block is split at an inter-paragraph gap, a first-line indent, or a short line before a full one; list items never start a split.
- **Hanging-indent labels** (`(a)`, `1.`, clause numbers in a left column) are read together with their clause, not as a separate column.
- **Word and line offsets into text**: `OcrLine.GetWordSpans()`, `OcrPage.GetWordSpans()` and `GetLineSpans()` return `OcrTextSpan` values for the text the page joins.

### Changed

- **Amounts have one canonical form (behaviour change).** Money values always carry at least two decimals: `126` becomes `126.00` and `36.5` becomes `36.50` in extracted fields, in `GroundedField.NormalizedValue` and in local-versus-LLM comparisons. Quantities and percentages keep the shortest exact decimal (`2` stays `2`, `7.50` is `7.5`). Code that compared these strings to whole-number text should compare decimal values.
- **Due date versus delivery date.** On an invoice that prints only a due date, the local model could tag it as the delivery date (`BT-72`). The label printed to the left of the value now decides between `BT-9` and `BT-72`.
- Indented `oneocr.page` JSON keeps bounds, quads and number lists on one line. The compact form is unchanged byte for byte.
- `RecursiveAI.OneOcr.Pdf` builds on `RecursiveAI.OnePdf` 1.25.0 (was 1.10.1).

## [1.0.0] - 2026-10-31

First release.

### Added

- **Local OCR** (`RecursiveAI.OneOcr`): a pure-managed reader for scans, born-digital renders, screenshots and phone photos of flat documents, with no dependency beyond the base class library and `RecursiveAI.OneImage`. Latin (including the EU languages, Turkish and Vietnamese), Greek and Cyrillic in one recognizer. Page orientation (0/90/180/270) and deskew are on by default.
- **Deterministic output.** Text, boxes and confidences are bit-identical on AVX-512, AVX2, 128-bit vectors (SSE and NEON) and the scalar path, and at every degree of parallelism. The numeric rules are specified in `docs/spec/numerics.md`.
- **The page model** (`OcrPage`, blocks, lines, words, characters, tables, key-value pairs, a `Source` and a calibrated `Confidence` on every element) and its versioned `oneocr.page` JSON, which shares member names with OnePDF's `onepdf.extraction`.
- **Calibrated confidence.** A word's confidence is the probability that the word is exactly right, calibrated per script on held-out data.
- **Local Markdown** from the page model: headings, lists, pipe tables, with headers and footers dropped.
- **Document extraction**: `ExtractInvoice()`, `ExtractReceipt()` and `Extract<T>()` over a small local model that labels words on the page, so a value always carries the words and boxes it was read from. Invoice fields carry their EN 16931 business-term ids. The model ships in `RecursiveAI.OneOcr.Models.Documents`.
- **LLM vision** (`RecursiveAI.OneOcr.Llm`): Anthropic, OpenAI, OpenAI-compatible servers, Gemini and Azure OpenAI over a plain `HttpClient`. Every returned value is checked against the words on the page and comes back `Grounded`, `GroundedNormalized`, `Derived` or `Ungrounded`. Cost estimates and caps, routing between local models and the LLM, an `Offline` switch, a redaction hook and no-retention flags. Nothing leaves the machine unless you configure a provider.
- **Fine-tuning** (`RecursiveAI.OneOcr.Training`): recognition and extraction fine-tuning on your own documents, on the CPU, deterministic and resumable, with a review queue for lines the reader and the LLM disagree on.
- **PDF pages** (`RecursiveAI.OneOcr.Pdf`): text-layer words where a page has them, local OCR for scanned regions and image-only pages, one page model.
- **The `oneocr` command-line tool** (`RecursiveAI.OneOcr.Tool`): `ocr`, `train`, `eval` and `review`.
- **The model container**, a versioned binary format for weights, specified in `docs/spec/model-container-v1.md`; the bundled recognizer, page detector and language models are embedded in `RecursiveAI.OneOcr.Models` and checked against their SHA-256 pins on every load.
- Parsers for model files, images, page JSON, PDFs, review files and LLM responses are bounds-checked, run under limits checked before anything large is allocated, and have fuzz targets. NativeAOT and trimming are supported.

### Known limits

- Not supported: CJK, Arabic and Indic scripts, handwriting, curved-page dewarping and scene text.
- The README's benchmark tables list every category where a competing engine reads better than OneOcr.
- A NativeAOT build for x64 needs an instruction-set level of at least `x86-64-v3`; the package sets it by default. Below it the reader runs its exact scalar path, about a hundred times slower, with the same text.
