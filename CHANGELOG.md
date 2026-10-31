# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-30

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
