# RecursiveAI.OneOcr.Pdf

PDF pages for `RecursiveAI.OneOcr`: words come from the text layer where a page has one, scanned regions and image-only pages are rendered and read by the local reader, and everything merges into one page model with a `Source` on every word. Built on `RecursiveAI.OnePdf` and `RecursiveAI.OneImage.Pdf`.

```sh
dotnet add package RecursiveAI.OneOcr.Pdf
```

## Searchable PDFs

A page image keeps its look and gets an invisible text layer, so selection, search and copy in any PDF reader match the words on the page.

```csharp
using OneOcr;
using OneOcr.Pdf;

OcrEngine.ConfigureDefault();

// One image (a decoded OneImage Image) or a multi-frame TIFF: one PDF, one page per frame.
image.ToSearchablePdf("scan.pdf");

// Many files into one PDF. A JPEG is placed as its own bytes, a 1-bit scan stays 1-bit, nothing is re-encoded lossily.
SearchablePdf.FromFiles(["p1.jpg", "p2.jpg", "p3.tif"], "book.pdf");

// A PDF in, a PDF out: pages with trusted text are left alone, scanned pages get a layer,
// and an invisible layer from an earlier OCR step is replaced instead of stacked.
SearchablePdf.MakeSearchable("scanned.pdf", "searchable.pdf");

// PDF/A-2b.
SearchablePdf.FromFiles(["p1.jpg"], "archive.pdf", options: new SearchablePdfOptions { Conformance = SearchablePdfConformance.PdfA2b });
```

The result says how many pages got a layer, how many words were written and what could not be done. The same input, options and engine give the same bytes. `PdfTextLayerWriter` draws one `OcrPage` into any page you already have. Details, tolerances and what is left alone: `docs/spec/searchable-pdf.md`.

Invisible text layers from earlier OCR runs are re-read by default, and garbled text layers are replaced by a fresh reading of the page. Visible text whose character map cannot be trusted is not removed, since it is part of what the page paints; it is wrapped in marked content with an empty `/ActualText` (ISO 32000-1, 14.9.4), so text extraction and copy skip it and return the recognized words instead. Rendering is unchanged.

From the command line, `oneocr pdf <images, folders or one PDF> --out file.pdf [--pdfa]` does the same (`dotnet tool install --global RecursiveAI.OneOcr.Tool`).

A file that opens but has no readable page tree throws `OcrFormatException` with `PdfOcrOptions.Strict`; otherwise `PdfOcr.Read` returns one empty 0 by 0 page whose diagnostics say so, never an empty list.

https://github.com/recursive-ai-oss/oneocr-csharp
