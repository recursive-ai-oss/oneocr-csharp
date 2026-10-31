# RecursiveAI.OneOcr.Pdf

PDF pages for `RecursiveAI.OneOcr`: words come from the text layer where a page has one, scanned regions and image-only pages are rendered and read by the local reader, and everything merges into one page model with a `Source` on every word. Built on `RecursiveAI.OnePdf` and `RecursiveAI.OneImage.Pdf`.

```sh
dotnet add package RecursiveAI.OneOcr.Pdf
```

Invisible text layers from earlier OCR runs are re-read by default, and garbled text layers are replaced by a fresh reading of the page.

https://github.com/recursive-ai-oss/oneocr-csharp
