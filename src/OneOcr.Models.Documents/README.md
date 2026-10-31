# RecursiveAI.OneOcr.Models.Documents

The invoice, receipt and key-value extraction model for `RecursiveAI.OneOcr`, embedded and checked against its SHA-256 pin. Install it next to `RecursiveAI.OneOcr` to use `ExtractInvoice()`, `ExtractReceipt()` and `Extract<T>()` offline.

```sh
dotnet add package RecursiveAI.OneOcr
dotnet add package RecursiveAI.OneOcr.Models.Documents
```

The model labels words that are on the page and deterministic code assembles the fields, so a value always has the words and boxes it was read from. The weights are licensed MIT OR Apache-2.0; training data credits are in THIRD-PARTY-NOTICES.md.

https://github.com/recursive-ai-oss/oneocr-csharp
