# RecursiveAI.OneOcr.Models

The weights for `RecursiveAI.OneOcr`: the line recognizer (Latin, Greek and Cyrillic), the page detector, and the character and word language models. They are embedded in the assembly and checked against their SHA-256 pins every time they are loaded. Most programs get this package through `RecursiveAI.OneOcr` and never reference it directly.

```sh
dotnet add package RecursiveAI.OneOcr.Models
```

`BundledModels` reports which models are pinned and which are present. The weights are licensed MIT OR Apache-2.0 like the code; the data they were trained on is credited in THIRD-PARTY-NOTICES.md.

https://github.com/recursive-ai-oss/oneocr-csharp
