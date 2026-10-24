# RecursiveAI.OneOcr.Training

Fine-tune the OneOcr line recognizer on your own documents.

```csharp
var llm = new LlmClient(new OpenAiProvider(key, "your-model"), new LlmOptions { SpendCapUsd = 2m });
var trainer = new OcrTrainer(new OcrTrainerOptions { ModelPath = "recognizer.onm", WorkDirectory = "run1" }, llm);
TrainingReport report = await trainer.RunAsync("scans/");
Console.WriteLine(OcrTrainer.ToJson(report));
```

How it works:

1. The local reader finds and reads the text lines of each page image; the language model you configured transcribes the page.
2. A line becomes training data **only when the two agree** (by default: the same text after whitespace is normalized; `MaxAgreementCer` allows a small difference). Lines they read differently, or that the model's text has no match for, go to the **review queue**.
3. The model is fine-tuned from the weights you start with (CPU only, deterministic: same labels and settings give the same weights on every machine and at every thread count). It is resumable: run it again with the same work directory and it continues from the last checkpoint, ending with exactly the weights an uninterrupted run would have.
4. A fraction of the documents is held out. The report gives character and word error rates of the starting and the tuned model on them, the agreement rate, and a suggested line-confidence threshold for `LocalFirst` routing. Held-out references are the language model's text (or your correction), so the numbers measure agreement with that text, not with the page.

Review queue (a data API, no UI): `LineReview.Export` writes `review.json` plus a picture of each line; set `text` on the items you checked and call `LineReview.Import`. Corrected lines train the next run. A review file is treated as untrusted input and read under the limits in `OcrLimits` (`MaxReviewImportBytes`, `MaxReviewItems`, `MaxReviewTextLength`).

The tuned model keeps the starting model's confidence calibration.

## Whose terms apply

The labels come from the provider you choose, under your own agreement with that provider. Some providers' terms limit using their output to build models: OpenAI's, for example, forbids using output to develop competing models but exempts models primarily meant to classify or organize data when they are not distributed or made available to others, which may cover a recognizer you keep for your own use. Other providers' terms differ, and some forbid it outright. Read the terms of the service you use before you train on its output. Labelling with an open-weight model you run yourself (through the OpenAI-compatible provider) avoids the question. OneOcr does not decide it for you.

Nothing leaves the machine unless you pass a labelling client; training, evaluation and review import never touch the network.
