# Review queue and fine-tuning work directory, version 1

Status: current. The recognition fine-tuner (`OcrTrainer`, `oneocr train`) keeps everything in one work directory, and the review queue is a pair of JSON files around it.

## Agreement rule

A page's lines come from the local reader (line finder, then the recognizer with greedy decoding). The labelling model's transcription of the page is split at whitespace. Walking the lines top to bottom, each line is compared (NFC, whitespace collapsed, case and punctuation kept) with every stretch of the model's words that starts within 80 words of the end of the previous match and is within 3 words of the line's own length; the stretch with the lowest character error rate wins (the earliest on ties). Character error rate is edit distance in code points over the line's length.

- **Agreed**: the best rate is at most `MaxAgreementCer` (default 0: the same text). The model's text is the training label.
- **Disagreed**: the best rate is above `MaxAgreementCer` but at most `AlignMaxCer` (default 0.5). The line goes to the review queue.
- **Unaligned**: no stretch is within `AlignMaxCer`, or the model gave no text. The line goes to the review queue.

Only agreed lines and lines a reviewer corrected train the model. Lines whose label has a character outside the model's charset are left out and counted.

## Work directory

`labels/<id>.json` (`oneocr.labels`, version 1): one file per document, `<id>` = first 16 hex digits of the SHA-256 of the image file. Holds each line's box, local reading, confidence, the matched model text, its state and the crop the recognizer saw (8-bit gray at the model's input height, base64). `reviewed.json` (`oneocr.reviewed`): the corrections by line id. `checkpoint.bin`: step, weights and AdamW moments (fp32, little-endian), tied to the run by a SHA-256 over the starting model, the settings and every label. `report.json`: the last run's report. A line id is `<document id>-<line index>`.

## Review file (`oneocr.review`, version 1)

```json
{ "format": "oneocr.review", "version": 1,
  "items": [ { "id": "0123456789abcdef-3", "document": "scan.png", "box": [10, 20, 300, 30],
               "local": "Tota1 due", "llm": "Total due", "state": "disagreed", "confidence": 0.81,
               "crop": "crops/0123456789abcdef-3.png", "text": null } ] }
```

The export writes this file and one PNG per line. A reviewer sets `text`. The importer reads only `id` and `text` and skips everything else. It is untrusted input: the whole file must be valid UTF-8 and at most `MaxReviewImportBytes` (16 MiB); there are at most `MaxReviewItems` items (100,000); `MaxJsonDepth` bounds nesting; a text is at most `MaxReviewTextLength` (1,024) UTF-16 units and, like an id (at most 128, not empty), holds no control characters and no U+FFFD; `items` appears once. A `text` that is missing or `null` leaves the line in the queue; an empty `text` says the line has no text and drops it from training. Ids that name no line of the work directory are ignored and counted. Text is stored with whitespace normalized.

## Training schedule

Batches are drawn from a seeded (SplitMix64) permutation of the training lines, reshuffled each pass. The learning rate rises linearly over the warm-up and then falls linearly to a tenth of the peak at the last step. Batches, schedule, loss and weights depend only on the seed, the labels and the settings, so a resumed run ends with exactly the weights of an uninterrupted one.
