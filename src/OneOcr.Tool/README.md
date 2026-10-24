# oneocr

The command-line tool for fine-tuning a OneOcr recognizer on your own documents.

```
dotnet tool install RecursiveAI.OneOcr.Tool
oneocr train --model recognizer.onm --work run1 --docs scans/ \
    --provider openai --llm-model <model> --key-from-env OPENAI_API_KEY --budget-usd 2
oneocr review export --work run1 --out review/
# edit review/review.json: set "text" on the lines you checked
oneocr review import --work run1 --file review/review.json
oneocr train --model recognizer.onm --work run1 --restart  # trains on the labels already in run1; no network
oneocr eval  --model run1/model.onm --work run1

# a custom document type: schema.json is a JSON Schema whose top-level scalar properties are the fields
oneocr train --extract schema.json --model docextract.onm --work po-run --docs purchase-orders/ \
    --provider openai-compatible --base-url http://localhost:8080/v1 --llm-model <local model>
```

`train --extract` fine-tunes the document-extraction model for your schema from values the labelling model found *on the page* (the report lists held-out accuracy before and after). `train` with `--docs` sends page images to the provider you name, under your own agreement with it. API keys come only from `--key` or from the environment variable you name with `--key-from-env`; nothing is read from the environment otherwise. `eval`, `review export|import` and `train` without `--docs` never touch the network. Using a provider's output to train a model is governed by your own agreement with it; a local open-weight model behind the OpenAI-compatible provider avoids the question. Run `oneocr help` for every option, and see the RecursiveAI.OneOcr.Training package for how lines are chosen for training and for a note on the terms of the providers you label with.
